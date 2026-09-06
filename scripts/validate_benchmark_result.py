#!/usr/bin/env python3
"""Fail closed when a benchmark run directory is incomplete or unsuccessful."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REQUIRED = ("manifest.json", "environment.json", "config.resolved.toml", "command.txt", "summary.json", "summary.md", "failures.jsonl")


def validate(root: Path) -> list[str]:
  errors = [f"missing {name}" for name in REQUIRED if not (root / name).is_file()]
  if errors:
    return errors
  summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
  manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
  if summary.get("status") != "ok":
    errors.append(f"status is {summary.get('status')!r}, not 'ok'")
  if any(summary.get("integrity", {}).get(key) for key in ("missing", "unexpected", "duplicates", "empty_hypotheses")):
    errors.append("integrity inventory is non-empty")
  if int(summary.get("num_files", -1)) != len(manifest.get("items", [])):
    errors.append("manifest/result count mismatch")
  if (root / "failures.jsonl").read_text(encoding="utf-8").strip():
    errors.append("failure inventory is non-empty")
  canonical = json.dumps(manifest.get("items", []), sort_keys=True, separators=(",", ":"))
  if hashlib.sha256(canonical.encode()).hexdigest() != manifest.get("manifest_sha256"):
    errors.append("manifest checksum mismatch")
  return errors


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("run_directory", type=Path)
  args = parser.parse_args()
  errors = validate(args.run_directory.resolve())
  if errors:
    print("INVALID\n" + "\n".join(f"- {error}" for error in errors))
    return 1
  print(f"VALID: {args.run_directory}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
