#!/usr/bin/env python3
"""Download and verify official LibriSpeech evaluation subsets explicitly."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmark_core import probe_duration


OPENSLR = "https://www.openslr.org/resources/12"
CHECKSUMS = {
  "dev-clean": "42e2234ba48799c1f50f24a7926300a1",
  "dev-other": "c8d0bcc9cca99d4f8b62fcc847357931",
  "test-clean": "32fa31d27d2e1cad72775fee3f4849a9",
  "test-other": "fb5a50374b501bb3bac4815ee91d3135",
}


def digest(path: Path, algorithm: str = "md5") -> str:
  value = hashlib.new(algorithm)
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      value.update(chunk)
  return value.hexdigest()


def safe_extract(archive: Path, root: Path) -> None:
  with tarfile.open(archive, "r:gz") as bundle:
    root_resolved = root.resolve()
    for member in bundle.getmembers():
      target = (root / member.name).resolve()
      if target != root_resolved and root_resolved not in target.parents:
        raise ValueError(f"unsafe archive member: {member.name}")
      if member.issym() or member.islnk():
        raise ValueError(f"links are not accepted in dataset archive: {member.name}")
    bundle.extractall(root, filter="data")


def subset_inventory(root: Path, subset: str) -> dict[str, object]:
  target = root / subset
  audio = sorted(target.rglob("*.flac")) if target.is_dir() else []
  transcripts = sorted(target.rglob("*.trans.txt")) if target.is_dir() else []
  refs: dict[str, str] = {}
  for path in transcripts:
    for line in path.read_text(encoding="utf-8").splitlines():
      if line.strip():
        key, text = line.split(maxsplit=1)
        if key in refs:
          raise ValueError(f"duplicate reference: {key}")
        refs[key] = text
  missing_refs = sorted(path.stem for path in audio if path.stem not in refs)
  unscored_refs = sorted(set(refs) - {path.stem for path in audio})
  manifest_lines = [f"{path.relative_to(root).as_posix()}\t{refs.get(path.stem, '')}" for path in audio]
  durations = [probe_duration(path) for path in audio]
  return {
    "subset": subset, "source_url": f"{OPENSLR}/{subset}.tar.gz",
    "archive_md5": CHECKSUMS[subset], "flac_files": len(audio),
    "transcript_files": len(transcripts), "references": len(refs),
    "total_duration_s": sum(durations),
    "missing_references": missing_refs, "references_without_audio": unscored_refs,
    "manifest_sha256": hashlib.sha256(("\n".join(manifest_lines) + "\n").encode()).hexdigest(),
    "ready": bool(audio) and not missing_refs and not unscored_refs,
  }


def prepare(root: Path, subset: str, download: bool) -> dict[str, object]:
  archives = root / "archives"
  archive = archives / f"{subset}.tar.gz"
  target = root / subset
  if not target.is_dir() and download:
    archives.mkdir(parents=True, exist_ok=True)
    curl = shutil.which("curl")
    if not curl:
      raise RuntimeError("curl is required for resumable downloads")
    proc = subprocess.run([curl, "--fail", "--location", "--continue-at", "-", "--output", str(archive), f"{OPENSLR}/{archive.name}"])
    if proc.returncode:
      raise RuntimeError(f"download failed: {archive.name}")
    actual = digest(archive)
    if actual != CHECKSUMS[subset]:
      raise ValueError(f"checksum mismatch for {archive.name}: {actual}")
    safe_extract(archive, root.parent)
  if archive.exists():
    actual = digest(archive)
    if actual != CHECKSUMS[subset]:
      raise ValueError(f"checksum mismatch for {archive.name}: {actual}")
  return subset_inventory(root, subset)


def main() -> int:
  parser = argparse.ArgumentParser(description="Prepare or verify official LibriSpeech evaluation subsets.")
  parser.add_argument("command", choices=("download", "verify"))
  parser.add_argument("--root", type=Path, default=Path("/data/datasets/LibriSpeech"))
  parser.add_argument("--subset", action="append", choices=tuple(CHECKSUMS), help="repeatable; default: all four")
  args = parser.parse_args()
  root = args.root.resolve()
  subsets = args.subset or list(CHECKSUMS)
  results = [prepare(root, subset, args.command == "download") for subset in subsets]
  root.mkdir(parents=True, exist_ok=True)
  (root / "inventory.json").write_text(json.dumps({"schema_version": 1, "subsets": results}, indent=2) + "\n", encoding="utf-8")
  for result in results:
    print(f"{result['subset']}: ready={result['ready']} flac={result['flac_files']} manifest={result['manifest_sha256']}")
  return 0 if all(bool(result["ready"]) for result in results) else 2


if __name__ == "__main__":
  raise SystemExit(main())
