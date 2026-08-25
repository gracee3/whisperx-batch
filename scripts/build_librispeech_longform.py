#!/usr/bin/env python3
"""Deterministically construct lossless chapter-level LibriSpeech FLAC files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmark_core import ManifestItem, manifest_checksum, probe_duration


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def load_references(root: Path) -> dict[str, str]:
  refs: dict[str, str] = {}
  for path in sorted(root.rglob("*.trans.txt")):
    for line in path.read_text(encoding="utf-8").splitlines():
      if line.strip():
        key, text = line.split(maxsplit=1)
        if key in refs:
          raise ValueError(f"duplicate reference: {key}")
        refs[key] = text
  return refs


def chapter_groups(root: Path, limit: int = 0) -> list[tuple[str, list[Path]]]:
  groups: dict[str, list[Path]] = {}
  for audio in sorted(root.rglob("*.flac")):
    parts = audio.stem.split("-")
    if len(parts) != 3:
      raise ValueError(f"unexpected utterance id: {audio.stem}")
    groups.setdefault("-".join(parts[:2]), []).append(audio)
  ordered = [(key, sorted(value, key=lambda p: p.stem)) for key, value in sorted(groups.items())]
  return ordered[:limit] if limit else ordered


def construct(source: Path, destination: Path, limit: int = 0, verify_only: bool = False, image: str = "whisperx:torch280-cu128") -> list[ManifestItem]:
  refs = load_references(source)
  destination.mkdir(parents=True, exist_ok=True)
  items: list[ManifestItem] = []
  for chapter, audio_files in chapter_groups(source, limit):
    missing = [p.stem for p in audio_files if p.stem not in refs]
    if missing:
      raise ValueError(f"missing references in {chapter}: {missing}")
    output = destination / f"{chapter}.flac"
    mapping = destination / f"{chapter}.json"
    sources = tuple(p.stem for p in audio_files)
    reference = " ".join(refs[p.stem] for p in audio_files)
    expected_duration = sum(probe_duration(path) for path in audio_files)
    expected = {"chapter_id": chapter, "source_utterances": list(sources), "reference": reference}
    output_valid = output.exists() and abs(probe_duration(output) - expected_duration) <= 0.02
    if not verify_only and not output_valid:
      output.unlink(missing_ok=True)
      with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", dir=destination, delete=False) as listing:
        for path in audio_files:
          escaped = str(path.resolve()).replace("'", "'\\''")
          listing.write(f"file '{escaped}'\n")
        listing.flush()
        listing_path = Path(listing.name)
      ffmpeg_args = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing_path), "-c:a", "flac", str(output)]
      if shutil.which("ffmpeg"):
        proc = subprocess.run(ffmpeg_args, stdin=subprocess.DEVNULL)
      else:
        shell_command = " ".join(shlex.quote(x) for x in ffmpeg_args)
        proc = subprocess.run(["docker", "run", "--rm", "--pull", "never", "-u", f"{os.getuid()}:{os.getgid()}",
          "-v", f"{source}:{source}:ro", "-v", f"{destination}:{destination}", image, shell_command], stdin=subprocess.DEVNULL)
      listing_path.unlink(missing_ok=True)
      if proc.returncode:
        raise RuntimeError(f"ffmpeg failed for {chapter}")
    if not output.exists():
      raise ValueError(f"incomplete chapter artifact: {chapter}")
    duration = probe_duration(output)
    if abs(duration - expected_duration) > 0.02:
      raise ValueError(f"chapter duration mismatch: {chapter}: output={duration} source_sum={expected_duration}")
    expected.update(duration_s=duration, audio_sha256=sha256(output))
    if not verify_only:
      mapping.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    if not mapping.exists() or json.loads(mapping.read_text(encoding="utf-8")) != expected:
      raise ValueError(f"mapping mismatch: {chapter}")
    items.append(ManifestItem(chapter, str(output.resolve()), reference, duration, sources))
  manifest = {"schema_version": 1, "mode": "constructed-librispeech-chapter", "source": str(source.resolve()), "items": [x.canonical() for x in items]}
  manifest["manifest_sha256"] = manifest_checksum(items)
  reference_index = destination / "chapters.trans.txt"
  expected_references = "".join(f"{item.item_id} {item.reference}\n" for item in items)
  if not verify_only:
    reference_index.write_text(expected_references, encoding="utf-8")
  if not reference_index.is_file() or reference_index.read_text(encoding="utf-8") != expected_references:
    raise ValueError("chapter reference index mismatch")
  (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
  return items


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--source", type=Path, required=True)
  parser.add_argument("--destination", type=Path, required=True)
  parser.add_argument("--limit-chapters", type=int, default=0)
  parser.add_argument("--verify-only", action="store_true")
  parser.add_argument("--image", default="whisperx:torch280-cu128", help="Container image used for ffmpeg when host ffmpeg is absent.")
  args = parser.parse_args()
  items = construct(args.source.resolve(), args.destination.resolve(), args.limit_chapters, args.verify_only, args.image)
  print(f"verified {len(items)} chapters; manifest={manifest_checksum(items)}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
