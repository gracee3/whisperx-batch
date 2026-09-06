"""Offline, deterministic primitives for auditable speech benchmarks."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class EditCounts:
  substitutions: int = 0
  deletions: int = 0
  insertions: int = 0
  hits: int = 0

  @property
  def reference_words(self) -> int:
    return self.hits + self.substitutions + self.deletions

  @property
  def hypothesis_words(self) -> int:
    return self.hits + self.substitutions + self.insertions

  @property
  def errors(self) -> int:
    return self.substitutions + self.deletions + self.insertions

  @property
  def wer(self) -> float | None:
    return self.errors / self.reference_words if self.reference_words else None

  def __add__(self, other: "EditCounts") -> "EditCounts":
    return EditCounts(*(getattr(self, k) + getattr(other, k) for k in (
      "substitutions", "deletions", "insertions", "hits"
    )))


@dataclass(frozen=True)
class ManifestItem:
  item_id: str
  audio_path: str
  reference: str
  duration_s: float
  source_utterances: tuple[str, ...] = ()

  def canonical(self) -> dict[str, object]:
    return asdict(self)


def edit_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> EditCounts:
  """Levenshtein alignment with deterministic S > D > I tie-breaking."""
  rows: list[list[tuple[int, EditCounts]]] = [[(0, EditCounts()) for _ in range(len(hypothesis) + 1)] for _ in range(len(reference) + 1)]
  for i in range(1, len(reference) + 1):
    rows[i][0] = (i, EditCounts(deletions=i))
  for j in range(1, len(hypothesis) + 1):
    rows[0][j] = (j, EditCounts(insertions=j))
  for i, ref in enumerate(reference, 1):
    for j, hyp in enumerate(hypothesis, 1):
      if ref == hyp:
        cost, prior = rows[i - 1][j - 1]
        rows[i][j] = (cost, prior + EditCounts(hits=1))
        continue
      candidates = (
        (rows[i - 1][j - 1][0] + 1, rows[i - 1][j - 1][1] + EditCounts(substitutions=1), 0),
        (rows[i - 1][j][0] + 1, rows[i - 1][j][1] + EditCounts(deletions=1), 1),
        (rows[i][j - 1][0] + 1, rows[i][j - 1][1] + EditCounts(insertions=1), 2),
      )
      best = min(candidates, key=lambda x: (x[0], x[2]))
      rows[i][j] = (best[0], best[1])
  return rows[-1][-1][1]


def aggregate_scores(pairs: Iterable[tuple[Sequence[str], Sequence[str]]]) -> dict[str, object]:
  total = EditCounts()
  per_file: list[float] = []
  empty_references = 0
  for ref, hyp in pairs:
    counts = edit_counts(ref, hyp)
    total += counts
    if counts.reference_words:
      per_file.append(counts.errors / counts.reference_words)
    else:
      empty_references += 1
  return {
    **asdict(total),
    "reference_words": total.reference_words,
    "hypothesis_words": total.hypothesis_words,
    "corpus_wer": total.wer,
    "macro_wer": statistics.fmean(per_file) if per_file else None,
    "scored_files": len(per_file),
    "empty_references": empty_references,
  }


def throughput(audio_s: float, wall_s: float, hypothesis_words: int, hypothesis_tokens: int | None = None) -> dict[str, float | int | None]:
  if audio_s <= 0 or wall_s <= 0:
    raise ValueError("audio and wall duration must be positive")
  return {
    "source_audio_s": audio_s,
    "wall_clock_processing_s": wall_s,
    "rtf": wall_s / audio_s,
    "x_realtime": audio_s / wall_s,
    "audio_s_per_s": audio_s / wall_s,
    "hypothesis_words": hypothesis_words,
    "words_per_s": hypothesis_words / wall_s,
    "hypothesis_tokens": hypothesis_tokens,
    "hypothesis_tokens_per_s_e2e": hypothesis_tokens / wall_s if hypothesis_tokens is not None else None,
  }


def manifest_checksum(items: Sequence[ManifestItem]) -> str:
  payload = json.dumps([i.canonical() for i in items], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
  return hashlib.sha256(payload.encode()).hexdigest()


def duration_balanced_shards(items: Sequence[ManifestItem], workers: int) -> list[list[ManifestItem]]:
  if workers < 1:
    raise ValueError("workers must be positive")
  shards: list[list[ManifestItem]] = [[] for _ in range(workers)]
  totals = [0.0] * workers
  for item in sorted(items, key=lambda x: (-x.duration_s, x.item_id)):
    target = min(range(workers), key=lambda i: (totals[i], i))
    shards[target].append(item)
    totals[target] += item.duration_s
  for shard in shards:
    shard.sort(key=lambda x: x.item_id)
  return shards


def validate_output_ids(expected: Sequence[str], produced: Sequence[str], empty: Sequence[str] = ()) -> dict[str, list[str]]:
  expected_set = set(expected)
  counts: dict[str, int] = {}
  for value in produced:
    counts[value] = counts.get(value, 0) + 1
  return {
    "missing": sorted(expected_set - set(counts)),
    "unexpected": sorted(set(counts) - expected_set),
    "duplicates": sorted(k for k, v in counts.items() if v > 1),
    "empty_hypotheses": sorted(set(empty)),
  }


def repetition_stats(values: Sequence[float]) -> dict[str, object]:
  if not values:
    raise ValueError("at least one repetition is required")
  return {
    "individual": list(values), "median": statistics.median(values),
    "min": min(values), "max": max(values), "mean": statistics.fmean(values),
    "stddev": statistics.pstdev(values),
  }


def aggregate_workers(workers: Sequence[dict[str, object]], shared_wall_s: float) -> dict[str, object]:
  if shared_wall_s <= 0:
    raise ValueError("shared wall duration must be positive")
  failed = [str(w.get("worker_id", "?")) for w in workers if w.get("status") != "ok"]
  if failed:
    raise ValueError(f"failed workers: {', '.join(failed)}")
  audio = sum(float(w["source_audio_s"]) for w in workers)
  if audio <= 0:
    raise ValueError("aggregate source duration must be positive")
  tokens = [w.get("hypothesis_tokens") for w in workers]
  token_total = sum(int(x) for x in tokens) if all(x is not None for x in tokens) else None
  substitutions = sum(int(w.get("substitutions", 0)) for w in workers)
  deletions = sum(int(w.get("deletions", 0)) for w in workers)
  insertions = sum(int(w.get("insertions", 0)) for w in workers)
  hits = sum(int(w.get("hits", 0)) for w in workers)
  reference_words = hits + substitutions + deletions
  hypothesis_words = hits + substitutions + insertions
  scored_files = sum(int(w.get("scored_files", w.get("num_files", 0))) for w in workers)
  macro_numerator = sum(float(w["macro_wer"]) * int(w.get("scored_files", w.get("num_files", 0))) for w in workers if w.get("macro_wer") is not None)
  return {
    "worker_count": len(workers), "shared_wall_clock_s": shared_wall_s,
    "total_source_audio_s": audio, "aggregate_rtf": shared_wall_s / audio,
    "aggregate_x_realtime": audio / shared_wall_s,
    "aggregate_hypothesis_tokens": token_total,
    "aggregate_tokens_per_s": token_total / shared_wall_s if token_total is not None else None,
    "substitutions": substitutions, "deletions": deletions, "insertions": insertions, "hits": hits,
    "reference_words": reference_words, "hypothesis_words": hypothesis_words,
    "corpus_wer": (substitutions + deletions + insertions) / reference_words if reference_words else None,
    "macro_wer": macro_numerator / scored_files if scored_files else None,
    "words_per_s": hypothesis_words / shared_wall_s,
  }


def probe_duration(path: Path, ffprobe: str = "ffprobe") -> float:
  if path.suffix.lower() == ".flac":
    with path.open("rb") as stream:
      if stream.read(4) != b"fLaC":
        raise ValueError(f"invalid FLAC signature: {path}")
      header = stream.read(4)
      if len(header) != 4 or (header[0] & 0x7f) != 0:
        raise ValueError(f"FLAC STREAMINFO is not first: {path}")
      length = int.from_bytes(header[1:4], "big")
      payload = stream.read(length)
      if len(payload) < 18:
        raise ValueError(f"truncated FLAC STREAMINFO: {path}")
      packed = int.from_bytes(payload[10:18], "big")
      sample_rate = packed >> 44
      total_samples = packed & ((1 << 36) - 1)
      duration = total_samples / sample_rate if sample_rate else 0.0
    if not math.isfinite(duration) or duration <= 0:
      raise ValueError(f"invalid duration for {path}: {duration}")
    return duration
  proc = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True)
  if proc.returncode:
    raise ValueError(f"ffprobe failed for {path}: {proc.stderr.strip()}")
  try:
    duration = float(proc.stdout.strip())
  except ValueError as exc:
    raise ValueError(f"invalid duration for {path}") from exc
  if not math.isfinite(duration) or duration <= 0:
    raise ValueError(f"invalid duration for {path}: {duration}")
  return duration
