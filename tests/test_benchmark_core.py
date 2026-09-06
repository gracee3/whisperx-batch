from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from benchmark_core import (
  ManifestItem, aggregate_scores, aggregate_workers, duration_balanced_shards, edit_counts,
  manifest_checksum, repetition_stats, throughput, validate_output_ids,
  probe_duration,
)


class ScoringTests(unittest.TestCase):
  def test_sdi_hits_and_corpus_macro_wer(self) -> None:
    counts = edit_counts("a b c".split(), "a x c y".split())
    self.assertEqual((counts.substitutions, counts.deletions, counts.insertions, counts.hits), (1, 0, 1, 2))
    scores = aggregate_scores([("a b".split(), "a x".split()), ("c d e f".split(), "c d e f".split())])
    self.assertEqual(scores["corpus_wer"], 1 / 6)
    self.assertEqual(scores["macro_wer"], 0.25)
    self.assertEqual(scores["reference_words"], 6)
    self.assertEqual(scores["hypothesis_words"], 6)

  def test_empty_reference_and_hypothesis_are_explicit(self) -> None:
    self.assertEqual(edit_counts([], ["extra"]).insertions, 1)
    self.assertEqual(edit_counts(["missing"], []).deletions, 1)
    scores = aggregate_scores([([], ["extra"])])
    self.assertIsNone(scores["corpus_wer"])
    self.assertEqual(scores["empty_references"], 1)


class AccountingTests(unittest.TestCase):
  def item(self, item_id: str, duration: float) -> ManifestItem:
    return ManifestItem(item_id, f"/{item_id}.flac", item_id, duration)

  def test_timing_formulas(self) -> None:
    metrics = throughput(100, 4, 20, 30)
    self.assertEqual(metrics["rtf"], 0.04)
    self.assertEqual(metrics["x_realtime"], 25)
    self.assertEqual(metrics["words_per_s"], 5)
    self.assertEqual(metrics["hypothesis_tokens_per_s_e2e"], 7.5)
    with self.assertRaises(ValueError):
      throughput(0, 1, 0)

  def test_flac_source_duration_comes_from_streaminfo(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      path = Path(temp_dir) / "fixture.flac"
      sample_rate = 16000
      total_samples = 40000
      packed = (sample_rate << 44) | (1 << 41) | (15 << 36) | total_samples
      payload = b"\x00" * 10 + packed.to_bytes(8, "big") + b"\x00" * 16
      path.write_bytes(b"fLaC" + bytes([0, 0, 0, len(payload)]) + payload)
      self.assertEqual(probe_duration(path), 2.5)

  def test_manifest_checksum_is_stable_and_sensitive(self) -> None:
    items = [self.item("b", 2), self.item("a", 1)]
    self.assertEqual(manifest_checksum(items), manifest_checksum(items))
    self.assertNotEqual(manifest_checksum(items), manifest_checksum(list(reversed(items))))

  def test_duration_balanced_sharding_is_deterministic_disjoint_and_complete(self) -> None:
    items = [self.item("a", 10), self.item("b", 8), self.item("c", 3), self.item("d", 2)]
    shards = duration_balanced_shards(items, 2)
    self.assertEqual(shards, duration_balanced_shards(items, 2))
    flat = [x.item_id for shard in shards for x in shard]
    self.assertEqual(sorted(flat), ["a", "b", "c", "d"])
    self.assertEqual([sum(x.duration_s for x in s) for s in shards], [12, 11])

  def test_output_integrity_and_partial_batch(self) -> None:
    result = validate_output_ids(["a", "b", "c"], ["a", "a", "c", "z"], ["c"])
    self.assertEqual(result, {"missing": ["b"], "unexpected": ["z"], "duplicates": ["a"], "empty_hypotheses": ["c"]})
    batches = [list(range(i, min(i + 2, 5))) for i in range(0, 5, 2)]
    self.assertEqual([len(x) for x in batches], [2, 2, 1])

  def test_repetition_statistics(self) -> None:
    stats = repetition_stats([1, 2, 3])
    self.assertEqual(stats["median"], 2)
    self.assertEqual(stats["mean"], 2)
    self.assertAlmostEqual(stats["stddev"], 0.816496580927726)

  def test_dual_worker_aggregate_and_failed_worker_gate(self) -> None:
    workers = [
      {"worker_id": "0", "status": "ok", "source_audio_s": 60, "hypothesis_tokens": 100, "hits": 8, "substitutions": 1},
      {"worker_id": "1", "status": "ok", "source_audio_s": 40, "hypothesis_tokens": 80, "hits": 9, "deletions": 1},
    ]
    result = aggregate_workers(workers, 5)
    self.assertEqual(result["aggregate_rtf"], 0.05)
    self.assertEqual(result["aggregate_x_realtime"], 20)
    self.assertEqual(result["aggregate_tokens_per_s"], 36)
    self.assertEqual(result["corpus_wer"], 2 / 19)
    workers[1]["status"] = "failed"
    with self.assertRaisesRegex(ValueError, "failed workers"):
      aggregate_workers(workers, 5)


if __name__ == "__main__":
  unittest.main()
