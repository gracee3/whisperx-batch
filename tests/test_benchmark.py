from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import benchmark_utils
from loader import load_entrypoint


BENCHMARK = load_entrypoint("whisperx-benchmark", "whisperx_batch_benchmark_test")


class BenchmarkParsingTests(unittest.TestCase):
  def test_parse_kv_items_normalizes_names_and_preserves_device_lists(self) -> None:
    parsed = benchmark_utils.parse_kv_items(
      ["batch_size=8", "cuda_devices=0,1"],
    )
    self.assertEqual(parsed["batch-size"], ["8"])
    self.assertEqual(parsed["cuda-devices"], ["0,1"])

  def test_parse_kv_items_rejects_missing_empty_and_duplicate_values(self) -> None:
    for values, message in (
      (["batch-size"], "key=value"),
      (["batch-size="], "empty value"),
      (["batch-size=8", "batch-size=16"], "duplicate key"),
    ):
      with self.subTest(values=values):
        with self.assertRaisesRegex(ValueError, message):
          benchmark_utils.parse_kv_items(values)

  def test_whisper_arg_for_sweep_handles_boolean_toggles(self) -> None:
    self.assertEqual(
      benchmark_utils.whisper_arg_for_sweep("no-align", "true"),
      ["--no-align"],
    )
    self.assertEqual(
      benchmark_utils.whisper_arg_for_sweep("no-align", "false"),
      [],
    )
    self.assertEqual(
      benchmark_utils.whisper_arg_for_sweep("beam-size", "8"),
      ["--beam-size", "8"],
    )


class BenchmarkScoringTests(unittest.TestCase):
  def test_normalize_and_wer_cover_exact_substitution_and_empty_reference(self) -> None:
    self.assertEqual(BENCHMARK.normalize("Hello, WORLD!"), ["hello", "world"])
    self.assertEqual(BENCHMARK.wer("one two", "one two"), 0.0)
    self.assertEqual(BENCHMARK.wer("one two", "one three"), 0.5)
    self.assertEqual(BENCHMARK.wer("", "unexpected"), 1.0)

  def test_librispeech_reference_loading_and_manifest_filtering(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      dataset = Path(temp_dir)
      chapter = dataset / "1" / "2"
      chapter.mkdir(parents=True)
      (chapter / "1-2-0001.wav").write_bytes(b"")
      (chapter / "1-2-0002.wav").write_bytes(b"")
      (chapter / "1-2-0003.wav").write_bytes(b"")
      (chapter / "1-2.trans.txt").write_text(
        "1-2-0001 FIRST REFERENCE\n1-2-0002 SECOND REFERENCE\n",
        encoding="utf-8",
      )
      refs = BENCHMARK.load_references(dataset)
      self.assertEqual(refs["1-2-0001"], "FIRST REFERENCE")
      manifest = BENCHMARK.build_manifest(dataset, "wav", refs, max_files=1)
      self.assertEqual(manifest, [(chapter / "1-2-0001.wav", "FIRST REFERENCE")])

  def test_txt_prediction_strips_speaker_labels(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      path = Path(temp_dir) / "prediction.txt"
      path.write_text(
        "[SPEAKER_00]: first line\n[SPEAKER_01]: second line\n",
        encoding="utf-8",
      )
      self.assertEqual(BENCHMARK.parse_txt_prediction(path), "first line\nsecond line")

  def test_json_prediction_returns_text_tokens_and_duration(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      path = Path(temp_dir) / "prediction.json"
      path.write_text(
        json.dumps(
          {
            "segments": [
              {
                "text": "hello world",
                "end": 1.5,
                "words": [
                  {"word": "hello", "end": 0.5},
                  {"word": "world", "end": 1.6},
                ],
              },
              {"text": "again", "end": 2.0},
            ],
          },
        ),
        encoding="utf-8",
      )
      self.assertEqual(
        BENCHMARK.parse_json_prediction_with_metrics(path),
        ("hello world again", 3, 2.0),
      )


class BenchmarkCommandTests(unittest.TestCase):
  def make_args(self, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
      "transcribe": "/repo/transcribe",
      "batch_size": 16,
      "output_format": "json",
      "skip_transcribe_existing": False,
      "no_diarize": True,
      "transcribe_config": "",
      "whisper_arg": [],
    }
    values.update(overrides)
    return argparse.Namespace(**values)

  def test_build_run_command_makes_diarization_intent_explicit(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      offline = BENCHMARK.build_run_command(
        self.make_args(no_diarize=True),
        root / "input",
        root / "offline",
        {"beam-size": "8"},
      )
      diarized = BENCHMARK.build_run_command(
        self.make_args(no_diarize=False),
        root / "input",
        root / "diarized",
        {},
      )
      self.assertIn("--no-diarize", offline)
      self.assertNotIn("--diarize", offline)
      self.assertEqual(offline[-2:], ["--beam-size", "8"])
      self.assertIn("--diarize", diarized)

  def test_trace_device_precedence_is_combo_then_command_then_config(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      config = Path(temp_dir) / "config.toml"
      config.write_text('[transcribe]\ncuda_devices = "3,4"\n', encoding="utf-8")
      self.assertEqual(
        BENCHMARK.resolve_trace_device_filter(
          ["transcribe", "--cuda-devices", "1,2"],
          str(config),
          {"cuda-devices": "0"},
        ),
        "0",
      )
      self.assertEqual(
        BENCHMARK.resolve_trace_device_filter(
          ["transcribe", "--cuda-devices", "1,2"],
          str(config),
          {},
        ),
        "1,2",
      )
      self.assertEqual(
        BENCHMARK.resolve_trace_device_filter(["transcribe"], str(config), {}),
        "3,4",
      )

  def test_gpu_summary_and_token_rates_are_deterministic(self) -> None:
    samples = [
      {
        "ts": 1.0,
        "gpus": [
          {
            "index": "0",
            "mem_used_mib": 100,
            "mem_total_mib": 1000,
            "utilization_pct": 20,
          },
        ],
      },
      {
        "ts": 2.0,
        "gpus": [
          {
            "index": "0",
            "mem_used_mib": 250,
            "mem_total_mib": 1000,
            "utilization_pct": 60,
          },
        ],
      },
    ]
    summary = BENCHMARK.summarize_gpu_samples(samples, 2.0)
    self.assertEqual(summary["trace_sample_count"], "2")
    self.assertEqual(summary["trace_gpu_peak_mem_mib"], "0:250")
    self.assertEqual(summary["trace_gpu_peak_mem_pct"], "0:25.0")
    self.assertEqual(summary["trace_gpu_avg_util_pct"], "0:40.0")
    self.assertEqual(
      BENCHMARK.compute_token_rate_stats([(10, 2.0), (20, 4.0)], 3.0),
      (10.0, 10.0),
    )

  def test_recorded_command_redacts_credentials(self) -> None:
    self.assertEqual(
      BENCHMARK.redact_command(["tool", "--hf-token", "hf_private", "--value", "safe", "sk-secret"]),
      ["tool", "--hf-token", "<redacted>", "--value", "safe", "<redacted>"],
    )


if __name__ == "__main__":
  unittest.main()
