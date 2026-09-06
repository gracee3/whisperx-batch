from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from loader import load_entrypoint


TRANSCRIBE = load_entrypoint("transcribe", "whisperx_batch_transcribe_test")


def whisper_kwargs() -> dict[str, object]:
  return {
    "model": "/models/whisper",
    "task": "transcribe",
    "language": "en",
    "output_format": "json",
    "device": "cuda",
    "compute_type": "float16",
    "batch_size": 16,
    "max_speakers": "",
    "align_model": "WAV2VEC2_ASR_BASE_960H",
    "interpolate_method": "nearest",
    "no_align": False,
    "return_char_alignments": False,
    "vad_method": "",
    "vad_onset": "",
    "vad_offset": "",
    "chunk_size": "",
    "diarize": False,
    "diarize_model": "",
    "speaker_embeddings": False,
    "temperature": "0",
    "best_of": "1",
    "beam_size": "1",
    "patience": "1.0",
    "length_penalty": "1.0",
    "suppress_tokens": "-1",
    "suppress_numerals": False,
    "initial_prompt": "",
    "condition_on_previous_text": "true",
    "fp16": "false",
    "temperature_increment_on_fallback": "0.2",
    "compression_ratio_threshold": "2.4",
    "logprob_threshold": "-1.0",
    "no_speech_threshold": "0.6",
    "max_line_width": "0",
    "max_line_count": "0",
    "highlight_words": False,
    "segment_resolution": "sentence",
    "min_speakers": "",
    "threads": "0",
    "print_progress": "false",
    "verbose": "",
    "log_level": "",
    "hotwords": "",
    "extra_args": ["--extra", "value with spaces"],
  }


class TranscribeControlPlaneTests(unittest.TestCase):
  def test_build_whisper_args_maps_values_and_boollikes(self) -> None:
    args = TRANSCRIBE.build_whisper_args(**whisper_kwargs())
    self.assertEqual(args[0], "whisperx")
    self.assertIn("--batch_size", args)
    self.assertEqual(args[args.index("--batch_size") + 1], "16")
    self.assertEqual(args[args.index("--condition_on_previous_text") + 1], "True")
    self.assertEqual(args[args.index("--fp16") + 1], "False")
    self.assertEqual(args[-2:], ["--extra", "value with spaces"])
    self.assertNotIn("--diarize", args)

  def test_build_whisper_args_adds_diarization_contract(self) -> None:
    values = whisper_kwargs()
    values.update(
      diarize=True,
      diarize_model="/models/diarize",
      speaker_embeddings=True,
    )
    args = TRANSCRIBE.build_whisper_args(**values)
    self.assertIn("--diarize", args)
    self.assertEqual(args[args.index("--diarize_model") + 1], "/models/diarize")
    self.assertIn("--speaker_embeddings", args)

  def test_without_diarization_args_removes_flag_value_and_embeddings(self) -> None:
    args = [
      "whisperx",
      "--model",
      "m",
      "--diarize_model",
      "d",
      "--diarize",
      "--speaker_embeddings",
      "audio.wav",
    ]
    self.assertEqual(
      TRANSCRIBE.without_diarization_args(args),
      ["whisperx", "--model", "m", "audio.wav"],
    )

  def test_duplicate_basenames_receive_stable_unique_names(self) -> None:
    inputs = [Path("a/sample.wav"), Path("b/sample.wav"), Path("c/other.wav")]
    mapped = TRANSCRIBE.build_container_input_names(inputs)
    self.assertEqual([name for _, name in mapped], ["sample.wav", "sample_1.wav", "other.wav"])

  def test_audio_discovery_respects_recursion_and_suffix_case(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      (root / "a.WAV").write_bytes(b"")
      (root / "ignore.txt").write_text("x", encoding="utf-8")
      nested = root / "nested"
      nested.mkdir()
      (nested / "b.flac").write_bytes(b"")
      self.assertEqual(TRANSCRIBE.find_audio_files(root, False), [root / "a.WAV"])
      self.assertEqual(
        TRANSCRIBE.find_audio_files(root, True),
        [root / "a.WAV", nested / "b.flac"],
      )

  def test_output_detection_uses_requested_extensions(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      output = Path(temp_dir)
      (output / "recording.json").write_text("{}", encoding="utf-8")
      self.assertTrue(TRANSCRIBE.output_exists(output, "recording.wav", ["json"]))
      self.assertFalse(TRANSCRIBE.output_exists(output, "recording.wav", ["txt"]))

  def test_chunk_inputs_is_bounded(self) -> None:
    items = [(Path(str(index)), str(index)) for index in range(5)]
    self.assertEqual([len(chunk) for chunk in TRANSCRIBE.chunk_inputs(items, 2)], [2, 2, 1])
    self.assertEqual(TRANSCRIBE.chunk_inputs(items, 0), [items])

  def test_large_v3_layout_is_fail_closed(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      model = Path(temp_dir) / "faster-whisper-large-v3"
      model.mkdir()
      with self.assertRaisesRegex(ValueError, "model.bin"):
        TRANSCRIBE.check_large_v3_layout(model)
      (model / "model.bin").write_bytes(b"")
      with self.assertRaisesRegex(ValueError, "preprocessor_config"):
        TRANSCRIBE.check_large_v3_layout(model)
      (model / "preprocessor_config.json").write_text("{}", encoding="utf-8")
      TRANSCRIBE.check_large_v3_layout(model)

  def test_collect_mount_paths_dedupes_existing_roots(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      model = root / "model"
      model.mkdir()
      file_path = model / "weights.bin"
      file_path.write_bytes(b"")
      mounts = TRANSCRIBE.collect_mount_ro_paths([str(model), str(file_path), "/missing"])
      self.assertEqual(mounts, [str(model.resolve())])

  def test_host_paths_expand_home_before_resolution(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      with mock.patch.dict(os.environ, {"HOME": temp_dir}):
        self.assertEqual(
          TRANSCRIBE.resolve_host_path("~/models/whisper"),
          Path(temp_dir) / "models" / "whisper",
        )

  def test_shell_quoting_preserves_single_quotes(self) -> None:
    self.assertEqual(
      TRANSCRIBE.build_shell_command(["whisperx", "it's a fixture.wav"]),
      "'whisperx' 'it'\\''s a fixture.wav'",
    )

  def test_docker_command_has_explicit_read_only_and_offline_boundaries(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      output = root / "output"
      cache = root / "cache"
      model = root / "model"
      audio = root / "audio.wav"
      for directory in (output, cache, model):
        directory.mkdir()
      audio.write_bytes(b"")
      cmd = TRANSCRIBE.build_docker_run_command(
        image="whisperx:test",
        pull_policy="missing",
        host_output_dir=str(output),
        host_cuda_devices=["1"],
        host_model_mounts=[str(model)],
        host_cache=str(cache),
        host_inputs=[(audio, "audio.wav")],
        whisper_args=["whisperx", "--model", str(model)],
        whisper_output_dir="/mnt/output",
        whisper_input_container_dir="/mnt/input",
      )
      self.assertEqual(cmd[:4], ["docker", "run", "--rm", "--pull"])
      self.assertNotIn("--interactive", cmd)
      self.assertIn("device=1", cmd)
      self.assertIn(f"{model.resolve()}:{model.resolve()}:ro", cmd)
      self.assertIn(f"{audio.resolve()}:/mnt/input/audio.wav:ro", cmd)
      self.assertIn("HF_HUB_OFFLINE=1", cmd)
      self.assertEqual(cmd[-2], "whisperx:test")
      self.assertIn("'/mnt/input/audio.wav'", cmd[-1])

  def test_diarization_failure_retries_without_diarization(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      output = root / "output"
      output.mkdir()
      cache = root / "cache"
      cache.mkdir()
      audio = root / "audio.wav"
      audio.write_bytes(b"")
      with mock.patch.object(TRANSCRIBE, "execute_docker", side_effect=[1, 0]) as execute:
        result = TRANSCRIBE.run_docker_batch(
          batch_inputs=[(audio, "audio.wav")],
          image="image",
          pull_policy="missing",
          output_dir=output,
          cache_scope=str(cache),
          cuda_devices=["0"],
          model_mounts=[],
          whisper_args=["whisperx", "--diarize_model", "model", "--diarize"],
          whisper_input_container_dir="/mnt/input",
          diarize=True,
        )
      self.assertEqual(result, 0)
      self.assertEqual(execute.call_count, 2)
      fallback = execute.call_args_list[1].args[0][-1]
      self.assertNotIn("--diarize", fallback)
      self.assertNotIn("diarize_model", fallback)

  def test_execute_docker_never_inherits_stdin(self) -> None:
    completed = mock.Mock(returncode=0, stdout="")
    with mock.patch.object(TRANSCRIBE.subprocess, "run", return_value=completed) as run:
      self.assertEqual(TRANSCRIBE.execute_docker(["docker", "run"]), 0)
    self.assertIs(run.call_args.kwargs["stdin"], TRANSCRIBE.subprocess.DEVNULL)


if __name__ == "__main__":
  unittest.main()
