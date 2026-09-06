from __future__ import annotations

import argparse
import ast
import tempfile
import tomllib
import unittest
from pathlib import Path

from loader import ROOT, load_entrypoint


CLEAN_AUDIO = load_entrypoint("scripts/clean_audio.py", "whisperx_batch_clean_audio_test")
CONVERT = load_entrypoint(
  "scripts/convert_flac_to_wav.py",
  "whisperx_batch_convert_audio_test",
)
PRESEED = load_entrypoint(
  "scripts/preseed_whisperx_cache.py",
  "whisperx_batch_preseed_cache_test",
)
SETUP_DATASET = load_entrypoint(
  "scripts/setup_librispeech_dataset.py",
  "whisperx_batch_setup_dataset_test",
)
LONGFORM = load_entrypoint(
  "scripts/build_librispeech_longform.py",
  "whisperx_batch_longform_test",
)


class HelperScriptTests(unittest.TestCase):
  def test_clean_audio_dry_run_builds_expected_output(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      source = root / "meeting.flac"
      source.write_bytes(b"")
      args = argparse.Namespace(
        suffix="_clean",
        output_dir="",
        overwrite=False,
        dry_run=True,
        channels=1,
        sample_rate=16000,
        sample_fmt="s16",
        audio_filter=CLEAN_AUDIO.DEFAULT_AUDIO_FILTER,
      )
      self.assertEqual(
        CLEAN_AUDIO.convert_one(str(source), args),
        ("planned", str(source), str(root / "meeting_clean.wav"), 0),
      )

  def test_flac_converter_dry_run_does_not_require_ffmpeg(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      source = Path(temp_dir) / "sample.flac"
      source.write_bytes(b"")
      args = argparse.Namespace(
        dry_run=True,
        overwrite=False,
        ffmpeg_threads=0,
        channels=1,
        sample_rate=16000,
        sample_fmt="s16",
      )
      self.assertEqual(
        CONVERT.convert_one(str(source), args),
        ("planned", "sample.flac -> sample.wav", CONVERT.DRY),
      )

  def test_dataset_settings_resolve_from_portable_config(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      config = root / "config.toml"
      config.write_text(
        "[librispeech]\n"
        f'root = "{root / "dataset"}"\n'
        'subset = "test-clean"\n'
        "convert_to_wav = false\n"
        "convert_num_proc = 2\n",
        encoding="utf-8",
      )
      args = argparse.Namespace(
        config=str(config),
        root=None,
        subset=None,
        url=None,
        archive=None,
        convert=None,
        num_proc=None,
        ffmpeg_threads=None,
        keep_archive=None,
        overwrite_wav=False,
        convert_if_missing_only=False,
        skip_download=True,
        skip_extract=True,
      )
      settings = SETUP_DATASET.resolve_settings(args)
      self.assertEqual(settings.root, root / "dataset")
      self.assertEqual(settings.subset, "test-clean")
      self.assertFalse(settings.convert_to_wav)
      self.assertEqual(settings.convert_num_proc, 2)
      self.assertTrue(settings.skip_download)

  def test_preseed_marker_requires_complete_punkt_data(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      marker_root = root / "tokenizers" / "punkt_tab"
      english = marker_root / "english"
      english.mkdir(parents=True)
      (marker_root / "README").write_text("ok", encoding="utf-8")
      (english / "sent_starters.txt").write_text("ok", encoding="utf-8")
      self.assertFalse(PRESEED.marker_exists(root))
      (english / "abbrev_types.txt").write_text("ok", encoding="utf-8")
      self.assertTrue(PRESEED.marker_exists(root))

  def test_longform_chapter_and_utterance_order_is_deterministic(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      chapter_b = root / "2" / "9"
      chapter_a = root / "1" / "3"
      chapter_b.mkdir(parents=True)
      chapter_a.mkdir(parents=True)
      for path in (chapter_b / "2-9-0002.flac", chapter_a / "1-3-0001.flac", chapter_b / "2-9-0001.flac"):
        path.write_bytes(b"fixture")
      groups = LONGFORM.chapter_groups(root)
      self.assertEqual([key for key, _ in groups], ["1-3", "2-9"])
      self.assertEqual([p.stem for p in groups[1][1]], ["2-9-0001", "2-9-0002"])

  def test_longform_reference_loading_supports_stitching_order(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      path = root / "1-2.trans.txt"
      path.write_text("1-2-0002 SECOND\n1-2-0001 FIRST\n", encoding="utf-8")
      refs = LONGFORM.load_references(root)
      ordered = ["1-2-0001", "1-2-0002"]
      self.assertEqual(" ".join(refs[key] for key in ordered), "FIRST SECOND")


class RepositoryBaselineTests(unittest.TestCase):
  def test_python_sources_parse_without_importing_optional_tools(self) -> None:
    paths = sorted(ROOT.glob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))
    paths += [ROOT / "transcribe", ROOT / "whisperx-benchmark"]
    for path in paths:
      with self.subTest(path=path.relative_to(ROOT)):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

  def test_config_template_is_valid_ignored_and_machine_neutral(self) -> None:
    template = ROOT / "config.example.toml"
    with template.open("rb") as stream:
      config = tomllib.load(stream)
    self.assertIn("transcribe", config)
    self.assertIn("benchmark", config)
    self.assertIn("config.local.toml", (ROOT / ".gitignore").read_text(encoding="utf-8"))
    local_home = "/" + "home" + "/" + "emmy"
    self.assertNotIn(local_home, template.read_text(encoding="utf-8"))


if __name__ == "__main__":
  unittest.main()
