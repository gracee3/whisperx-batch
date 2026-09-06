# Changelog

This file separates historical stack checkpoints from current unreleased work.

## Unreleased

- Detached Docker transcription from the caller's standard input so shell
  loops can process every queued file without Docker consuming the remaining
  input records.
- Added the repository's intended MIT license and dependency/data provenance
  boundaries.
- Replaced tracked personal host paths with a portable `config.example.toml` and
  ignored `config.local.toml`.
- Added offline behavioral tests for config, argument construction, discovery,
  Docker command generation, sweeps, WER, manifests, and trace summaries.
- Added Python 3.11/3.14 offline CI with exceptional resources explicitly absent.
- Reframed the historical benchmark defaults as an observation pending a
  publication-grade evidence bundle.
- Added current architecture, CLI, benchmark, privacy/publication, and
  contribution guidance.

## v0.2.0 — 2026-03-14

Historical stack checkpoint: Python 3.11, CUDA 12.8, Torch 2.8, pyannote 4,
CTranslate2 4.7.1, Whisper large-v3 support, and the benchmark harness. This tag
was not accompanied by a GitHub release or a compatibility/support policy.

## v0.1.0 — 2025-12-14

Historical stack checkpoint: Python 3.10, CUDA 12.1, Torch 2.4.1, pyannote
3.3.2, and CTranslate2 4.4. This tag was not accompanied by a GitHub release or
a compatibility/support policy.
