# whisperx-batch

A small, practical toolkit for cleaning audio and batch-transcribing it with
[WhisperX](https://github.com/m-bain/whisperx).

Built for:
- real-world audio (meetings, lectures, interviews)
- stable diarization
- GPU systems that *shouldn’t* break every time Torch updates
- fast re-runs without redoing work

Nothing fancy — just stuff that works.

---

## What this does

1. Cleans audio with `ffmpeg`
   - mono, 16 kHz PCM WAV
   - band-pass filtering
   - light denoise
   - loudness normalization
2. Transcribes with WhisperX
   - GPU-accelerated
   - optional diarization
3. Skips work intelligently on re-runs
   - cached clean audio (with filter hash validation)
   - cached transcripts

---

## Requirements

### General
- Linux
- `ffmpeg`
- Bash

### GPU transcription
- NVIDIA GPU
- NVIDIA drivers

### Recommended (Docker mode)
- Docker
- NVIDIA Container Toolkit (`--gpus all`)

### Diarization
- Hugging Face token with access to required models

```bash
export HUGGINGFACE_TOKEN="hf_..."
````

> ⚠️ Tokens are **never** stored in this repo or scripts.

---

## Quick start (Docker – recommended)

Build the image once:

```bash
docker build -f Dockerfile.whisperx-cu121-torch241 \
  -t whisperx:torch241-cu121 .
```

Then from a directory with audio files:

```bash
transcribe.sh m4a
```

Outputs:

* cleaned audio → `./clean/`
* transcripts → `./output/`

---

## Usage

```bash
transcribe.sh [mode] <wav|m4a|mp3> [options]
```

### Modes

* `docker` – Dockerized WhisperX (default if Docker is available)
* `native` – Local WhisperX install (advanced users only)

---

## Common options

```text
-j, --jobs N                     ffmpeg parallelism (default: nproc)
--whisper-jobs N                 whisper parallelism (default: 1)
--input-dir DIR                  input directory (default: cwd)
--clean-dir DIR                  cleaned WAV directory (default: ./clean)
--output-dir DIR                 transcript output directory (default: ./output)

--skip-clean-existing             skip ffmpeg if cached clean audio is valid
--force-clean                     force re-cleaning audio
--skip-transcribe-existing        skip WhisperX if output already exists

--no-diarize                      disable diarization
```

---

## Examples

Fast incremental rerun (recommended):

```bash
transcribe.sh m4a -j 8 \
  --skip-clean-existing \
  --skip-transcribe-existing
```

Force re-clean audio (filter changed, bad denoise, etc.):

```bash
transcribe.sh wav --force-clean
```

Docker explicitly:

```bash
transcribe.sh docker mp3
```

---

## Caching behavior (important)

### Clean audio cache

For each cleaned file, a sidecar metadata file is written:

```text
clean/foo_clean.wav
clean/foo_clean.wav.meta
```

The `.meta` file records:

* input file size
* input mtime
* hash of the ffmpeg filter chain

If any of those change, the file is re-cleaned automatically.

---

### Docker model cache

Models are cached between runs at:

```text
~/.cache/whisperx-docker
```

This avoids re-downloading Whisper / Hugging Face models every time.

---

## Project layout

```text
.
├── transcribe.sh
├── run_whisperx_docker.sh
├── Dockerfile.whisperx-cu121-torch241
├── clean/        # generated (ignored)
├── output/       # generated (ignored)
└── README.md
```

---

## Notes

* Diarization is GPU + CPU heavy — `--whisper-jobs 1` is recommended.
* Docker mode avoids Torch / CUDA dependency churn.
* Native mode is supported, but Docker is strongly recommended unless you enjoy debugging ML stacks.

---

## License

MIT.

Use it, modify it, break it, fix it.

```

---

If you want next:
- a `CHANGELOG.md`
- GitHub Actions for linting / shellcheck
- a `make build / make transcribe` flow
- badges (CUDA, Docker, WhisperX version)

Just say the word 🙂
