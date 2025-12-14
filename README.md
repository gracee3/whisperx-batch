# whisperx-batch

A small, practical toolkit for cleaning audio and batch-transcribing it with
[WhisperX](https://github.com/m-bain/whisperx).

Designed for:
- running on local machines with NVIDIA GPUs
- stable diarization
- reproducible installs (Docker-first, native optional)
- batch processing of real-world audio files

Nothing fancy — just stuff that works.

---

## Features

- Batch audio cleaning with `ffmpeg`
  - mono, 16kHz PCM WAV
  - high/low-pass filtering
  - light denoise + loudness normalization
- WhisperX transcription with optional diarization
- Docker-based runtime to avoid CUDA / Torch dependency churn
- Parallel audio preprocessing
- Safe handling of filenames with spaces
- No secrets checked into the repo

---

## Requirements

### General
- `ffmpeg`
- Bash (Linux / macOS)
- NVIDIA GPU + drivers (for GPU transcription)

### For Docker mode (recommended)
- Docker
- NVIDIA Container Toolkit (`--gpus all` support)

### For diarization
- A Hugging Face token with access to the required models

```bash
export HUGGINGFACE_TOKEN="hf_..."
````

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

* `docker` – use Dockerized WhisperX (default if Docker is available)
* `native` – use local WhisperX install

### Common options

```text
-j, --jobs N            ffmpeg parallelism (default: nproc)
--whisper-jobs N        whisper parallelism (default: 1)
--input-dir DIR         input directory (default: cwd)
--clean-dir DIR         cleaned audio dir (default: ./clean)
--output-dir DIR        output dir (default: ./output)
--no-diarize            disable diarization
```

### Examples

```bash
# Transcribe all .m4a files in the current directory
transcribe.sh m4a

# Use docker explicitly
transcribe.sh docker wav

# Faster audio cleaning, no diarization
transcribe.sh mp3 -j 8 --no-diarize
```

---

## Docker cache (important)

Models are cached between runs at:

```text
~/.cache/whisperx-docker
```

This avoids re-downloading Whisper / Hugging Face models every time.

---

## Project layout

```text
.
├── transcribe.sh              # main entry point
├── run_whisperx_docker.sh     # docker runner helper
├── Dockerfile.whisperx-cu121-torch241
├── clean/                     # generated cleaned WAVs
└── output/                    # transcripts
```

---

## Notes

* Diarization is GPU + CPU heavy — `--whisper-jobs 1` is recommended for stability.
* The Docker image pins known-good versions of Torch, WhisperX, and CUDA libs.
* Native mode is supported, but Docker is strongly recommended if you value your time.

---

## License

MIT. Use it, modify it, break it, fix it.



