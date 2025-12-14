#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

usage() {
  cat <<'EOF'
Usage:
  run_whisperx_docker.sh [options] <input_wav> <output_dir>

Options:
  --image NAME           Docker image (default: whisperx:torch241-cu121)
  --cache-dir DIR        Cache dir mounted into container (default: ~/.cache/whisperx-docker)
  --workdir DIR          Host workdir to mount as /work (default: current directory)
  --device DEV           whisperx --device (default: cuda)
  --compute-type TYPE    whisperx --compute_type (default: float16)
  --model NAME           whisperx --model (default: large-v2)
  --batch-size N         whisperx --batch_size (default: 16)
  --diarize              Enable diarization (default: on)
  --no-diarize           Disable diarization
  -h, --help             Show help

Environment:
  HUGGINGFACE_TOKEN      Required when diarization is enabled.

Example:
  run_whisperx_docker.sh --image whisperx:torch241-cu121 \
    "clean/foo_clean.wav" "output"
EOF
}

IMAGE="whisperx:torch241-cu121"
CACHE_DIR="${HOME}/.cache/whisperx-docker"
WORKDIR="$PWD"

DEVICE="cuda"
COMPUTE_TYPE="float16"
MODEL="large-v2"
BATCH_SIZE="16"
DIARIZE=1

if [[ $# -lt 2 ]]; then usage; exit 2; fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) IMAGE="$2"; shift 2 ;;
    --cache-dir) CACHE_DIR="$2"; shift 2 ;;
    --workdir) WORKDIR="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --compute-type) COMPUTE_TYPE="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --diarize) DIARIZE=1; shift ;;
    --no-diarize) DIARIZE=0; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    *) break ;;
  esac
done

INPUT_WAV="${1:?missing input wav}"
OUTPUT_DIR="${2:?missing output dir}"

if [[ "$DIARIZE" -eq 1 ]]; then
  : "${HUGGINGFACE_TOKEN:?ERROR: HUGGINGFACE_TOKEN is not set (required for diarization)}"
fi

mkdir -p "$CACHE_DIR"

# Ensure paths are absolute for docker mount reliability
WORKDIR_ABS="$(cd "$WORKDIR" && pwd)"
CACHE_ABS="$(cd "$CACHE_DIR" && pwd)"

# We pass output_dir as /work/<relative> so it lands in the mounted WORKDIR
# Require INPUT_WAV and OUTPUT_DIR to be relative to WORKDIR, or absolute inside WORKDIR.
# We'll compute container paths by stripping WORKDIR prefix if applicable.
container_path() {
  local p="$1"
  # If absolute and under WORKDIR, convert to /work/relative
  if [[ "$p" == /* && "$p" == "$WORKDIR_ABS"* ]]; then
    echo "/work/${p#"$WORKDIR_ABS"/}"
  elif [[ "$p" == /* ]]; then
    # absolute but not under workdir: refuse (would not be mounted)
    echo "ERROR_ABS_OUTSIDE"
  else
    echo "/work/$p"
  fi
}

IN_C="$(container_path "$INPUT_WAV")"
OUT_C="$(container_path "$OUTPUT_DIR")"

if [[ "$IN_C" == "ERROR_ABS_OUTSIDE" || "$OUT_C" == "ERROR_ABS_OUTSIDE" ]]; then
  echo "ERROR: input/output must be inside workdir mount: $WORKDIR_ABS" >&2
  exit 2
fi

DIARIZE_ARGS=()
if [[ "$DIARIZE" -eq 1 ]]; then
  DIARIZE_ARGS+=(--diarize --hf_token "$HUGGINGFACE_TOKEN")
fi

docker run --rm -i --gpus all \
  -v "${WORKDIR_ABS}:/work" \
  -v "${CACHE_ABS}:/cache" \
  -e HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-}" \
  -e HF_HOME=/cache/hf \
  -e TRANSFORMERS_CACHE=/cache/hf \
  -e TORCH_HOME=/cache/torch \
  -e XDG_CACHE_HOME=/cache/xdg \
  "$IMAGE" \
  whisperx "$IN_C" \
    --model "$MODEL" \
    --device "$DEVICE" \
    --compute_type "$COMPUTE_TYPE" \
    --batch_size "$BATCH_SIZE" \
    --output_dir "$OUT_C" \
    "${DIARIZE_ARGS[@]}"

