#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

usage() {
  cat <<'EOF'
Usage:
  transcribe.sh [mode] <wav|m4a|mp3> [options]

Modes:
  docker        Use dockerized whisperx (default if docker is available)
  native        Use local whisperx binary in PATH/venv

Options:
  -j, --jobs N                 CPU parallelism for ffmpeg cleaning (default: nproc)
  --whisper-jobs N             Parallelism for whisper step (default: 1; not recommended >1)
  --input-dir DIR              Input directory (default: cwd)
  --clean-dir DIR              Clean WAV output directory (default: ./clean)
  --output-dir DIR             WhisperX output directory (default: ./output)
  --filter STR                 Override ffmpeg -af filter chain
  --model NAME                 WhisperX model (default: large-v2)
  --device DEV                 WhisperX device (default: cuda)
  --compute-type TYPE          WhisperX compute type (default: float16)
  --batch-size N               WhisperX batch size (default: 16)
  --diarize                    Enable diarization (default: on)
  --no-diarize                 Disable diarization

  --skip-clean-existing         Skip ffmpeg when clean wav exists AND metadata matches (filter hash + input mtime/size)
  --force-clean                 Always re-run ffmpeg cleaning (overrides skip-clean-existing)
  --skip-transcribe-existing    Skip whisperx when output JSON or SRT already exists

Docker options (mode=docker):
  --docker-image NAME          Docker image (default: whisperx:torch241-cu121)
  --docker-cache DIR           Host cache dir for HF/torch models (default: ~/.cache/whisperx-docker)
  --docker-runner PATH         Path to run_whisperx_docker.sh (default: ./run_whisperx_docker.sh)

Environment:
  HUGGINGFACE_TOKEN            Required when diarization is enabled.

Examples:
  transcribe.sh m4a -j 8 --skip-clean-existing --skip-transcribe-existing
  transcribe.sh docker wav --force-clean
  transcribe.sh native mp3 --no-diarize --skip-transcribe-existing
EOF
}

need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "ERROR: missing command: $1" >&2; exit 1; }; }

# Defaults
MODE="auto"
JOBS="$(command -v nproc >/dev/null 2>&1 && nproc || echo 4)"
WHISPER_JOBS=1

INPUT_DIR="$PWD"
CLEAN_DIR="./clean"
OUTPUT_DIR="./output"

AUDIO_FILTER_DEFAULT='highpass=f=80,lowpass=f=8000,afftdn=nf=-25,loudnorm=I=-16:LRA=11:TP=-2'
AUDIO_FILTER="$AUDIO_FILTER_DEFAULT"

WHISPER_MODEL="large-v2"
DEVICE="cuda"
COMPUTE_TYPE="float16"
BATCH_SIZE="16"
DIARIZE=1

SKIP_CLEAN_EXISTING=1
FORCE_CLEAN=0
SKIP_TRANSCRIBE_EXISTING=0

DOCKER_IMAGE="whisperx:torch241-cu121"
DOCKER_CACHE="${HOME}/.cache/whisperx-docker"
DOCKER_RUNNER="./run_whisperx_docker.sh"

# Parse optional mode
if [[ $# -lt 1 ]]; then usage; exit 2; fi
case "${1,,}" in
  docker|native) MODE="${1,,}"; shift ;;
esac

# Require extension
if [[ $# -lt 1 ]]; then usage; exit 2; fi
ext="${1,,}"; shift
case "$ext" in wav|m4a|mp3) ;; *) echo "ERROR: unsupported extension: $ext" >&2; usage; exit 2 ;; esac

# Parse options
while [[ $# -gt 0 ]]; do
  case "$1" in
    -j|--jobs) JOBS="$2"; shift 2 ;;
    --whisper-jobs) WHISPER_JOBS="$2"; shift 2 ;;
    --input-dir) INPUT_DIR="$2"; shift 2 ;;
    --clean-dir) CLEAN_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --filter) AUDIO_FILTER="$2"; shift 2 ;;
    --model) WHISPER_MODEL="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --compute-type) COMPUTE_TYPE="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --diarize) DIARIZE=1; shift ;;
    --no-diarize) DIARIZE=0; shift ;;

    --skip-clean-existing) SKIP_CLEAN_EXISTING=1; shift ;;
    --force-clean) FORCE_CLEAN=1; shift ;;
    --skip-transcribe-existing) SKIP_TRANSCRIBE_EXISTING=1; shift ;;

    --docker-image) DOCKER_IMAGE="$2"; shift 2 ;;
    --docker-cache) DOCKER_CACHE="$2"; shift 2 ;;
    --docker-runner) DOCKER_RUNNER="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

# Validate numeric args
re='^[0-9]+$'
[[ "$JOBS" =~ $re ]] || { echo "ERROR: --jobs must be an integer" >&2; exit 2; }
[[ "$WHISPER_JOBS" =~ $re ]] || { echo "ERROR: --whisper-jobs must be an integer" >&2; exit 2; }
(( JOBS >= 1 )) || { echo "ERROR: --jobs must be >= 1" >&2; exit 2; }
(( WHISPER_JOBS >= 1 )) || { echo "ERROR: --whisper-jobs must be >= 1" >&2; exit 2; }

# Resolve mode=auto
if [[ "$MODE" == "auto" ]]; then
  if command -v docker >/dev/null 2>&1; then MODE="docker"; else MODE="native"; fi
fi

# Checks
need_cmd ffmpeg
need_cmd find
need_cmd xargs
need_cmd sha256sum
need_cmd stat

if [[ "$DIARIZE" -eq 1 ]]; then
  : "${HUGGINGFACE_TOKEN:?ERROR: HUGGINGFACE_TOKEN is not set (required for diarization)}"
fi

# For docker mode, ensure runner exists
if [[ "$MODE" == "docker" ]]; then
  need_cmd docker
  [[ -x "$DOCKER_RUNNER" ]] || { echo "ERROR: docker runner not found/executable: $DOCKER_RUNNER" >&2; exit 1; }
else
  need_cmd whisperx
fi

mkdir -p "$CLEAN_DIR" "$OUTPUT_DIR"

INPUT_DIR_ABS="$(cd "$INPUT_DIR" && pwd)"

echo "Mode: $MODE"
echo "Input:  $INPUT_DIR_ABS (.$ext)"
echo "Clean:  $CLEAN_DIR"
echo "Output: $OUTPUT_DIR"
echo "ffmpeg jobs=$JOBS, whisper jobs=$WHISPER_JOBS"
echo "skip-clean=$SKIP_CLEAN_EXISTING force-clean=$FORCE_CLEAN skip-transcribe=$SKIP_TRANSCRIBE_EXISTING"

shopt -s nullglob nocaseglob

# Compute a stable hash for the current filter chain
FILTER_SHA="$(printf "%s" "$AUDIO_FILTER" | sha256sum | awk '{print $1}')"

meta_path_for() {
  local out_wav="$1"
  echo "${out_wav}.meta"
}

write_clean_meta() {
  local in="$1"
  local out="$2"
  local meta
  meta="$(meta_path_for "$out")"
  local sz mt
  sz="$(stat -c '%s' "$in")"
  mt="$(stat -c '%Y' "$in")"
  cat > "$meta" <<EOF
filter_sha=$FILTER_SHA
input_size=$sz
input_mtime=$mt
EOF
}

clean_meta_matches() {
  local in="$1"
  local out="$2"
  local meta
  meta="$(meta_path_for "$out")"
  [[ -f "$out" && -f "$meta" ]] || return 1

  # shellcheck disable=SC1090
  source "$meta" || return 1

  local sz mt
  sz="$(stat -c '%s' "$in")"
  mt="$(stat -c '%Y' "$in")"

  [[ "${filter_sha:-}" == "$FILTER_SHA" ]] || return 1
  [[ "${input_size:-}" == "$sz" ]] || return 1
  [[ "${input_mtime:-}" == "$mt" ]] || return 1
  return 0
}

convert_one() {
  local in="$1"
  local base stem out
  base="$(basename "$in")"
  stem="${base%.*}"
  out="${CLEAN_DIR%/}/${stem}_clean.wav"

  if [[ "$FORCE_CLEAN" -eq 0 && "$SKIP_CLEAN_EXISTING" -eq 1 ]]; then
    if clean_meta_matches "$in" "$out"; then
      echo "Skipping clean (match): $out"
      return 0
    fi
    if [[ -f "$out" ]]; then
      echo "Re-cleaning (metadata mismatch or missing): $out"
    fi
  fi

  echo "Cleaning: $in -> $out"
  ffmpeg -hide_banner -loglevel error -y \
    -i "$in" -vn \
    -ac 1 -ar 16000 -c:a pcm_s16le \
    -af "$AUDIO_FILTER" \
    "$out"

  write_clean_meta "$in" "$out"
}

output_exists_for_clean_wav() {
  local w="$1"
  local base
  base="$(basename "$w")"
  base="${base%.*}"  # e.g., foo_clean

  [[ -f "${OUTPUT_DIR%/}/${base}.json" ]] && return 0
  [[ -f "${OUTPUT_DIR%/}/${base}.srt" ]] && return 0
  return 1
}

export -f convert_one write_clean_meta clean_meta_matches meta_path_for output_exists_for_clean_wav
export AUDIO_FILTER CLEAN_DIR FILTER_SHA SKIP_CLEAN_EXISTING FORCE_CLEAN OUTPUT_DIR SKIP_TRANSCRIBE_EXISTING

# Cleaning step (parallel)
num_inputs="$(find "$INPUT_DIR_ABS" -maxdepth 1 -type f -iname "*.${ext}" | wc -l | tr -d ' ')"
if [[ "$num_inputs" == "0" ]]; then
  echo "No input files found for extension: .$ext in $INPUT_DIR_ABS"
  exit 0
fi

find "$INPUT_DIR_ABS" -maxdepth 1 -type f -iname "*.${ext}" -print0 \
  | xargs -0 -I {} -P "$JOBS" bash -c 'convert_one "$@"' _ "{}"

# Whisper step
num_cleaned="$(find "$CLEAN_DIR" -type f -name "*_clean.wav" | wc -l | tr -d ' ')"
if [[ "$num_cleaned" == "0" ]]; then
  echo "No cleaned wavs found in $CLEAN_DIR (unexpected)."
  exit 1
fi

transcribe_one_native() {
  local w="$1"

  if [[ "$SKIP_TRANSCRIBE_EXISTING" -eq 1 ]] && output_exists_for_clean_wav "$w"; then
    echo "Skipping transcribe (exists): $w"
    return 0
  fi

  echo "Transcribing (native): $w"
  if [[ "$DIARIZE" -eq 1 ]]; then
    whisperx "$w" \
      --model "$WHISPER_MODEL" \
      --device "$DEVICE" \
      --compute_type "$COMPUTE_TYPE" \
      --batch_size "$BATCH_SIZE" \
      --diarize \
      --hf_token "$HUGGINGFACE_TOKEN" \
      --output_dir "$OUTPUT_DIR"
  else
    whisperx "$w" \
      --model "$WHISPER_MODEL" \
      --device "$DEVICE" \
      --compute_type "$COMPUTE_TYPE" \
      --batch_size "$BATCH_SIZE" \
      --output_dir "$OUTPUT_DIR"
  fi
}

transcribe_one_docker() {
  local w="$1"

  if [[ "$SKIP_TRANSCRIBE_EXISTING" -eq 1 ]] && output_exists_for_clean_wav "$w"; then
    echo "Skipping transcribe (exists): $w"
    return 0
  fi

  echo "Transcribing (docker): $w"
  local diarize_flag="--diarize"
  [[ "$DIARIZE" -eq 0 ]] && diarize_flag="--no-diarize"

  if ! "$DOCKER_RUNNER" \
    --image "$DOCKER_IMAGE" \
    --cache-dir "$DOCKER_CACHE" \
    --workdir "$PWD" \
    --model "$WHISPER_MODEL" \
    --device "$DEVICE" \
    --compute-type "$COMPUTE_TYPE" \
    --batch-size "$BATCH_SIZE" \
    $diarize_flag \
    "$w" "$OUTPUT_DIR"
  then
    echo "WARN: diarization run failed for $w" >&2
    if [[ "$DIARIZE" -eq 1 ]]; then
      echo "Retrying without diarization so outputs are produced..." >&2
      "$DOCKER_RUNNER" \
        --image "$DOCKER_IMAGE" \
        --cache-dir "$DOCKER_CACHE" \
        --workdir "$PWD" \
        --model "$WHISPER_MODEL" \
        --device "$DEVICE" \
        --compute-type "$COMPUTE_TYPE" \
        --batch-size "$BATCH_SIZE" \
        --no-diarize \
        "$w" "$OUTPUT_DIR"
    else
      return 1
    fi
  fi
}

export -f transcribe_one_native transcribe_one_docker
export WHISPER_MODEL DEVICE COMPUTE_TYPE BATCH_SIZE DIARIZE DOCKER_IMAGE DOCKER_CACHE DOCKER_RUNNER HUGGINGFACE_TOKEN

if [[ "$WHISPER_JOBS" -eq 1 ]]; then
  while IFS= read -r -d '' w; do
    if [[ "$MODE" == "docker" ]]; then
      transcribe_one_docker "$w"
    else
      transcribe_one_native "$w"
    fi
  done < <(find "$CLEAN_DIR" -type f -name "*_clean.wav" -print0)
else
  if [[ "$MODE" == "docker" ]]; then
    find "$CLEAN_DIR" -type f -name "*_clean.wav" -print0 \
      | xargs -0 -I {} -P "$WHISPER_JOBS" bash -c 'transcribe_one_docker "$@"' _ "{}"
  else
    find "$CLEAN_DIR" -type f -name "*_clean.wav" -print0 \
      | xargs -0 -I {} -P "$WHISPER_JOBS" bash -c 'transcribe_one_native "$@"' _ "{}"
  fi
fi

echo "Done. Cleaned audio: $CLEAN_DIR   Outputs: $OUTPUT_DIR"
