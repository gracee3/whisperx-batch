# CLI behavior

The entrypoints are executable Python scripts and require Python 3.11 or newer.
Use `--help` as the exact option reference.

## Configuration precedence

Both commands default to ignored `config.local.toml`. A missing file is treated
as an empty config; required runtime paths then fail during validation rather
than triggering a download.

Precedence is:

1. an explicit CLI value;
2. the relevant TOML section;
3. a portable built-in default.

`--no-config` applies to `transcribe`. The benchmark accepts an explicit
`--config` and can pass a separate `--transcribe-config` into each run.

## Input and output behavior

Supported audio suffixes are WAV, M4A, MP3, and FLAC, case-insensitively.
Discovery is nonrecursive unless `--recursive` is set. Duplicate basenames are
renamed inside the container (`name.wav`, `name_1.wav`, and so on) so one batch
does not mount two inputs at the same path.

`--skip-transcribe-existing` skips an input when any requested output extension
already exists for the mapped stem. It does not validate output completeness,
input hashes, config identity, or model identity.

## Docker boundary

- Docker receives no inherited standard input and runs without interactive
  mode. This preserves the remaining records when a caller drives transcription
  from a pipe or a shell loop.
- Input files and discovered model/alignment/diarization directories are
  mounted read-only.
- The output directory and cache root are writable.
- Hugging Face, Transformers, and dataset offline environment flags are set in
  the container.
- `--docker-pull-policy missing` avoids an automatic pull when the image exists;
  `always` is explicitly networked behavior.
- Large input sets are split to keep the Docker command below a practical mount
  count.

## Sweep behavior

`--set key=value` fixes one value. `--sweep key=a,b` supplies candidates, and
multiple sweep keys form a Cartesian product. Underscores in keys normalize to
hyphens. CUDA-device lists are kept as one value instead of being mistaken for a
sweep axis.

Known Boolean toggles become present/absent flags. Other Boolean-looking values
remain key/value arguments because WhisperX has options whose Boolean state is
encoded as text.
