# whisperX-batch

**Status:** Active stabilization. Tags `v0.1.0` and `v0.2.0` are historical
stack checkpoints, not a promise that current `main` is a compatible packaged
release. The next release boundary will be cut only after offline CI and a
reproducible GPU validation record agree.

`whisperX-batch` is the Docker-first batch transcription and benchmark harness I
use for local GPU work. It turns an explicit config into a WhisperX invocation,
keeps models and caches outside the image, processes one visible GPU per run,
and records sweep-level WER and throughput data.

The interesting part is the control plane around WhisperX: predictable
arguments, read-only inputs/model mounts, offline cache behavior, bounded Docker
command sizes, resume semantics, and benchmark bookkeeping. It does not contain
a speech model and does not claim a new ASR method.

## Design boundary

```text
config.local.toml + CLI
          |
          v
  transcribe control plane ----> one Docker process / one visible GPU
          |                             |
          |                             +--> local model + cache mounts
          |                             +--> read-only audio mounts
          v
  transcript artifacts <-------- /mnt/output
          |
          v
  whisperx-benchmark ----> WER, throughput, optional GPU trace
```

- Each `transcribe` invocation is single-process and owns one selected GPU.
- Multi-GPU work is explicit: launch independent invocations over independent
  shards rather than relying on a hidden scheduler.
- The image contains the runtime, not model snapshots, datasets, credentials, or
  personal audio.
- File-level output detection can skip completed work; it is not a transactional
  job database.
- Docker, GPU, model, dataset, audio, and network operations are never part of
  the default test target.

See [Architecture](docs/ARCHITECTURE.md) for the component and failure model.

## Stack checkpoint

The current Dockerfile is based on CUDA 12.8.1 and Python 3.11 with pinned
Torch 2.8.0, torchaudio 2.8.0, CTranslate2 4.7.1, faster-whisper 1.2.1,
WhisperX 3.8.2, and pyannote-audio 4.0.4. Triton and Transformers currently use
lower bounds rather than exact pins, so rebuilding the image is not yet an
immutable reproduction of the historical v0.2 stack.

That remaining dependency lock is a release-hardening item, not something the
current README quietly calls reproducible.

## Local configuration

The tracked file is a portable template. The real config is ignored because it
usually contains local paths and may reveal model, dataset, and host layout.

```bash
cp config.example.toml config.local.toml
${EDITOR:-vi} config.local.toml
```

Both entrypoints default to `config.local.toml`. An explicit config can be used
instead:

```bash
./transcribe --config /path/to/config.local.toml --input-dir /path/to/audio
./whisperx-benchmark --config /path/to/config.local.toml \
  --dataset /path/to/LibriSpeech/dev-clean
```

## Build and run

Building the runtime image downloads packages and requires separate network and
Docker authorization:

```bash
docker build \
  -f Dockerfile.whisperx-torch280-cu128 \
  -t whisperx:torch280-cu128 .
```

Validate local runtime prerequisites, then install symlinks into
`~/.local/bin`:

```bash
make preflight
make install
```

A typical non-diarized run is:

```bash
transcribe \
  --input-dir /path/to/audio \
  --output-dir /path/to/output \
  --cuda-devices 0 \
  --batch-size 16 \
  --no-diarize \
  --skip-transcribe-existing
```

Use `transcribe --help` and `whisperx-benchmark --help` for the current option
surface. [CLI notes](docs/CLI.md) describe precedence, mounts, output detection,
and sweep behavior without duplicating generated help text here.

## Offline development check

```bash
make test
```

This uses only Python's standard library. It tests config precedence, argument
construction, file discovery and output detection, Docker command construction,
sweep parsing, manifests, WER, result parsing, GPU-trace summaries, and safe
helper behavior. It does not build or start a container and does not access a
GPU, model, dataset, audio file, or network.

## Benchmarks

`whisperx-benchmark` supports original LibriSpeech FLAC input, Cartesian
parameter sweeps, corpus and macro WER with S/D/I/H counts, source-duration-based
RTF and x-realtime throughput, strict output accounting, run manifests, CSV/JSON
results, and optional `nvidia-smi` traces. A benchmark
is publishable evidence only when it records the repository commit, image
digest, dependency/model/corpus revisions, exact config and command, hardware
context, repetitions, correctness result, resource measurements, failures, and
limitations.

The old `batch_size=16`, `beam_size=1`, `best_of=1`, `temperature=0.0`,
`suppress_numerals=false`, non-diarized settings are retained as a historical
maintainer observation from a 200-file LibriSpeech `dev-clean` RTX 3090 sweep on
2026-03-14. The repository does not currently contain enough raw evidence to
present that observation as a reproducible comparative result.

See [Benchmark publication](docs/BENCHMARKING.md).

Prepare and verify the four official evaluation subsets explicitly (never as a
test prerequisite):

```bash
make dataset-setup
make dataset-verify
python3 scripts/build_librispeech_longform.py \
  --source /data/datasets/LibriSpeech/dev-clean \
  --destination /data/datasets/LibriSpeech-longform/dev-clean
```

`corpus_wer` is the primary word-error metric; `macro_wer` is the mean of
per-file WERs. `rtf` is wall time divided by source audio time, while
`x_realtime` is its inverse. Words are never called tokens: exact tokenizer
throughput is left unavailable unless the exact model tokenizer is invoked.

## Privacy and limitations

- Audio, transcripts, model caches, tokens, local configs, and benchmark outputs
  are intentionally ignored and must be reviewed before sharing.
- The helpers can download public datasets or cache artifacts, but only when run
  explicitly. Dataset/model terms still apply.
- Diarization can process sensitive voice identity information. This repository
  supplies mechanics, not consent or a retention policy for someone else's
  recordings.
- WER on one public corpus does not establish accuracy for other speakers,
  languages, recording conditions, or high-stakes use.
- No filtering, queue service, distributed scheduler, or automatic multi-GPU
  coordination is promised.

See [Publication, privacy, and research notes](docs/PUBLICATION.md) and
[provenance](docs/PROVENANCE.md).

## Project posture

This is a personal tool. Focused fixes and reproducible reports are useful, but
there is no support or response-time commitment. The aim is a small harness that
works predictably on its documented local stack, not broad packaging or adoption.

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CHANGELOG.md](CHANGELOG.md).

## License

The repository's original orchestration code and documentation are available
under the [MIT License](LICENSE). Models, datasets, base images, and Python
packages retain their own terms.
