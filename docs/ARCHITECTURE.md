# Architecture

## Why this shape

WhisperX already owns model execution, alignment, and diarization. This
repository exists to make a particular local batch workflow predictable around
that engine. The control plane stays plain Python and the resource-heavy runtime
stays in a container so model/CUDA changes do not silently become orchestration
changes.

## Components

- `config_utils.py` loads TOML and resolves typed Boolean/value precedence.
- `script_defaults.py` maps local config and CLI values into the two entrypoint
  configurations.
- `transcribe` discovers input files, assigns collision-free container names,
  builds the exact WhisperX argument vector, groups mounts into bounded batches,
  and starts Docker.
- `whisperx-benchmark` builds public-corpus manifests, expands parameter sweeps,
  calls `transcribe`, scores output, and optionally samples `nvidia-smi`.
- `benchmark_utils.py` owns fixed/sweep option parsing and Boolean flag mapping;
  `benchmark_core.py` owns offline scoring, timing formulas, manifest hashes,
  integrity inventories, repetition statistics, and deterministic sharding.
- `scripts/` contains explicit dataset, cache, conversion, and cleaning helpers;
  none run as an implicit prerequisite.

## Data and control flow

Configuration precedence is CLI, then the selected TOML section, then a portable
built-in default. `config.local.toml` is ignored because paths can reveal host
and data layout.

`transcribe` enumerates supported audio, gives duplicate basenames stable unique
container names, optionally removes files with an existing requested output,
and chunks the remaining mount list. Each chunk becomes one Docker invocation.
Inputs and model paths are mounted read-only. Output and cache paths are the only
writable persistent mounts.

The constructed container shell command is fully quoted, but it remains an
argument to the image's shell entrypoint. Extra WhisperX arguments are an
advanced trusted-user interface, not a boundary for hostile input.

## Concurrency and GPU ownership

One invocation exposes one configured GPU (or Docker's `all` only when the
caller has not selected a CUDA device). WhisperX owns compute concurrency inside
the container. Multi-GPU throughput comes from explicit independent shards.
There is no shared queue, coordinator, or ordering contract between shards.

That choice keeps failures and output ownership obvious. Scaling claims must
measure aggregate I/O and GPU contention rather than multiplying a single-run
number.

## Failure and resume behavior

- Invalid/missing directories, models, diarization models, or large-v3 layout
  fail before Docker starts.
- A nonzero diarized Docker run is retried once without `--diarize`; this is a
  coarse recovery path and must remain visible in logs/evidence.
- `--skip-transcribe-existing` is artifact-presence detection, not content hash
  validation. Partial or stale artifacts can therefore require manual removal.
- Benchmark runs preserve a per-combination directory and mark missing outputs;
  they do not prove that failures are safe to resume automatically.

## Performance-sensitive boundaries

Model inference dominates normal runtime, but the architecture also exposes
mount-list size, Docker startup, input copying in the benchmark harness, output
serialization, WER scoring, and trace sampling. Optimize those only after
profiling a pinned workload. Correct transcript association and complete result
accounting are gates for any throughput improvement.

## Trust and privacy

The operator is trusted. Audio, transcript text, local paths, model terms, and
tokens are sensitive even on an offline host. Docker isolation here is a
packaging boundary, not a sandbox claim. Images and Python/model dependencies
execute code with access to the explicitly mounted data.
