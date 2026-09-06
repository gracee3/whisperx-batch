# Benchmark publication

## Metric definitions and integrity gate

The primary accuracy metric is corpus WER: `(S + D + I) / reference_words`,
with substitutions, deletions, insertions, hits, reference words, and hypothesis
words retained. `macro_wer` is the arithmetic mean of scored file WERs and is a
secondary diagnostic. A run with a missing, duplicate, unexpected, empty, or
unscored output is failed; it is never silently included in a successful result.

Source duration comes from `ffprobe` on the input media. End-to-end wall time is
measured with `time.perf_counter()` around the complete `transcribe` child
process. Conventional `rtf = wall_s / audio_s`; `x_realtime` and
`audio_s_per_s = audio_s / wall_s`. Internal startup, model load, VAD,
transcription, alignment, and diarization durations remain unavailable unless
the runtime supplies trustworthy instrumentation; they are not inferred by
subtraction.

The removed historical `*_toks_per_sec` columns counted normalized words and
were misleading. Current results report words separately. Token throughput is
null and labeled unavailable unless the exact model tokenizer is explicitly
invoked; post-hoc tokenizer counts, when added, must be labeled final-hypothesis
tokenizer throughput and never raw decoder-kernel throughput.

## Dataset modes

Standard mode consumes original LibriSpeech FLAC utterances and official
references. `scripts/librispeech_corpus.py` downloads the four evaluation
archives from OpenSLR with resume support, verifies OpenSLR-published MD5s,
safely extracts them, checks one-to-one audio/reference membership, and writes a
checksum-addressed inventory. It performs no network operation in default tests.

Constructed long-form mode uses `scripts/build_librispeech_longform.py`. It
groups utterances by speaker/chapter, sorts official utterance IDs, concatenates
audio through a lossless FLAC decode/re-encode into one valid stream, requires
the chapter duration to equal the sum of source durations, stitches references
in the same order, and writes a per-chapter source mapping plus manifest checksum. These are
constructed chapter-level results, not an official upstream long-form task.

Batch size controls CTranslate2 batching inside one process on one visible GPU.
GPU worker count instead means independent duration-balanced, disjoint shards,
one process and output directory per physical GPU. Aggregate claims require a
shared wall interval, successful workers, combined integrity validation, and a
measured comparison with the equivalent one-GPU workload.

Development configuration may be selected only on `dev-clean`/`dev-other`.
After locking, `test-clean`/`test-other` are evaluation-only. Final evidence
uses an excluded warmup and at least three measured repetitions where practical,
reporting individual values plus median, min, max, mean, and population standard
deviation. Cache state and run order must be explicit.

The benchmark harness is useful for local tuning, but a CSV alone is not a
portable result.

For a publishable run, record:

- repository commit and clean-tree state;
- Dockerfile plus immutable image digest and resolved package inventory;
- driver, CUDA runtime, GPU model/count, CPU, RAM, and relevant storage;
- model, alignment, diarization, corpus, and scoring revisions and licenses;
- exact redacted config and command, including device/shard ownership;
- input subset, file count, duration distribution, and excluded/failed files;
- warmup policy, repetitions, ordering/randomization, and summary dispersion;
- transcript correctness/WER alongside runtime, throughput, VRAM, utilization,
  and trace sampling interval;
- whether caches were cold/warm and whether any network access occurred;
- raw machine-readable results small and safe enough for review.

Compare equivalent workloads. Independent GPU shards do not establish linear
scaling unless aggregate throughput, I/O contention, failures, and result
integrity were measured together.

## Historical observation

A 2026-03-14 maintainer sweep over 200 LibriSpeech `dev-clean` files on one RTX
3090 led to the current tuning defaults: batch size 16, beam size 1, best-of 1,
temperature 0, numeral suppression off, and diarization off. The repository does
not currently preserve the complete command, immutable image, raw repetitions,
failure inventory, or machine-readable evidence required to reproduce that
comparison. Treat the settings as a starting point, not a performance claim.
