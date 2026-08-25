#!/usr/bin/env python3
"""Run excluded warmups plus measured repetitions and summarize dispersion."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmark_core import repetition_stats


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", type=Path, required=True)
  parser.add_argument("--output-root", type=Path, required=True)
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--gpu", default="0")
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--beam-size", type=int, default=1)
  parser.add_argument("--warmups", type=int, default=1)
  parser.add_argument("--repetitions", type=int, default=3)
  parser.add_argument("--alignment", choices=("enabled", "disabled"), default="disabled")
  parser.add_argument("--cache-state-label", choices=("cold-cache", "warm-cache"), default="warm-cache",
    help="Recorded operator-controlled cache state; this command never deletes caches")
  parser.add_argument("--trace-interval", type=float, default=2.0)
  args = parser.parse_args()
  if args.warmups < 0 or args.repetitions < 1:
    parser.error("warmups must be nonnegative and repetitions positive")
  root = args.output_root.resolve()
  records = []
  total = args.warmups + args.repetitions
  for index in range(total):
    measured = index >= args.warmups
    label = f"measured-{index - args.warmups + 1:02d}" if measured else f"warmup-{index + 1:02d}"
    output = root / label
    cmd = [str(Path(__file__).resolve().parent.parent / "whisperx-benchmark"), "--config", str(args.config.resolve()),
      "--dataset", str(args.dataset.resolve()), "--ext", "flac", "--output-root", str(output),
      "--batch-size", str(args.batch_size), "--set", f"cuda-devices={args.gpu}", "--set", f"beam-size={args.beam_size}",
      "--set", f"no-align={'false' if args.alignment == 'enabled' else 'true'}", "--trace", "--trace-interval", str(args.trace_interval)]
    proc = subprocess.run(cmd, stdin=subprocess.DEVNULL)
    summary_path = output / "run_001" / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"status": "failed"}
    records.append({"label": label, "measured": measured, "returncode": proc.returncode, "summary": summary, "command": cmd})
    if proc.returncode or summary.get("status") != "ok":
      (root / "repetitions.jsonl").parent.mkdir(parents=True, exist_ok=True)
      with (root / "repetitions.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
          stream.write(json.dumps(record) + "\n")
      print(f"failed at {label}", file=sys.stderr)
      return 1
  measured_records = [record["summary"] for record in records if record["measured"]]
  metrics = {}
  for key in ("runtime_s", "rtf", "x_realtime", "words_per_s", "hypothesis_tokens_per_s_e2e"):
    values = [float(record[key]) for record in measured_records if record.get(key) is not None]
    metrics[key] = repetition_stats(values) if values else None
  corpus_wers = {record.get("corpus_wer") for record in measured_records}
  payload = {"status": "ok", "warmups_excluded": args.warmups, "measured_repetitions": args.repetitions,
    "cache_state": args.cache_state_label,
    "run_order": [record["label"] for record in records], "metrics": metrics,
    "corpus_wer_values": sorted(corpus_wers), "configuration": {"gpu": args.gpu, "batch_size": args.batch_size, "beam_size": args.beam_size, "alignment": args.alignment}}
  root.mkdir(parents=True, exist_ok=True)
  with (root / "repetitions.jsonl").open("w", encoding="utf-8") as stream:
    for record in records:
      stream.write(json.dumps(record) + "\n")
  (root / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
  print(json.dumps(payload, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
