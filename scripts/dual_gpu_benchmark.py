#!/usr/bin/env python3
"""Run two independent, explicitly assigned benchmark workers."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmark_core import ManifestItem, aggregate_workers, duration_balanced_shards, manifest_checksum, probe_duration


def references(dataset: Path) -> dict[str, str]:
  result: dict[str, str] = {}
  for path in sorted(dataset.rglob("*.trans.txt")):
    for line in path.read_text(encoding="utf-8").splitlines():
      if line.strip():
        key, text = line.split(maxsplit=1)
        if key in result:
          raise ValueError(f"duplicate reference: {key}")
        result[key] = text
  return result


def make_shards(dataset: Path, root: Path, ext: str, limit: int) -> list[list[ManifestItem]]:
  refs = references(dataset)
  audio = sorted(dataset.rglob(f"*.{ext}"))
  if limit:
    audio = audio[:limit]
  items = [ManifestItem(p.stem, str(p.resolve()), refs[p.stem], probe_duration(p)) for p in audio]
  shards = duration_balanced_shards(items, 2)
  for index, shard in enumerate(shards):
    shard_root = root / "shards" / f"worker-{index}" / "input"
    shard_root.mkdir(parents=True, exist_ok=True)
    lines = []
    for item in shard:
      destination = shard_root / Path(item.audio_path).name
      if not destination.exists():
        destination.hardlink_to(item.audio_path)
      lines.append(f"{item.item_id} {item.reference}")
    (shard_root / "shard.trans.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (shard_root.parent / "manifest.json").write_text(json.dumps({"items": [x.canonical() for x in shard], "manifest_sha256": manifest_checksum(shard)}, indent=2) + "\n", encoding="utf-8")
  return shards


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", type=Path, required=True)
  parser.add_argument("--output-root", type=Path, required=True)
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--gpus", default="0,1")
  parser.add_argument("--ext", choices=("flac", "wav"), default="flac")
  parser.add_argument("--max-files", type=int, default=0)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--beam-size", type=int, default=1)
  parser.add_argument("--no-align", action="store_true")
  parser.add_argument("--single-baseline-summary", type=Path, help="Equivalent one-GPU summary.json used for measured speedup.")
  args = parser.parse_args()
  gpu_ids = [x.strip() for x in args.gpus.split(",") if x.strip()]
  if len(gpu_ids) != 2 or gpu_ids[0] == gpu_ids[1]:
    parser.error("--gpus must name two distinct physical GPU indices")
  output = args.output_root.resolve()
  gpu_identities = []
  for gpu in gpu_ids:
    proc = subprocess.run(["nvidia-smi", "-i", gpu, "--query-gpu=index,uuid,name", "--format=csv,noheader"], capture_output=True, text=True)
    if proc.returncode or not proc.stdout.strip():
      parser.error(f"could not resolve physical GPU identity for index {gpu}")
    index, uuid, name = [value.strip() for value in proc.stdout.strip().split(",", 2)]
    gpu_identities.append({"index": index, "uuid": uuid, "name": name})
  shards = make_shards(args.dataset.resolve(), output, args.ext, args.max_files)
  commands = []
  for index, gpu in enumerate(gpu_ids):
    cmd = [str(Path(__file__).resolve().parent.parent / "whisperx-benchmark"), "--config", str(args.config.resolve()),
      "--dataset", str(output / "shards" / f"worker-{index}" / "input"), "--output-root", str(output / "shards" / f"worker-{index}" / "run"),
      "--ext", args.ext, "--batch-size", str(args.batch_size), "--set", f"cuda-devices={gpu}", "--set", f"beam-size={args.beam_size}", "--trace"]
    if args.no_align:
      cmd += ["--set", "no-align=true"]
    commands.append(cmd)
  start = time.perf_counter()
  processes = [subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for cmd in commands]
  outputs = [process.communicate() for process in processes]
  shared_wall = time.perf_counter() - start
  workers = []
  for index, (process, (stdout, _)) in enumerate(zip(processes, outputs)):
    (output / "shards" / f"worker-{index}" / "worker.log").write_text(stdout or "", encoding="utf-8")
    summary_path = output / "shards" / f"worker-{index}" / "run" / "run_001" / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary.update(worker_id=str(index), returncode=process.returncode)
    if process.returncode or summary.get("status") != "ok":
      summary["status"] = "failed"
    workers.append(summary)
  try:
    aggregate = aggregate_workers(workers, shared_wall)
    if args.single_baseline_summary:
      baseline = json.loads(args.single_baseline_summary.read_text(encoding="utf-8"))
      baseline_x = float(baseline["x_realtime"])
      aggregate["measured_speedup_vs_single_gpu"] = float(aggregate["aggregate_x_realtime"]) / baseline_x
      aggregate["single_gpu_baseline_summary"] = str(args.single_baseline_summary.resolve())
    status = "ok"
  except ValueError as exc:
    aggregate = {"error": str(exc)}
    status = "failed"
  payload = {"status": status, "gpu_identities": gpu_identities, "workers": workers, "aggregate": aggregate, "commands": commands}
  (output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
  print(json.dumps({"status": status, "aggregate": aggregate}, indent=2))
  return 0 if status == "ok" else 1


if __name__ == "__main__":
  raise SystemExit(main())
