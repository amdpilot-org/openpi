#!/usr/bin/env python3
"""Stage0 harness for issue-openpi-1-r37.

Runs the issue-specified OpenPI Pi0 B=1 policy inference benchmark and extracts
P50 latency plus throughput. The harness does not alter workload size or metric
semantics. It maps the assigned physical GPU to logical cuda:0 via the container
HIP_VISIBLE_DEVICES setting and therefore passes --gpu 0 to the benchmark.
"""
import json
import os
import re
import subprocess
import sys

cmd = [
    sys.executable,
    "scripts/benchmark_policy_inference.py",
    "--batch-size", os.environ.get("OPENPI_BATCH_SIZE", "1"),
    "--warmup", os.environ.get("OPENPI_WARMUP", "10"),
    "--iterations", os.environ.get("OPENPI_ITERATIONS", "30"),
    "--timing", os.environ.get("OPENPI_TIMING", "cuda_event"),
]
if "OPENPI_GPU_ARG" in os.environ:
    cmd.extend(["--gpu", os.environ["OPENPI_GPU_ARG"]])
print("Running:", " ".join(cmd), flush=True)
proc = subprocess.run(
    cmd,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    timeout=int(os.environ.get("OPENPI_BENCH_TIMEOUT", "5400")),
)
print(proc.stdout)
if proc.returncode != 0:
    sys.exit(proc.returncode)

p50 = re.search(r"^P50:\s*([0-9.]+)\s*ms", proc.stdout, re.M)
hz = re.search(r"^Throughput:\s*([0-9.]+)\s*Hz", proc.stdout, re.M)
shape = re.search(r"^Actions shape:\s*(\([^\n]+\))", proc.stdout, re.M)
if not p50:
    raise SystemExit("metric_not_extracted: P50 latency not found")
metric = float(p50.group(1))
if metric <= 0:
    raise SystemExit(f"invalid P50 latency {metric}")
result = {
    "metric_name": "p50_latency_ms",
    "metric_value": metric,
    "metric_direction": "lower",
    "throughput_hz": float(hz.group(1)) if hz else None,
    "actions_shape": shape.group(1) if shape else None,
}
print("STAGE0_RESULT " + json.dumps(result, sort_keys=True), flush=True)
