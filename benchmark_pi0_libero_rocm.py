#!/usr/bin/env python3
"""Canonical pi0 LIBERO ROCm benchmark for OpenPI.

This script intentionally keeps the issue workload fixed: config pi0_libero,
PyTorch checkpoint, bf16/default model settings, batch=1 random LIBERO example,
3 warmup and 20 timed inference calls unless overridden by CLI.
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="pi0_libero")
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-warmup", type=int, default=3)
    p.add_argument("--num-runs", type=int, default=20)
    p.add_argument("--backend", default="pytorch")
    p.add_argument("--json-out", default="")
    args = p.parse_args()
    if args.backend != "pytorch":
        raise SystemExit("Only --backend pytorch is supported for this issue")

    import torch
    from openpi.policies import libero_policy
    from openpi.policies import policy_config
    from openpi.training import config as openpi_config

    if not torch.cuda.is_available():
        raise SystemExit("ERROR: torch.cuda/ROCm device is not available")
    ckpt = Path(os.path.expanduser(args.checkpoint_dir))
    if not (ckpt / "model.safetensors").exists():
        raise SystemExit(f"ERROR: PyTorch checkpoint missing model.safetensors under {ckpt}")
    if not (ckpt / "assets").exists():
        raise SystemExit(f"ERROR: checkpoint assets directory missing under {ckpt}")

    np.random.seed(2)
    torch.manual_seed(2)
    print(f"Loading OpenPI config={args.config} checkpoint={ckpt} device={args.device}", flush=True)
    train_cfg = openpi_config.get_config(args.config)
    policy = policy_config.create_trained_policy(train_cfg, ckpt, pytorch_device=args.device)
    obs = libero_policy.make_libero_example()

    # Warmup includes any first-call torch.compile/autotune cost, matching the issue contract.
    warmup_times = []
    for i in range(args.num_warmup):
        t0 = time.perf_counter()
        out = policy.infer(obs)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        warmup_times.append(dt)
        print(f"warmup {i+1}/{args.num_warmup}: {dt:.6f}s action_shape={np.asarray(out['actions']).shape}", flush=True)

    times = []
    for i in range(args.num_runs):
        t0 = time.perf_counter()
        out = policy.infer(obs)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"run {i+1}/{args.num_runs}: {dt:.6f}s", flush=True)

    actions = np.asarray(out["actions"])
    expected_shape = (train_cfg.model.action_horizon, 7)
    if actions.shape != expected_shape:
        raise SystemExit(f"ERROR: expected LIBERO action shape {expected_shape}, got {actions.shape}")
    mean_s = statistics.mean(times)
    result = {
        "schema_version": "openpi_pi0_libero_bench.v1",
        "config": args.config,
        "backend": args.backend,
        "checkpoint_dir": str(ckpt),
        "device": args.device,
        "num_warmup": args.num_warmup,
        "num_runs": args.num_runs,
        "mean_latency_s": mean_s,
        "median_latency_s": statistics.median(times),
        "min_latency_s": min(times),
        "max_latency_s": max(times),
        "throughput_inf_s": 1.0 / mean_s,
        "warmup_total_s": sum(warmup_times),
        "torch_version": torch.__version__,
        "torch_hip": torch.version.hip,
        "gpu_name": torch.cuda.get_device_name(0),
        "action_shape": list(actions.shape),
        "action_horizon": int(train_cfg.model.action_horizon),
        "action_checksum": float(np.sum(actions)),
        "metric_name": "throughput_inf_s",
        "metric_value": 1.0 / mean_s,
        "metric_direction": "higher",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"METRIC throughput_inf_s={result['throughput_inf_s']:.6f}")
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
