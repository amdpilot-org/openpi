#!/usr/bin/env python3
"""Canonical pi0_libero PyTorch ROCm benchmark harness.

The issue names this script and CLI as the benchmark contract. Upstream OpenPI
at the pinned commit does not ship this exact file, so Stage0 provides the
minimal wrapper that loads the upstream model/policy path and reports the
required throughput metric.
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.policies.libero_policy import make_libero_example
from openpi.policies.policy import Policy
from openpi.training import config as training_config


def load_model(checkpoint_dir: str, device: str):
    train_config = training_config.get_config("pi0_libero")
    model_config = train_config.model
    logging.info(
        "Model config: action_dim=%s action_horizon=%s compile_mode=%s",
        model_config.action_dim,
        model_config.action_horizon,
        model_config.pytorch_compile_mode,
    )
    model = PI0Pytorch(model_config).to(device).eval()
    checkpoint_path = Path(checkpoint_dir).expanduser() / "model.safetensors"
    if checkpoint_path.exists():
        logging.info("Loading checkpoint from %s", checkpoint_path)
        from safetensors.torch import load_file

        state_dict = load_file(str(checkpoint_path), device=device)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logging.info("Checkpoint loaded: missing=%d unexpected=%d", len(missing), len(unexpected))
    else:
        logging.warning("Checkpoint not found at %s; using initialized weights", checkpoint_path)
    return model, train_config


def create_policy(model, train_config, checkpoint_dir: str, device: str):
    ckpt = Path(checkpoint_dir).expanduser()
    assets_dir = ckpt / "assets"
    data_config = train_config.data.create(assets_dir, train_config.model)
    input_transforms = (*data_config.data_transforms.inputs, *data_config.model_transforms.inputs)
    output_transforms = data_config.data_transforms.outputs
    logging.info("Input transforms: %s", [type(t).__name__ for t in input_transforms])
    logging.info("Output transforms: %s", [type(t).__name__ for t in output_transforms])
    return Policy(
        model,
        transforms=input_transforms,
        output_transforms=output_transforms,
        pytorch_device=device,
        is_pytorch=True,
    )


def benchmark(policy, num_warmup: int, num_runs: int):
    input_data = make_libero_example()
    logging.info("Running warmup: %d", num_warmup)
    for i in range(num_warmup):
        _ = policy.infer(input_data)
        torch.cuda.synchronize()
        logging.info("Warmup %d/%d complete", i + 1, num_warmup)

    logging.info("Running timed runs: %d", num_runs)
    times_ms = []
    for i in range(num_runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = policy.infer(input_data)
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)
        times_ms.append(float(elapsed))
        logging.info("Run %d/%d: %.3f ms", i + 1, num_runs, elapsed)

    mean_ms = float(np.mean(times_ms))
    std_ms = float(np.std(times_ms))
    throughput = 1000.0 / mean_ms
    return throughput, mean_ms, std_ms, times_ms


def main():
    parser = argparse.ArgumentParser(description="Benchmark pi0_libero inference on ROCm")
    parser.add_argument("--config", type=str, default="pi0_libero")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-warmup", type=int, default=3)
    parser.add_argument("--num-runs", type=int, default=20)
    parser.add_argument("--result-json", type=str, default="/tmp/openpi_result.json")
    args = parser.parse_args()
    if args.config != "pi0_libero":
        raise ValueError("Stage0 benchmark contract requires --config pi0_libero")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info("Using torch=%s hip=%s cuda_available=%s device_count=%s", torch.__version__, torch.version.hip, torch.cuda.is_available(), torch.cuda.device_count())
    logging.info("Using device: %s", args.device)
    logging.info("TORCH_BLAS_PREFER_HIPBLASLT=%s", os.environ.get("TORCH_BLAS_PREFER_HIPBLASLT"))

    model, train_config = load_model(args.checkpoint_dir, args.device)
    policy = create_policy(model, train_config, args.checkpoint_dir, args.device)
    started = time.time()
    throughput, mean_ms, std_ms, times_ms = benchmark(policy, args.num_warmup, args.num_runs)
    elapsed_s = time.time() - started

    print("\n=== Benchmark Results ===")
    print(f"Mean inference time: {mean_ms:.2f} ms")
    print(f"Std inference time: {std_ms:.2f} ms")
    print(f"Throughput: {throughput:.2f} inf/s")
    print(f"SCORE: {throughput:.4f} throughput_inf_s")

    result = {
        "throughput_inf_s": throughput,
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "times_ms": times_ms,
        "elapsed_s": elapsed_s,
        "num_warmup": args.num_warmup,
        "num_runs": args.num_runs,
        "device": args.device,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "compile_mode": train_config.model.pytorch_compile_mode,
    }
    Path(args.result_json).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
