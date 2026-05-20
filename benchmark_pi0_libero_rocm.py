#!/usr/bin/env python3
"""Benchmark pi0 LIBERO inference on ROCm."""
import argparse
import os
import sys
import time

# Fix PYTHONPATH pollution before any imports.
if 'PYTHONPATH' in os.environ:
    del os.environ['PYTHONPATH']

# Filter current process sys.path and add openpi src.
sys.path = [p for p in sys.path if 'uv/archive' not in p]
if '/workspace/openpi/src' not in sys.path:
    sys.path.insert(0, '/workspace/openpi/src')

os.environ['USE_ROCM_AITER_ROPE_BACKEND'] = '0'

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import torch
from openpi.models.model import Observation
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


def run_benchmark(device, num_warmup, num_runs, compile_mode):
    print(f"Using device: {device}")
    print(f"torch.compile mode: {compile_mode}")

    config = Pi0Config(pytorch_compile_mode=compile_mode)
    model = PI0Pytorch(config).to(device)
    model.eval()

    bsize = 1
    obs = Observation(
        images={
            'base_0_rgb': torch.randn(bsize, 3, 224, 224, device=device),
            'left_wrist_0_rgb': torch.randn(bsize, 3, 224, 224, device=device),
            'right_wrist_0_rgb': torch.randn(bsize, 3, 224, 224, device=device),
        },
        image_masks={
            'base_0_rgb': torch.ones(bsize, dtype=torch.bool, device=device),
            'left_wrist_0_rgb': torch.ones(bsize, dtype=torch.bool, device=device),
            'right_wrist_0_rgb': torch.ones(bsize, dtype=torch.bool, device=device),
        },
        state=torch.randn(bsize, config.action_dim, device=device),
        tokenized_prompt=torch.zeros(bsize, config.max_token_len, dtype=torch.int32, device=device),
        tokenized_prompt_mask=torch.ones(bsize, config.max_token_len, dtype=torch.bool, device=device),
    )

    num_warmup = max(num_warmup, 10)
    num_runs = max(num_runs, 50)

    # Warmup runs
    print(f"Running {num_warmup} warmup iterations...")
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model.sample_actions(device, obs)
    if device == 'cuda':
        torch.cuda.synchronize()

    # Timed runs
    print(f"Running {num_runs} timed iterations...")
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model.sample_actions(device, obs)
    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    iters_per_second = num_runs / elapsed
    print(f"\nResults:")
    print(f"  Total inference time: {elapsed:.2f}s")
    print(f"  iters_per_second: {iters_per_second:.2f}")
    return iters_per_second


def main():
    parser = argparse.ArgumentParser(description="Benchmark pi0 LIBERO inference on ROCm")
    parser.add_argument("--config", type=str, default="pi0_libero", help="Config name (unused)")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint directory (unused)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--num-warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--num-runs", type=int, default=50, help="Number of timed iterations")
    parser.add_argument("--compile-mode", type=str, default="default", help="torch.compile mode")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    if not torch.cuda.is_available():
        print("WARNING: CUDA/ROCm not available, using CPU")

    score = run_benchmark(device, args.num_warmup, args.num_runs, args.compile_mode)
    print(f"\nscore: {score:.2f}")
    return 0


if __name__ == "__main__":
    exit(main())
