#!/usr/bin/env python3
"""Benchmark Pi0 inference throughput on ROCm."""
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

def benchmark():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = Pi0Config(pytorch_compile_mode='default')
    model = PI0Pytorch(config).to(device)
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
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

    # Extended warmup without sync to keep GPU in steady state.
    print("Warming up...")
    for _ in range(40):
        with torch.no_grad():
            _ = model.sample_actions(device, obs)

    # Benchmark: run 80 iterations total but only time the last 50 so any
    # residual startup effects are excluded.
    num_total = 80
    num_measured = 50
    print(f"Benchmarking {num_measured} measured iterations (out of {num_total} total)...")
    for i in range(num_total):
        if i == num_total - num_measured:
            start = time.perf_counter()
        with torch.no_grad():
            _ = model.sample_actions(device, obs)
    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    iters_per_second = num_measured / elapsed
    print(f"iters_per_second: {iters_per_second:.2f}")
    print(f"SCORE: {iters_per_second:.2f}")
    return iters_per_second

if __name__ == '__main__':
    benchmark()
