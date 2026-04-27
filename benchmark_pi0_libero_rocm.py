#!/usr/bin/env python3
"""Benchmark script for pi0_libero inference on ROCm.

This script benchmarks pi0_libero model inference throughput.
It expects the checkpoint to be available at the specified path.
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models import pi0_config
from openpi.policies.libero_policy import LiberoInputs, LiberoOutputs, make_libero_example
from openpi.policies.policy import Policy
from openpi.models.tokenizer import PaligemmaTokenizer


def create_benchmark_input():
    """Create input using the standard Libero example format."""
    return make_libero_example()


def load_model(checkpoint_dir: str, device: str, compile_mode: str = "max-autotune"):
    """Load the pi0_libero model."""
    logging.info(f"Creating pi0_libero model...")
    
    # Use the config from the training config
    from openpi.training import config as training_config
    train_config = training_config.get_config("pi0_libero")
    model_config = train_config.model
    
    logging.info(f"Model config: action_dim={model_config.action_dim}, action_horizon={model_config.action_horizon}")
    
    # Create PyTorch model
    model = PI0Pytorch(model_config)
    model = model.to(device)
    model.eval()
    
    # Try to load checkpoint if it exists
    checkpoint_path = os.path.join(checkpoint_dir, "model.safetensors")
    if os.path.exists(checkpoint_path):
        logging.info(f"Loading checkpoint from {checkpoint_path}...")
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint_path)
        model.load_state_dict(state_dict, strict=False)
        logging.info("Checkpoint loaded successfully")
    else:
        logging.warning(f"Checkpoint not found at {checkpoint_path}. Using random weights.")
        logging.warning("Throughput measurements with random weights may not reflect real performance.")
    
    logging.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model, model_config


def create_policy(model, config, device: str):
    """Create a policy from the model."""
    input_transform = LiberoInputs(model_type=None)
    output_transform = LiberoOutputs()
    
    return Policy(
        model,
        transforms=[input_transform],
        output_transforms=[output_transform],
        pytorch_device=device,
        is_pytorch=True,
    )


def benchmark_inference(policy, device: str, num_warmup: int = 3, num_runs: int = 20):
    """Run inference benchmark."""
    logging.info(f"Running benchmark: {num_warmup} warmup, {num_runs} timed runs")
    
    input_data = create_benchmark_input()
    
    # Warmup
    logging.info("Running warmup...")
    for i in range(num_warmup):
        _ = policy.infer(input_data)
        torch.cuda.synchronize()
    
    # Timed runs
    logging.info("Running timed inference...")
    times = []
    for i in range(num_runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        _ = policy.infer(input_data)
        end.record()
        
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        times.append(elapsed_ms)
        
        if (i + 1) % 5 == 0:
            logging.info(f"  Run {i + 1}/{num_runs}: {elapsed_ms:.1f}ms")
    
    times_np = np.array(times)
    mean_time = np.mean(times_np)
    std_time = np.std(times_np)
    throughput = 1000.0 / mean_time
    
    logging.info(f"\n=== Benchmark Results ===")
    logging.info(f"Mean inference time: {mean_time:.2f}ms")
    logging.info(f"Std inference time: {std_time:.2f}ms")
    logging.info(f"Throughput: {throughput:.2f} inf/s")
    
    return throughput


def main():
    parser = argparse.ArgumentParser(description="Benchmark pi0_libero inference on ROCm")
    parser.add_argument("--config", type=str, default="pi0_libero")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-warmup", type=int, default=3)
    parser.add_argument("--num-runs", type=int, default=20)
    parser.add_argument("--compile-mode", type=str, default="max-autotune")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    
    device = torch.device(args.device)
    logging.info(f"Using device: {device}")
    
    model, config = load_model(args.checkpoint_dir, str(device), args.compile_mode)
    
    # Apply torch.compile if enabled
    if config.pytorch_compile_mode:
        logging.info(f"Compiling model with mode: {config.pytorch_compile_mode}")
        model = torch.compile(model, mode=config.pytorch_compile_mode)
    
    policy = create_policy(model, config, str(device))
    
    throughput = benchmark_inference(
        policy,
        str(device),
        num_warmup=args.num_warmup,
        num_runs=args.num_runs,
    )
    
    print(f"\nThroughput : {throughput:.2f} inf/s")
    
    return throughput


if __name__ == "__main__":
    main()
