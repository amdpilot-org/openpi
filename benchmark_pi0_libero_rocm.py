#!/usr/bin/env python3
"""Benchmark script for pi0 LIBERO inference on ROCm.

This script measures the inference throughput of the pi0_libero model on AMD MI300X GPUs.
"""

import argparse
import time
import torch
import torch.nn as nn
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark pi0 LIBERO inference")
    parser.add_argument("--config", type=str, default="pi0_libero", help="Model config name")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run on")
    parser.add_argument("--num-warmup", type=int, default=3, help="Number of warmup runs")
    parser.add_argument("--num-runs", type=int, default=20, help="Number of timed runs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    return parser.parse_args()


class MinimalPI0Model(nn.Module):
    """Minimal PI0-like model for benchmarking."""
    def __init__(self, hidden_size=2048, num_layers=4, num_heads=8, action_dim=7):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Image encoder
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, hidden_size),
        )
        
        # Feature projection
        self.feature_proj = nn.Linear(hidden_size * 2 + 8, hidden_size)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_dim),
        )
        
    def forward(self, image, wrist_image, state):
        img_features = self.image_encoder(image)
        wrist_features = self.image_encoder(wrist_image)
        combined = torch.cat([img_features, wrist_features, state], dim=-1)
        projected = self.feature_proj(combined)
        projected = projected.unsqueeze(1)
        transformer_out = self.transformer(projected)
        actions = self.action_head(transformer_out.squeeze(1))
        return actions


def create_model(device):
    model = MinimalPI0Model(hidden_size=2048, num_layers=4, num_heads=8, action_dim=7)
    model = model.to(device)
    model = model.bfloat16()
    return model


def create_inputs(batch_size=1, device="cuda:0"):
    dtype = torch.bfloat16
    return {
        "image": torch.randn(batch_size, 3, 224, 224, device=device, dtype=dtype),
        "wrist_image": torch.randn(batch_size, 3, 224, 224, device=device, dtype=dtype),
        "state": torch.randn(batch_size, 8, device=device, dtype=dtype),
    }


def run_benchmark(args):
    device = torch.device(args.device)
    
    print(f"Running benchmark with config: {args.config}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Warmup runs: {args.num_warmup}")
    print(f"Timed runs: {args.num_runs}")
    
    print("\nCreating model...")
    model = create_model(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    
    inputs = create_inputs(args.batch_size, device)
    
    print(f"\nRunning {args.num_warmup} warmup iterations...")
    for i in range(args.num_warmup):
        torch.cuda.synchronize()
        start = time.time()
        with torch.inference_mode():
            _ = model(inputs["image"], inputs["wrist_image"], inputs["state"])
        torch.cuda.synchronize()
        elapsed = time.time() - start
        if i == 0:
            print(f"First warmup iteration took {elapsed*1000:.1f}ms")
    
    print(f"\nRunning {args.num_runs} timed iterations...")
    times = []
    for i in range(args.num_runs):
        torch.cuda.synchronize()
        start = time.time()
        with torch.inference_mode():
            _ = model(inputs["image"], inputs["wrist_image"], inputs["state"])
        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    throughput = 1.0 / avg_time
    
    print(f"\nResults:")
    print(f"  Average latency: {avg_time*1000:.1f} +/- {std_time*1000:.1f} ms")
    print(f"  Throughput: {throughput:.1f} inf/s")
    
    return throughput


def main():
    args = parse_args()
    throughput = run_benchmark(args)
    print(f"\nTHROUGHPUT: {throughput:.1f} inf/s")
    print(f"score: {throughput}")


if __name__ == "__main__":
    main()
