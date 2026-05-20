#!/usr/bin/env python3
"""
Benchmark script for pi0 LIBERO inference on ROCm.
Measures iters_per_second for a representative Pi0 model forward pass.
"""
import argparse
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.mlp_dim = int(hidden_dim * mlp_ratio)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        self.up_proj = nn.Linear(hidden_dim, self.mlp_dim)
        self.down_proj = nn.Linear(self.mlp_dim, hidden_dim)

    def forward(self, x):
        residual = x
        x = self.attn_norm(x)
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_dim)
        x = self.o_proj(attn_out) + residual
        residual = x
        x = self.mlp_norm(x)
        x = self.up_proj(x)
        x = F.gelu(x)
        x = self.down_proj(x) + residual
        return x


class Pi0LiberoModel(nn.Module):
    def __init__(self, action_dim=7, action_horizon=10,
                 vision_hidden_dim=2048, vision_num_layers=18, vision_num_tokens=256,
                 action_hidden_dim=1024, action_num_layers=18, num_heads=8):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.vision_hidden_dim = vision_hidden_dim
        self.action_hidden_dim = action_hidden_dim
        self.vision_num_tokens = vision_num_tokens

        self.state_encoder = nn.Sequential(
            nn.Linear(8, vision_hidden_dim),
            nn.LayerNorm(vision_hidden_dim),
            nn.GELU(),
            nn.Linear(vision_hidden_dim, vision_hidden_dim)
        )
        self.image_conv = nn.Conv2d(3, vision_hidden_dim, kernel_size=14, stride=14, padding=0)
        self.image_proj = nn.Linear(vision_hidden_dim, vision_hidden_dim)
        self.wrist_conv = nn.Conv2d(3, vision_hidden_dim, kernel_size=14, stride=14, padding=0)
        self.wrist_proj = nn.Linear(vision_hidden_dim, vision_hidden_dim)
        self.vision_num_tokens = 256
        self.vision_transformer = nn.ModuleList([
            TransformerBlock(vision_hidden_dim, num_heads=num_heads, mlp_ratio=4.0)
            for _ in range(vision_num_layers)
        ])
        self.fusion = nn.Linear(vision_hidden_dim + vision_hidden_dim, action_hidden_dim)
        self.action_transformer = nn.ModuleList([
            TransformerBlock(action_hidden_dim, num_heads=num_heads, mlp_ratio=4.0)
            for _ in range(action_num_layers)
        ])
        self.action_decoder = nn.Linear(action_hidden_dim, action_dim)

    def forward(self, state, image, wrist_image):
        state_feat = self.state_encoder(state)
        image_feat = self.image_conv(image)
        image_feat = image_feat.flatten(2).transpose(1, 2)
        image_feat = self.image_proj(image_feat)
        wrist_feat = self.wrist_conv(wrist_image)
        wrist_feat = wrist_feat.flatten(2).transpose(1, 2)
        wrist_feat = self.wrist_proj(wrist_feat)
        visual_feat = image_feat + wrist_feat
        state_token = state_feat.unsqueeze(1)
        x = torch.cat([state_token, visual_feat], dim=1)
        for block in self.vision_transformer:
            x = block(x)
        vision_output = x.mean(dim=1)
        state_output = x[:, 0]
        fused = torch.cat([vision_output, state_output], dim=-1)
        action_input = self.fusion(fused)
        action_input = action_input.unsqueeze(1).expand(-1, self.action_horizon, -1)
        for block in self.action_transformer:
            action_input = block(action_input)
        actions = self.action_decoder(action_input)
        return actions


def run_benchmark(args):
    device = args.device
    torch.set_float32_matmul_precision('high')
    model = Pi0LiberoModel(
        action_dim=args.action_dim,
        action_horizon=args.action_horizon,
        vision_hidden_dim=args.vision_hidden_dim,
        vision_num_layers=args.vision_num_layers,
        vision_num_tokens=args.vision_num_tokens,
        action_hidden_dim=args.action_hidden_dim,
        action_num_layers=args.action_num_layers,
        num_heads=args.num_heads,
    ).to(device=device, dtype=torch.bfloat16)
    model.eval()

    compile_mode = args.compile_mode
    if torch.version.hip is not None and compile_mode == "max-autotune":
        compile_mode = "default"

    print(f"Compiling model with mode={compile_mode}...")
    compiled_model = torch.compile(model, mode=compile_mode)
    dummy_state = torch.randn(1, 8, device=device, dtype=torch.bfloat16)
    dummy_image = torch.randn(1, 3, 224, 224, device=device, dtype=torch.bfloat16)
    dummy_wrist = torch.randn(1, 3, 224, 224, device=device, dtype=torch.bfloat16)

    with torch.inference_mode():
        _ = compiled_model(dummy_state, dummy_image, dummy_wrist)

    print(f"Warmup {args.num_warmup}...")
    for i in range(args.num_warmup):
        s = torch.randn(1, 8, device=device, dtype=torch.bfloat16)
        img = torch.randn(1, 3, 224, 224, device=device, dtype=torch.bfloat16)
        w = torch.randn(1, 3, 224, 224, device=device, dtype=torch.bfloat16)
        with torch.inference_mode():
            _ = compiled_model(s, img, w)
    torch.cuda.synchronize(device)

    print(f"Timed runs {args.num_runs}...")
    start_time = time.monotonic()
    for i in range(args.num_runs):
        s = torch.randn(1, 8, device=device, dtype=torch.bfloat16)
        img = torch.randn(1, 3, 224, 224, device=device, dtype=torch.bfloat16)
        w = torch.randn(1, 3, 224, 224, device=device, dtype=torch.bfloat16)
        with torch.inference_mode():
            _ = compiled_model(s, img, w)
    torch.cuda.synchronize(device)
    elapsed_time = time.monotonic() - start_time

    throughput = args.num_runs / elapsed_time
    print(f"iters_per_second: {throughput:.2f}")
    return throughput


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="pi0_libero")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-warmup", type=int, default=3)
    parser.add_argument("--num-runs", type=int, default=20)
    parser.add_argument("--compile-mode", type=str, default="default")
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--vision-hidden-dim", type=int, default=2048)
    parser.add_argument("--vision-num-layers", type=int, default=18)
    parser.add_argument("--vision-num-tokens", type=int, default=256)
    parser.add_argument("--action-hidden-dim", type=int, default=1024)
    parser.add_argument("--action-num-layers", type=int, default=18)
    parser.add_argument("--num-heads", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("iters_per_second: 0.0")
        return 1

    try:
        run_benchmark(args)
        return 0
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("iters_per_second: 0.0")
        return 1


if __name__ == "__main__":
    exit(main())
