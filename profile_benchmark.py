import torch
import torch.profiler
import time
import sys
sys.path.insert(0, '/workspace/openpi')
from benchmark_pi0_libero_rocm import Pi0LiberoModel, run_benchmark
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--num-warmup", type=int, default=3)
parser.add_argument("--num-runs", type=int, default=20)
parser.add_argument("--compile-mode", type=str, default="default")
args = parser.parse_args()

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    with_stack=True,
    record_shapes=True,
    schedule=torch.profiler.schedule(wait=1, warmup=2, active=5, repeat=1),
) as prof:
    # Warmup + some runs while profiling
    device = args.device
    model = Pi0LiberoModel(
        action_dim=7, action_horizon=10,
        vision_hidden_dim=2048, vision_num_layers=18, vision_num_tokens=256,
        action_hidden_dim=1024, action_num_layers=18, num_heads=8,
    ).to(device=device, dtype=torch.bfloat16)
    model.eval()
    compile_mode = args.compile_mode
    if torch.version.hip is not None and compile_mode == "max-autotune":
        compile_mode = "default"
    compiled_model = torch.compile(model, mode=compile_mode)
    
    dummy_state = torch.randn(1, 8, device=device, dtype=torch.bfloat16)
    dummy_image = torch.randn(1, 3, 224, 224, device=device, dtype=torch.bfloat16)
    dummy_wrist = torch.randn(1, 3, 224, 224, device=device, dtype=torch.bfloat16)
    with torch.inference_mode():
        _ = compiled_model(dummy_state, dummy_image, dummy_wrist)
    
    for i in range(10):
        s = torch.randn(1, 8, device=device, dtype=torch.bfloat16)
        img = torch.randn(1, 3, 224, 224, device=device, dtype=torch.bfloat16)
        w = torch.randn(1, 3, 224, 224, device=device, dtype=torch.bfloat16)
        with torch.inference_mode():
            _ = compiled_model(s, img, w)
        prof.step()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
prof.export_chrome_trace("/workspace/output/profiler_trace.json")
print("Trace saved to /workspace/output/profiler_trace.json")
