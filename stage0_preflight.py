import os
from pathlib import Path
import torch

print(f"PRECHECK torch={torch.__version__} hip={torch.version.hip} cuda_available={torch.cuda.is_available()} count={torch.cuda.device_count()}")
print(f"PRECHECK visible HIP={os.environ.get('HIP_VISIBLE_DEVICES')} ROCR={os.environ.get('ROCR_VISIBLE_DEVICES')} CUDA={os.environ.get('CUDA_VISIBLE_DEVICES')}")
ckpt = Path.home()/'.cache/openpi/openpi-assets/checkpoints/pi0_libero_pytorch/model.safetensors'
norm = Path.home()/'.cache/openpi/openpi-assets/checkpoints/pi0_libero_pytorch/assets/physical-intelligence/libero/norm_stats.json'
tok = Path.home()/'.cache/openpi/big_vision/paligemma_tokenizer.model'
print(f"PRECHECK checkpoint={ckpt} exists={ckpt.exists()} size={ckpt.stat().st_size if ckpt.exists() else 0}")
print(f"PRECHECK norm_stats={norm} exists={norm.exists()} size={norm.stat().st_size if norm.exists() else 0}")
print(f"PRECHECK tokenizer={tok} exists={tok.exists()} size={tok.stat().st_size if tok.exists() else 0}")
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit('No ROCm/CUDA-visible GPU')
if not ckpt.exists():
    raise SystemExit('Missing pi0_libero_pytorch checkpoint cache mount')
if not norm.exists():
    raise SystemExit('Missing pi0_libero_pytorch norm_stats.json cache mount')
if not tok.exists():
    raise SystemExit('Missing paligemma tokenizer cache mount')
