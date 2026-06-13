#!/usr/bin/env bash
set -euo pipefail
cd /opt/openpi
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
export ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH=/opt/openpi/src:/opt/openpi/.venv310/lib/python3.10/site-packages:/opt/venv/lib/python3.10/site-packages:${PYTHONPATH:-}
export TORCH_BLAS_PREFER_HIPBLASLT=${TORCH_BLAS_PREFER_HIPBLASLT:-1}
export GPU_COREDUMP_ENABLE=${GPU_COREDUMP_ENABLE:-0}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export OPENPI_STAGE0_RESULT_JSON=${OPENPI_STAGE0_RESULT_JSON:-/tmp/openpi_result.json}
rm -f "$OPENPI_STAGE0_RESULT_JSON" /tmp/openpi_benchmark.log
python3 /opt/openpi/stage0_preflight.py
.venv/bin/python benchmark_pi0_libero_rocm.py \
  --config pi0_libero \
  --checkpoint-dir ~/.cache/openpi/openpi-assets/checkpoints/pi0_libero_pytorch \
  --device cuda:0 \
  --num-warmup 3 \
  --num-runs 20 \
  --result-json "$OPENPI_STAGE0_RESULT_JSON" | tee /tmp/openpi_benchmark.log
python3 /opt/openpi/stage0_parse_metric.py /tmp/openpi_benchmark.log "$OPENPI_STAGE0_RESULT_JSON"
python3 - <<'PY'
import json
p='/tmp/openpi_result.json'
data=json.load(open(p))
print(f"STAGE0_METRIC throughput_inf_s={data['throughput_inf_s']}")
PY
