# amdpilot-org/openpi#2 Stage0 task

## Objective

Optimize Pi0 LIBERO inference throughput on a single AMD MI300X GPU using the upstream `amdpilot-org/openpi` codebase and the canonical ROCm PyTorch benchmark command from the issue. This Stage0 bundle freezes a verified starting point only; executor agents should optimize from this state and must preserve the benchmark contract.

## Fixed workload contract

Run exactly the Pi0 LIBERO PyTorch benchmark workload:

```bash
.venv/bin/python benchmark_pi0_libero_rocm.py \
  --config pi0_libero \
  --checkpoint-dir ~/.cache/openpi/openpi-assets/checkpoints/pi0_libero_pytorch \
  --device cuda:0 \
  --num-warmup 3 \
  --num-runs 20
```

Required conditions: model `pi0_libero`, PyTorch backend, bf16 default precision, batch size 1, 10 denoising steps, image size 224x224, 2 cameras, state dim 8, action horizon 10, single GPU exposed as `cuda:0`, and torch compile mode `max-autotune` unless intentionally changed by a later optimization trial and measured.

## Starting point

- Base image: `rocm/sgl-dev:v0.5.10.post1-rocm720-mi30x-20260424`.
- Source repo: `https://github.com/amdpilot-org/openpi.git` at commit `c23745b5ad24e98f66967ea795a07b2588ed6c79`.
- Host cache expected for verification: `/home/amd/openpi_cache` mounted to `/root/.cache/openpi`.
- Prior hill-climb memory reports best-known throughput `24.06 inf/s` after replacing the unstable denoising `while time >= -dt / 2` loop with a fixed `for _ in range(num_steps)` loop. This bundle treats that best-known state as the candidate starting point and does not add additional optimizations.

## Metric

- Metric name: `throughput_inf_s`
- Direction: higher is better
- Acceptance target from issue comments: at least 23 inf/s; stretch 25 inf/s.

## Verification

The included `test_harness.py` runs `/opt/openpi/bench_openpi.sh`, extracts `SCORE: ... throughput_inf_s`, and writes `/tmp/openpi_result.json`. The harness fails if the metric is missing, if ROCm torch/GPU visibility fails, or if the benchmark command exits nonzero.
