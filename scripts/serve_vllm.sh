#!/usr/bin/env bash
# Serve the pilot model from WSL2. Run inside WSL: bash scripts/serve_vllm.sh
# Exact settings are recorded in NOTES.md and in every run log.
set -euo pipefail
source ~/venvs/vllm/bin/activate
MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3.6-35B-A3B-AWQ}"
SERVED_NAME="${SERVED_NAME:-qwen3.6-35b-a3b-awq}"
mkdir -p ~/vllm_logs
# No CUDA toolkit in WSL, so FlashInfer cannot JIT its sampler; use the PyTorch sampler.
export VLLM_USE_FLASHINFER_SAMPLER=0
exec vllm serve "$MODEL_DIR" \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.90 \
  --dtype "${DTYPE:-bfloat16}" \
  --max-num-seqs 64 \
  --language-model-only \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --api-key local \
  2>&1 | tee ~/vllm_logs/serve_$(date +%Y%m%d_%H%M%S).log
