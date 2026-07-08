#!/usr/bin/env bash
# Phase 1: 0.8B baseline — verify pipeline end-to-end with SGLang backend
set -xeuo pipefail

CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=/home/LLM/Qwen/Qwen3.5-0.8B \
MAX_PROMPT_LENGTH=24576 \
MAX_RESPONSE_LENGTH=1024 \
PPO_MAX_TOKEN_LEN_PER_GPU=28672 \
TRAIN_BATCH_SIZE=32 \
PPO_MINI_BATCH_SIZE=8 \
ROLLOUT_N=4 \
ROLLOUT_GPU_MEM_UTIL=0.50 \
ACTOR_LR=2e-6 \
KL_LOSS_COEF=0.001 \
ENTROPY_COEFF=0 \
LORA_RANK=32 \
LORA_ALPHA=16 \
TOTAL_EPOCHS=10 \
SAVE_FREQ=100 \
TEST_FREQ=10 \
bash examples/grpo_trainer/run_review_deficiency_qwen35_0_8b_lora_sglang.sh
