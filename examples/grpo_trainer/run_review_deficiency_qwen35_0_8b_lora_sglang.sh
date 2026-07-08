#!/usr/bin/env bash
# =============================================================================
# GRPO + LoRA | Review Deficiency Classification | Qwen3.5-0.8B | SGLang | Single GPU
#
# Fine-tunes Qwen3.5-0.8B to classify peer reviews into c0–c6 categories.
# Uses LoRA for parameter efficiency and GRPO with a custom two-part reward:
#   1. Format reward: response contains parseable JSON with required schema
#   2. Accuracy reward: is_high_quality + defect_type correctness
#
# SGLang backend — replaces vLLM which had stability issues on this setup.
#
# Usage:
#   cd /home/ROVER-claudecode/projects/verl-rover
#   bash examples/grpo_trainer/run_review_deficiency_qwen35_0_8b_lora_sglang.sh
#
# Overrides (env vars):
#   MODEL_PATH       — base model directory
#   NGPUS_PER_NODE   — number of GPUs (default: 1)
#   TRAIN_BATCH_SIZE — training batch size
#   TOTAL_EPOCHS     — number of training epochs
#   LORA_RANK        — LoRA rank
#   LORA_ALPHA       — LoRA alpha
#   ACTOR_LR         — learning rate
#   ROLLOUT_N        — number of rollouts per prompt
# =============================================================================

set -xeuo pipefail

########################### user-adjustable ###########################
MODEL_PATH=${MODEL_PATH:-/home/LLM/Qwen/Qwen3.5-0.8B}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}

# -- 80GB single GPU: model ~0.8B → massive headroom; use larger batches & LoRA rank
train_batch_size=${TRAIN_BATCH_SIZE:-64}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-32}
max_prompt_length=${MAX_PROMPT_LENGTH:-4096}
max_response_length=${MAX_RESPONSE_LENGTH:-1024}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}

actor_lr=${ACTOR_LR:-2e-6}
kl_loss_coef=${KL_LOSS_COEF:-0.001}
entropy_coeff=${ENTROPY_COEFF:-0}

lora_rank=${LORA_RANK:-128}
lora_alpha=${LORA_ALPHA:-64}

rollout_tp=${ROLLOUT_TP:-1}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.85}
rollout_n=${ROLLOUT_N:-6}

total_epochs=${TOTAL_EPOCHS:-10}
save_freq=${SAVE_FREQ:-10}
test_freq=${TEST_FREQ:-5}

project_name=${PROJECT_NAME:-verl_grpo_review_deficiency}
experiment_name=${EXPERIMENT_NAME:-qwen35_0_8B_lora_review_deficiency_sglang_$(date +%Y%m%d_%H%M)}
########################### end user-adjustable ###########################

# Paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REWARD_FN_PATH="$SCRIPT_DIR/../../verl/utils/reward_score/review_deficiency.py"
TRAIN_DATA="$SCRIPT_DIR/../../data/review_deficiency/train.parquet"
TEST_DATA="$SCRIPT_DIR/../../data/review_deficiency/test.parquet"

# ── Parameter arrays ───────────────────────────────────────────

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="['$TRAIN_DATA']"
    data.val_files="['$TEST_DATA']"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.lora_rank=${lora_rank}
    actor_rollout_ref.model.lora_alpha=${lora_alpha}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.trust_remote_code=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.freeze_vision_tower=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=sglang
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.layered_summon=False
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.prompt_length=${max_prompt_length}
    actor_rollout_ref.rollout.response_length=${max_response_length}
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.ref.fsdp_config.param_offload=False
)

TRAINER=(
    trainer.balance_batch=True
    trainer.use_v1=False
    trainer.logger='["console"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=False
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
)

# Custom reward function — uses json5 for robust JSON parsing
REWARD=(
    reward.custom_reward_function.path="$REWARD_FN_PATH"
    reward.custom_reward_function.name=compute_score
)

# SGLang engine kwargs — GPU auto-detects attention backend (flashinfer preferred).
# For Ascend NPU, set: +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=ascend
EXTRA=()

########################### launch ###########################
echo "=============================================================================="
echo " GRPO + LoRA — Review Deficiency Classification (SGLang backend)"
echo "=============================================================================="
echo " Model:       $MODEL_PATH"
echo " Train data:  $TRAIN_DATA"
echo " Test data:   $TEST_DATA"
echo " GPUs:        $NGPUS_PER_NODE"
echo " LoRA rank:   $lora_rank  alpha: $lora_alpha"
echo " Batch size:  $train_batch_size"
echo " LR:          $actor_lr"
echo " Epochs:      $total_epochs"
echo " Reward fn:   $REWARD_FN_PATH"
echo " Backend:     sglang (tp=$rollout_tp, gpu_mem=$rollout_gpu_mem_util)"
echo "=============================================================================="

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${REWARD[@]}" \
    "${EXTRA[@]}" \
    "$@"
