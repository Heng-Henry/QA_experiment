#!/usr/bin/env bash
#
# Cobra pure-text LongBench runner (dummy-image path inside run_longbench_eval.py).
#
# Required env:
#   TASK=hotpotqa|2wikimqa|qmsum|narrativeqa
#   MODE=fp16|w4a4|w4a8|w8a8
#   MODEL_PATH=cobra+3b or cobra+3b-ptq-<bits>-fake
#   RUN_TAG=<unique_run_tag>
#
# Optional env:
#   OUTPUT_DIR=/work/u9907741/project_0209/runs/cobra_textqa_eval
#   MAX_NEW_TOKENS=64
#   MAX_CONTEXT_TOKENS=8192
#   NUM_SAMPLES=<int>  # smoke only; unset for full
#   FORCE_COBRA=1
#
#SBATCH -J cobra_textqa_lb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH -p dev
#SBATCH -A MST114205
#SBATCH -o /work/u9907741/project_0209/runs/slurm/%x.%j.out
#SBATCH -e /work/u9907741/project_0209/runs/slurm/%x.%j.err

set -euo pipefail
IFS=$'\n\t'

module load cuda/12.4

set +u
source "${CONDA_SH_PATH:-/work/u9907741/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV_PATH:-/work/u9907741/miniconda3/envs/cobra-1}"
set -u

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export COBRA_USE_CFZO_MAMBA="${COBRA_USE_CFZO_MAMBA:-1}"
export COBRA_LLM_MIXER_REQUIRE_SUCCESS="${COBRA_LLM_MIXER_REQUIRE_SUCCESS:-1}"

REPO_ROOT="${REPO_ROOT:-/work/u9907741/project_0209/cobra_1115}"
RUNS_ROOT="${RUNS_ROOT:-/work/u9907741/project_0209/runs}"
HF_TOKEN="${HF_TOKEN:-${REPO_ROOT}/.hf_token}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUNS_ROOT}/cobra_textqa_eval}"

TASK="${TASK:-hotpotqa}"
MODE="${MODE:-fp16}"
MODEL_PATH="${MODEL_PATH:-cobra+3b}"
RUN_TAG="${RUN_TAG:-cobra_textqa_${TASK}_${MODE}_$(date +%Y%m%d_%H%M%S)}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-8192}"
SEED="${SEED:-42}"
NUM_SAMPLES="${NUM_SAMPLES:-}"
FORCE_COBRA="${FORCE_COBRA:-1}"
COBRA_RUN_DIR="${COBRA_RUN_DIR:-${REPO_ROOT}}"
STRICT_MIN_TOKENS="${STRICT_MIN_TOKENS:-4096}"
STRICT_MAX_TOKENS="${STRICT_MAX_TOKENS:-8192}"
STRICT_SAMPLE_SIZE="${STRICT_SAMPLE_SIZE:-200}"
STRICT_PRECHECK_SIZE="${STRICT_PRECHECK_SIZE:-500}"
STRICT_MIN_AVG="${STRICT_MIN_AVG:-5500}"
STRICT_MAX_TRIES="${STRICT_MAX_TRIES:-20}"

PROMPT_TEMPLATE="${PROMPT_TEMPLATE:-}"
if [[ -z "${PROMPT_TEMPLATE}" ]]; then
  PROMPT_TEMPLATE=$'Answer the question based on the given passages.\n\nContext: {context}\n\nQuestion: {input}\n\nAnswer:'
fi

mkdir -p "${RUNS_ROOT}/slurm" "${OUTPUT_DIR}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export COBRA_1115_ROOT="${COBRA_1115_ROOT:-${REPO_ROOT}}"

echo "[INFO] TASK=${TASK} MODE=${MODE} MODEL_PATH=${MODEL_PATH}"
echo "[INFO] RUN_TAG=${RUN_TAG}"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[INFO] HF_TOKEN=${HF_TOKEN}"
echo "[INFO] BACKEND=${BACKEND:-} BITS=${BITS:-}"
echo "[INFO] COBRA_LLM_MIXER_ROTATE=${COBRA_LLM_MIXER_ROTATE:-0}"
echo "[INFO] COBRA_LLM_MIXER_HADAMARD=${COBRA_LLM_MIXER_HADAMARD:-0}"
echo "[INFO] COBRA_LLM_MIXER_ACT_KLT=${COBRA_LLM_MIXER_ACT_KLT:-0}"
echo "[INFO] COBRA_LLM_MIXER_OUT_TRANSFORM=${COBRA_LLM_MIXER_OUT_TRANSFORM:-0}"
echo "[INFO] COBRA_LLM_MIXER_TARGETS=${COBRA_LLM_MIXER_TARGETS:-out_proj}"
echo "[INFO] COBRA_LLM_MIXER_BLOCK=${COBRA_LLM_MIXER_BLOCK:-512}"
echo "[INFO] ACT_KLT_OUTPROJ_IN=${ACT_KLT_OUTPROJ_IN:-}"
echo "[INFO] ACT_KLT_OUTPROJ_OUT=${ACT_KLT_OUTPROJ_OUT:-}"
echo "[INFO] COBRA_DISABLE_MAMBA_FAST_PATH=${COBRA_DISABLE_MAMBA_FAST_PATH:-}"
echo "[INFO] COBRA_USE_CFZO_MAMBA=${COBRA_USE_CFZO_MAMBA}"
echo "[INFO] COBRA_LLM_MIXER_REQUIRE_SUCCESS=${COBRA_LLM_MIXER_REQUIRE_SUCCESS}"
echo "[INFO] STRICT_MIN_TOKENS=${STRICT_MIN_TOKENS} STRICT_MAX_TOKENS=${STRICT_MAX_TOKENS}"
echo "[INFO] STRICT_SAMPLE_SIZE=${STRICT_SAMPLE_SIZE} STRICT_PRECHECK_SIZE=${STRICT_PRECHECK_SIZE}"
echo "[INFO] STRICT_MIN_AVG=${STRICT_MIN_AVG} STRICT_MAX_TRIES=${STRICT_MAX_TRIES}"

ARGS=(
  scripts/run_longbench_eval.py
  --task "${TASK}"
  --model-path "${MODEL_PATH}"
  --hf-token "${HF_TOKEN}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --max-context-tokens "${MAX_CONTEXT_TOKENS}"
  --seed "${SEED}"
  --output-dir "${OUTPUT_DIR}"
  --prompt-template "${PROMPT_TEMPLATE}"
  --run-tag "${RUN_TAG}"
  --cobra-run-dir "${COBRA_RUN_DIR}"
  --strict-min-tokens "${STRICT_MIN_TOKENS}"
  --strict-max-tokens "${STRICT_MAX_TOKENS}"
  --strict-sample-size "${STRICT_SAMPLE_SIZE}"
  --strict-precheck-size "${STRICT_PRECHECK_SIZE}"
  --strict-min-avg "${STRICT_MIN_AVG}"
  --strict-max-tries "${STRICT_MAX_TRIES}"
)

if [[ -n "${NUM_SAMPLES}" ]]; then
  ARGS+=(--num-samples "${NUM_SAMPLES}")
fi
if [[ "${FORCE_COBRA}" == "1" ]]; then
  ARGS+=(--force-cobra)
fi

set -x
srun -u python "${ARGS[@]}"
set +x

RESULT_DIR="${OUTPUT_DIR}/${TASK}_${RUN_TAG}"
echo "[INFO] RESULT_DIR=${RESULT_DIR}"
if [[ -f "${RESULT_DIR}/metrics.json" ]]; then
  echo "[INFO] metrics.json:"
  cat "${RESULT_DIR}/metrics.json"
fi
