#!/bin/bash
#SBATCH --job-name=vlm_eval
#SBATCH --account=MST114205
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH -o outputs/slurm/%x_%j.out
#SBATCH -e outputs/slurm/%x_%j.err

set -euo pipefail

# ==============================
# 1. Runtime environment
# ==============================
module load cuda/12.4

# conda activate (nounset-safe)
set +u
source "${CONDA_SH_PATH:-/work/u9907741/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV_PATH:-/work/u9907741/miniconda3/envs/cobra-1}"
set -u

export ADDR2LINE=${ADDR2LINE:-$(command -v addr2line || true)}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
# Keep Cobra on the stable custom Mamba path. This avoids decode-time
# assertion failures seen with upstream mamba_simple in current env.
export COBRA_USE_CFZO_MAMBA="${COBRA_USE_CFZO_MAMBA:-1}"

# ==============================
# 2. Cobra repo wiring
# ==============================

# cobra_1115 project root (required for fake-quant)
export COBRA_1115_ROOT="${COBRA_1115_ROOT:-/work/u9907741/project_0209/cobra_1115}"
export PYTHONPATH="${COBRA_1115_ROOT}:${PYTHONPATH:-}"

# base float model id (for cobra.load)
export COBRA_MODEL_BASE_ID="${COBRA_MODEL_BASE_ID:-cobra+3b}"

# Baseline:
#   - weights: fake-quant (W8/W4/W2)
#   - activations: float (no pct_hi_lo, no rotation, no fusion)
export BACKEND="${BACKEND:-fake}"   # float | fake
export BITS="${BITS:-W8}"           # e.g., W8 | W4 | W2 | W4A8 | A8 (passed through as-is)

BACKEND_LOWER="$(echo "${BACKEND}" | tr 'A-Z' 'a-z')"
case "${BACKEND_LOWER}" in
  float|fake) ;;
  *)
    echo "[ERROR] Unsupported BACKEND='${BACKEND}'. Only BACKEND=float|fake is supported." >&2
    exit 1
    ;;
esac

# Keep BITS as a single source of truth (no normalization here).
# Downstream (vlm_eval/models/quant_cobra.py + cobra runtime) will interpret it.
BITS_TAG="$(echo "${BITS}" | tr 'A-Z' 'a-z')"

# ==============================
# 3. Project roots
# ==============================

VLM_ROOT="${VLM_ROOT:-/work/u9907741/project_0209/cobra_1115-evaluation}"
cd "${VLM_ROOT}"

mkdir -p outputs/slurm
mkdir -p results

# ==============================
# 3.5 Optional Text-QA Sweep Trigger
# ==============================
# Use this script as a unified trigger for pure-text QA sweeps:
#   PIPELINE_KIND=textqa sbatch vlm_eval.sh
#
# Optional overrides:
#   TEXTQA_TASKS=longbench_hotpotqa,longbench_2wikimqa,longbench_qmsum,longbench_narrativeqa
#   TEXTQA_MODES=fp16,w8a8,w4a8
#   TEXTQA_BATCH_SIZE=16 TEXTQA_MAX_LENGTH=8192
PIPELINE_KIND="${PIPELINE_KIND:-vlm}"   # vlm | textqa | cobra_textqa

if [[ "${PIPELINE_KIND}" == "cobra_textqa" ]]; then
  PROJECT_ROOT="${PROJECT_ROOT:-/work/u9907741/project_0209}"
  COBRA_TEXTQA_SLURM_SCRIPT="${COBRA_TEXTQA_SLURM_SCRIPT:-${VLM_ROOT}/scripts/slurm_cobra_textqa_longbench.sh}"
  COBRA_TEXTQA_TASKS="${COBRA_TEXTQA_TASKS:-longbench_hotpotqa,longbench_2wikimqa,longbench_qmsum,longbench_narrativeqa}"
  COBRA_TEXTQA_MODES="${COBRA_TEXTQA_MODES:-fp16,w4a4,w4a8,w8a8}"
  COBRA_TEXTQA_OUTPUT_DIR="${COBRA_TEXTQA_OUTPUT_DIR:-${PROJECT_ROOT}/runs/cobra_textqa_eval}"
  COBRA_TEXTQA_MAX_NEW_TOKENS="${COBRA_TEXTQA_MAX_NEW_TOKENS:-64}"
  COBRA_TEXTQA_MAX_CONTEXT_TOKENS="${COBRA_TEXTQA_MAX_CONTEXT_TOKENS:-8192}"
  COBRA_TEXTQA_NUM_SAMPLES="${COBRA_TEXTQA_NUM_SAMPLES:-}"
  COBRA_TEXTQA_FORCE_COBRA="${COBRA_TEXTQA_FORCE_COBRA:-1}"
  COBRA_TEXTQA_SEED="${COBRA_TEXTQA_SEED:-42}"
  COBRA_TEXTQA_STRICT_MIN_TOKENS="${COBRA_TEXTQA_STRICT_MIN_TOKENS:-4096}"
  COBRA_TEXTQA_STRICT_MAX_TOKENS="${COBRA_TEXTQA_STRICT_MAX_TOKENS:-8192}"
  COBRA_TEXTQA_STRICT_SAMPLE_SIZE="${COBRA_TEXTQA_STRICT_SAMPLE_SIZE:-200}"
  COBRA_TEXTQA_STRICT_PRECHECK_SIZE="${COBRA_TEXTQA_STRICT_PRECHECK_SIZE:-500}"
  COBRA_TEXTQA_STRICT_MIN_AVG="${COBRA_TEXTQA_STRICT_MIN_AVG:-5500}"
  COBRA_TEXTQA_STRICT_MAX_TRIES="${COBRA_TEXTQA_STRICT_MAX_TRIES:-20}"
  COBRA_TEXTQA_INCLUDE_OUT_TRANSFORM="${COBRA_TEXTQA_INCLUDE_OUT_TRANSFORM:-1}"
  COBRA_TEXTQA_INCLUDE_ACT_KLT="${COBRA_TEXTQA_INCLUDE_ACT_KLT:-1}"
  COBRA_TEXTQA_SBATCH_PARTITION="${COBRA_TEXTQA_SBATCH_PARTITION:-dev}"
  COBRA_TEXTQA_SBATCH_ACCOUNT="${COBRA_TEXTQA_SBATCH_ACCOUNT:-MST114205}"
  COBRA_TEXTQA_SBATCH_TIME="${COBRA_TEXTQA_SBATCH_TIME:-06:00:00}"
  COBRA_TEXTQA_CONDA_ENV_PATH="${COBRA_TEXTQA_CONDA_ENV_PATH:-/work/u9907741/miniconda3/envs/cobra-1}"
  COBRA_TEXTQA_HF_TOKEN="${COBRA_TEXTQA_HF_TOKEN:-${COBRA_1115_ROOT}/.hf_token}"
  COBRA_TEXTQA_PROMPT_TEMPLATE="${COBRA_TEXTQA_PROMPT_TEMPLATE:-}"
  if [[ -z "${COBRA_TEXTQA_PROMPT_TEMPLATE}" ]]; then
    COBRA_TEXTQA_PROMPT_TEMPLATE=$'Answer the question based on the given passages.\n\nContext: {context}\n\nQuestion: {input}\n\nAnswer:'
  fi

  MIXER_BLOCK="${COBRA_LLM_MIXER_BLOCK:-512}"
  MIXER_TARGETS="${COBRA_LLM_MIXER_TARGETS:-out_proj}"
  MIXER_REQUIRE_SUCCESS="${COBRA_LLM_MIXER_REQUIRE_SUCCESS:-1}"
  ACT_IN_DEFAULT="${COBRA_1115_ROOT}/outputs/quantize/act_klt_outproj_in_bs${MIXER_BLOCK}/act_klt_outproj_in.pt"
  ACT_OUT_DEFAULT="${COBRA_1115_ROOT}/outputs/quantize/act_klt_outproj_out_bs${MIXER_BLOCK}/act_klt_outproj_out.pt"
  ACT_IN="${ACT_KLT_OUTPROJ_IN:-${ACT_IN_DEFAULT}}"
  ACT_OUT="${ACT_KLT_OUTPROJ_OUT:-${ACT_OUT_DEFAULT}}"

  TS="$(date +%Y%m%d_%H%M%S)"
  MAP_PATH="${PROJECT_ROOT}/runs/slurm/cobra_textqa_ablation_sweep_${TS}.tsv"
  mkdir -p "${PROJECT_ROOT}/runs/slurm" "${COBRA_TEXTQA_OUTPUT_DIR}"
  printf 'status\treason\tframework\tmode\ttask\ttask_short\tablation\trotate\thadamard\tact_klt\tout_transform\tbackend\tbits\tmodel_path\tjobid\trun_tag\tresult_dir\tslurm_out\tslurm_err\tact_klt_in\tact_klt_out\n' > "${MAP_PATH}"

  if [[ ! -f "${COBRA_TEXTQA_SLURM_SCRIPT}" ]]; then
    echo "[ERROR] Missing COBRA_TEXTQA_SLURM_SCRIPT=${COBRA_TEXTQA_SLURM_SCRIPT}" >&2
    exit 1
  fi

  declare -a COBRA_ABLATIONS=(
    # name:ROTATE:HADAMARD:ACT_KLT:OUT_TRANSFORM
    "base_no_rotate:0:0:0:0"
    "rotate_only:1:0:0:0"
    "rotate_hadamard:1:1:0:0"
  )
  if [[ "${COBRA_TEXTQA_INCLUDE_ACT_KLT}" == "1" ]]; then
    COBRA_ABLATIONS+=("rotate_hadamard_actklt:1:1:1:0")
  fi
  if [[ "${COBRA_TEXTQA_INCLUDE_OUT_TRANSFORM}" == "1" ]]; then
    COBRA_ABLATIONS+=("rotate_hadamard_actklt_out:1:1:1:1")
  fi

  resolve_mode() {
    local mode="$1"
    case "${mode}" in
      fp16)
        echo "float FP16 ${COBRA_MODEL_BASE_ID}"
        ;;
      w8a8)
        echo "fake W8A8 ${COBRA_MODEL_BASE_ID}-ptq-w8a8-fake"
        ;;
      w4a8)
        echo "fake W4A8 ${COBRA_MODEL_BASE_ID}-ptq-w4a8-fake"
        ;;
      w4a4)
        echo "fake W4A4 ${COBRA_MODEL_BASE_ID}-ptq-w4a4-fake"
        ;;
      *)
        return 1
        ;;
    esac
  }

  IFS=',' read -r -a TASK_ARR <<< "${COBRA_TEXTQA_TASKS}"
  IFS=',' read -r -a MODE_ARR <<< "${COBRA_TEXTQA_MODES}"

  echo "[INFO] PIPELINE_KIND=cobra_textqa"
  echo "[INFO] COBRA_TEXTQA_TASKS=${COBRA_TEXTQA_TASKS}"
  echo "[INFO] COBRA_TEXTQA_MODES=${COBRA_TEXTQA_MODES}"
  echo "[INFO] COBRA_TEXTQA_OUTPUT_DIR=${COBRA_TEXTQA_OUTPUT_DIR}"
  echo "[INFO] COBRA_TEXTQA_SLURM_SCRIPT=${COBRA_TEXTQA_SLURM_SCRIPT}"
  echo "[INFO] COBRA_TEXTQA_SBATCH_PARTITION=${COBRA_TEXTQA_SBATCH_PARTITION}"
  echo "[INFO] COBRA_TEXTQA_SBATCH_ACCOUNT=${COBRA_TEXTQA_SBATCH_ACCOUNT}"
  echo "[INFO] COBRA_TEXTQA_SBATCH_TIME=${COBRA_TEXTQA_SBATCH_TIME}"
  echo "[INFO] COBRA_TEXTQA_CONDA_ENV_PATH=${COBRA_TEXTQA_CONDA_ENV_PATH}"
  echo "[INFO] COBRA_TEXTQA_HF_TOKEN=${COBRA_TEXTQA_HF_TOKEN}"
  echo "[INFO] STRICT_MIN_TOKENS=${COBRA_TEXTQA_STRICT_MIN_TOKENS} STRICT_MAX_TOKENS=${COBRA_TEXTQA_STRICT_MAX_TOKENS}"
  echo "[INFO] STRICT_SAMPLE_SIZE=${COBRA_TEXTQA_STRICT_SAMPLE_SIZE} STRICT_PRECHECK_SIZE=${COBRA_TEXTQA_STRICT_PRECHECK_SIZE}"
  echo "[INFO] STRICT_MIN_AVG=${COBRA_TEXTQA_STRICT_MIN_AVG} STRICT_MAX_TRIES=${COBRA_TEXTQA_STRICT_MAX_TRIES}"
  echo "[INFO] MIXER_BLOCK=${MIXER_BLOCK} MIXER_TARGETS=${MIXER_TARGETS}"
  echo "[INFO] MIXER_REQUIRE_SUCCESS=${MIXER_REQUIRE_SUCCESS}"
  echo "[INFO] ACT_KLT_OUTPROJ_IN=${ACT_IN}"
  echo "[INFO] ACT_KLT_OUTPROJ_OUT=${ACT_OUT}"

  for task in "${TASK_ARR[@]}"; do
    task_short="${task#longbench_}"
    for mode in "${MODE_ARR[@]}"; do
      if ! mode_cfg="$(resolve_mode "${mode}")"; then
        echo "[WARN] Unsupported COBRA_TEXTQA mode='${mode}', skipping."
        continue
      fi
      read -r backend bits model_path <<< "${mode_cfg}"

      for cfg in "${COBRA_ABLATIONS[@]}"; do
        IFS=':' read -r ablation rotate hadamard act_klt out_transform <<< "${cfg}"

        reason="-"
        if [[ "${act_klt}" == "1" && ! -f "${ACT_IN}" ]]; then
          reason="missing_act_klt_in"
        fi
        if [[ "${out_transform}" == "1" && ! -f "${ACT_OUT}" ]]; then
          reason="missing_act_klt_out"
        fi

        run_tag="cobra_${mode}_${task_short}_${ablation}_${TS}"
        result_dir="${COBRA_TEXTQA_OUTPUT_DIR}/${task_short}_${run_tag}"

        if [[ "${reason}" != "-" ]]; then
          printf 'SKIPPED\t%s\tcobra\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t-\t%s\t%s\t-\t-\t%s\t%s\n' \
            "${reason}" "${mode}" "${task}" "${task_short}" "${ablation}" \
            "${rotate}" "${hadamard}" "${act_klt}" "${out_transform}" \
            "${backend}" "${bits}" "${model_path}" "${run_tag}" "${result_dir}" "${ACT_IN}" "${ACT_OUT}" >> "${MAP_PATH}"
          continue
        fi

        job_name="cobra_textqa_${task_short}_${mode}_${ablation}"
        exports=(
          "ALL"
          "TASK=${task_short}"
          "MODE=${mode}"
          "MODEL_PATH=${model_path}"
          "RUN_TAG=${run_tag}"
          "OUTPUT_DIR=${COBRA_TEXTQA_OUTPUT_DIR}"
          "MAX_NEW_TOKENS=${COBRA_TEXTQA_MAX_NEW_TOKENS}"
          "MAX_CONTEXT_TOKENS=${COBRA_TEXTQA_MAX_CONTEXT_TOKENS}"
          "NUM_SAMPLES=${COBRA_TEXTQA_NUM_SAMPLES}"
          "SEED=${COBRA_TEXTQA_SEED}"
          "STRICT_MIN_TOKENS=${COBRA_TEXTQA_STRICT_MIN_TOKENS}"
          "STRICT_MAX_TOKENS=${COBRA_TEXTQA_STRICT_MAX_TOKENS}"
          "STRICT_SAMPLE_SIZE=${COBRA_TEXTQA_STRICT_SAMPLE_SIZE}"
          "STRICT_PRECHECK_SIZE=${COBRA_TEXTQA_STRICT_PRECHECK_SIZE}"
          "STRICT_MIN_AVG=${COBRA_TEXTQA_STRICT_MIN_AVG}"
          "STRICT_MAX_TRIES=${COBRA_TEXTQA_STRICT_MAX_TRIES}"
          "FORCE_COBRA=${COBRA_TEXTQA_FORCE_COBRA}"
          "PROMPT_TEMPLATE=${COBRA_TEXTQA_PROMPT_TEMPLATE}"
          "CONDA_ENV_PATH=${COBRA_TEXTQA_CONDA_ENV_PATH}"
          "REPO_ROOT=${COBRA_1115_ROOT}"
          "COBRA_1115_ROOT=${COBRA_1115_ROOT}"
          "COBRA_RUN_DIR=${COBRA_1115_ROOT}"
          "RUNS_ROOT=${PROJECT_ROOT}/runs"
          "HF_TOKEN=${COBRA_TEXTQA_HF_TOKEN}"
          "BACKEND=${backend}"
          "BITS=${bits}"
          "COBRA_LLM_MIXER_ROTATE=${rotate}"
          "COBRA_LLM_MIXER_HADAMARD=${hadamard}"
          "COBRA_LLM_MIXER_ACT_KLT=${act_klt}"
          "COBRA_LLM_MIXER_OUT_TRANSFORM=${out_transform}"
          "COBRA_LLM_MIXER_TARGETS=${MIXER_TARGETS}"
          "COBRA_LLM_MIXER_BLOCK=${MIXER_BLOCK}"
          "COBRA_LLM_MIXER_REQUIRE_SUCCESS=${MIXER_REQUIRE_SUCCESS}"
          "COBRA_DISABLE_MAMBA_FAST_PATH=1"
          "ACT_KLT_OUTPROJ_IN=${ACT_IN}"
          "ACT_KLT_OUTPROJ_OUT=${ACT_OUT}"
          "COBRA_USE_CFZO_MAMBA=${COBRA_USE_CFZO_MAMBA}"
        )

        submit_out=$(
          sbatch \
            --account="${COBRA_TEXTQA_SBATCH_ACCOUNT}" \
            --partition="${COBRA_TEXTQA_SBATCH_PARTITION}" \
            --time="${COBRA_TEXTQA_SBATCH_TIME}" \
            --job-name="${job_name}" \
            --output="${PROJECT_ROOT}/runs/slurm/%x.%j.out" \
            --error="${PROJECT_ROOT}/runs/slurm/%x.%j.err" \
            --export="$(IFS=,; echo "${exports[*]}")" \
            "${COBRA_TEXTQA_SLURM_SCRIPT}"
        )
        echo "${submit_out}"
        jobid="${submit_out##* }"
        slurm_out="${PROJECT_ROOT}/runs/slurm/${job_name}.${jobid}.out"
        slurm_err="${PROJECT_ROOT}/runs/slurm/${job_name}.${jobid}.err"

        printf 'SUBMITTED\t-\tcobra\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
          "${mode}" "${task}" "${task_short}" "${ablation}" \
          "${rotate}" "${hadamard}" "${act_klt}" "${out_transform}" \
          "${backend}" "${bits}" "${model_path}" "${jobid}" "${run_tag}" "${result_dir}" \
          "${slurm_out}" "${slurm_err}" "${ACT_IN}" "${ACT_OUT}" >> "${MAP_PATH}"
      done
    done
  done

  echo "[INFO] Submitted cobra_textqa ablation sweep jobs:"
  cat "${MAP_PATH}"
  echo "[INFO] Mapping file: ${MAP_PATH}"
  exit 0
fi

if [[ "${PIPELINE_KIND}" == "textqa" ]]; then
  PROJECT_ROOT="${PROJECT_ROOT:-/work/u9907741/project_0209}"
  INCLUDE_PATH="${INCLUDE_PATH:-${PROJECT_ROOT}/runs/lm_eval_tasks}"
  TEXTQA_TASKS="${TEXTQA_TASKS:-longbench_hotpotqa,longbench_2wikimqa,longbench_qmsum,longbench_narrativeqa}"
  TEXTQA_MODES="${TEXTQA_MODES:-fp16,w8a8,w4a8}"
  TEXTQA_BATCH_SIZE="${TEXTQA_BATCH_SIZE:-16}"
  TEXTQA_MAX_LENGTH="${TEXTQA_MAX_LENGTH:-8192}"
  # Canonical text-QA checkpoint used across frameworks for fair comparison.
  # Quamba implementation notes and MambaQuant loaders are aligned on HF-converted checkpoints.
  TEXTQA_CANONICAL_CHECKPOINT="${TEXTQA_CANONICAL_CHECKPOINT:-state-spaces/mamba-1.4b-hf}"
  TEXTQA_ENFORCE_SAME_CHECKPOINT="${TEXTQA_ENFORCE_SAME_CHECKPOINT:-1}"
  QUAMBA_MODEL="${QUAMBA_MODEL:-${TEXTQA_CANONICAL_CHECKPOINT}}"
  MAMBAQUANT_MODEL_PATH="${MAMBAQUANT_MODEL_PATH:-${TEXTQA_CANONICAL_CHECKPOINT}}"
  if [[ "${TEXTQA_ENFORCE_SAME_CHECKPOINT}" == "1" ]]; then
    QUAMBA_MODEL="${TEXTQA_CANONICAL_CHECKPOINT}"
    MAMBAQUANT_MODEL_PATH="${TEXTQA_CANONICAL_CHECKPOINT}"
  fi
  QUAMBA_CONDA_ENV_PATH="${QUAMBA_CONDA_ENV_PATH:-/work/u9907741/miniconda3/envs/cobra-1}"
  MAMBAQUANT_CONDA_ENV_PATH="${MAMBAQUANT_CONDA_ENV_PATH:-/work/u9907741/miniconda3/envs/cobra-1}"
  QUAMBA_STRICT_ENV="${QUAMBA_STRICT_ENV:-0}"
  TS="$(date +%Y%m%d_%H%M%S)"
  MAP_PATH="${PROJECT_ROOT}/runs/slurm/textqa_sweep_${TS}.tsv"
  mkdir -p "${PROJECT_ROOT}/runs/slurm"
  printf 'framework\tmode\ttask\tjobid\trun_tag\tcanonical_checkpoint\tquamba_model\tmambaquant_model_path\n' > "${MAP_PATH}"

  submit_quamba_fp16() {
    local task="$1"
    local run_tag="quamba_fp16_${task}_${TS}"
    local out
    out=$(cd "${PROJECT_ROOT}/Quamba" && sbatch --export=ALL,TASK_LIST="${task}",RUN_TAG="${run_tag}",BATCH_SIZE="${TEXTQA_BATCH_SIZE}",MAX_LENGTH="${TEXTQA_MAX_LENGTH}",INCLUDE_PATH="${INCLUDE_PATH}",MODEL="${QUAMBA_MODEL}",CONDA_ENV_PATH="${QUAMBA_CONDA_ENV_PATH}",QUAMBA_STRICT_ENV="${QUAMBA_STRICT_ENV}" slurm_quamba_fp16_longbench.sh)
    local jobid="${out##* }"
    printf 'quamba\tfp16\t%s\t%s\t%s\t%s\t%s\t%s\n' "${task}" "${jobid}" "${run_tag}" "${TEXTQA_CANONICAL_CHECKPOINT}" "${QUAMBA_MODEL}" "${MAMBAQUANT_MODEL_PATH}" >> "${MAP_PATH}"
  }

  submit_quamba_quant() {
    local mode="$1"
    local task="$2"
    local w_bits="$3"
    local a_bits="$4"
    local run_tag="quamba_${mode}_${task}_${TS}"
    local run_dir="${PROJECT_ROOT}/runs/quamba_eval/${run_tag}"
    local pretrained_dir="${PROJECT_ROOT}/runs/quamba_pretrained_sweep/${run_tag}"
    local log_dir="${run_dir}/logs"
    local quantize_embedding="${QUAMBA_QUANTIZE_EMBEDDING:-0}"
    local quantize_lm_head="${QUAMBA_QUANTIZE_LM_HEAD:-0}"
    local calib_data_num="${QUAMBA_CALIB_DATA_NUM:-512}"
    local calib_seqlen="${QUAMBA_CALIB_SEQLEN:-512}"
    local apply_gptq="${QUAMBA_APPLY_GPTQ:-}"
    if [[ -z "${apply_gptq}" ]]; then
      if [[ "${w_bits}" == "4" ]]; then
        apply_gptq=1
      else
        apply_gptq=0
      fi
    fi
    local out
    out=$(cd "${PROJECT_ROOT}/Quamba" && sbatch --export=ALL,TASK_LIST="${task}",RUN_TAG="${run_tag}",BATCH_SIZE="${TEXTQA_BATCH_SIZE}",MAX_LENGTH="${TEXTQA_MAX_LENGTH}",INCLUDE_PATH="${INCLUDE_PATH}",MODEL="${QUAMBA_MODEL}",W_BITS="${w_bits}",A_BITS="${a_bits}",PRETRAINED_DIR="${pretrained_dir}",LOG_DIR="${log_dir}",QUANTIZE_EMBEDDING="${quantize_embedding}",QUANTIZE_LM_HEAD="${quantize_lm_head}",CALIB_DATA_NUM="${calib_data_num}",CALIB_SEQLEN="${calib_seqlen}",APPLY_GPTQ="${apply_gptq}",CONDA_ENV_PATH="${QUAMBA_CONDA_ENV_PATH}",QUAMBA_STRICT_ENV="${QUAMBA_STRICT_ENV}" slurm_quamba_w8a8.sh)
    local jobid="${out##* }"
    printf 'quamba\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${mode}" "${task}" "${jobid}" "${run_tag}" "${TEXTQA_CANONICAL_CHECKPOINT}" "${QUAMBA_MODEL}" "${MAMBAQUANT_MODEL_PATH}" >> "${MAP_PATH}"
  }

  submit_mambaquant() {
    local mode="$1"
    local task="$2"
    local run_tag="mambaquant_${mode}_${task}_${TS}"
    local out
    if [[ "${mode}" == "fp16" ]]; then
      out=$(cd "${PROJECT_ROOT}/MambaQuant" && sbatch --export=ALL,TASKS="${task}",INCLUDE_PATH="${INCLUDE_PATH}",MODEL_PATH="${MAMBAQUANT_MODEL_PATH}",CONDA_ENV_PATH="${MAMBAQUANT_CONDA_ENV_PATH}",QUANT_WEIGHT=0,QUANT_ACT=0,USE_HADAMARD=0,USE_HADAMARD_R1=0,USE_HADAMARD_R5=0,USE_KLT=0,USE_PERTOKEN=0,W_PERCHANNEL=0,MODEL_DTYPE=float16,BATCH_SIZE="${TEXTQA_BATCH_SIZE}",RUN_TAG="${run_tag}" model_llm/tests/slurm_mambaquant_w8a8.sh)
    elif [[ "${mode}" == "w8a8" ]]; then
      out=$(cd "${PROJECT_ROOT}/MambaQuant" && sbatch --export=ALL,TASKS="${task}",INCLUDE_PATH="${INCLUDE_PATH}",MODEL_PATH="${MAMBAQUANT_MODEL_PATH}",CONDA_ENV_PATH="${MAMBAQUANT_CONDA_ENV_PATH}",QUANT_WEIGHT=1,QUANT_ACT=1,W_BIT=8,A_BIT=8,MODEL_DTYPE=float16,BATCH_SIZE="${TEXTQA_BATCH_SIZE}",RUN_TAG="${run_tag}" model_llm/tests/slurm_mambaquant_w8a8.sh)
    elif [[ "${mode}" == "w4a8" ]]; then
      out=$(cd "${PROJECT_ROOT}/MambaQuant" && sbatch --export=ALL,TASKS="${task}",INCLUDE_PATH="${INCLUDE_PATH}",MODEL_PATH="${MAMBAQUANT_MODEL_PATH}",CONDA_ENV_PATH="${MAMBAQUANT_CONDA_ENV_PATH}",QUANT_WEIGHT=1,QUANT_ACT=1,W_BIT=4,A_BIT=8,MODEL_DTYPE=float16,BATCH_SIZE="${TEXTQA_BATCH_SIZE}",RUN_TAG="${run_tag}" model_llm/tests/slurm_mambaquant_w8a8.sh)
    else
      echo "[WARN] Unsupported mode='${mode}' for MambaQuant, skipping."
      return 0
    fi
    local jobid="${out##* }"
    printf 'mambaquant\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${mode}" "${task}" "${jobid}" "${run_tag}" "${TEXTQA_CANONICAL_CHECKPOINT}" "${QUAMBA_MODEL}" "${MAMBAQUANT_MODEL_PATH}" >> "${MAP_PATH}"
  }

  IFS=',' read -r -a TASK_ARR <<< "${TEXTQA_TASKS}"
  IFS=',' read -r -a MODE_ARR <<< "${TEXTQA_MODES}"

  echo "[INFO] PIPELINE_KIND=textqa"
  echo "[INFO] PROJECT_ROOT=${PROJECT_ROOT}"
  echo "[INFO] INCLUDE_PATH=${INCLUDE_PATH}"
  echo "[INFO] TEXTQA_TASKS=${TEXTQA_TASKS}"
  echo "[INFO] TEXTQA_MODES=${TEXTQA_MODES}"
  echo "[INFO] TEXTQA_BATCH_SIZE=${TEXTQA_BATCH_SIZE} TEXTQA_MAX_LENGTH=${TEXTQA_MAX_LENGTH}"
  echo "[INFO] TEXTQA_CANONICAL_CHECKPOINT=${TEXTQA_CANONICAL_CHECKPOINT}"
  echo "[INFO] TEXTQA_ENFORCE_SAME_CHECKPOINT=${TEXTQA_ENFORCE_SAME_CHECKPOINT}"
  echo "[INFO] QUAMBA_MODEL(resolved)=${QUAMBA_MODEL}"
  echo "[INFO] MAMBAQUANT_MODEL_PATH(resolved)=${MAMBAQUANT_MODEL_PATH}"
  if [[ "${QUAMBA_MODEL}" != "${MAMBAQUANT_MODEL_PATH}" ]]; then
    echo "[WARN] Framework model sources differ: QUAMBA_MODEL='${QUAMBA_MODEL}' vs MAMBAQUANT_MODEL_PATH='${MAMBAQUANT_MODEL_PATH}'"
  fi
  echo "[INFO] QUAMBA_CONDA_ENV_PATH=${QUAMBA_CONDA_ENV_PATH}"
  echo "[INFO] MAMBAQUANT_CONDA_ENV_PATH=${MAMBAQUANT_CONDA_ENV_PATH}"
  echo "[INFO] QUAMBA_STRICT_ENV=${QUAMBA_STRICT_ENV}"
  echo "[INFO] QUAMBA_QUANTIZE_EMBEDDING=${QUAMBA_QUANTIZE_EMBEDDING:-0} QUAMBA_QUANTIZE_LM_HEAD=${QUAMBA_QUANTIZE_LM_HEAD:-0}"
  echo "[INFO] QUAMBA_CALIB_DATA_NUM=${QUAMBA_CALIB_DATA_NUM:-512} QUAMBA_CALIB_SEQLEN=${QUAMBA_CALIB_SEQLEN:-512}"
  echo "[INFO] QUAMBA_APPLY_GPTQ=${QUAMBA_APPLY_GPTQ:-auto(1 for w4, 0 for w8)}"

  for task in "${TASK_ARR[@]}"; do
    for mode in "${MODE_ARR[@]}"; do
      case "${mode}" in
        fp16)
          submit_quamba_fp16 "${task}"
          submit_mambaquant fp16 "${task}"
          ;;
        w8a8)
          submit_quamba_quant w8a8 "${task}" 8 8
          submit_mambaquant w8a8 "${task}"
          ;;
        w4a8)
          submit_quamba_quant w4a8 "${task}" 4 8
          submit_mambaquant w4a8 "${task}"
          ;;
        *)
          echo "[WARN] Unsupported TEXTQA mode='${mode}', skipping."
          ;;
      esac
    done
  done

  echo "[INFO] Submitted text-QA sweep jobs:"
  cat "${MAP_PATH}"
  echo "[INFO] Mapping file: ${MAP_PATH}"
  exit 0
fi

# ==============================
# 4. Controls
# ==============================

MODE="${MODE:-full}"  # prepare | evaluate | score | full

DATA_ROOT="${DATA_ROOT:-/work/u9907741/project/vlm-evaluation}"
DATASET_FAMILY="${DATASET_FAMILY:-text-vqa}"
DATASET_KEY="${DATASET_KEY:-TEXTVQA_SLIM}"

MODEL_FAMILY="${MODEL_FAMILY:-cobra}"

# MODEL_ID:
#   float: cobra+3b
#   fake : cobra+3b-ptq-w8-fake   (bits tag derived from BITS)
case "${BACKEND_LOWER}" in
  float)
    MODEL_ID="${MODEL_ID:-${COBRA_MODEL_BASE_ID}}"
    MODEL_DIR=""
    ;;
  fake)
    MODEL_ID="${MODEL_ID:-${COBRA_MODEL_BASE_ID}-ptq-${BITS_TAG}-${BACKEND_LOWER}}"
    MODEL_DIR="${MODEL_DIR:-${COBRA_1115_ROOT}}"
    ;;
esac

RESULTS_BASE_DIR="${RESULTS_DIR:-results}"
RUN_TAG="${RUN_TAG:-}"
AUTO_RUN_TAG="${AUTO_RUN_TAG:-0}"
if [[ -z "${RUN_TAG}" && "${AUTO_RUN_TAG}" == "1" ]]; then
  RUN_TAG="$(date +%Y%m%d_%H%M%S)"
fi
if [[ -n "${RUN_TAG}" ]]; then
  RESULTS_DIR="${RESULTS_BASE_DIR}/${RUN_TAG}"
else
  RESULTS_DIR="${RESULTS_BASE_DIR}"
fi
HF_TOKEN_PATH="${HF_TOKEN_PATH:-/work/u9907741/project/vlm-evaluation/.hf_token}"

export MODE DATA_ROOT DATASET_FAMILY DATASET_KEY
export MODEL_FAMILY MODEL_ID MODEL_DIR
export RESULTS_DIR RESULTS_BASE_DIR RUN_TAG AUTO_RUN_TAG HF_TOKEN_PATH
export BITS BACKEND COBRA_MODEL_BASE_ID COBRA_1115_ROOT

# ==============================
# 4.5 Optional LLM mixer rotation controls (default OFF)
# ==============================
# Enable by setting:
#   export COBRA_LLM_MIXER_ROTATE=1
#
# Optional knobs:
#   COBRA_LLM_MIXER_ACT_KLT=1                 # enable act-KLT (IN side)
#   COBRA_LLM_MIXER_OUT_TRANSFORM=1           # enable output-side transform (requires OUT payload)
#   COBRA_LLM_MIXER_BLOCK=512                 # block size
#   ACT_KLT_OUTPROJ_IN=/path/to/in.pt         # required if ACT_KLT=1
#   ACT_KLT_OUTPROJ_OUT=/path/to/out.pt       # required if OUT_TRANSFORM=1
#   COBRA_LLM_MIXER_TARGETS=out_proj          # keep default
#   COBRA_LLM_MIXER_KLT_DTYPE=fp32|fp16       # optional (fp32 default)
#
# Strongly recommended when rotation is enabled:
#   COBRA_DISABLE_MAMBA_FAST_PATH=1           # avoids bypassing nn.Linear hooks
#
export COBRA_LLM_MIXER_ROTATE="${COBRA_LLM_MIXER_ROTATE:-1}"

if [[ "${COBRA_LLM_MIXER_ROTATE}" == "1" ]]; then
  export COBRA_LLM_MIXER_HADAMARD="${COBRA_LLM_MIXER_HADAMARD:-1}"
  export COBRA_LLM_MIXER_TARGETS="${COBRA_LLM_MIXER_TARGETS:-out_proj}"
  export COBRA_LLM_MIXER_BLOCK="${COBRA_LLM_MIXER_BLOCK:-512}"
  export COBRA_LLM_MIXER_REQUIRE_SUCCESS="${COBRA_LLM_MIXER_REQUIRE_SUCCESS:-1}"

  # act-KLT (IN)
  export COBRA_LLM_MIXER_ACT_KLT="${COBRA_LLM_MIXER_ACT_KLT:-1}"
  if [[ "${COBRA_LLM_MIXER_ACT_KLT}" == "1" ]]; then
    export ACT_KLT_OUTPROJ_IN="${ACT_KLT_OUTPROJ_IN:-${COBRA_1115_ROOT}/outputs/quantize/act_klt_outproj_in_bs${COBRA_LLM_MIXER_BLOCK}/act_klt_outproj_in.pt}"
  fi

  # output-side transform (optional)
  export COBRA_LLM_MIXER_OUT_TRANSFORM="${COBRA_LLM_MIXER_OUT_TRANSFORM:-0}"
  if [[ "${COBRA_LLM_MIXER_OUT_TRANSFORM}" == "1" ]]; then
    export ACT_KLT_OUTPROJ_OUT="${ACT_KLT_OUTPROJ_OUT:-${COBRA_1115_ROOT}/outputs/quantize/act_klt_outproj_out_bs${COBRA_LLM_MIXER_BLOCK}/act_klt_outproj_out.pt}"
  fi

  # ensure mamba slow path so hooks are honored
  export COBRA_DISABLE_MAMBA_FAST_PATH="${COBRA_DISABLE_MAMBA_FAST_PATH:-1}"

  echo "[INFO] COBRA_LLM_MIXER_ROTATE=1"
  echo "[INFO] COBRA_LLM_MIXER_HADAMARD=${COBRA_LLM_MIXER_HADAMARD}"
  echo "[INFO] COBRA_LLM_MIXER_TARGETS=${COBRA_LLM_MIXER_TARGETS}"
  echo "[INFO] COBRA_LLM_MIXER_BLOCK=${COBRA_LLM_MIXER_BLOCK}"
  echo "[INFO] COBRA_LLM_MIXER_REQUIRE_SUCCESS=${COBRA_LLM_MIXER_REQUIRE_SUCCESS}"
  echo "[INFO] COBRA_LLM_MIXER_ACT_KLT=${COBRA_LLM_MIXER_ACT_KLT}"
  if [[ "${COBRA_LLM_MIXER_ACT_KLT}" == "1" ]]; then
    echo "[INFO] ACT_KLT_OUTPROJ_IN=${ACT_KLT_OUTPROJ_IN}"
  fi
  echo "[INFO] COBRA_LLM_MIXER_OUT_TRANSFORM=${COBRA_LLM_MIXER_OUT_TRANSFORM}"
  if [[ "${COBRA_LLM_MIXER_OUT_TRANSFORM}" == "1" ]]; then
    echo "[INFO] ACT_KLT_OUTPROJ_OUT=${ACT_KLT_OUTPROJ_OUT}"
  fi
  echo "[INFO] COBRA_DISABLE_MAMBA_FAST_PATH=${COBRA_DISABLE_MAMBA_FAST_PATH}"
else
  echo "[INFO] COBRA_LLM_MIXER_ROTATE=0"
fi

echo "[INFO] MODE=${MODE}"
echo "[INFO] DATASET_FAMILY=${DATASET_FAMILY}, DATASET_KEY=${DATASET_KEY}"
echo "[INFO] MODEL_FAMILY=${MODEL_FAMILY}"
echo "[INFO] MODEL_ID=${MODEL_ID}"
echo "[INFO] MODEL_DIR(run_dir)=${MODEL_DIR}"
echo "[INFO] RUN_TAG=${RUN_TAG:-<none>}"
echo "[INFO] RESULTS_BASE_DIR=${RESULTS_BASE_DIR}"
echo "[INFO] RESULTS_DIR=${RESULTS_DIR}"
echo "[INFO] HF_TOKEN_PATH=${HF_TOKEN_PATH}"
echo "[INFO] VLM_ROOT=${VLM_ROOT}"
echo "[INFO] COBRA_1115_ROOT=${COBRA_1115_ROOT}"
echo "[INFO] BACKEND=${BACKEND}, BITS=${BITS}"
echo "[INFO] COBRA_USE_CFZO_MAMBA=${COBRA_USE_CFZO_MAMBA}"

# ==============================
# 5. Dataset prepare (scripts/datasets/prepare.py)
# ==============================
if [[ "${MODE}" == "prepare" || "${MODE}" == "full" ]]; then
  echo "[STEP] Running vlm-evaluation dataset prepare (family=${DATASET_FAMILY})..."

  python - << 'PY'
import os
from pathlib import Path

from scripts.datasets.prepare import DatasetPreparationConfig, prepare

dataset_family = os.environ.get("DATASET_FAMILY", "vqa-v2")
data_root = Path(os.environ.get("DATA_ROOT", "/work/u9907741/project/vlm-evaluation"))
hf_token_path = Path(os.environ.get("HF_TOKEN_PATH", ".hf_token"))

if not hf_token_path.is_file():
    raise SystemExit(
        f"[ERROR] HF token file not found: {hf_token_path}\n"
        f"        Please create it (one-line token) or set HF_TOKEN_PATH correctly."
    )

cfg = DatasetPreparationConfig(
    dataset_family=dataset_family,
    root_dir=data_root,
    hf_token=hf_token_path,
)

prepare(cfg)
PY

  echo "[STEP] Dataset prepare finished for family=${DATASET_FAMILY}"
fi

# ==============================
# 6. Evaluate (scripts/evaluate.py)
# ==============================
if [[ "${MODE}" == "evaluate" || "${MODE}" == "full" ]]; then
  echo "[STEP] Running vlm-evaluation evaluate.py ..."

  python - << 'PY'
import os
from pathlib import Path

from vlm_eval.conf.datasets import DatasetConfig, DatasetRegistry
from scripts.evaluate import EvaluationConfig, evaluate

dataset_key = os.environ.get("DATASET_KEY", "VQAv2_FULL")
data_root = Path(os.environ.get("DATA_ROOT", "/work/u9907741/project/vlm-evaluation"))

model_family = os.environ.get("MODEL_FAMILY", "cobra")
model_id = os.environ.get("MODEL_ID", "cobra+3b")
model_dir_str = os.environ.get("MODEL_DIR", "").strip()
results_dir = Path(os.environ.get("RESULTS_DIR", "results"))
hf_token_path = Path(os.environ.get("HF_TOKEN_PATH", ".hf_token"))

hf_token = hf_token_path

try:
    ds_enum = DatasetRegistry[dataset_key]
except KeyError as e:
    raise SystemExit(
        f"[ERROR] Unknown DATASET_KEY={dataset_key!r}. "
        f"Please verify DatasetRegistry in vlm_eval/conf/datasets.py."
    ) from e

ds_cls = DatasetConfig.get_choice_class(ds_enum.dataset_id)
dataset_cfg = ds_cls()
dataset_cfg.root_dir = data_root

model_dir = Path(model_dir_str) if model_dir_str else None

cfg = EvaluationConfig(
    dataset=dataset_cfg,
    model_family=model_family,
    model_id=model_id,
    model_dir=model_dir,
    results_dir=results_dir,
    hf_token=hf_token,
)

evaluate(cfg)
PY

  echo "[STEP] Evaluate finished for MODEL_FAMILY=${MODEL_FAMILY}, MODEL_ID=${MODEL_ID}, DATASET_KEY=${DATASET_KEY}"
fi

# ==============================
# 7. Score (scripts/score.py)
# ==============================
if [[ "${MODE}" == "score" || "${MODE}" == "full" ]]; then
  echo "[STEP] Running vlm-evaluation score.py ..."

  python - << 'PY'
import os
import inspect
from pathlib import Path

from scripts.score import ScoreConfig, score
from vlm_eval.conf.datasets import DatasetConfig, DatasetRegistry

dataset_key = os.environ.get("DATASET_KEY", "VQAv2_FULL")
results_dir = Path(os.environ.get("RESULTS_DIR", "results"))
model_id = os.environ.get("MODEL_ID", "cobra+3b")
data_root = Path(os.environ.get("DATA_ROOT", "/work/u9907741/project/vlm-evaluation"))

sig = inspect.signature(ScoreConfig)
params = sig.parameters

kwargs = {}

def _build_dataset_cfg(key: str):
    try:
        ds_enum = DatasetRegistry[key]
    except KeyError as e:
        raise SystemExit(
            f"[ERROR] Unknown DATASET_KEY={key!r}. "
            f"Please verify DatasetRegistry in vlm_eval/conf/datasets.py."
        ) from e

    ds_cls = DatasetConfig.get_choice_class(ds_enum.dataset_id)
    cfg = ds_cls()
    cfg.root_dir = data_root
    return cfg

# dataset selector (API differs across repo revisions)
# Prefer passing DatasetConfig instance when possible; score.py often expects cfg.dataset.dataset_id
if "dataset" in params:
    kwargs["dataset"] = _build_dataset_cfg(dataset_key)
elif "dataset_key" in params:
    kwargs["dataset_key"] = dataset_key
elif "dataset_name" in params:
    kwargs["dataset_name"] = dataset_key
elif "dataset_id" in params:
    kwargs["dataset_id"] = dataset_key

# results dir (API differs across repo revisions)
if "results_dir" in params:
    kwargs["results_dir"] = results_dir
elif "root_dir" in params:
    kwargs["root_dir"] = results_dir
elif "results_root" in params:
    kwargs["results_root"] = results_dir
elif "predictions_dir" in params:
    kwargs["predictions_dir"] = results_dir

# model selector (API differs across repo revisions)
if "model_id" in params:
    kwargs["model_id"] = model_id
elif "model" in params:
    kwargs["model"] = model_id

try:
    cfg = ScoreConfig(**kwargs)
except TypeError as e:
    raise SystemExit(
        "[ERROR] Failed to construct ScoreConfig with inferred kwargs.\n"
        f"  inferred kwargs={kwargs}\n"
        f"  ScoreConfig signature={sig}\n"
        f"  original error: {e}"
    )

score(cfg)
PY

  echo "[STEP] Score finished for DATASET_KEY=${DATASET_KEY}"
fi
echo "[INFO] vlm-evaluation script completed."
