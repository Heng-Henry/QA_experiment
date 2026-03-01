#!/usr/bin/env bash

set -euo pipefail

# Submit Cobra architecture ablations by overriding env vars consumed by vlm_eval.sh.
#
# Usage:
#   bash scripts/submit_cobra_arch_sweep.sh
#
# Optional overrides:
#   DATASET_FAMILY=text-vqa DATASET_KEY=TEXTVQA_SLIM bash scripts/submit_cobra_arch_sweep.sh
#   BACKEND=fake BITS=W8 bash scripts/submit_cobra_arch_sweep.sh
#   INCLUDE_OUT_TRANSFORM=1 bash scripts/submit_cobra_arch_sweep.sh

VLM_ROOT="${VLM_ROOT:-/work/u9907741/project_0209/cobra_1115-evaluation}"
COBRA_1115_ROOT="${COBRA_1115_ROOT:-/work/u9907741/project_0209/cobra_1115}"
RUNS_ROOT="${RUNS_ROOT:-/work/u9907741/project_0209/runs}"
SLURM_LOG_ROOT="${SLURM_LOG_ROOT:-${RUNS_ROOT}/slurm}"

DATA_ROOT="${DATA_ROOT:-/work/u9907741/project/vlm-evaluation}"
DATASET_FAMILY="${DATASET_FAMILY:-text-vqa}"
DATASET_KEY="${DATASET_KEY:-TEXTVQA_SLIM}"

MODE="${MODE:-evaluate}"              # prepare | evaluate | score | full
MODEL_FAMILY="${MODEL_FAMILY:-cobra}" # cobra | cobra-quant
BACKEND="${BACKEND:-float}"           # float | fake
BITS="${BITS:-W8}"                    # used if BACKEND=fake
RUN_TAG_BASE="${RUN_TAG_BASE:-}"
AUTO_RUN_TAG="${AUTO_RUN_TAG:-0}"

HF_TOKEN_PATH="${HF_TOKEN_PATH:-/work/u9907741/project/vlm-evaluation/.hf_token}"
RESULTS_DIR="${RESULTS_DIR:-${RUNS_ROOT}/cobra_vlm_eval}"
SBATCH_PARTITION="${SBATCH_PARTITION:-normal}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-MST114205}"
SBATCH_TIME="${SBATCH_TIME:-08:00:00}"
DRY_RUN="${DRY_RUN:-0}"
COBRA_USE_CFZO_MAMBA="${COBRA_USE_CFZO_MAMBA:-1}"

INCLUDE_OUT_TRANSFORM="${INCLUDE_OUT_TRANSFORM:-0}"
MIXER_BLOCK="${MIXER_BLOCK:-512}"
MIXER_TARGETS="${MIXER_TARGETS:-out_proj}"

VLM_EVAL_SH="${VLM_ROOT}/vlm_eval.sh"
ACT_KLT_OUTPROJ_IN_DEFAULT="${COBRA_1115_ROOT}/outputs/quantize/act_klt_outproj_in_bs${MIXER_BLOCK}/act_klt_outproj_in.pt"
ACT_KLT_OUTPROJ_OUT_DEFAULT="${COBRA_1115_ROOT}/outputs/quantize/act_klt_outproj_out_bs${MIXER_BLOCK}/act_klt_outproj_out.pt"

mkdir -p "${SLURM_LOG_ROOT}" "${RESULTS_DIR}"

SWEEP_TS="${SWEEP_TS:-$(date +%Y%m%d_%H%M%S)}"
MANIFEST_PATH="${MANIFEST_PATH:-${SLURM_LOG_ROOT}/cobra_arch_sweep_${DATASET_KEY,,}_${BACKEND,,}_${BITS,,}_${SWEEP_TS}.tsv}"

if [[ ! -f "${MANIFEST_PATH}" ]]; then
  printf 'status\treason\tjob_name\tjob_id\tbackend\tbits\tmode\tdataset_family\tdataset_key\tablation\trotate\thadamard\tact_klt\tout_transform\tmixer_targets\tmixer_block\trun_tag\tresults_dir\tslurm_out\tslurm_err\tact_klt_in\tact_klt_out\n' > "${MANIFEST_PATH}"
fi

if [[ ! -f "${VLM_EVAL_SH}" ]]; then
  echo "[ERROR] vlm_eval.sh not found: ${VLM_EVAL_SH}" >&2
  exit 1
fi

declare -a CONFIGS=(
  # name:ROTATE:HADAMARD:ACT_KLT:OUT_TRANSFORM
  "base_no_rotate:0:0:0:0"
  "rotate_only:1:0:0:0"
  "rotate_hadamard:1:1:0:0"
  "rotate_hadamard_actklt:1:1:1:0"
)

if [[ "${INCLUDE_OUT_TRANSFORM}" == "1" ]]; then
  CONFIGS+=("rotate_hadamard_actklt_out:1:1:1:1")
fi

echo "[INFO] Sweep submit start"
echo "[INFO] VLM_ROOT=${VLM_ROOT}"
echo "[INFO] COBRA_1115_ROOT=${COBRA_1115_ROOT}"
echo "[INFO] DATASET_FAMILY=${DATASET_FAMILY} DATASET_KEY=${DATASET_KEY}"
echo "[INFO] MODE=${MODE} MODEL_FAMILY=${MODEL_FAMILY} BACKEND=${BACKEND} BITS=${BITS}"
echo "[INFO] COBRA_USE_CFZO_MAMBA=${COBRA_USE_CFZO_MAMBA}"
echo "[INFO] RESULTS_DIR=${RESULTS_DIR}"
echo "[INFO] MANIFEST_PATH=${MANIFEST_PATH}"

for cfg in "${CONFIGS[@]}"; do
  IFS=":" read -r NAME ROTATE HADAMARD ACT_KLT OUT_TRANSFORM <<< "${cfg}"

  ACT_IN="${ACT_KLT_OUTPROJ_IN:-${ACT_KLT_OUTPROJ_IN_DEFAULT}}"
  ACT_OUT="${ACT_KLT_OUTPROJ_OUT:-${ACT_KLT_OUTPROJ_OUT_DEFAULT}}"

  if [[ "${ACT_KLT}" == "1" && ! -f "${ACT_IN}" ]]; then
    echo "[WARN] Skip ${NAME}: missing ACT_KLT_OUTPROJ_IN=${ACT_IN}"
    printf 'SKIPPED\tmissing_act_klt_in\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "cobra_${DATASET_KEY,,}_${NAME}" "-" "${BACKEND}" "${BITS}" "${MODE}" "${DATASET_FAMILY}" "${DATASET_KEY}" "${NAME}" \
      "${ROTATE}" "${HADAMARD}" "${ACT_KLT}" "${OUT_TRANSFORM}" "${MIXER_TARGETS}" "${MIXER_BLOCK}" "-" "${RESULTS_DIR}" "-" "-" "${ACT_IN}" "${ACT_OUT}" >> "${MANIFEST_PATH}"
    continue
  fi
  if [[ "${OUT_TRANSFORM}" == "1" && ! -f "${ACT_OUT}" ]]; then
    echo "[WARN] Skip ${NAME}: missing ACT_KLT_OUTPROJ_OUT=${ACT_OUT}"
    printf 'SKIPPED\tmissing_act_klt_out\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "cobra_${DATASET_KEY,,}_${NAME}" "-" "${BACKEND}" "${BITS}" "${MODE}" "${DATASET_FAMILY}" "${DATASET_KEY}" "${NAME}" \
      "${ROTATE}" "${HADAMARD}" "${ACT_KLT}" "${OUT_TRANSFORM}" "${MIXER_TARGETS}" "${MIXER_BLOCK}" "-" "${RESULTS_DIR}" "-" "-" "${ACT_IN}" "${ACT_OUT}" >> "${MANIFEST_PATH}"
    continue
  fi

  JOB_NAME="cobra_${DATASET_KEY,,}_${NAME}"
  RUN_TAG_THIS=""
  if [[ -n "${RUN_TAG_BASE}" ]]; then
    RUN_TAG_THIS="${RUN_TAG_BASE}_${NAME}"
  elif [[ "${AUTO_RUN_TAG}" == "1" ]]; then
    RUN_TAG_THIS="$(date +%Y%m%d_%H%M%S)_${NAME}"
  fi

  EXPORTS=(
    "ALL"
    "VLM_ROOT=${VLM_ROOT}"
    "COBRA_1115_ROOT=${COBRA_1115_ROOT}"
    "DATA_ROOT=${DATA_ROOT}"
    "DATASET_FAMILY=${DATASET_FAMILY}"
    "DATASET_KEY=${DATASET_KEY}"
    "MODE=${MODE}"
    "MODEL_FAMILY=${MODEL_FAMILY}"
    "BACKEND=${BACKEND}"
    "BITS=${BITS}"
    "HF_TOKEN_PATH=${HF_TOKEN_PATH}"
    "RESULTS_DIR=${RESULTS_DIR}"
    "RUN_TAG=${RUN_TAG_THIS}"
    "AUTO_RUN_TAG=0"
    "COBRA_LLM_MIXER_ROTATE=${ROTATE}"
    "COBRA_LLM_MIXER_HADAMARD=${HADAMARD}"
    "COBRA_LLM_MIXER_ACT_KLT=${ACT_KLT}"
    "COBRA_LLM_MIXER_OUT_TRANSFORM=${OUT_TRANSFORM}"
    "COBRA_LLM_MIXER_TARGETS=${MIXER_TARGETS}"
    "COBRA_LLM_MIXER_BLOCK=${MIXER_BLOCK}"
    "COBRA_DISABLE_MAMBA_FAST_PATH=1"
    "COBRA_USE_CFZO_MAMBA=${COBRA_USE_CFZO_MAMBA}"
    "ACT_KLT_OUTPROJ_IN=${ACT_IN}"
    "ACT_KLT_OUTPROJ_OUT=${ACT_OUT}"
  )

  echo "[INFO] Submit ${JOB_NAME} (rotate=${ROTATE}, hadamard=${HADAMARD}, act_klt=${ACT_KLT}, out_transform=${OUT_TRANSFORM})"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "sbatch --account=${SBATCH_ACCOUNT} --partition=${SBATCH_PARTITION} --time=${SBATCH_TIME} --job-name=${JOB_NAME} --output=${SLURM_LOG_ROOT}/%x.%j.out --error=${SLURM_LOG_ROOT}/%x.%j.err --export=$(IFS=,; echo "${EXPORTS[*]}") ${VLM_EVAL_SH}"
    printf 'DRY_RUN\t-\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${JOB_NAME}" "-" "${BACKEND}" "${BITS}" "${MODE}" "${DATASET_FAMILY}" "${DATASET_KEY}" "${NAME}" \
      "${ROTATE}" "${HADAMARD}" "${ACT_KLT}" "${OUT_TRANSFORM}" "${MIXER_TARGETS}" "${MIXER_BLOCK}" "${RUN_TAG_THIS}" "${RESULTS_DIR}" "-" "-" "${ACT_IN}" "${ACT_OUT}" >> "${MANIFEST_PATH}"
  else
    SUBMIT_OUT=$(
    sbatch \
      --account="${SBATCH_ACCOUNT}" \
      --partition="${SBATCH_PARTITION}" \
      --time="${SBATCH_TIME}" \
      --job-name="${JOB_NAME}" \
      --output="${SLURM_LOG_ROOT}/%x.%j.out" \
      --error="${SLURM_LOG_ROOT}/%x.%j.err" \
      --export="$(IFS=,; echo "${EXPORTS[*]}")" \
      "${VLM_EVAL_SH}"
    )
    echo "${SUBMIT_OUT}"
    JOB_ID="$(awk '{print $NF}' <<< "${SUBMIT_OUT}")"
    SLURM_OUT_PATH="${SLURM_LOG_ROOT}/${JOB_NAME}.${JOB_ID}.out"
    SLURM_ERR_PATH="${SLURM_LOG_ROOT}/${JOB_NAME}.${JOB_ID}.err"
    printf 'SUBMITTED\t-\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${JOB_NAME}" "${JOB_ID}" "${BACKEND}" "${BITS}" "${MODE}" "${DATASET_FAMILY}" "${DATASET_KEY}" "${NAME}" \
      "${ROTATE}" "${HADAMARD}" "${ACT_KLT}" "${OUT_TRANSFORM}" "${MIXER_TARGETS}" "${MIXER_BLOCK}" "${RUN_TAG_THIS}" "${RESULTS_DIR}" "${SLURM_OUT_PATH}" "${SLURM_ERR_PATH}" "${ACT_IN}" "${ACT_OUT}" >> "${MANIFEST_PATH}"
  fi
done

echo "[INFO] Sweep submit done"
echo "[INFO] Manifest: ${MANIFEST_PATH}"
