#!/usr/bin/env bash
#
# Submit with:
#   sbatch scripts/slurm_longbench_mamba.sh
#
# Env overrides (examples):
#   TASK=hotpotqa MODEL_PATH=state-spaces/mamba-1.4b-hf NUM_SAMPLES=50 sbatch scripts/slurm_longbench_mamba.sh
#
# This script runs the HF-only Mamba LongBench baseline (no cobra imports).
#
#SBATCH -J mamba_lb_eval
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH -p dev
#SBATCH -A MST114205
#SBATCH -o /work/u9907741/project_0209/runs/slurm/%x.%j.out
#SBATCH -e /work/u9907741/project_0209/runs/slurm/%x.%j.err

set -eo pipefail
IFS=$'\n\t'

# ---------- Env / modules ----------
module load miniconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
# Update this to your env name/path if needed.
conda activate /work/u9907741/miniconda3/envs/cobra-1

# Optional: auto-install missing runtime deps in current env.
# Enabled by default so jobs self-heal dependency issues on cluster nodes.
AUTO_INSTALL_DEPS=${AUTO_INSTALL_DEPS:-1}

install_compatible_stack() {
  # Keep a known-good HF stack for this benchmark script.
  python -m pip install \
    "numpy" \
    "transformers==4.39.3" \
    "tokenizers==0.15.2" \
    "huggingface-hub>=0.19.3,<1.0" \
    "datasets>=2.14,<3.0"
}

ensure_causal_conv1d() {
  if python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("causal_conv1d") else 1)
PY
  then
    return 0
  fi

  echo "[WARN] causal_conv1d is missing; installing a torch2.1-compatible build ..."
  PIP_NO_BUILD_ISOLATION=1 python -m pip install --no-build-isolation "causal-conv1d==1.2.0.post2"
}

# ---------- Paths ----------
export REPO_ROOT=${REPO_ROOT:-/work/u9907741/project_0209/cobra_1115}
export RUNS_ROOT=${RUNS_ROOT:-/work/u9907741/project_0209/runs}
export HF_HOME=${HF_HOME:-"${REPO_ROOT}/.hf_cache"}
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_TOKEN=${HF_TOKEN:-"${REPO_ROOT}/.hf_token"}

mkdir -p "${HF_HOME}" "${RUNS_ROOT}/slurm" "${RUNS_ROOT}/longbench_eval"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ---------- Dependency preflight ----------
MISSING_DEPS=$(python - <<'PY'
import importlib.util
required = ("numpy", "datasets", "torch", "transformers")
missing = [name for name in required if importlib.util.find_spec(name) is None]
print(" ".join(missing))
PY
)

if [[ -n "${MISSING_DEPS}" ]]; then
  echo "[WARN] Missing Python deps in current env: ${MISSING_DEPS}"
  if [[ "${AUTO_INSTALL_DEPS}" == "1" ]]; then
    echo "[INFO] Installing compatible dependency stack via pip..."
    install_compatible_stack
  else
    echo "[ERROR] Missing deps. Re-submit with AUTO_INSTALL_DEPS=1, or install manually:"
    echo "        conda activate /work/u9907741/miniconda3/envs/cobra-1"
    echo "        python -m pip install \"numpy\" \"transformers==4.39.3\" \"tokenizers==0.15.2\" \"huggingface-hub>=0.19.3,<1.0\" \"datasets>=2.14,<3.0\""
    exit 1
  fi
fi

COMPAT_ISSUES=$(python - <<'PY'
from importlib.metadata import version, PackageNotFoundError

def parse(v: str):
    core = v.split("+", 1)[0]
    parts = core.split(".")
    out = []
    for p in parts:
        n = ""
        for ch in p:
            if ch.isdigit():
                n += ch
            else:
                break
        out.append(int(n) if n else 0)
    while len(out) < 3:
        out.append(0)
    return tuple(out[:3])

issues = []
try:
    tr_v = parse(version("transformers"))
except PackageNotFoundError:
    tr_v = None
if tr_v is not None and tr_v[:2] != (4, 39):
    issues.append("transformers not in 4.39.x")

try:
    tk_v = parse(version("tokenizers"))
except PackageNotFoundError:
    tk_v = None
if tk_v is not None and tk_v[:2] != (0, 15):
    issues.append("tokenizers not in 0.15.x")

try:
    hub_v = parse(version("huggingface-hub"))
except PackageNotFoundError:
    hub_v = None
if hub_v is not None and not ((0, 19, 3) <= hub_v < (1, 0, 0)):
    issues.append("huggingface-hub not in [0.19.3,1.0)")

try:
    ds_v = parse(version("datasets"))
except PackageNotFoundError:
    ds_v = None
if ds_v is not None and ds_v[0] >= 3:
    issues.append("datasets major version must be < 3")

print(" | ".join(issues))
PY
)

if [[ -n "${COMPAT_ISSUES}" ]]; then
  echo "[WARN] Dependency compatibility issue: ${COMPAT_ISSUES}"
  if [[ "${AUTO_INSTALL_DEPS}" == "1" ]]; then
    echo "[INFO] Fixing HF stack compatibility ..."
    install_compatible_stack
  else
    echo "[ERROR] Version conflict detected. Re-submit with AUTO_INSTALL_DEPS=1, or run:"
    echo "        conda activate /work/u9907741/miniconda3/envs/cobra-1"
    echo "        python -m pip install \"numpy\" \"transformers==4.39.3\" \"tokenizers==0.15.2\" \"huggingface-hub>=0.19.3,<1.0\" \"datasets>=2.14,<3.0\""
    exit 1
  fi
fi

# Ensure causal_conv1d exists so Mamba can avoid naive-path fallback.
ensure_causal_conv1d

# ---------- Params ----------
TASK=${TASK:-hotpotqa}                    # hotpotqa | 2wikimqa | narrativeqa | qmsum
MODEL_PATH=${MODEL_PATH:-state-spaces/mamba-1.4b-hf}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
NUM_SAMPLES=${NUM_SAMPLES:-}             # optional limit for smoke; empty = full set
OUTPUT_DIR=${OUTPUT_DIR:-"${RUNS_ROOT}/longbench_eval"}
PROMPT_TEMPLATE=${PROMPT_TEMPLATE:-"Answer the question based on the given passages.\n\nContext: {context}\n\nQuestion: {input}\n\nAnswer:"}
MAX_CONTEXT_TOKENS=${MAX_CONTEXT_TOKENS:-8192}
MIN_TOKENS=${MIN_TOKENS:-4096}
MAX_TOKENS=${MAX_TOKENS:-8192}
DROP_CALIBRATION=${DROP_CALIBRATION:-10}
SEED=${SEED:-42}

ARGS=(
  scripts/run_longbench_baseline_mamba.py
  --task "${TASK}"
  --model-path "${MODEL_PATH}"
  --hf-token "${HF_TOKEN}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --output-dir "${OUTPUT_DIR}"
  --prompt-template "${PROMPT_TEMPLATE}"
  --max-context-tokens "${MAX_CONTEXT_TOKENS}"
  --min-tokens "${MIN_TOKENS}"
  --max-tokens "${MAX_TOKENS}"
  --drop-calibration "${DROP_CALIBRATION}"
  --seed "${SEED}"
)
if [[ -n "${NUM_SAMPLES}" ]]; then
  ARGS+=( --num-samples "${NUM_SAMPLES}" )
fi

echo "[INFO] Running LongBench (Mamba HF baseline):"
echo "      TASK=${TASK} MODEL_PATH=${MODEL_PATH} NUM_SAMPLES=${NUM_SAMPLES:-full} MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "      MIN_TOKENS=${MIN_TOKENS} MAX_TOKENS=${MAX_TOKENS} DROP_CALIBRATION=${DROP_CALIBRATION}"
echo "      OUTPUT_DIR=${OUTPUT_DIR}"

set -x
srun -u python "${ARGS[@]}"
set +x

echo "[DONE] Eval finished. Check ${OUTPUT_DIR}"
