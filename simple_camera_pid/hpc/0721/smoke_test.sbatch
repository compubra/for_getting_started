#!/bin/bash
#SBATCH --job-name=lsac-py-smoke
#SBATCH --output=lsac_py_smoke_%j.out
#SBATCH --error=lsac_py_smoke_%j.err
#SBATCH --time=00:20:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
# ============================================================================
#  smoke_test.sbatch — verify the PYTHON translation end-to-end on a GPU node:
#  EGL camera render + full env stepping + a 5-episode micro-training.
#
#  Submit:   sbatch smoke_test.sbatch        (from train_bundle/python/)
# ============================================================================
set -uo pipefail

echo "=== LSAC python smoke test ==="
echo "host: $(hostname)  date: $(date)  job: ${SLURM_JOB_ID:-?}"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || echo "no nvidia-smi"

source "$HOME/LSAC/activate_mujoco_rl.sh"
export MUJOCO_GL=egl

# sbatch runs a spool copy of this script, so $0 is useless — use the
# submission directory (submit from train_bundle/python/).
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

echo "--- unit tests (incl. camera render + env stepping) ---"
python tests/test_translation.py || exit 1

echo "--- PID-only rollout (no agent) ---"
python evaluate_agent.py --pid-only --map-key training --episodes 1 --max-steps 300 || exit 1

echo "--- 5-episode micro-training ---"
python train_sac_lyapunov.py --max-episodes 5 --max-steps 60 --map-key training \
    --save-dir "simulation_data/smoke_${SLURM_JOB_ID:-manual}" || exit 1

echo "=== SMOKE TEST PASSED ==="
