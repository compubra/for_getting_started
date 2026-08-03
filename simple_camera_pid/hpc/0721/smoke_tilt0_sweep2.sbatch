#!/bin/bash
#SBATCH --job-name=lsac-smoke-sweep2
#SBATCH --output=lsac_smoke_sweep2_%j.out
#SBATCH --error=lsac_smoke_sweep2_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
# ============================================================================
#  One-off validation for submit_tilt0_sweep2.sh's 12 runs (ellipse/capsule
#  shape tracking, action-bound extremes, SAC hyperparameter construction)
#  before submitting the full round-2 sweep.
# ============================================================================
set -uo pipefail

echo "=== LSAC tilt0 sweep2 smoke ==="
echo "host: $(hostname)  date: $(date)  job: ${SLURM_JOB_ID:-?}"

source "$HOME/LSAC/activate_mujoco_rl.sh"
export MUJOCO_GL=egl

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

python smoke_tilt0_sweep2.py
rc=$?
echo "=== smoke finished (rc=$rc) ==="
exit $rc
