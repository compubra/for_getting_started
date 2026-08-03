#!/bin/bash
#SBATCH --job-name=lsac-smoke-tilt0
#SBATCH --output=lsac_smoke_tilt0_%j.out
#SBATCH --error=lsac_smoke_tilt0_%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
# ============================================================================
#  One-off validation for the genpath_roi30_tilt0 sweep (cam_pitch_deg=0,
#  roi_bottom_fraction=0.3, gen-track random/0.4) before submitting the full
#  9-run OFAT matrix. Not part of the general smoke_residual.sbatch gate.
# ============================================================================
set -uo pipefail

echo "=== LSAC tilt0/roi30 smoke ==="
echo "host: $(hostname)  date: $(date)  job: ${SLURM_JOB_ID:-?}"

source "$HOME/LSAC/activate_mujoco_rl.sh"
export MUJOCO_GL=egl

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

python smoke_tilt0_roi30.py
rc=$?
echo "=== smoke finished (rc=$rc) ==="
exit $rc
