#!/bin/bash
#SBATCH --job-name=lsac-res-smoke
#SBATCH --output=lsac_res_smoke_%j.out
#SBATCH --error=lsac_res_smoke_%j.err
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
# ============================================================================
#  smoke_residual.sbatch — gate for the full SAC/PPO residual trainings.
#  Verifies the .slx-derived spec (8-D obs, 2-D action [delta_v, delta_omega]
#  since 2026-07-21, SteeringSign=-1 + upright camera, ROIFraction=0.5):
#  PID must still track the line, both agents must train.
#  The full jobs are submitted with --dependency=afterok on this job.
# ============================================================================
set -uo pipefail

echo "=== LSAC residual smoke ==="
echo "host: $(hostname)  date: $(date)  job: ${SLURM_JOB_ID:-?}"

source "$HOME/LSAC/activate_mujoco_rl.sh"
export MUJOCO_GL=egl

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

python smoke_residual.py
rc=$?
echo "=== smoke finished (rc=$rc) ==="
exit $rc
