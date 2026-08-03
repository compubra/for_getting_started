#!/bin/bash
#SBATCH --job-name=lsac-gentrack-smoke
#SBATCH --output=lsac_gentrack_smoke_%j.out
#SBATCH --error=lsac_gentrack_smoke_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
# ============================================================================
#  smoke_gentrack.sbatch — one-off gate for the --gen-track (open-domain /
#  random-path) code path before the 2000x500 full run. Exercises
#  track_gen.py's skimage-based rasterizer (scikit-image was missing from
#  ~/envs/mujoco_rl until this session) end-to-end with EGL rendering, plus a
#  short SAC train loop with domain randomization + gen-track both on.
# ============================================================================
set -uo pipefail

echo "=== LSAC gen-track smoke ==="
echo "host: $(hostname)  date: $(date)  job: ${SLURM_JOB_ID:-?}"

source "$HOME/LSAC/activate_mujoco_rl.sh"
export MUJOCO_GL=egl

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

python train_sac_residual.py \
    --max-episodes 5 \
    --max-steps 30 \
    --gen-track --gen-track-shape random --gen-track-difficulty 0.4 \
    --checkpoint-every 100 \
    --seed 0 \
    --save-dir simulation_data/sac_residual_training/_gentrack_smoke
rc=$?
echo "=== gen-track smoke finished (rc=$rc) ==="
exit $rc
