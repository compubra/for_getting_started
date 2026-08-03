#!/bin/bash
#SBATCH --job-name=lsac-sac-res
#SBATCH --output=lsac_sac_residual_%j.out
#SBATCH --error=lsac_sac_residual_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
# ============================================================================
#  train_sac_residual.sbatch — quadratic-reward SAC residual training
#  (python translation of sac_training_config.m; 500 eps x 200 steps, matching
#  the 2026-07-20 real-car-speed MATLAB run; domain randomization ON with map
#  fixed to track_hard).
#
#  Submit:   sbatch train_sac_residual.sbatch    (from this directory)
# ============================================================================
set -uo pipefail

echo "=== LSAC python SAC residual training ==="
echo "host: $(hostname)  date: $(date)  job: ${SLURM_JOB_ID:-?}"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || echo "no nvidia-smi"

source "$HOME/LSAC/activate_mujoco_rl.sh"
export MUJOCO_GL=egl

# sbatch runs a spool copy of this script, so $0 is useless — use the
# submission directory (submit from train_bundle/python/).
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

# LSAC_GENTRACK=1 trains on fresh procedurally generated tracks instead of
# the fixed LSAC_MAPKEY map — recommended after this session's generalization
# check found a fixed-map agent got WORSE than the plain PID baseline on an
# unseen track. LSAC_MAPKEY is then ignored.
gen_track_args=()
if [ "${LSAC_GENTRACK:-0}" = "1" ]; then
    gen_track_args=(--gen-track
                     --gen-track-shape "${LSAC_GENTRACK_SHAPE:-random}"
                     --gen-track-difficulty "${LSAC_GENTRACK_DIFFICULTY:-0.4}")
fi

python train_sac_residual.py \
    --max-episodes "${LSAC_EPISODES:-500}" \
    --max-steps    "${LSAC_STEPS:-200}" \
    --map-key      "${LSAC_MAPKEY:-track_hard}" \
    --seed         "${LSAC_SEED:-0}" \
    "${gen_track_args[@]}"

echo "=== training job finished (rc=$?) ==="
