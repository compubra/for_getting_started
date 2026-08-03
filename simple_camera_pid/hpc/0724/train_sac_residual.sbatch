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
#  (python translation of sac_training_config.m, updated 2026-07-23:
#  Reward_Calculation gained Q_Lateral/Q_Heading terms, and GenTrack now
#  defaults ON with a 3-way Generator choice (simple/complex/wandering/random)
#  -- see train_sac_residual.py's module docstring. 2-D action
#  [delta_v, delta_omega] via the shared Diff_Drive_Kinematics conversion
#  -- see lsac/controller.py + lsac/env.py).
#
#  ⚠️ The 2026-07-21 MATLAB run with LSAC_BASE_SPEED_MAX at the full
#  Burger-rated 7.333 never converged (flat ~-260 avg reward, 500 episodes) —
#  too little steering headroom left at high sampled base speeds. Consider
#  LSAC_BASE_SPEED_MAX=5.6 or lower until this is re-validated.
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

# LSAC_GENTRACK defaults to 1 (ON), matching rl_dr_defaults.m's actual
# GenTrack.Enable=true (see train_sac_residual.py's module docstring — the
# MATLAB source's own "default off" comment is stale). Set LSAC_GENTRACK=0 to
# train on the fixed LSAC_MAPKEY map instead. LSAC_GENTRACK_GENERATOR selects
# which of the three track_gen algorithms runs each episode ("random" picks
# one of the three per episode, matching dr.GenTrack.Generator).
gen_track_args=()
if [ "${LSAC_GENTRACK:-1}" = "0" ]; then
    gen_track_args=(--no-gen-track)
else
    gen_track_args=(--gen-track-generator "${LSAC_GENTRACK_GENERATOR:-random}"
                     --gen-track-shape "${LSAC_GENTRACK_SHAPE:-random}"
                     --gen-track-difficulty "${LSAC_GENTRACK_DIFFICULTY:-0.4}")
fi

# LSAC_RANDMAP=1 randomizes --map-key every episode from the 7-map domain-rand
# pool (DRConfig.maps) instead of training on a single fixed map. Ignored
# unless LSAC_GENTRACK=0 is also set (gen-track takes priority, same as
# MATLAB).
rand_map_args=()
if [ "${LSAC_RANDMAP:-0}" = "1" ]; then
    rand_map_args=(--rand-map)
fi

# LSAC_CAM_PITCH_DEG overrides the front camera's downward mount pitch (deg);
# unset keeps the scene XML's baked-in 15 deg untouched. See
# train_sac_residual.py --cam-pitch-deg and lsac/env.py's _apply_cam_pitch.
cam_pitch_args=()
if [ -n "${LSAC_CAM_PITCH_DEG:-}" ]; then
    cam_pitch_args=(--cam-pitch-deg "$LSAC_CAM_PITCH_DEG")
fi

# LSAC_SAVE_DIR overrides the default timestamp-only save dir. Two runs
# submitted together as separate sbatch jobs can otherwise start in the same
# second and collide on simulation_data/sac_residual_training/<ts>/ (identical
# checkpoint filenames, silent overwrite/corruption) -- always set this when
# submitting concurrent runs.
save_dir_args=()
if [ -n "${LSAC_SAVE_DIR:-}" ]; then
    save_dir_args=(--save-dir "$LSAC_SAVE_DIR")
fi

# LSAC_RESUME_FROM=/path/to/checkpoint_epNNNNN.zip warm-starts the policy from
# a saved checkpoint instead of training from scratch -- e.g. to continue a
# run that crashed partway through (see track_gen.py's 2026-07-22 atomic-write
# fix). LSAC_START_EPISODE must be set to the episode count that checkpoint
# represents so numbering/stop-condition continue correctly instead of
# restarting at episode 1.
resume_args=()
if [ -n "${LSAC_RESUME_FROM:-}" ]; then
    resume_args=(--resume-from "$LSAC_RESUME_FROM"
                 --start-episode "${LSAC_START_EPISODE:?LSAC_START_EPISODE must be set when LSAC_RESUME_FROM is used}")
fi

python train_sac_residual.py \
    --max-episodes    "${LSAC_EPISODES:-500}" \
    --max-steps       "${LSAC_STEPS:-200}" \
    --checkpoint-every "${LSAC_CHECKPOINT_EVERY:-50}" \
    --map-key         "${LSAC_MAPKEY:-simple}" \
    --seed            "${LSAC_SEED:-0}" \
    --ent-init        "${LSAC_ENT_INIT:-0.2}" \
    --lr              "${LSAC_LR:-3e-4}" \
    --gamma           "${LSAC_GAMMA:-0.99}" \
    --tau             "${LSAC_TAU:-5e-3}" \
    --q-lateral       "${LSAC_Q_LATERAL:-0.5}" \
    --q-heading       "${LSAC_Q_HEADING:-0.5}" \
    --p-lost          "${LSAC_P_LOST:-10.0}" \
    --max-delta-v      "${LSAC_MAX_DELTA_V:-1.0}" \
    --max-delta-omega  "${LSAC_MAX_DELTA_OMEGA:-1.0}" \
    --base-speed-min   "${LSAC_BASE_SPEED_MIN:-0.0}" \
    --base-speed-max   "${LSAC_BASE_SPEED_MAX:-7.3333}" \
    --roi-fraction     "${LSAC_ROI_FRACTION:-0.5}" \
    ${gen_track_args[@]+"${gen_track_args[@]}"} \
    ${rand_map_args[@]+"${rand_map_args[@]}"} \
    ${cam_pitch_args[@]+"${cam_pitch_args[@]}"} \
    ${save_dir_args[@]+"${save_dir_args[@]}"} \
    ${resume_args[@]+"${resume_args[@]}"}

echo "=== training job finished (rc=$?) ==="
