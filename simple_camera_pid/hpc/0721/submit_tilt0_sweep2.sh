#!/bin/bash
# ============================================================================
#  submit_tilt0_sweep2.sh — round 2 of the genpath_roi30_tilt0 OFAT sweep.
#  Same fixed spec as round 1 (cam_pitch_deg=0, roi_fraction=0.3, gen-track
#  procedural path, 500 eps x 200 steps), and reuses round 1's "baseline" run
#  (job 10987067: base-speed-max=5.6, ent-init=0.2, gen-track-difficulty=0.4,
#  seed=0) as the center point instead of re-training it.
#
#  This round perturbs 6 NEW axes one at a time from that same baseline:
#  lr, gamma, tau, max-delta-v/omega (moved together), p-lost, gen-track-shape.
#
#  Not gated on a smoke job -- round 1 already validated the tilt=0/roi=0.3
#  rendering+ROI combination works; these 12 runs only change SAC/reward
#  hyperparameters, which don't affect whether the line is visible.
#
#  Usage:  bash submit_tilt0_sweep2.sh          (from this directory)
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

BASE_DIR="simulation_data/sac_residual_training/genpath_roi30_tilt0"

# name : LSAC_LR : LSAC_GAMMA : LSAC_TAU : LSAC_MAX_DELTA_V=OMEGA : LSAC_P_LOST : LSAC_GENTRACK_SHAPE
runs=(
  "lr1e-4        1e-4  0.99  5e-3  1.0  10.0  random"
  "lr1e-3        1e-3  0.99  5e-3  1.0  10.0  random"
  "gamma0.95     3e-4  0.95  5e-3  1.0  10.0  random"
  "gamma0.999    3e-4  0.999 5e-3  1.0  10.0  random"
  "tau1e-3       3e-4  0.99  1e-3  1.0  10.0  random"
  "tau2e-2       3e-4  0.99  2e-2  1.0  10.0  random"
  "delta0.5      3e-4  0.99  5e-3  0.5  10.0  random"
  "delta2.0      3e-4  0.99  5e-3  2.0  10.0  random"
  "plost3        3e-4  0.99  5e-3  1.0  3.0   random"
  "plost30       3e-4  0.99  5e-3  1.0  30.0  random"
  "shape_ellipse 3e-4  0.99  5e-3  1.0  10.0  ellipse"
  "shape_capsule 3e-4  0.99  5e-3  1.0  10.0  capsule"
)

for row in "${runs[@]}"; do
    read -r name lr gamma tau delta plost shape <<< "$row"
    save_dir="${BASE_DIR}/${name}"
    echo "submitting ${name}: lr=${lr} gamma=${gamma} tau=${tau} " \
         "max-delta-v/omega=${delta} p-lost=${plost} gen-track-shape=${shape} -> ${save_dir}"
    sbatch \
        --export=ALL,LSAC_GENTRACK=1,LSAC_GENTRACK_SHAPE=${shape},LSAC_GENTRACK_DIFFICULTY=0.4,LSAC_ROI_FRACTION=0.3,LSAC_CAM_PITCH_DEG=0,LSAC_EPISODES=500,LSAC_STEPS=200,LSAC_BASE_SPEED_MAX=5.6,LSAC_ENT_INIT=0.2,LSAC_SEED=0,LSAC_LR=${lr},LSAC_GAMMA=${gamma},LSAC_TAU=${tau},LSAC_MAX_DELTA_V=${delta},LSAC_MAX_DELTA_OMEGA=${delta},LSAC_P_LOST=${plost},LSAC_SAVE_DIR=${save_dir} \
        train_sac_residual.sbatch
done
