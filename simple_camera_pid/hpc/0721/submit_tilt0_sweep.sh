#!/bin/bash
# ============================================================================
#  submit_tilt0_sweep.sh — 9-run OFAT sweep for the genpath_roi30_tilt0
#  ablation: cam_pitch_deg=0 (tilt cancelled), roi_fraction=0.3, gen-track
#  random-shape procedural path all held fixed; one baseline run plus one
#  variable perturbed at a time (base-speed-max, ent-init,
#  gen-track-difficulty, seed).
#
#  Gated on the tilt0/roi30 smoke check (job 10987005, smoke_tilt0_roi30.sbatch)
#  via --dependency=afterok, same convention as smoke_residual.sbatch ->
#  train_sac_residual.sbatch: these 9 jobs sit PENDING and only actually run
#  if that smoke job exits 0. If it fails, cancel these before re-submitting
#  (scancel) rather than letting them burn GPU time on a broken config.
#
#  Usage:  bash submit_tilt0_sweep.sh          (from this directory)
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

SMOKE_JOB=10987005
BASE_DIR="simulation_data/sac_residual_training/genpath_roi30_tilt0"

# name : LSAC_BASE_SPEED_MAX : LSAC_ENT_INIT : LSAC_GENTRACK_DIFFICULTY : LSAC_SEED
runs=(
  "baseline    5.6    0.2   0.4  0"
  "bsmax7.33   7.3333 0.2   0.4  0"
  "bsmax4.0    4.0    0.2   0.4  0"
  "ent0.05     5.6    0.05  0.4  0"
  "ent0.5      5.6    0.5   0.4  0"
  "diff0.2     5.6    0.2   0.2  0"
  "diff0.7     5.6    0.2   0.7  0"
  "seed1       5.6    0.2   0.4  1"
  "seed2       5.6    0.2   0.4  2"
)

for row in "${runs[@]}"; do
    read -r name bsmax ent diff seed <<< "$row"
    save_dir="${BASE_DIR}/${name}"
    echo "submitting ${name}: base-speed-max=${bsmax} ent-init=${ent} " \
         "gen-track-difficulty=${diff} seed=${seed} -> ${save_dir}"
    sbatch --dependency=afterok:${SMOKE_JOB} \
        --export=ALL,LSAC_GENTRACK=1,LSAC_GENTRACK_SHAPE=random,LSAC_ROI_FRACTION=0.3,LSAC_CAM_PITCH_DEG=0,LSAC_EPISODES=500,LSAC_STEPS=200,LSAC_BASE_SPEED_MAX=${bsmax},LSAC_ENT_INIT=${ent},LSAC_GENTRACK_DIFFICULTY=${diff},LSAC_SEED=${seed},LSAC_SAVE_DIR=${save_dir} \
        train_sac_residual.sbatch
done
