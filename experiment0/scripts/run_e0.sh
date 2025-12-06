#!/usr/bin/env bash

# Configuration
ECLAIR_DIR=${ECLAIR_DIR:-/data/eclair}
DALES_DIR=${DALES_DIR:-/data/dales}
OUTROOT=${OUTROOT:-runs/E0}
CFG=./configs/train_e0.yaml
EPOCHS=80
EVAL_EVERY=5

# GPU settings
#export CUDA_VISIBLE_DEVICES=0

# ================================================
# PHASE 1: Loss Ablation at Default Voxel Size (0.15m)
# ================================================
echo "==================================="
echo "PHASE 1: Loss Ablation (v=0.15)"
echo "==================================="

VOXEL=0.15

# E0a: Cross-Entropy
# python train_baseline.py \
#   --eclair_dir $ECLAIR_DIR \
#   --dales_dir $DALES_DIR \
#   --output_dir $OUTROOT/e0a_ce_v${VOXEL} \
#   --config_file $CFG \
#   --loss_name ce \
#   --voxel_size $VOXEL \
#   --epochs $EPOCHS --eval_every $EVAL_EVERY --amp True \
#   --num_workers 0

# # E0b: Focal Loss
# python train_baseline.py \
#   --eclair_dir $ECLAIR_DIR \
#   --dales_dir $DALES_DIR \
#   --output_dir $OUTROOT/e0b_focal_v${VOXEL} \
#   --config_file $CFG \
#   --loss_name focal --focal_gamma 2.0 \
#   --voxel_size $VOXEL \
#   --epochs $EPOCHS --eval_every $EVAL_EVERY --amp True \
#   --num_workers 0

# # E0c: DICE Loss
# python train_baseline.py \
#   --eclair_dir $ECLAIR_DIR \
#   --dales_dir $DALES_DIR \
#   --output_dir $OUTROOT/e0c_dice_v${VOXEL} \
#   --config_file $CFG \
#   --loss_name dice --dice_smooth 1.0 \
#   --voxel_size $VOXEL \
#   --epochs $EPOCHS --eval_every $EVAL_EVERY --amp True \
#   --num_workers 0

# # E0d: Focal + DICE
# python train_baseline.py \
#   --eclair_dir $ECLAIR_DIR \
#   --dales_dir $DALES_DIR \
#   --output_dir $OUTROOT/e0d_focal_dice_v${VOXEL} \
#   --config_file $CFG \
#   --loss_name focal_dice --focal_gamma 2.0 --dice_smooth 1.0 \
#   --voxel_size $VOXEL \
#   --epochs $EPOCHS --eval_every $EVAL_EVERY --amp True \
#   --num_workers 0

# E0e: Class-Balanced Loss (NEW)
python train_baseline.py \
  --eclair_dir $ECLAIR_DIR \
  --dales_dir $DALES_DIR \
  --output_dir $OUTROOT/e0e_cb_v${VOXEL} \
  --config_file $CFG \
  --loss_name class_balanced \
  --voxel_size $VOXEL \
  --epochs $EPOCHS --eval_every $EVAL_EVERY --amp True \
  --num_workers 0

# E0f: Lovász Loss (NEW)
# python train_baseline.py \
#   --eclair_dir $ECLAIR_DIR \
#   --dales_dir $DALES_DIR \
#   --output_dir $OUTROOT/e0f_lovasz_v${VOXEL} \
#   --config_file $CFG \
#   --loss_name lovasz \
#   --voxel_size $VOXEL \
#   --epochs $EPOCHS --eval_every $EVAL_EVERY --amp True \
#   --num_workers 0

# # E0g: DICE + Lovász (NEW)
# python train_baseline.py \
#   --eclair_dir $ECLAIR_DIR \
#   --dales_dir $DALES_DIR \
#   --output_dir $OUTROOT/e0g_dice_lovasz_v${VOXEL} \
#   --config_file $CFG \
#   --loss_name dice_lovasz --dice_smooth 1.0 \
#   --voxel_size $VOXEL \
#   --epochs $EPOCHS --eval_every $EVAL_EVERY --amp True \
#   --num_workers 0

# ================================================
# PHASE 2: Voxel Size Ablation with Best Loss
# ================================================
echo "==================================="
echo "PHASE 2: Voxel Ablation"
echo "==================================="
echo "Run this AFTER identifying best loss from Phase 1"
echo "Example: If focal_dice won, run:"
echo ""
echo "for VOXEL in 0.10 0.20; do"
echo "  python train_baseline.py \\"
echo "    --loss_name focal_dice \\"
echo "    --voxel_size \$VOXEL \\"
echo "    --output_dir $OUTROOT/e0_best_v\${VOXEL} \\"
echo "    ..."
echo "done"
echo "==================================="

# To run voxel ablation manually:
# bash scripts/run_e0_voxel_ablation.sh <best_loss_name>
