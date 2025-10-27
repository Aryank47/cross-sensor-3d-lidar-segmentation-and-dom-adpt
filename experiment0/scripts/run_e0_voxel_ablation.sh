#!/usr/bin/env bash
# ================================================
# E0 Voxel Size Ablation
# Usage: bash scripts/run_e0_voxel_ablation.sh <loss_name>
# Example: bash scripts/run_e0_voxel_ablation.sh focal_dice
# ================================================

if [ -z "$1" ]; then
  echo "Usage: $0 <loss_name>"
  echo "Example: $0 focal_dice"
  exit 1
fi

LOSS_NAME=$1
ECLAIR_DIR=${ECLAIR_DIR:-/data/eclair}
DALES_DIR=${DALES_DIR:-/data/dales}
OUTROOT=${OUTROOT:-runs/E0}
CFG=./configs/train_e0.yaml

echo "Running voxel ablation for loss: $LOSS_NAME"

for VOXEL in 0.10 0.15 0.20; do
  echo ""
  echo "========================================="
  echo "Training with voxel_size = ${VOXEL}m"
  echo "========================================="
  
  python train_baseline.py \
    --eclair_dir $ECLAIR_DIR \
    --dales_dir $DALES_DIR \
    --output_dir $OUTROOT/${LOSS_NAME}_v${VOXEL} \
    --config_file $CFG \
    --loss_name $LOSS_NAME \
    --voxel_size $VOXEL \
    --epochs 80 --eval_every 5 --amp True \
    $([ "$LOSS_NAME" == "focal" ] && echo "--focal_gamma 2.0") \
    $([ "$LOSS_NAME" == "dice" ] && echo "--dice_smooth 1.0") \
    $([ "$LOSS_NAME" == "focal_dice" ] && echo "--focal_gamma 2.0 --dice_smooth 1.0")
done

echo ""
echo "Voxel ablation complete. Results in: $OUTROOT/"