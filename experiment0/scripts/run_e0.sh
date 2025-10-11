# ================================================
# FILE: scripts/run_e0.sh
# ================================================
#!/usr/bin/env bash
# Example usage for each loss variant (adjust paths)
ECLAIR_DIR=/data/eclair
DALES_DIR=/data/dales
OUTROOT=runs/E0
CFG=./configs/train_e0.yaml

# CE
python train_baseline.py \
  --eclair_dir $ECLAIR_DIR \
  --dales_dir $DALES_DIR \
  --output_dir $OUTROOT/ce \
  --config_file $CFG \
  --loss_name ce \
  --epochs 80 --eval_every 5 --amp True

# Focal
python train_baseline.py \
  --eclair_dir $ECLAIR_DIR \
  --dales_dir $DALES_DIR \
  --output_dir $OUTROOT/focal \
  --config_file $CFG \
  --loss_name focal --focal_gamma 2.0 \
  --epochs 80 --eval_every 5 --amp True

# DICE
python train_baseline.py \
  --eclair_dir $ECLAIR_DIR \
  --dales_dir $DALES_DIR \
  --output_dir $OUTROOT/dice \
  --config_file $CFG \
  --loss_name dice --dice_smooth 1.0 \
  --epochs 80 --eval_every 5 --amp True

# Focal + DICE
python train_baseline.py \
  --eclair_dir $ECLAIR_DIR \
  --dales_dir $DALES_DIR \
  --output_dir $OUTROOT/focal_dice \
  --config_file $CFG \
  --loss_name focal_dice --focal_gamma 2.0 --dice_smooth 1.0 \
  --epochs 80 --eval_every 5 --amp True
