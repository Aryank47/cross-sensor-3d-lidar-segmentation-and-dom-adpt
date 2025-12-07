# #!/usr/bin/env bash

# # Configuration
# ECLAIR_DIR=${ECLAIR_DIR:-/data/eclair}
# DALES_DIR=${DALES_DIR:-/data/dales}
# OUTROOT=${OUTROOT:-runs/E0}
# CFG=./configs/train_e0.yaml
# EPOCHS=2
# EVAL_EVERY=1

# # GPU settings
# #export CUDA_VISIBLE_DEVICES=0

# # ================================================
# # PHASE 1: Loss Ablation at Default Voxel Size (0.15m)
# # ================================================
# echo "==================================="
# echo "PHASE 1: Loss Ablation (v=0.15)"
# echo "==================================="

# VOXEL=0.15

# # E0a: Cross-Entropy
# python train_baseline.py \
#   --eclair_dir $ECLAIR_DIR \
#   --dales_dir $DALES_DIR \
#   --output_dir $OUTROOT/e0a_ce_v${VOXEL} \
#   --config_file $CFG \
#   --use_cache True \
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
#   --use_cache True \
#   --loss_name focal \
#   --focal_gamma 2.0 \
#   --focal_use_cb_alpha True \
#   --cb_beta 0.9999 \
#   --voxel_size $VOXEL \
#   --epochs $EPOCHS --eval_every $EVAL_EVERY --amp True \
#   --num_workers 0

# # E0c: DICE Loss
# python train_baseline.py \
#   --eclair_dir $ECLAIR_DIR \
#   --dales_dir $DALES_DIR \
#   --output_dir $OUTROOT/e0c_dice_v${VOXEL} \
#   --config_file $CFG \
#   --use_cache True \
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
#   --use_cache True \
#   --loss_name focal_dice --focal_gamma 2.0 --focal_use_cb_alpha True \
#   --cb_beta 0.9999 \
#   --dice_smooth 1.0 \
#   --voxel_size $VOXEL \
#   --epochs $EPOCHS --eval_every $EVAL_EVERY --amp True \
#   --num_workers 0

# # E0e: Class-Balanced Loss (NEW)
# python train_baseline.py \
#   --eclair_dir $ECLAIR_DIR \
#   --dales_dir $DALES_DIR \
#   --output_dir $OUTROOT/e0e_cb_v${VOXEL} \
#   --config_file $CFG \
#   --use_cache True \
#   --loss_name class_balanced \
#   --voxel_size $VOXEL \
#   --epochs $EPOCHS --eval_every $EVAL_EVERY --amp True \
#   --num_workers 0

# # E0f: Lovász Loss (NEW)
# python train_baseline.py \
#   --eclair_dir $ECLAIR_DIR \
#   --dales_dir $DALES_DIR \
#   --output_dir $OUTROOT/e0f_lovasz_v${VOXEL} \
#   --config_file $CFG \
#   --use_cache True \
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
#   --use_cache True \
#   --loss_name dice_lovasz --dice_smooth 1.0 \
#   --voxel_size $VOXEL \
#   --epochs $EPOCHS --eval_every $EVAL_EVERY --amp True \
#   --num_workers 0

# # ================================================
# # PHASE 2: Voxel Size Ablation with Best Loss
# # ================================================
# echo "==================================="
# echo "PHASE 2: Voxel Ablation"
# echo "==================================="
# echo "Run this AFTER identifying best loss from Phase 1"
# echo "Example: If focal_dice won, run:"
# echo ""
# echo "for VOXEL in 0.10 0.20; do"
# echo "  python train_baseline.py \\"
# echo "    --loss_name focal_dice \\"
# echo "    --voxel_size \$VOXEL \\"
# echo "    --output_dir $OUTROOT/e0_best_v\${VOXEL} \\"
# echo "    ..."
# echo "done"
# echo "==================================="

# # To run voxel ablation manually:
# # bash scripts/run_e0_voxel_ablation.sh <best_loss_name>


#!/usr/bin/env bash

# Configuration
ECLAIR_DIR=${ECLAIR_DIR:-/data/eclair}
DALES_DIR=${DALES_DIR:-/data/dales}
OUTROOT=${OUTROOT:-runs/E0}
CFG=./configs/train_e0.yaml
EPOCHS=2
EVAL_EVERY=1

VOXEL=0.15

# Auto-detect number of GPUs (fallback to 1)
NUM_GPUS=${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-1}}
echo "[run_e0] Using ${NUM_GPUS} GPUs for DDP"


run_and_eval() {
  local LOSS_NAME="$1"      # e.g. "ce"
  local OUTDIR="$2"         # e.g. "$OUTROOT/e0a_ce_v${VOXEL}"
  shift 2                   # remaining args are extra flags to train_baseline

  echo "==================================="
  echo "TRAIN: ${OUTDIR} (loss=${LOSS_NAME})"
  echo "==================================="

  CACHE_ROOT="/scratch/m23csa510/eclair_cache"

  torchrun --standalone --nproc_per_node="${NUM_GPUS}" train_baseline.py \
    --eclair_dir "${ECLAIR_DIR}" \
    --dales_dir "${DALES_DIR}" \
    --output_dir "${OUTDIR}" \
    --config_file "${CFG}" \
    --use_cache True \
    --cache_root "${CACHE_ROOT}" \
    --loss_name "${LOSS_NAME}" \
    --voxel_size "${VOXEL}" \
    --epochs "${EPOCHS}" \
    --eval_every "${EVAL_EVERY}" \
    --amp True \
    --num_workers 0 \
    "$@"

  echo "Done training: ${OUTDIR}"

  # Now run cross-domain eval on the best checkpoint for this run
  local CKPT="${OUTDIR}/checkpoints/best.pth"
  if [ -f "${CKPT}" ]; then
    echo "Running eval_cross_domain.py on ${CKPT}"
    python eval_cross_domain.py \
      --eclair_dir "${ECLAIR_DIR}" \
      --dales_dir "${DALES_DIR}" \
      --checkpoint "${CKPT}" \
      --config_file "${CFG}" \
      > "${OUTDIR}/eval_cross_domain.log" 2>&1
    echo "Saved cross-domain eval to ${OUTDIR}/eval_cross_domain.log"
  else
    echo "WARNING: best checkpoint not found at ${CKPT} – skipping eval_cross_domain"
  fi
}

# ================================================
# PHASE 1: Loss Ablation at Default Voxel Size
# ================================================
echo "==================================="
echo "PHASE 1: Loss Ablation (v=${VOXEL})"
echo "==================================="

# E0a: Cross-Entropy
run_and_eval "ce" "${OUTROOT}/e0a_ce_v${VOXEL}"

# E0b: Focal Loss
run_and_eval "focal" "${OUTROOT}/e0b_focal_v${VOXEL}" \
  --focal_gamma 2.0 \
  --focal_use_cb_alpha True \
  --cb_beta 0.9999

# E0c: DICE Loss
run_and_eval "dice" "${OUTROOT}/e0c_dice_v${VOXEL}" \
  --dice_smooth 1.0

# E0d: Focal + DICE
run_and_eval "focal_dice" "${OUTROOT}/e0d_focal_dice_v${VOXEL}" \
  --focal_gamma 2.0 \
  --focal_use_cb_alpha True \
  --cb_beta 0.9999 \
  --dice_smooth 1.0

# E0e: Class-Balanced Loss
run_and_eval "class_balanced" "${OUTROOT}/e0e_cb_v${VOXEL}"

# E0f: Lovász Loss
run_and_eval "lovasz" "${OUTROOT}/e0f_lovasz_v${VOXEL}"

# E0g: DICE + Lovász
run_and_eval "dice_lovasz" "${OUTROOT}/e0g_dice_lovasz_v${VOXEL}" \
  --dice_smooth 1.0

# ================================================
# PHASE 2: Voxel Size Ablation with Best Loss
# ================================================
echo "==================================="
echo "PHASE 2: Voxel Ablation"
echo "==================================="
echo "Run this AFTER identifying best loss from Phase 1."
echo "Example: If focal_dice won, run:"
echo
echo "for VOXEL in 0.10 0.20; do"
echo "  python train_baseline.py \\"
echo "    --loss_name focal_dice \\"
echo "    --voxel_size \$VOXEL \\"
echo "    --output_dir $OUTROOT/e0_best_v\${VOXEL} \\"
echo "    ... (and then call eval_cross_domain.py with the new checkpoint)"
echo "done"
echo "==================================="
