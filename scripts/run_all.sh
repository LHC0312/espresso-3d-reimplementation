#!/bin/bash
# Full recipe used for the released weights (one RTX PRO 6000 96GB, ~4 h):
#   1) Espresso-3D pretraining on Affogato-750K (PRE_MIN wall-clock minutes; 185 min ≈ 3.2 epochs)
#   2) LASO finetuning, Seen and Unseen run concurrently (<= 50 epochs, FT_MIN minutes each)
#   3) LASO test evaluation (aIoU / AUC / SIM / MAE) of the final and best-val checkpoints
# Every stage resumes from its last checkpoint if re-run.
set -uo pipefail
cd "$(dirname "$0")/.."
DATA_DIR=${DATA_DIR:-data}
RUNS=${RUNS:-runs}
PRE_MIN=${PRE_MIN:-185}
FT_MIN=${FT_MIN:-45}
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ ! -f $RUNS/pretrain_affogato/ckpt/final.pt ]; then
  python train.py --dataset affogato --data_root $DATA_DIR/affogato --out $RUNS/pretrain_affogato \
    --epochs 50 --time_budget_min $PRE_MIN --bs 80 --lr 1e-4 --text_lr 5e-6 --warmup 500 --workers 12 \
    --val_every 0.25 --backup_every 0.25 --keep_backups 2 --log_every 25 --resume
fi

for S in seen unseen; do
  [ -f $RUNS/ft_laso_$S/ckpt/final.pt ] && continue
  python train.py --dataset laso --setting $S --data_root $DATA_DIR/LASO_dataset \
    --init $RUNS/pretrain_affogato/ckpt/final.pt --out $RUNS/ft_laso_$S --epochs 50 \
    --time_budget_min $FT_MIN --bs 64 --lr 1e-4 --text_lr 5e-6 --warmup 200 --workers 4 \
    --val_every 1 --backup_every 5 --keep_backups 2 --log_every 25 --cache_bs 32 --resume \
    > $RUNS/ft_laso_$S.log 2>&1 &
done
wait

for S in seen unseen; do
  for C in final best; do
    python eval_laso.py --ckpt $RUNS/ft_laso_$S/ckpt/$C.pt --data_root $DATA_DIR/LASO_dataset \
      --setting $S --out $RUNS/eval/${S}_$C --bs 32
  done
done
