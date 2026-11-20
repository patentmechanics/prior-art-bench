#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
#
# The Anticipate-0.6B recipe: three seeds of full fine-tuning from harrier-oss-v1-0.6b, stopped at step 75 of a 900-step cosine
# schedule, then the uniform average of the three step-75 checkpoints, exported in the base model's layout.
#
#   bash train/run_anticipate_0.6b.sh PACK OUT [NPROC]
#
# PACK = a training pack (train/pack.py output: queries.parquet + docs.parquet). Ours is not shipped (its mined negatives
# are not part of the release); build/README.md and train/README.md describe how to make one. OUT = output directory.
# Settings are the ones of the release runs: global batch 64 queries (split over the GPUs), 3 hard negatives per query,
# query 4,096 / document 8,192 tokens, GPU mini-batches of up to 16,384 padded tokens (GradCache), SDPA attention,
# temperature 0.02, AdamW lr 1e-5 (weight decay 0.01, clip 1.0), 50 warm-up steps, cosine over 900 steps, stop at 75,
# positives drawn with grade weights 3=1.0, 2=0.7.
set -euo pipefail
PACK=${1:?pack} OUT=${2:?out} NPROC=${3:-1}
cd "$(dirname "$0")/.."
BASE_ID=microsoft/harrier-oss-v1-0.6b BASE_REV=f9b9dc8d367d443f2479d27aa5d8d2850c0774ee
BASE=$OUT/base
[ -f $BASE/config.json ] || python -c "from huggingface_hub import snapshot_download as s; s('$BASE_ID', revision='$BASE_REV', local_dir='$BASE')"
LAUNCH=python; [ "$NPROC" -gt 1 ] && LAUNCH="torchrun --standalone --nproc_per_node $NPROC"
for SEED in 0 1 2; do
  $LAUNCH train/train.py --pack $PACK --model $BASE --out $OUT/seed$SEED --batch 64 --hard-negs 3 \
      --q-len 4096 --d-len 8192 --chunk-tokens 16384 --attn sdpa --temp 0.02 --seed $SEED \
      --lr 1e-5 --warmup 50 --steps 900 --stop-at 75 --save-every 25 --log-every 10
done
python train/soup.py --ckpts $OUT/seed0/step-75 $OUT/seed1/step-75 $OUT/seed2/step-75 --out $OUT/soup
python train/export.py --base $BASE --trained $OUT/soup --out $OUT/export
python train/make_model_repo.py --export $OUT/export --base $BASE --out $OUT/model > $OUT/model_manifest.json
echo "model: $OUT/model  (evaluate: python evaluate.py encode --model $OUT/model --config configs/anticipate-0.6b.json --out emb/)"
