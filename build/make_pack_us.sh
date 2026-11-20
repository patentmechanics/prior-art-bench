#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
# Training pack variants for the US-only ablations; the release pack is TAG ha (see train/README.md).
# Same queries (mine/queries_100k), filters (known + co-cited cocited_w1 excluded, 0.95 x best-positive cap) and pack
# arguments as the base pack (pack_us_tacd); only the named setting differs.
#   build/make_pack_us.sh TAG RUN LO HI [tacd [PACK_FLAGS]]   negatives from RUN, ranks LO < r <= HI, documents with descriptions
#   build/make_pack_us.sh TAG - - - tac            base negatives (mine/negatives.parquet), documents without descriptions
# Output: $PAB_WORK/packs/pack_us_<TAG>/{docs,queries}.parquet + meta.json, and negatives in mine/neg_<TAG>.parquet
set -euo pipefail
TAG=$1 RUN=$2 LO=$3 HI=$4 TEXT=${5:-tacd} PACK_EXTRA=${6:-}   # 6th: extra train/pack.py flags (e.g. --y-indep-only)
W=${PAB_WORK:?set PAB_WORK} PY=${PY:-python}
cd "$(dirname "$0")/.."
OUT=$W/packs/pack_us_$TAG
mkdir -p $W/packs
if [ "$RUN" = - ]; then
  NEG=$W/mine/negatives.parquet
else
  NEG=$W/mine/neg_$TAG.parquet
  $PY train/mine.py --run "$RUN" --data $W --out $NEG --lo "$LO" --hi "$HI" --n 32 --pos-cap 0.95 \
      --cocited $W/labels/cocited_w1.parquet --memory 10GB
fi
if [ "$TEXT" = tacd ]; then DOCS=$W/tacd/train-us.parquet; else DOCS="$W/tac/shard-*.parquet"; fi
rm -rf $OUT
$PY train/pack.py --data $W --out $OUT --sources us_oa=1 --negatives $NEG --max-negs 32 --query-field tac_text --split train \
    --min-grade 2 --threads 6 --docs-glob "$DOCS" --text-col ${TEXT}_text --memory 10GB --query-ids $W/mine/queries_100k.parquet $PACK_EXTRA
$PY - <<EOF
import json, duckdb
m = json.load(open("$OUT/meta.json"))
n = duckdb.connect().execute("SELECT count(*), avg(len(neg_ids)) FROM '$NEG'").fetchall()[0]
print(json.dumps({"tag": "$TAG", "negatives_queries": n[0], "avg_negs": round(n[1], 1), "docs": m["docs"], "queries": m["queries_by_source"]}))
EOF
