#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
"""Slim US-only evaluation data set (local files only, nothing from EPO or DOCDB).

  PAB_WORK=$WORK python build/eval_pool_us.py [--out $WORK/eval_us] [--per-file 2500]

Layout written under --out:
  queries/queries.parquet      the evaluation queries only (splits gold, eval_ptab), same schema as queries/queries.parquet
  labels/{splits,labels,known_relevant,devtest}.parquet   rows of those queries only (devtest = dev/test halves, salt devtest-us-v1)
  family_map/families.parquet  family_id, earliest_pd, in_scope (= true: every family is a US family)
  tacd/part-NNN.parquet        the 250K-family evaluation pool, title+abstract+claims+description, 2,500 documents per file
  tac/part-NNN.parquet         the same pool, title+abstract+claims only
Small files let evaluate.py encode shard over many machines (files sorted, shard i takes files i::N).
"""
import argparse, glob, json, os, time

import duckdb
import pyarrow as pa, pyarrow.parquet as pq

W = os.environ.get("PAB_WORK", "work")
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=f"{W}/eval_us")
ap.add_argument("--per-file", type=int, default=2500)
a = ap.parse_args()
t0 = time.time()
for d in ("queries", "labels", "family_map", "tacd", "tac"):
    os.makedirs(f"{a.out}/{d}", exist_ok=True)
c = duckdb.connect()
c.execute("SET threads=3; SET memory_limit='12GB'; SET preserve_insertion_order=false")   # the 7.7 GB queries file has huge text row groups
c.execute(f"CREATE TABLE ev AS SELECT query_id FROM '{W}/labels/splits.parquet' WHERE split IN ('gold', 'eval_ptab')")
c.execute(f"COPY (SELECT q.* FROM '{W}/queries/queries.parquet' q JOIN ev USING (query_id)) TO '{a.out}/queries/queries.parquet' (FORMAT parquet, COMPRESSION zstd)")
for f in ("splits", "labels", "known_relevant"):
    c.execute(f"COPY (SELECT x.* FROM '{W}/labels/{f}.parquet' x JOIN ev USING (query_id)) TO '{a.out}/labels/{f}.parquet' (FORMAT parquet, COMPRESSION zstd)")
c.execute(f"COPY (SELECT x.* FROM '{W}/labels/devtest.parquet' x JOIN ev USING (query_id)) TO '{a.out}/labels/devtest.parquet' (FORMAT parquet)")
c.execute(f"""COPY (SELECT family_id, earliest_pd, true AS in_scope FROM '{W}/family_map/families.parquet')
              TO '{a.out}/family_map/families.parquet' (FORMAT parquet, COMPRESSION zstd)""")


def cut(files, col, out):
    w, n, part, total = None, 0, 0, 0
    schema = pa.schema([("family_id", pa.int64()), (col, pa.string())])
    for f in files:
        for b in pq.ParquetFile(f).iter_batches(batch_size=500, columns=["family_id", col]):
            t = pa.Table.from_batches([b]).cast(schema)
            i = 0
            while i < t.num_rows:
                if w is None or n >= a.per_file:
                    if w:
                        w.close()
                    w = pq.ParquetWriter(f"{out}/part-{part:03d}.parquet", schema, compression="zstd")
                    part += 1
                    n = 0
                k = min(a.per_file - n, t.num_rows - i)
                w.write_table(t.slice(i, k))
                n += k; i += k; total += k
    if w:
        w.close()
    return total, part


n_d, p_d = cut(sorted(glob.glob(f"{W}/tacd/subset-us-*.parquet")), "tacd_text", f"{a.out}/tacd")
n_t, p_t = cut(sorted(glob.glob(f"{W}/tac_subset_250k/shard-*.parquet")), "tac_text", f"{a.out}/tac")
fam_d = c.execute(f"SELECT count(DISTINCT family_id) FROM '{a.out}/tacd/*.parquet'").fetchone()[0]
fam_t = c.execute(f"SELECT count(DISTINCT family_id) FROM '{a.out}/tac/*.parquet'").fetchone()[0]
rel_missing = c.execute(f"""SELECT count(DISTINCT k.family_id) FROM '{a.out}/labels/known_relevant.parquet' k
                            ANTI JOIN (SELECT DISTINCT family_id FROM '{a.out}/tacd/*.parquet') p USING (family_id)""").fetchone()[0]
stats = {"eval_queries": c.execute("SELECT count(*) FROM ev").fetchone()[0],
         "by_split_part": c.execute(f"SELECT source, part, count(*) FROM '{a.out}/labels/devtest.parquet' GROUP BY 1, 2 ORDER BY 1, 2").fetchall(),
         "tacd_docs": n_d, "tacd_files": p_d, "tacd_families": fam_d, "tac_docs": n_t, "tac_files": p_t, "tac_families": fam_t,
         "known_relevant_families_missing_from_pool": rel_missing, "seconds": round(time.time() - t0)}
json.dump(stats, open(f"{a.out}/eval_stats.json", "w"), indent=1)
print(json.dumps(stats))
