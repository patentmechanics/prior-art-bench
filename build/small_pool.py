#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
"""Small fixed evaluation collection (50,000 families) for learning-rate / checkpoint checks; local files only.

  PAB_WORK=$WORK python build/small_pool.py [--size 50000] [--salt small50k-us-v1]

Collection = every family relevant (labels v2, keep) to a dev-half evaluation query + the first (size - relevant) other
families of pool v2 ordered by md5(family_id || salt). Nested: small ⊂ pool v2 ⊂ full corpus. Used only to choose a
checkpoint (dev half); published numbers use pool v2 / the full corpus. Same layout as eval_us_v2 (labels, queries and
family map are hard links), 2,500 documents per file.
Output: $PAB_WORK/eval_us_v2_small, small_pool_manifest.json
"""
import argparse, glob, hashlib, json, os, shutil, time

import duckdb
import pyarrow as pa, pyarrow.parquet as pq

W = os.environ.get("PAB_WORK", "work")
ap = argparse.ArgumentParser()
ap.add_argument("--src", default=f"{W}/eval_us_v2")
ap.add_argument("--out", default=f"{W}/eval_us_v2_small")
ap.add_argument("--size", type=int, default=50000)
ap.add_argument("--salt", default="small50k-us-v1")
ap.add_argument("--per-file", type=int, default=2500)
a = ap.parse_args()
t0 = time.time()
if os.path.exists(a.out):
    shutil.rmtree(a.out)
for d in ("queries", "labels", "family_map", "tacd", "tac"):
    os.makedirs(f"{a.out}/{d}")
for d in ("queries", "labels", "family_map"):
    for f in glob.glob(f"{a.src}/{d}/*.parquet"):
        os.link(f, f"{a.out}/{d}/{os.path.basename(f)}")
c = duckdb.connect()
c.execute("SET threads=4; SET memory_limit='8GB'")
c.execute(f"CREATE TABLE pool AS SELECT DISTINCT family_id FROM '{a.src}/tacd/part-*.parquet'")
c.execute(f"""CREATE TABLE rel AS SELECT DISTINCT l.target_family_id AS family_id FROM '{a.src}/labels/labels.parquet' l
              JOIN '{a.src}/labels/devtest.parquet' d USING (query_id) WHERE l.keep AND d.part = 'dev'""")
n_rel = c.execute("SELECT count(*) FROM rel").fetchone()[0]
missing = c.execute("SELECT count(*) FROM rel ANTI JOIN pool USING (family_id)").fetchone()[0]
c.execute(f"""CREATE TABLE keepf AS SELECT family_id FROM rel
              UNION ALL SELECT family_id FROM (SELECT family_id FROM pool ANTI JOIN rel USING (family_id)
                                               ORDER BY md5(family_id::VARCHAR || '{a.salt}'), family_id LIMIT {a.size - n_rel})""")
n_keep = c.execute("SELECT count(*) FROM keepf").fetchone()[0]
for col, d in (("tacd_text", "tacd"), ("tac_text", "tac")):
    t = c.execute(f"""SELECT t.family_id, t.{col} FROM '{a.src}/{d}/part-*.parquet' t SEMI JOIN keepf USING (family_id)
                      ORDER BY t.family_id""").to_arrow_table()
    for i in range(0, t.num_rows, a.per_file):
        pq.write_table(t.slice(i, a.per_file), f"{a.out}/{d}/part-{i // a.per_file:03d}.parquet", compression="zstd")
h = hashlib.md5("".join(f"{f}\n" for (f,) in c.execute("SELECT family_id FROM keepf ORDER BY family_id").fetchall()).encode()).hexdigest()
m = {"definition": "dev-half relevant families (labels v2) + pool v2 families by md5(family_id || salt)", "salt": a.salt, "size": n_keep,
     "dev_relevant_families": n_rel, "dev_relevant_missing_from_pool": missing, "family_ids_md5_sorted_newline": h,
     "tacd_files": len(glob.glob(f"{a.out}/tacd/*.parquet")), "tac_files": len(glob.glob(f"{a.out}/tac/*.parquet")),
     "seconds": round(time.time() - t0)}
json.dump(m, open(f"{a.out}/small_pool_manifest.json", "w"), indent=1)
print(json.dumps(m))
