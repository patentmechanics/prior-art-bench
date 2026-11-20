#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
"""Fixed fallback evaluation collection (1,000,000 US families), defined before any test-half number is seen.

  PAB_WORK=$WORK python build/fallback_collection_us.py [--size 1000000] [--salt fallback-1m-us-v1]

Purpose (pre-registered evaluation plan): if the budget does not allow encoding
the full corpus (9.44M families) with the final models, the test half is scored on this collection instead. It is
fixed now so that it cannot be chosen after seeing results.

Definition (deterministic, no random-number library involved):
  full corpus  = every family of the US-only corpus (tac shards, one representative document per family)
  250K pool    = the development pool (eval_us/tacd), already containing every family relevant to an evaluation query
  collection   = 250K pool  +  the first (size - |pool|) other families of the full corpus ordered by md5(family_id || salt)
So 250K pool ⊂ 1M collection ⊂ full corpus (nested). Relevant families added later by labels v2 (WO citations resolved
to US families) are added to every collection by rule at scoring time, and are not part of this list's selection.
Output: $PAB_WORK/collections/fallback_1m/{families.parquet (family_id, origin = pool | sampled), manifest.json}
"""
import argparse, glob, hashlib, json, os, time

import duckdb

W = os.environ.get("PAB_WORK", "work")
ap = argparse.ArgumentParser()
ap.add_argument("--size", type=int, default=1_000_000)
ap.add_argument("--salt", default="fallback-1m-us-v1")
ap.add_argument("--out", default=f"{W}/collections/fallback_1m")
a = ap.parse_args()
t0 = time.time()
os.makedirs(a.out, exist_ok=True)
c = duckdb.connect()
c.execute(f"SET threads=4; SET memory_limit='6GB'; SET temp_directory='{W}/tmp'; SET preserve_insertion_order=false")
c.execute(f"CREATE TABLE corpus AS SELECT DISTINCT family_id FROM '{W}/tac/shard-*.parquet'")
c.execute(f"CREATE TABLE pool AS SELECT DISTINCT family_id FROM '{W}/eval_us/tacd/part-*.parquet'")
n_corpus, n_pool = c.execute("SELECT count(*) FROM corpus").fetchone()[0], c.execute("SELECT count(*) FROM pool").fetchone()[0]
outside = c.execute("SELECT count(*) FROM pool ANTI JOIN corpus USING (family_id)").fetchone()[0]
assert outside == 0, f"{outside} pool families are not in the full corpus"
need = a.size - n_pool
c.execute(f"""CREATE TABLE sampled AS
    SELECT family_id FROM corpus ANTI JOIN pool USING (family_id)
    ORDER BY md5(family_id::VARCHAR || '{a.salt}'), family_id LIMIT {need}""")
c.execute(f"""COPY (SELECT family_id, 'pool' AS origin FROM pool UNION ALL SELECT family_id, 'sampled' FROM sampled ORDER BY family_id)
              TO '{a.out}/families.parquet' (FORMAT parquet, COMPRESSION zstd)""")
n_out, n_dist = c.execute(f"SELECT count(*), count(DISTINCT family_id) FROM '{a.out}/families.parquet'").fetchone()
assert n_out == n_dist == a.size, (n_out, n_dist)
# content hash independent of parquet encoding: md5 over the sorted family ids
h = hashlib.md5()
for (f,) in c.execute(f"SELECT family_id FROM '{a.out}/families.parquet' ORDER BY family_id").fetchall():
    h.update(f"{f}\n".encode())
known = f"{W}/eval_us/labels/known_relevant.parquet"
rel_missing = c.execute(f"SELECT count(DISTINCT family_id) FROM '{known}' ANTI JOIN '{a.out}/families.parquet' USING (family_id)").fetchone()[0]
m = {"definition": "250K pool + first (size - pool) other full-corpus families by md5(family_id || salt)", "salt": a.salt,
     "size": a.size, "full_corpus_families": n_corpus, "pool_families": n_pool, "sampled_families": need,
     "family_ids_md5_sorted_newline": h.hexdigest(), "file_md5": hashlib.md5(open(f"{a.out}/families.parquet", "rb").read()).hexdigest(),
     "known_relevant_families_missing": rel_missing, "created": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "seconds": round(time.time() - t0)}
json.dump(m, open(f"{a.out}/manifest.json", "w"), indent=1)
print(json.dumps(m))
