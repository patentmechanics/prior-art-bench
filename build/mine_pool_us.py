#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
"""US-only inputs for hard-negative mining (after build/corpus.py tac and build/tacd.py). Local files only, no EP data.

  PAB_WORK=$WORK python build/mine_pool_us.py ids   --data $WORK --work $WORK/mine [--n-queries 200000]
  PAB_WORK=$WORK python build/mine_pool_us.py parts --data $WORK --work $WORK/mine \
        --tacd $WORK/tacd/train-us.parquet [--per-part 20000]

ids:   the queries to mine for = the first --n-queries US office-action training queries in pack.py's md5 order
       (same selection rule as the training pack), and `need.parquet` = every positive family of those queries
       (the mining pool, and the only documents that need a description text).
parts: pool_tac/part-*.parquet (family_id, tac_text; from tac/shard-*.parquet) and pool_tacd/part-*.parquet
       (family_id, tacd_text; from the tacd train file), `pool_stats.json`. Negatives must come from this pool so that every
       training variant has the text.
Then: train/retrieve.py encode + search (encode the pool, search the queries) -> train/mine.py -> train/pack.py --negatives.
"""
import argparse, glob, json, os, subprocess, sys, time

import duckdb
import pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq


def ids(a):
    code = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(a.work, exist_ok=True)
    view = f"{a.work}/ids_view"
    os.makedirs(view, exist_ok=True)
    for d in ("queries", "labels"):
        if not os.path.islink(f"{view}/{d}"):
            os.symlink(f"{a.data}/{d}", f"{view}/{d}")
    out = f"{a.work}/ids_us"
    cmd = [sys.executable, f"{code}/train/pack.py", "--data", view, "--out", out, "--sources", "us_oa=1", "--ids-only",
           "--threads", "6", "--memory", "10GB", "--source-caps", f"us_oa={a.n_queries}"]
    print(" ".join(cmd), flush=True)
    print(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip(), flush=True)
    c = duckdb.connect()
    c.execute(f"COPY (SELECT DISTINCT query_id, source FROM '{out}/queries.parquet' ORDER BY 1) TO '{a.work}/queries.parquet' (FORMAT parquet)")
    n_q = c.execute(f"SELECT count(*) FROM '{a.work}/queries.parquet'").fetchone()[0]
    n_d = c.execute(f"SELECT count(*) FROM '{out}/need.parquet'").fetchone()[0]
    print(json.dumps({"queries": n_q, "pool_families": n_d}), flush=True)


def write_parts(tables, col, out, per):
    os.makedirs(out, exist_ok=True)
    schema = pa.schema([("family_id", pa.int64()), (col, pa.string())])
    w, n, part, total = None, 0, 0, 0
    for t in tables:
        for b in t.to_batches(max_chunksize=2000):
            if w is None or n >= per:
                if w:
                    w.close()
                w = pq.ParquetWriter(f"{out}/part-{part:03d}.parquet", schema, compression="zstd")
                part += 1
                n = 0
            w.write_table(pa.Table.from_batches([b]).select(["family_id", col]).cast(schema))
            n += b.num_rows
            total += b.num_rows
    if w:
        w.close()
    return total, part


def parts(a):
    t0 = time.time()
    need = pq.read_table(f"{a.work}/ids_us/need.parquet").column("family_id")

    def from_files(files, col):
        for f in files:
            pf = pq.ParquetFile(f)
            for i in range(pf.num_row_groups):
                t = pf.read_row_group(i, columns=["family_id", col])
                t = t.filter(pc.is_in(t.column("family_id"), value_set=need))
                if t.num_rows:
                    yield t

    n_tacd, p_tacd = write_parts(from_files([a.tacd], "tacd_text"), "tacd_text", f"{a.work}/pool_tacd", a.per_part)
    n_tac, p_tac = write_parts(from_files(sorted(glob.glob(f"{a.data}/tac/shard-*.parquet")), "tac_text"), "tac_text",
                               f"{a.work}/pool_tac", a.per_part)
    c = duckdb.connect()
    dup = {n: c.execute(f"SELECT count(*) - count(DISTINCT family_id) FROM '{a.work}/{n}/*.parquet'").fetchone()[0]
           for n in ("pool_tac", "pool_tacd")}
    stats = {"pool_families": len(need), "pool_tac_docs": n_tac, "pool_tac_parts": p_tac, "pool_tacd_docs": n_tacd,
             "pool_tacd_parts": p_tacd, "duplicate_rows": dup, "seconds": round(time.time() - t0)}
    json.dump(stats, open(f"{a.work}/pool_stats.json", "w"), indent=1)
    print(json.dumps(stats), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["ids", "parts"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--n-queries", type=int, default=200000)
    ap.add_argument("--tacd", default=None)
    ap.add_argument("--per-part", type=int, default=20000)
    a = ap.parse_args()
    if a.stage == "parts" and not a.tacd:
        ap.error("parts needs --tacd")
    {"ids": ids, "parts": parts}[a.stage](a)


if __name__ == "__main__":
    main()
