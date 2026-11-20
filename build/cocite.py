#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
"""Co-citation edges for hard-negative mining: documents examiners cited together (X/Y) on other patents.

  python build/cocite.py --labels DIR [--min-queries 2] [--top 50]

Two families are co-cited when the same query (an EP search report, a US office action, a PTAB or opposition case) cites
both with grade >= 2 (X or Y). Edge weight = number of distinct queries citing both. Kept: weight >= --min-queries, and
the --top strongest partners per family. train/mine.py adds the partners of a query's positives to its known set, so a
document that examiners elsewhere treat as equivalent to its citations never becomes its negative.
Only training-split queries are used (evaluation citations must not shape training).
Output: DIR/<out-name>.parquet  family_id, partner_id, n_queries
"""
import argparse, json, os, time

import duckdb

WORK = os.environ.get("PAB_WORK", "work")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=f"{WORK}/labels")
    ap.add_argument("--min-queries", type=int, default=2)
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--out-name", default="cocited")
    ap.add_argument("--max-cites", type=int, default=50, help="skip queries citing more X/Y families than this (noise, n^2 pairs)")
    ap.add_argument("--memory", default="48GB", help="DuckDB memory limit")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    t0 = time.time()
    c = duckdb.connect()
    c.execute(f"SET threads={a.threads}; SET memory_limit='{a.memory}'; SET temp_directory='{WORK}/tmp'; SET preserve_insertion_order=false")
    c.execute(f"""CREATE TABLE xy AS SELECT DISTINCT query_id, target_family_id AS f FROM '{a.labels}/labels.parquet'
                  WHERE grade >= 2 AND target_family_id IS NOT NULL
                  AND query_id IN (SELECT query_id FROM '{a.labels}/splits.parquet' WHERE split = 'train')""")   # never evaluation labels
    c.execute(f"""CREATE TABLE xy2 AS SELECT query_id, f FROM xy WHERE query_id IN
                  (SELECT query_id FROM xy GROUP BY 1 HAVING count(*) BETWEEN 2 AND {a.max_cites})""")
    c.execute(f"""CREATE TABLE edges AS SELECT x.f AS family_id, y.f AS partner_id, count(*) AS n_queries
                  FROM xy2 x JOIN xy2 y ON x.query_id = y.query_id AND x.f <> y.f GROUP BY 1, 2
                  HAVING count(*) >= {a.min_queries}""")
    c.execute(f"""COPY (SELECT family_id, partner_id, n_queries FROM
                    (SELECT *, row_number() OVER (PARTITION BY family_id ORDER BY n_queries DESC, partner_id) AS r FROM edges)
                  WHERE r <= {a.top}) TO '{a.labels}/{a.out_name}.parquet' (FORMAT parquet, COMPRESSION zstd)""")
    st = c.execute(f"""SELECT count(*), count(DISTINCT family_id), round(avg(n_queries), 2), max(n_queries)
                       FROM '{a.labels}/{a.out_name}.parquet'""").fetchone()
    deg = c.execute(f"""SELECT quantile_cont(n, [0.5, 0.9, 0.99]) FROM
                        (SELECT count(*) AS n FROM '{a.labels}/{a.out_name}.parquet' GROUP BY family_id)""").fetchone()[0]
    print(json.dumps({"xy_rows": c.execute("SELECT count(*) FROM xy").fetchone()[0], "edges": st[0], "families": st[1],
                      "avg_weight": st[2], "max_weight": st[3], "partners_p50_p90_p99": deg, "seconds": round(time.time() - t0)}))


if __name__ == "__main__":
    main()
