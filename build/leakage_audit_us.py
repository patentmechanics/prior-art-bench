#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
"""Leakage audit (US-only): exact and near-exact text overlap between evaluation queries and training queries / pool
documents. Local files only. Writes labels/leakage_audit.json.

  PAB_WORK=$WORK python build/leakage_audit_us.py

Checks, each counting evaluation queries (by split) that match a *different-family* training item:
  full_tac        md5 of the query text (title + abstract + claims) equals a training query's
  claim1          normalised first claim equals a training query's
  title_abstract  normalised title + abstract equals a training query's
  pool document   query text equals the text of a training-pool document (positives of the mining queries)
plus the train / evaluation family overlap (must be 0).
"""
import duckdb, json, os

W = os.environ.get("PAB_WORK", "work")
EVAL = "('gold', 'eval_ptab', 'dev', 'holdout_time')"
c = duckdb.connect()
c.execute("SET threads=6; SET memory_limit='10GB'")
c.execute(f"""CREATE TABLE q AS SELECT q.query_id, s.split, q.family_id, md5(q.tac_text) AS h_tac,
    md5(lower(regexp_replace(coalesce(q.claim1, ''), '\\s+', ' ', 'g'))) AS h_c1,
    md5(lower(regexp_replace(coalesce(q.title, '') || ' ' || coalesce(q.abstract, ''), '\\s+', ' ', 'g'))) AS h_ta
    FROM '{W}/queries/queries.parquet' q JOIN '{W}/labels/splits.parquet' s USING (query_id)""")
empty = {"h_c1": c.execute("SELECT md5(lower(regexp_replace('', '\\s+', ' ', 'g')))").fetchone()[0],
         "h_ta": c.execute("SELECT md5(lower(regexp_replace(' ', '\\s+', ' ', 'g')))").fetchone()[0],
         "h_tac": c.execute("SELECT md5('')").fetchone()[0]}
out = {"queries_by_split": dict(c.execute("SELECT split, count(*) FROM q GROUP BY 1").fetchall())}
for col, name in (("h_tac", "full_tac"), ("h_c1", "claim1"), ("h_ta", "title_abstract")):
    rows = c.execute(f"""SELECT e.split, count(DISTINCT e.query_id) FROM q e JOIN q t ON e.{col} = t.{col}
        AND t.split = 'train' AND e.family_id <> t.family_id
        WHERE e.split IN {EVAL} AND e.{col} <> '{empty[col]}' GROUP BY 1""").fetchall()
    out[f"eval_queries_matching_a_train_query_{name}"] = dict(rows)
rows = c.execute(f"""SELECT e.split, count(DISTINCT e.query_id) FROM q e JOIN
    (SELECT family_id, md5(tac_text) AS h FROM read_parquet('{W}/mine/pool_tac/*.parquet')) p
    ON e.h_tac = p.h AND e.family_id <> p.family_id WHERE e.split IN {EVAL} GROUP BY 1""").fetchall()
out["eval_queries_matching_a_pool_document_full_tac"] = dict(rows)
out["train_eval_family_overlap"] = c.execute(f"""SELECT count(*) FROM (SELECT family_id FROM q WHERE split = 'train'
    INTERSECT SELECT family_id FROM q WHERE split IN {EVAL})""").fetchone()[0]
json.dump(out, open(f"{W}/labels/leakage_audit.json", "w"), indent=1)
print(json.dumps(out))
