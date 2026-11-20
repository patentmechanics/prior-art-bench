#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
"""Near-duplicate guard (US-only): training queries whose text matches an evaluation query of a different family are taken
out of training (split 'excluded_near_dup'); the evaluation sets stay as defined. Match = identical full query text, or
identical normalised first claim, or identical normalised title + abstract (see leakage_audit_us.py).

  PAB_WORK=$WORK python build/dedup_splits_us.py

labels/splits.parquet is backed up once to labels/splits.pre_dedup.parquet and then rewritten; the list of removed
training queries is kept in labels/near_dup_train_queries.parquet. Re-running is safe (the backup is the input).
"""
import duckdb, json, os, shutil

W = os.environ.get("PAB_WORK", "work")
L = f"{W}/labels"
if not os.path.exists(f"{L}/splits.pre_dedup.parquet"):
    shutil.copy(f"{L}/splits.parquet", f"{L}/splits.pre_dedup.parquet")
c = duckdb.connect()
c.execute("SET threads=6; SET memory_limit='10GB'")
c.execute(f"""CREATE TABLE q AS SELECT q.query_id, s.split, q.family_id, md5(q.tac_text) AS h_tac,
    md5(lower(regexp_replace(coalesce(q.claim1, ''), '\\s+', ' ', 'g'))) AS h_c1,
    md5(lower(regexp_replace(coalesce(q.title, '') || ' ' || coalesce(q.abstract, ''), '\\s+', ' ', 'g'))) AS h_ta
    FROM '{W}/queries/queries.parquet' q JOIN '{L}/splits.pre_dedup.parquet' s USING (query_id)""")
e_c1 = c.execute("SELECT md5(lower(regexp_replace('', '\\s+', ' ', 'g')))").fetchone()[0]
e_ta = c.execute("SELECT md5(lower(regexp_replace(' ', '\\s+', ' ', 'g')))").fetchone()[0]
c.execute("CREATE TABLE ev AS SELECT * FROM q WHERE split IN ('gold', 'eval_ptab', 'dev', 'holdout_time')")
c.execute(f"""CREATE TABLE bad AS SELECT DISTINCT t.query_id,
      max(CASE WHEN t.h_tac = e.h_tac THEN 1 ELSE 0 END) AS m_tac,
      max(CASE WHEN t.h_c1 = e.h_c1 AND t.h_c1 <> '{e_c1}' THEN 1 ELSE 0 END) AS m_claim1,
      max(CASE WHEN t.h_ta = e.h_ta AND t.h_ta <> '{e_ta}' THEN 1 ELSE 0 END) AS m_ta
    FROM q t JOIN ev e ON e.family_id <> t.family_id AND (t.h_tac = e.h_tac OR (t.h_c1 = e.h_c1 AND t.h_c1 <> '{e_c1}')
         OR (t.h_ta = e.h_ta AND t.h_ta <> '{e_ta}'))
    WHERE t.split = 'train' GROUP BY t.query_id""")
c.execute(f"""COPY (SELECT s.query_id, CASE WHEN b.query_id IS NOT NULL THEN 'excluded_near_dup' ELSE s.split END AS split,
                     s.* EXCLUDE (query_id, split) FROM '{L}/splits.pre_dedup.parquet' s LEFT JOIN bad b USING (query_id))
              TO '{L}/splits.parquet' (FORMAT parquet)""")
c.execute(f"COPY (SELECT * FROM bad) TO '{L}/near_dup_train_queries.parquet' (FORMAT parquet)")
st = {"train_queries_removed": c.execute("SELECT count(*) FROM bad").fetchone()[0],
      "by_rule": c.execute("SELECT sum(m_tac), sum(m_claim1), sum(m_ta) FROM bad").fetchone(),
      "splits_after": dict(c.execute(f"SELECT split, count(*) FROM '{L}/splits.parquet' GROUP BY 1").fetchall())}
st["by_rule"] = dict(zip(["full_tac", "claim1", "title_abstract"], [int(x) for x in st["by_rule"]]))
json.dump(st, open(f"{L}/near_dup_stats.json", "w"), indent=1)
print(json.dumps(st))
