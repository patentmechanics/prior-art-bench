#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
"""Hard negatives from a retrieval run (evaluate.py search over training queries), with false-negative safeguards.

  python train/mine.py --run RUN.parquet --data DATA --out NEG.parquet [--lo 30 --hi 100 --n 32]
        [--pos-cap 0.95 --cap-ref max] [--cocited DATA/labels/cocited.parquet] [--audit AUDIT.parquet]

Examiners cite enough to reject a claim, not every relevant document, so the top uncited documents of a query are partly
unlabeled prior art. A candidate becomes a negative only if all of these hold:
  1. rank window: lo < rank <= hi (1-based; the very top ranks are the likeliest unlabeled prior art);
  2. not known: not cited for the query in any source, phase or category (known_relevant), not the query's own family;
  3. not co-cited: not an examiner co-citation partner of any of the query's positives (build/cocite.py, training
     queries only) - documents cited together with its citations elsewhere;
  4. positive-aware cap: score <= pos_cap * the query's positive score (cap-ref max: its best-scoring positive, min: its
     weakest), applied only when a positive is in the run's list. Queries whose positives are all outside the list are not
     capped: the LLM audit (2026-11-09, 1,481 judged pairs) found their candidates the least likely to be unlabeled prior
     art (0.4% judged X/Y vs 1.8% for kept candidates and 23% for real citations), and they are the queries the model
     misses. Recommended co-citation file: cocited_w1 (co-cited by >= 1 training query; as precise as >= 2, 18x coverage).
Output: query_id, neg_ids list<int64> (rank order) -> train/pack.py --negatives. --audit writes every candidate in the
window with the reason it was kept or dropped (for audits and filter tuning).
"""
import argparse, json, time

import duckdb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels-dir", default="labels")
    ap.add_argument("--lo", type=int, default=30)
    ap.add_argument("--hi", type=int, default=100)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--pos-cap", type=float, default=0.95, help="0 disables the cap")
    ap.add_argument("--cap-ref", choices=["max", "min"], default="max")
    ap.add_argument("--cocited", default=None, help="parquet family_id, partner_id (build/cocite.py); omit to disable")
    ap.add_argument("--audit", default=None)
    ap.add_argument("--memory", default="16GB")
    a = ap.parse_args()
    t0 = time.time()
    L = f"{a.data}/{a.labels_dir}"
    c = duckdb.connect()
    c.execute(f"SET memory_limit='{a.memory}'")
    c.execute(f"""CREATE TABLE r AS SELECT query_id, unnest(family_ids) AS family_id, unnest(scores) AS score,
                  generate_subscripts(family_ids, 1) AS rank FROM '{a.run}'""")
    c.execute("CREATE TABLE r1 AS SELECT query_id, family_id, min(rank) AS rank, max(score) AS score FROM r GROUP BY 1, 2")
    c.execute("CREATE TABLE last AS SELECT query_id, min(score) AS last_score FROM r GROUP BY 1")
    c.execute(f"""CREATE TABLE pos AS SELECT query_id, target_family_id AS family_id FROM '{L}/labels.parquet'
                  WHERE keep AND grade >= 2 AND query_id IN (SELECT query_id FROM last)""")
    agg = "max" if a.cap_ref == "max" else "min"
    c.execute(f"""CREATE TABLE pos_score AS SELECT p.query_id, {agg}(r1.score) AS pos_score   -- NULL: no positive in the list
                  FROM pos p LEFT JOIN r1 USING (query_id, family_id) GROUP BY 1""")
    c.execute(f"""CREATE TABLE known AS SELECT query_id, family_id FROM '{L}/known_relevant.parquet' WHERE query_id IN (SELECT query_id FROM last)
                  UNION SELECT query_id, family_id FROM '{a.data}/queries/queries.parquet'
                        WHERE family_id IS NOT NULL AND query_id IN (SELECT query_id FROM last)""")
    if a.cocited:
        c.execute(f"""CREATE TABLE cocit AS SELECT DISTINCT p.query_id, e.partner_id AS family_id
                      FROM pos p JOIN '{a.cocited}' e ON e.family_id = p.family_id""")
    else:
        c.execute("CREATE TABLE cocit(query_id VARCHAR, family_id BIGINT)")
    cap = f"r1.score > {a.pos_cap} * ps.pos_score" if a.pos_cap > 0 else "FALSE"
    c.execute(f"""CREATE TABLE cand AS
        SELECT r1.query_id, r1.family_id, r1.rank, r1.score, ps.pos_score,
               CASE WHEN k.family_id IS NOT NULL THEN 'known'
                    WHEN cc.family_id IS NOT NULL THEN 'cocited'
                    WHEN ps.query_id IS NULL THEN 'no_positive'
                    WHEN ps.pos_score IS NOT NULL AND {cap} THEN 'above_cap'
                    ELSE 'kept' END AS status
        FROM r1 LEFT JOIN pos_score ps USING (query_id)
        LEFT JOIN known k USING (query_id, family_id) LEFT JOIN cocit cc USING (query_id, family_id)
        WHERE r1.rank > {a.lo} AND r1.rank <= {a.hi}""")
    c.execute(f"""COPY (SELECT query_id, list(family_id ORDER BY rank)[1:{a.n}] AS neg_ids FROM cand WHERE status = 'kept' GROUP BY 1)
                  TO '{a.out}' (FORMAT parquet)""")
    if a.audit:
        c.execute(f"COPY cand TO '{a.audit}' (FORMAT parquet, COMPRESSION zstd)")
    st = dict(c.execute("SELECT status, count(*) FROM cand GROUP BY 1").fetchall())
    nq = c.execute("SELECT count(*) FROM last").fetchone()[0]
    out = c.execute(f"SELECT count(*), avg(len(neg_ids)) FROM '{a.out}'").fetchone()
    print(json.dumps({"queries": nq, "queries_with_negatives": out[0], "avg_negs": round(out[1] or 0, 1),
                      "candidates_by_status": st, "window": [a.lo, a.hi], "pos_cap": a.pos_cap, "cap_ref": a.cap_ref,
                      "cocited": a.cocited, "seconds": round(time.time() - t0)}))


if __name__ == "__main__":
    main()
