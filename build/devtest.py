#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
"""Dev/test split of the evaluation queries (2026-11-09): every decision is made on dev; test is scored once, for the
final public numbers.

  python build/devtest.py --labels $WORK/labels --salt devtest-us-v1      # the salt of the US benchmark

Evaluation queries = splits gold (EP, US), eval_ptab, eval_opp_amended, eval_opp_revoked of labels/splits.parquet. Each
query goes to dev or test by md5(salt + its own family id) (query id if it has no family): all queries of one family
(e.g. a patent in both PTAB and an opposition) land on the same side; ~50/50 per source. The existing EP/US "dev" splits
are not used: only ~7% of their cited documents are in the evaluation subset (runs/subset_v2).
Output: DATA/labels/devtest.parquet (query_id, source, split, family_id, part). The same file serves labels_b (same query
ids; B's older PTAB/opposition half is training data there and never scored). train/evaluate.py metrics/compare --part.
"""
import argparse, hashlib, json

import duckdb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--salt", default="devtest-v1")
    a = ap.parse_args()
    c = duckdb.connect()
    rows = c.execute(f"""SELECT query_id, source, split, family_id FROM '{a.labels}/splits.parquet'
                         WHERE split IN ('gold', 'eval_ptab', 'eval_opp_amended', 'eval_opp_revoked')""").fetchall()
    out = []
    for qid, src, split, fam in rows:
        key = f"{a.salt}:{fam if fam is not None else qid}"
        out.append((qid, src, split, fam, "dev" if int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % 2 == 0 else "test"))
    c.execute("CREATE TABLE dt(query_id VARCHAR, source VARCHAR, split VARCHAR, family_id BIGINT, part VARCHAR)")
    c.executemany("INSERT INTO dt VALUES (?, ?, ?, ?, ?)", out)
    c.execute(f"COPY (SELECT * FROM dt ORDER BY query_id) TO '{a.labels}/devtest.parquet' (FORMAT parquet)")
    st = c.execute("SELECT source, part, count(*) FROM dt GROUP BY 1, 2 ORDER BY 1, 2").fetchall()
    leak = c.execute("SELECT count(*) FROM (SELECT family_id FROM dt WHERE family_id IS NOT NULL GROUP BY 1 HAVING count(DISTINCT part) > 1)").fetchone()[0]
    print(json.dumps({"queries": len(out), "by_source_part": [list(r) for r in st], "families_in_both_parts": leak, "salt": a.salt}))


if __name__ == "__main__":
    main()
