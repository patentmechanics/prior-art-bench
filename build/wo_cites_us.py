#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
"""US-only metrics v2, step 1 (reads $PAB_SRC/uspto_oa/v1 only): first-action X/Y office-action citations of the
evaluation queries that are NOT US-numbered, so labels v2 can resolve WO publications to US families.

  PAB_SRC=$SRC python build/wo_cites_us.py --splits $WORK/eval_us/labels/splits.parquet \
      --out $WORK/v2in/oa_foreign_cites.parquet

Output rows (one per citation of an evaluation office-action query): query_id, ref_country, ref_number_docdb, ref_raw,
category (X/Y), grade (3/2), wo_year, wo_serial (WO only, parsed from the many number formats, see wo_key()).
Also <out>.counts.json: citations by country (the "left out and counted" figures of the metric rule).
"""
import argparse, json, re

import duckdb

import os
W = os.environ.get("PAB_SRC", "src")


def wo_key(num):
    """WO publication number in any of the observed forms -> (year, serial) or None.
    WO2012019643 / WO2012/019643 -> (2012, 19643); WO9710214 -> (1997, 10214); WO06080951 -> (2006, 80951);
    WO20150192507 -> (2015, 192507); WO201482599 -> (2014, 82599); WO025702 (raw WO 2002/05702) -> (2002, 5702)."""
    if not num:
        return None
    s = re.sub(r"^\s*WO\s*", "", num.upper())
    s = re.sub(r"\s*[A-Z]\d?\s*$", "", s)          # kind code (A1, A2, A) at the end
    s = re.sub(r"[^0-9/]", "", s)
    if "/" in s:
        y, _, n = s.partition("/")
        n = re.sub(r"\D", "", n)
    else:
        if len(s) >= 9 and s[:2] in ("19", "20") and int(s[:4]) >= 1978:
            y, n = s[:4], s[4:]
        else:
            y, n = s[:2], s[2:]
    if not y.isdigit() or not n.isdigit() or not n:
        return None
    y = int(y)
    if y < 100:
        y += 1900 if y >= 78 else 2000
    if not 1978 <= y <= 2030:
        return None
    return y, int(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True, help="evaluation splits.parquet (query_id, source, split)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    c = duckdb.connect()
    c.execute("SET threads=3; SET memory_limit='5GB'; SET preserve_insertion_order=false")
    c.execute(f"""CREATE TABLE ev AS SELECT replace(query_id, 'us_oa:', '') AS app, query_id FROM '{a.splits}'
                  WHERE source = 'us_oa' AND split IN ('gold', 'eval_ptab')""")
    c.execute(f"""CREATE TABLE fc AS
        SELECT ev.query_id, c.ref_country, c.ref_number_docdb, c.ref_raw, c.category, CASE c.category WHEN 'X' THEN 3 ELSE 2 END AS grade
        FROM '{W}/uspto_oa/v1/oa_citations.parquet' c JOIN '{W}/uspto_oa/v1/oa_actions.parquet' x USING (oa_id) JOIN ev ON ev.app = x.app_number
        WHERE x.is_first_action AND c.category IN ('X', 'Y') AND c.ref_country IS NOT NULL AND c.ref_country <> 'US'""")
    rows = c.execute("SELECT * FROM fc").fetchall()
    keys = [wo_key(r[2] or r[3]) if r[1] == "WO" else None for r in rows]
    import pyarrow as pa, pyarrow.parquet as pq
    cols = list(zip(*rows)) if rows else [[] for _ in range(6)]
    t = pa.table({"query_id": cols[0], "ref_country": cols[1], "ref_number_docdb": cols[2], "ref_raw": cols[3],
                  "category": cols[4], "grade": pa.array(cols[5], pa.int32()),
                  "wo_year": pa.array([k[0] if k else None for k in keys], pa.int32()),
                  "wo_serial": pa.array([k[1] if k else None for k in keys], pa.int64())})
    pq.write_table(t, a.out)
    counts = c.execute("SELECT ref_country, count(*), count(DISTINCT query_id) FROM fc GROUP BY 1 ORDER BY 2 DESC").fetchall()
    st = {"evaluation_oa_queries": c.execute("SELECT count(*) FROM ev").fetchone()[0], "foreign_citations": len(rows),
          "wo_citations": sum(1 for r in rows if r[1] == "WO"), "wo_parsed": sum(1 for k in keys if k),
          "by_country": [{"country": k, "citations": n, "queries": q} for k, n, q in counts]}
    json.dump(st, open(a.out + ".counts.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in st.items() if k != "by_country"}))


if __name__ == "__main__":
    main()
