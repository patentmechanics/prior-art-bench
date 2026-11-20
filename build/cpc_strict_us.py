#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
"""Strict CPC tables for the CPC-strict fusion stage (USPTO data only).

The first fusion test (cpc_us.py) used the CURRENT Master Classification File: reclassification after the query date and
family members published later could leak. This builds, per document, the CPC as PRINTED at publication (parsed USPTO
full text, `cpc` field) next to the current MCF symbols, plus the publication date, so the evaluation can restrict every
candidate family to members published before the query's priority date:

  doc_cpc.parquet    family_id, doc_id, pub_date, cpc_printed, cpc_current   (documents of the evaluation-pool families)
  query_cpc_strict.parquet  query_id, source, qdate (priority date), cpc_printed (printed on the query's own document),
                            cpc_current

  PAB_WORK=$WORK python build/cpc_strict_us.py [--cpc $WORK/eval_us_v2_cpc]
Writes summary_strict.json (coverage). Reads only the doc_id / pub_date / cpc columns of the parsed corpus.
"""
import argparse, json, os, resource, time

import duckdb

W = os.environ.get("PAB_WORK", "work")
ap = argparse.ArgumentParser()
ap.add_argument("--cpc", default=f"{W}/eval_us_v2_cpc")
a = ap.parse_args()
resource.setrlimit(resource.RLIMIT_NOFILE, (resource.getrlimit(resource.RLIMIT_NOFILE)[1],) * 2)   # ~4,000 weekly files
t0 = time.time()
mcf = f"{a.cpc}/mcf.parquet"
c = duckdb.connect()
c.execute("SET threads=4; SET memory_limit='8GB'; SET preserve_insertion_order=false")
c.execute(f"""CREATE TABLE d AS SELECT family_id::BIGINT AS family_id, doc_id, pub_date
              FROM '{W}/family_map/docs.parquet'
              WHERE family_id::BIGINT IN (SELECT DISTINCT family_id::BIGINT FROM '{W}/eval_us_v2/tacd/*.parquet')""")
c.execute(f"CREATE TABLE q AS SELECT query_id, source, query_doc_id, priority_date AS qdate FROM '{W}/eval_us_v2/queries/queries.parquet'")
c.execute("CREATE TABLE want AS SELECT doc_id FROM d UNION SELECT query_doc_id FROM q")
# printed CPC: one pass over the parsed corpus, two columns only (one schema for every product, so no union_by_name)
c.execute(f"""CREATE TABLE printed AS SELECT p.doc_id, p.cpc FROM read_parquet('{W}/us/*/*/*.parquet') p
              SEMI JOIN want USING (doc_id)""")
c.execute(f"CREATE TABLE cur AS SELECT key, cpc FROM '{mcf}' WHERE key IN (SELECT DISTINCT regexp_extract(doc_id, '^(US[0-9]+)', 1) FROM want)")
c.execute(f"""COPY (SELECT d.family_id, d.doc_id, d.pub_date,
                           CASE WHEN len(p.cpc) > 0 THEN p.cpc END AS cpc_printed, m.cpc AS cpc_current
                    FROM d LEFT JOIN printed p USING (doc_id)
                           LEFT JOIN cur m ON m.key = regexp_extract(d.doc_id, '^(US[0-9]+)', 1))
              TO '{a.cpc}/doc_cpc.parquet' (FORMAT parquet, COMPRESSION zstd)""")
c.execute(f"""COPY (SELECT q.query_id, q.source, q.qdate, CASE WHEN len(p.cpc) > 0 THEN p.cpc END AS cpc_printed, m.cpc AS cpc_current
                    FROM q LEFT JOIN printed p ON p.doc_id = q.query_doc_id
                           LEFT JOIN cur m ON m.key = regexp_extract(q.query_doc_id, '^(US[0-9]+)', 1))
              TO '{a.cpc}/query_cpc_strict.parquet' (FORMAT parquet, COMPRESSION zstd)""")
D, Q = f"'{a.cpc}/doc_cpc.parquet'", f"'{a.cpc}/query_cpc_strict.parquet'"
s = {"pool_docs": c.execute(f"SELECT count(*) FROM {D}").fetchone()[0],
     "pool_docs_printed": c.execute(f"SELECT count(*) FILTER (WHERE cpc_printed IS NOT NULL) FROM {D}").fetchone()[0],
     "pool_docs_current": c.execute(f"SELECT count(*) FILTER (WHERE cpc_current IS NOT NULL) FROM {D}").fetchone()[0],
     "pool_families_printed": c.execute(f"SELECT count(DISTINCT family_id) FILTER (WHERE cpc_printed IS NOT NULL) FROM {D}").fetchone()[0],
     "printed_by_decade": dict(c.execute(f"""SELECT (pub_date // 100000) * 10 AS dec, round(avg((cpc_printed IS NOT NULL)::INT), 3)
                                             FROM {D} GROUP BY 1 ORDER BY 1""").fetchall()),
     "queries_printed": dict(c.execute(f"SELECT source, count(*) FILTER (WHERE cpc_printed IS NOT NULL) FROM {Q} GROUP BY 1").fetchall()),
     "queries_current": dict(c.execute(f"SELECT source, count(*) FILTER (WHERE cpc_current IS NOT NULL) FROM {Q} GROUP BY 1").fetchall()),
     "minutes": round((time.time() - t0) / 60, 1)}
json.dump(s, open(f"{a.cpc}/summary_strict.json", "w"), indent=1)
print(json.dumps(s), flush=True)
