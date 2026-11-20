#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
"""US-only: slim extract of the USPTO office-action citation table (reads $PAB_SRC/uspto_oa/v1 from build/oa_build.py).

Keeps citations to US documents (ref_country = 'US'), with the reference normalised to the family-graph tokens of
family_us.py:  U:<11-digit pre-grant publication number>  or  G:<patent number>.  Foreign and non-patent references are
dropped (the US-only release never resolves them). Output: oa_us.parquet
(oa_id, app_number, is_first_action, mail_date, category, related_claims, tok, ref_number).
"""
import duckdb, os, sys
W = os.environ.get("PAB_SRC", "src"); OUT = os.environ.get("PAB_WORK", "work") + "/oa"; TMP = os.environ.get("PAB_WORK", "work") + "/tmp"
os.makedirs(OUT, exist_ok=True)
c = duckdb.connect(); c.execute(f"SET threads=10; SET memory_limit='40GB'; SET temp_directory='{TMP}'")
os.makedirs(TMP, exist_ok=True)
c.execute(f"""COPY (
  SELECT c.oa_id, c.app_number, a.is_first_action, a.mail_date, c.category, array_to_string(c.related_claims, ',') AS related_claims,
    CASE WHEN regexp_matches(c.ref_number_docdb, '^US\\d{{4}}\\d{{6}}$') THEN 'U:' || substr(c.ref_number_docdb, 3, 4) || '0' || substr(c.ref_number_docdb, 7)
         WHEN regexp_matches(c.ref_number_docdb, '^US\\d{{7,8}}$') THEN 'G:' || regexp_replace(substr(c.ref_number_docdb, 3), '^0+', '')
         WHEN regexp_matches(c.ref_number_docdb, '^USRE\\d+$') THEN 'G:RE' || regexp_replace(substr(c.ref_number_docdb, 5), '^0+', '')
         ELSE NULL END AS tok,
    c.ref_number_docdb AS ref_number
  FROM '{W}/uspto_oa/v1/oa_citations.parquet' c JOIN '{W}/uspto_oa/v1/oa_actions.parquet' a USING (oa_id)
  WHERE c.ref_country = 'US' AND c.ref_number_docdb IS NOT NULL
) TO '{OUT}/oa_us.parquet' (FORMAT parquet, COMPRESSION zstd)""")
print(c.execute(f"SELECT count(*), count(tok), count(DISTINCT app_number) FROM '{OUT}/oa_us.parquet'").fetchone())
print(c.execute(f"SELECT category, is_first_action, count(*) FROM '{OUT}/oa_us.parquet' GROUP BY ALL ORDER BY 3 DESC LIMIT 12").fetchall())
