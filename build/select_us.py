#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
"""US-only step 2: integer family ids, family table and one representative document per US family.

Reads family/docs.parquet and family/family_dates.parquet (family_us.py) and meta/us.parquet (read-only, for the source file
of each document). Writes into $OUT:
  family_map/families.parquet   family_id (BIGINT, dense, ordered by the family key), fam_key, strict_id, priority_date,
                                priority_src, earliest_pd (earliest publication date of any member), n_docs
  family_map/docs.parquet       doc_id -> family_id, strict_id, pub_tok, app_tok, ... (all US documents)
  reps/reps_us.parquet          one representative per family: utility/reissue, published 1976 on (same rule as the
                                US+EP pipeline: has claims, grant before application, longer description, later date)
Same column names as build/corpus.py expects (family_id, doc_id, file ...), so build/corpus.py build runs unchanged with
PAB_WORK pointing at the US-only work directory.
"""
import json, os, time
import duckdb

OUT = os.environ.get("PAB_WORK", "work")
META = os.environ.get("PAB_SRC", "src") + "/meta/us.parquet"   # build/corpus.py meta --src us (per-document metadata)
FAM = f"{OUT}/family"
os.makedirs(f"{OUT}/family_map", exist_ok=True); os.makedirs(f"{OUT}/reps", exist_ok=True)
c = duckdb.connect(); c.execute(f"SET threads=20; SET memory_limit='60GB'; SET temp_directory='{OUT}/tmp'; SET preserve_insertion_order=false")
c.execute(f"""CREATE TABLE fk AS SELECT fam_key, (row_number() OVER (ORDER BY fam_key))::BIGINT AS family_id FROM (SELECT DISTINCT fam_ext AS fam_key FROM '{FAM}/docs.parquet')""")
c.execute(f"""CREATE TABLE sk AS SELECT fam_key, (row_number() OVER (ORDER BY fam_key))::BIGINT AS strict_id FROM (SELECT DISTINCT fam_strict AS fam_key FROM '{FAM}/docs.parquet')""")
c.execute(f"""CREATE TABLE d AS
    SELECT d.*, fk.family_id, sk.strict_id, m.file
    FROM '{FAM}/docs.parquet' d JOIN fk ON fk.fam_key = d.fam_ext JOIN sk ON sk.fam_key = d.fam_strict
    LEFT JOIN (SELECT doc_id, any_value(file) AS file FROM '{META}' GROUP BY 1) m ON m.doc_id = d.doc_id""")
c.execute(f"COPY d TO '{OUT}/family_map/docs.parquet' (FORMAT parquet, COMPRESSION zstd)")
c.execute(f"""COPY (
    SELECT fk.family_id, fk.fam_key, any_value(d.strict_id) AS strict_id, p.priority_date, p.priority_src,
           min(d.pub_date) AS earliest_pd, count(*) AS n_docs
    FROM d JOIN fk USING (family_id) LEFT JOIN '{FAM}/family_dates.parquet' p ON p.fam_ext = fk.fam_key
    GROUP BY fk.family_id, fk.fam_key, p.priority_date, p.priority_src
  ) TO '{OUT}/family_map/families.parquet' (FORMAT parquet, COMPRESSION zstd)""")
c.execute("""CREATE TABLE reps AS
    SELECT * FROM d WHERE coalesce(patent_type, '') NOT IN ('design', 'plant') AND pub_date >= 19760101 AND file IS NOT NULL
    QUALIFY row_number() OVER (PARTITION BY family_id
        ORDER BY (n_claims > 0) DESC, (doc_class = 'grant') DESC, desc_chars DESC, pub_date DESC, doc_id) = 1""")
c.execute(f"COPY reps TO '{OUT}/reps/reps_us.parquet' (FORMAT parquet, COMPRESSION zstd)")
s = {"docs": c.execute("SELECT count(*) FROM d").fetchone()[0],
     "docs_without_file": c.execute("SELECT count(*) FROM d WHERE file IS NULL").fetchone()[0],
     "families": c.execute("SELECT count(*) FROM fk").fetchone()[0],
     "families_with_rep": c.execute("SELECT count(*) FROM reps").fetchone()[0],
     "reps_by_class": c.execute("SELECT doc_class, count(*), count_if(n_claims = 0) FROM reps GROUP BY 1").fetchall(),
     "families_without_priority_date": c.execute(f"SELECT count(*) FROM '{OUT}/family_map/families.parquet' WHERE priority_date IS NULL").fetchone()[0]}
json.dump(s, open(f"{OUT}/reps/select_stats.json", "w"), indent=1, default=str)
print(json.dumps(s, default=str))
