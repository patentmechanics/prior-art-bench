#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
"""CPC codes of the US-only evaluation pool and queries (USPTO data only: the CPC Master Classification Files
for U.S. grants and pre-grant publications, USPTO Open Data Portal products CPCMCPT / CPCMCAPP, release 2026-11-01).
For the CPC fusion stage: dense score + w x CPC agreement, as examiners search by CPC.

  PAB_WORK=$WORK python build/cpc_us.py [--zips $WORK/cpc] [--out $WORK/eval_us_v2_cpc]

1. Stream every XML file of both zips (iterparse; ~67 GB unzipped, never on disk) -> one row per publication:
   key "US<number>" (grant number without leading zeros / 11-digit publication number), CPC symbols (main + further,
   "B61C11/04"). Written to $out/mcf.parquet.
2. Our documents: doc_id "US<number><kind>" (family_map/docs.parquet) -> key by stripping the kind code.
   - pool families (eval_us_v2/tacd): the union of the CPC symbols of every document in the family -> fam_cpc.parquet
   - queries (eval_us_v2/queries): the CPC of the query's own document (query_doc_id: the published application for
     office-action queries, the granted patent for PTAB queries) -> query_cpc.parquet
   The MCF holds CURRENT classifications (reclassified since publication); noted in the results.
Writes summary.json (coverage).
"""
import argparse, glob, json, os, re, time, zipfile
import xml.etree.ElementTree as ET

import duckdb
import pyarrow as pa, pyarrow.parquet as pq

W = os.environ.get("PAB_WORK", "work")
ap = argparse.ArgumentParser()
ap.add_argument("--zips", default=f"{W}/cpc")
ap.add_argument("--out", default=f"{W}/eval_us_v2_cpc")
a = ap.parse_args()
t0 = time.time()
os.makedirs(a.out, exist_ok=True)
NS = {"pat": "http://www.wipo.int/standards/XMLSchema/ST96/Patent", "com": "http://www.wipo.int/standards/XMLSchema/ST96/Common",
      "uspat": "patent:uspto:doc:us:gov"}
P, C = "{%s}" % NS["pat"], "{%s}" % NS["com"]
REC = "{%s}CPCMasterClassificationRecord" % NS["uspat"]


def symbol(c):
    g = lambda t: (c.findtext(P + t) or "").strip()
    sec, cls, sub, mg, sg = g("CPCSection"), g("Class"), g("Subclass"), g("MainGroup"), g("Subgroup")
    return f"{sec}{cls}{sub}{mg}/{sg}" if sec and cls and sub and mg else None


schema = pa.schema([("key", pa.string()), ("cpc", pa.list_(pa.string()))])
mcf = f"{a.out}/mcf.parquet"
if not os.path.exists(mcf):
    writer, keys, cpcs, n = pq.ParquetWriter(mcf + ".tmp", schema, compression="zstd"), [], [], 0
    for z in sorted(glob.glob(f"{a.zips}/US_*_CPC_MCF_XML_*.zip")):
        zf = zipfile.ZipFile(z)
        for name in sorted(x for x in zf.namelist() if x.endswith(".xml")):
            with zf.open(name) as fh:
                root = None
                for ev, el in ET.iterparse(fh, events=("start", "end")):
                    if root is None:
                        root = el   # cleared after each record: finished records must not stay attached (memory)
                    if ev != "end" or el.tag != REC:
                        continue
                    num = (el.findtext(f".//{P}PatentNumber") or el.findtext(f".//{P}PublicationNumber") or "").strip()
                    syms = sorted({s for c in el.iter(P + "CPCClassification") if (s := symbol(c))})
                    if num and syms:
                        keys.append("US" + (num.lstrip("0") if len(num) < 11 else num))
                        cpcs.append(syms)
                    root.clear()
                    if len(keys) >= 200000:
                        writer.write_table(pa.table({"key": keys, "cpc": cpcs}, schema=schema))
                        n += len(keys)
                        keys, cpcs = [], []
            print(json.dumps({"file": name.rsplit("/", 1)[-1], "records": n + len(keys), "minutes": round((time.time() - t0) / 60, 1)}), flush=True)
    if keys:
        writer.write_table(pa.table({"key": keys, "cpc": cpcs}, schema=schema))
        n += len(keys)
    writer.close()
    os.replace(mcf + ".tmp", mcf)
c = duckdb.connect()
c.execute("SET threads=4; SET memory_limit='8GB'; SET preserve_insertion_order=false")
c.execute(f"""CREATE TABLE d AS SELECT family_id::BIGINT AS family_id, regexp_extract(doc_id, '^(US[0-9]+)', 1) AS key, doc_id
              FROM '{W}/family_map/docs.parquet'
              WHERE family_id::BIGINT IN (SELECT DISTINCT family_id::BIGINT FROM '{W}/eval_us_v2/tacd/*.parquet')""")
c.execute(f"""COPY (SELECT d.family_id, list_sort(list_distinct(flatten(list(m.cpc)))) AS cpc
                    FROM d JOIN '{mcf}' m USING (key) GROUP BY d.family_id)
              TO '{a.out}/fam_cpc.parquet' (FORMAT parquet, COMPRESSION zstd)""")
c.execute(f"""COPY (SELECT q.query_id, q.source, m.cpc FROM '{W}/eval_us_v2/queries/queries.parquet' q
                    LEFT JOIN '{mcf}' m ON m.key = regexp_extract(q.query_doc_id, '^(US[0-9]+)', 1))
              TO '{a.out}/query_cpc.parquet' (FORMAT parquet, COMPRESSION zstd)""")
s = {"mcf_records": c.execute(f"SELECT count(*) FROM '{mcf}'").fetchone()[0],
     "pool_families": c.execute(f"SELECT count(DISTINCT family_id) FROM '{W}/eval_us_v2/tacd/*.parquet'").fetchone()[0],
     "pool_families_with_cpc": c.execute(f"SELECT count(*) FROM '{a.out}/fam_cpc.parquet'").fetchone()[0],
     "queries": c.execute(f"SELECT count(*) FROM '{a.out}/query_cpc.parquet'").fetchone()[0],
     "queries_with_cpc": dict(c.execute(f"SELECT source, count(*) FILTER (WHERE cpc IS NOT NULL) FROM '{a.out}/query_cpc.parquet' GROUP BY 1").fetchall()),
     "avg_symbols_family": c.execute(f"SELECT round(avg(len(cpc)), 1) FROM '{a.out}/fam_cpc.parquet'").fetchone()[0],
     "minutes": round((time.time() - t0) / 60, 1)}
json.dump(s, open(f"{a.out}/summary.json", "w"), indent=1)
print(json.dumps(s), flush=True)
