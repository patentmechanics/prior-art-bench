#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
"""Query texts of the US benchmark: the US stages of our query builder (its EP and opposition stages are not used here and
are left out).

  PAB_SRC=$SRC python build/queries_us.py ptab --ptab-jsonl $SRC/inputs/ptab_eval_full.jsonl
  PAB_SRC=$SRC python build/queries_us.py oa_apps
  PAB_SRC=$SRC python build/queries_us.py oa_a1

  ptab     granted claims of each challenged patent of an instituted PTAB trial with references, from the parsed US full
           text ($PAB_SRC/meta/us.parquet + $PAB_SRC/us/) -> $PAB_SRC/queries/ptab_docs.parquet
  oa_apps  applications whose first office action cites at least one reference ($PAB_SRC/uspto_oa/v1 from
           build/oa_build.py) -> $PAB_SRC/queries/oa_apps.parquet
  oa_a1    the published application (claims as filed) of each of those applications, from the parsed US application
           files ($PAB_SRC/us/APPXML/) -> $PAB_SRC/queries_oa/oa_a1.parquet
build/labels_us.py then assembles queries, labels and splits from these files.
"""
import argparse, glob, json, os, re, time

import duckdb

WORK = os.environ.get("PAB_SRC", "src")
INPUTS = f"{WORK}/inputs"


def log(stage, msg):
    os.makedirs(f"{WORK}/logs", exist_ok=True)
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(f"{WORK}/logs/{stage}.log", "a") as f:
        f.write(line + "\n")


def con(threads=8, mem="24GB"):
    c = duckdb.connect()
    c.execute(f"SET threads={threads}; SET memory_limit='{mem}'; SET temp_directory='{WORK}/tmp'")
    return c


def ptab(a):
    stage = "q_ptab"
    os.makedirs(f"{WORK}/queries", exist_ok=True)
    pats = {}
    for line in open(a.ptab_jsonl):
        r = json.loads(line)
        if r.get("instituted") and r.get("reference_numbers") and r.get("challenged_patent"):
            pats.setdefault(re.sub(r"^US0*", "US", r["challenged_patent"]), []).append(r["trial"])
    c = con(threads=a.threads)
    c.execute("CREATE TABLE want(pn_nokind VARCHAR)")
    c.executemany("INSERT INTO want VALUES (?)", [(p,) for p in pats])
    c.execute(f"""CREATE TABLE m AS SELECT * FROM '{WORK}/meta/us.parquet'
                  WHERE doc_class = 'grant' AND regexp_replace(pn, '[A-Z][0-9]?$', '') IN (SELECT pn_nokind FROM want)
                  QUALIFY row_number() OVER (PARTITION BY regexp_replace(pn, '[A-Z][0-9]?$', '') ORDER BY pub_date DESC) = 1""")
    files = [r[0] for r in c.execute("SELECT DISTINCT file FROM m").fetchall()]
    log(stage, f"start: {len(pats)} challenged patents, {c.execute('SELECT count(*) FROM m').fetchone()[0]} found in US text, {len(files)} files")
    paths = [f"{WORK}/us/{f}" for f in files]
    c.execute(f"""COPY (SELECT d.doc_id, regexp_replace(m.pn, '[A-Z][0-9]?$', '') AS pn_nokind, d.title, d.abstract, d.claims,
                         d.pub_date, d.filing_date, d.app_number
                  FROM read_parquet({json.dumps(paths)}) d JOIN m USING (doc_id))
                  TO '{WORK}/queries/ptab_docs.parquet' (FORMAT parquet, COMPRESSION zstd)""")
    n = c.execute(f"SELECT count(*) FROM '{WORK}/queries/ptab_docs.parquet'").fetchone()[0]
    log(stage, f"DONE {n} challenged patents with granted claims (of {len(pats)})")


OA = f"{WORK}/uspto_oa/v1"
# normalized US doc_id (US20200000001A1 / US0123456B1) -> DOCDB publication number (US2020000001A1 / US123456B1); same as corpus.py
US_PN_SQL = r"""CASE WHEN regexp_matches(doc_id, '^US\d{4}0\d{6}[A-Z]\d?$') THEN 'US' || substr(doc_id, 3, 4) || substr(doc_id, 8)
     WHEN regexp_matches(doc_id, '^US[A-Z]{0,2}0\d+[A-Z]\d?$') THEN regexp_replace(doc_id, '^US([A-Z]{0,2})0+(\d+[A-Z]\d?)$', 'US\1\2')
     ELSE doc_id END"""


def oa_apps(a):
    """applications whose first office action cites at least one reference -> queries/oa_apps.parquet"""
    stage = "q_oa_apps"
    os.makedirs(f"{WORK}/queries", exist_ok=True)
    c = con(threads=a.threads)
    c.execute(f"""COPY (SELECT DISTINCT a.app_number FROM '{OA}/oa_actions.parquet' a JOIN '{OA}/oa_citations.parquet' c USING (oa_id)
                        WHERE a.is_first_action AND c.ref_number_docdb IS NOT NULL ORDER BY 1)
                  TO '{WORK}/queries/oa_apps.parquet' (FORMAT parquet)""")
    n = c.execute(f"SELECT count(*) FROM '{WORK}/queries/oa_apps.parquet'").fetchone()[0]
    log(stage, f"DONE {n} applications")


def oa_a1(a):
    """the published application (claims as filed) of each office-action application, from the US source files
    (the corpus keeps one document per family, usually the grant). Earliest publication per application, A1 before A2/A9."""
    stage = "q_oa_a1"
    c = con(threads=a.threads, mem="24GB")
    files = sorted(glob.glob(f"{WORK}/us/APPXML/*/*.parquet"))
    log(stage, f"start: {len(files)} application files")
    os.makedirs(f"{WORK}/queries_oa", exist_ok=True)
    c.execute(f"CREATE TABLE apps AS SELECT app_number FROM '{WORK}/queries/oa_apps.parquet'")
    c.execute(f"""COPY (
        SELECT doc_id, {US_PN_SQL} AS pn, app_number, kind, pub_date, filing_date, title, abstract, claims
        FROM read_parquet({json.dumps(files)}, union_by_name = true)
        WHERE app_number IN (SELECT app_number FROM apps) AND len(claims) > 0
        QUALIFY row_number() OVER (PARTITION BY app_number ORDER BY (kind = 'A1') DESC, pub_date, doc_id) = 1
      ) TO '{WORK}/queries_oa/oa_a1.parquet' (FORMAT parquet, COMPRESSION zstd)""")
    n = c.execute(f"SELECT count(*), count(DISTINCT app_number) FROM '{WORK}/queries_oa/oa_a1.parquet'").fetchone()
    log(stage, f"DONE {n[0]} published applications ({n[1]} distinct)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["ptab", "oa_apps", "oa_a1"])
    ap.add_argument("--ptab-jsonl", default=f"{INPUTS}/ptab_eval_full.jsonl")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(f"{WORK}/tmp", exist_ok=True)
    {"ptab": ptab, "oa_apps": oa_apps, "oa_a1": oa_a1}[a.stage](a)


if __name__ == "__main__":
    main()
