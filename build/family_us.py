#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
"""US-only step 1: patent families from USPTO data only (no DOCDB, no EPO data).

Inputs (read only):
  $SRC/{APPXML,PTGRXML,PTGRAPS,PTGRSGM}/**/*.parquet   USPTO full-text bulk (columns doc_id, kind, doc_class, patent_type,
                                                          pub_date, app_number, filing_date, n_claims_parsed, desc_chars)
  $PV/g_us_rel_doc, pg_us_rel_doc                       PatentsView continuity (continuation, division, CIP, reissue,
                                                          substitution, provisional, related-publication, PCT parent)
  $PV/g_pct_data, pg_pct_data, g_foreign_priority, pg_foreign_priority

Method: a graph whose nodes are identifier tokens
  A:<8-digit application number> (provisionals included)   G:<patent number>   U:<11-digit pre-grant publication number>
  P:<PCT application number>                                F:<country><foreign priority number> (extended variant only)
Each corpus document joins its own pub token to its application token. Each continuity / PCT / priority row joins the
document's token to the related token. Connected components = families.
  fam_strict = continuity + provisional + PCT links
  fam_ext    = fam_strict + shared foreign priority (more conservative for train/test separation)

Outputs ($OUT): docs.parquet (one row per US document with its tokens and both family ids), family_dates.parquet
(per fam_ext: earliest valid filing/priority date and its source), family_stats.json.
"""
import argparse, glob, json, os, time

import duckdb
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

SRC = os.environ.get("PAB_US_TEXT", "src/us")          # parsed USPTO full text (see build/README.md)
PV = os.environ.get("PAB_PV", os.environ.get("PAB_WORK", "work") + "/pv")   # PatentsView tables
OUT = os.environ.get("PAB_FAMILY_OUT", os.environ.get("PAB_WORK", "work") + "/family")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--mem", default="60GB")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(f"{OUT}/../tmp", exist_ok=True)
    c = duckdb.connect()
    c.execute(f"SET threads={a.threads}; SET memory_limit='{a.mem}'; SET preserve_insertion_order=false; "
              f"SET temp_directory='{OUT}/../tmp'")

    files = sorted(glob.glob(f"{SRC}/**/*.parquet", recursive=True))
    log(f"{len(files)} source files")
    # real USPTO application series codes (01-19 utility, 29 design, 35 reissue, 60-64 provisional, 90/95 reexam); 00xxxxxx is junk
    c.execute("""CREATE MACRO valid_app(x) AS (x ~ '^(0[1-9]|1[0-9]|29|35|6[0-4]|90|95)\\d{6}$')""")
    c.execute(f"""CREATE TABLE docs AS
        SELECT doc_id, kind, doc_class, patent_type, pub_date, app_number, filing_date,
               coalesce(n_claims_parsed, 0) AS n_claims, coalesce(desc_chars, 0) AS desc_chars
        FROM read_parquet({json.dumps(files)}, union_by_name = true)
        QUALIFY row_number() OVER (PARTITION BY doc_id ORDER BY coalesce(n_claims_parsed, 0) DESC, coalesce(desc_chars, 0) DESC) = 1""")
    log("docs", c.execute("SELECT count(*), count(DISTINCT doc_id) FROM docs").fetchone())
    # tokens of each document
    c.execute(r"""CREATE TABLE doc_tok AS
        SELECT doc_id, app_number,
          CASE WHEN regexp_matches(doc_id, '^US\d{4}0\d{6}[A-Z]\d?$') THEN 'U:' || substr(doc_id, 3, 11)
               WHEN regexp_matches(doc_id, '^US\d+[A-Z]\d?$') THEN 'G:' || regexp_replace(substr(doc_id, 3), '^0+|[A-Z]\d?$', '', 'g')
               WHEN regexp_matches(doc_id, '^USRE\d+[A-Z]\d?$') THEN 'G:RE' || regexp_replace(substr(doc_id, 5), '^0+|[A-Z]\d?$', '', 'g')
               ELSE NULL END AS pub_tok,
          CASE WHEN valid_app(app_number) THEN 'A:' || app_number ELSE NULL END AS app_tok
        FROM docs""")
    log("doc tokens", c.execute("SELECT count(*), count(pub_tok), count(app_tok) FROM doc_tok").fetchone())

    # normalisers for the related-number column
    rel = r"""CASE WHEN {kind} = 'parent-pct-document' OR {num} LIKE 'PCT/%' THEN 'P:' || upper(replace({num}, ' ', ''))
                  WHEN {typ} = 'related-publication' AND {num} ~ '^\d{{11}}$' THEN 'U:' || {num}
                  WHEN {kind} = 'parent-grant-document' THEN 'G:' || regexp_replace({num}, '^0+', '')
                  WHEN valid_app({num}) THEN 'A:' || {num}
                  ELSE NULL END"""
    g = rel.format(kind="related_doc_kind", num="related_doc_number", typ="related_doc_type")
    pg = rel.format(kind="related_doc_kind", num="related_doc_number", typ="related_doc_type")
    c.execute(f"""CREATE TABLE edges_strict AS
        SELECT pub_tok AS a, app_tok AS b FROM doc_tok WHERE pub_tok IS NOT NULL AND app_tok IS NOT NULL
        UNION ALL
        SELECT 'G:' || regexp_replace(patent_id, '^0+', ''), 'A:' || application_id FROM '{PV}/g_application.parquet' WHERE valid_app(application_id)
        UNION ALL
        SELECT 'G:' || regexp_replace(patent_id, '^0+', ''), {g} FROM '{PV}/g_us_rel_doc.parquet'
        UNION ALL
        SELECT 'U:' || pgpub_id, {pg.replace('related_doc_kind', "''")} FROM '{PV}/pg_us_rel_doc.parquet'
        UNION ALL
        SELECT 'G:' || regexp_replace(patent_id, '^0+', ''), 'P:' || upper(replace(pct_doc_number, ' ', ''))
        FROM '{PV}/g_pct_data.parquet' WHERE pct_doc_number LIKE 'PCT/%'
        UNION ALL
        SELECT 'U:' || pgpub_id, 'P:' || upper(replace(pct_doc_number, ' ', ''))
        FROM '{PV}/pg_pct_data.parquet' WHERE pct_doc_number LIKE 'PCT/%'""")
    c.execute("DELETE FROM edges_strict WHERE a IS NULL OR b IS NULL")
    c.execute(f"""CREATE TABLE edges_fp AS
        SELECT 'G:' || regexp_replace(patent_id, '^0+', '') AS a,
               'F:' || foreign_country_filed || upper(regexp_replace(foreign_application_id, '[^A-Za-z0-9]', '', 'g')) AS b
        FROM '{PV}/g_foreign_priority.parquet' WHERE foreign_application_id <> '' AND foreign_country_filed <> ''
        UNION ALL
        SELECT 'U:' || pgpub_id, 'F:' || foreign_country_filed || upper(regexp_replace(foreign_application_id, '[^A-Za-z0-9]', '', 'g'))
        FROM '{PV}/pg_foreign_priority.parquet' WHERE foreign_application_id <> '' AND foreign_country_filed <> ''""")
    log("edges", c.execute("SELECT (SELECT count(*) FROM edges_strict), (SELECT count(*) FROM edges_fp)").fetchone())

    def components(name, tables):
        sel = " UNION ALL ".join(f"SELECT a, b FROM {t}" for t in tables)
        c.execute(f"CREATE TABLE e_{name} AS {sel}")
        c.execute(f"""CREATE TABLE tok_{name} AS
            SELECT tok, (row_number() OVER ())::INTEGER - 1 AS id FROM (SELECT a AS tok FROM e_{name} UNION SELECT b FROM e_{name})""")
        ea = c.execute(f"SELECT ta.id AS x, tb.id AS y FROM e_{name} e JOIN tok_{name} ta ON ta.tok = e.a JOIN tok_{name} tb ON tb.tok = e.b").fetchnumpy()
        n = c.execute(f"SELECT count(*) FROM tok_{name}").fetchone()[0]
        x, y = ea["x"], ea["y"]
        m = coo_matrix((np.ones(len(x), dtype=np.int8), (x, y)), shape=(n, n))
        k, lab = connected_components(m, directed=False)
        log(f"{name}: {n} tokens, {len(x)} edges, {k} components")
        c.execute(f"CREATE TABLE comp_{name}(id INTEGER, comp INTEGER)")
        c.register("lab_np", {"id": np.arange(n, dtype=np.int32), "comp": lab.astype(np.int32)})
        c.execute(f"INSERT INTO comp_{name} SELECT id, comp FROM lab_np")
        c.unregister("lab_np")
        # family id = smallest document-side token id would be unstable: use the component's min token string
        c.execute(f"""CREATE TABLE fam_{name} AS
            SELECT comp, min(tok) AS fam_key FROM comp_{name} JOIN tok_{name} USING (id) GROUP BY comp""")
    components("strict", ["edges_strict"])
    components("ext", ["edges_strict", "edges_fp"])

    c.execute(f"""CREATE TABLE out_docs AS
        SELECT d.*, t.pub_tok, t.app_tok, coalesce(fs.fam_key, t.pub_tok, d.doc_id) AS fam_strict, coalesce(fe.fam_key, t.pub_tok, d.doc_id) AS fam_ext
        FROM docs d JOIN doc_tok t USING (doc_id)
        LEFT JOIN tok_strict ts ON ts.tok = coalesce(t.app_tok, t.pub_tok) LEFT JOIN comp_strict cs ON cs.id = ts.id LEFT JOIN fam_strict fs ON fs.comp = cs.comp
        LEFT JOIN tok_ext te ON te.tok = coalesce(t.app_tok, t.pub_tok) LEFT JOIN comp_ext ce ON ce.id = te.id LEFT JOIN fam_ext fe ON fe.comp = ce.comp""")
    c.execute(f"COPY out_docs TO '{OUT}/docs.parquet' (FORMAT parquet, COMPRESSION zstd)")

    # family dates: earliest valid filing date of members, foreign priority dates, PCT filing dates
    c.execute(f"""CREATE TABLE dates AS
        SELECT fam_ext, filing_date AS d, 'us_filing' AS src FROM out_docs WHERE fam_ext IS NOT NULL AND filing_date BETWEEN 18360101 AND pub_date
        UNION ALL
        SELECT o.fam_ext, strftime(p.filing_date::DATE, '%Y%m%d')::INTEGER, 'foreign_priority'
        FROM '{PV}/g_foreign_priority.parquet' p JOIN out_docs o ON o.pub_tok = 'G:' || regexp_replace(p.patent_id, '^0+', '')
        WHERE try_cast(p.filing_date AS DATE) BETWEEN DATE '1836-01-01' AND DATE '2026-12-31' AND o.fam_ext IS NOT NULL
        UNION ALL
        SELECT o.fam_ext, strftime(p.filing_date::DATE, '%Y%m%d')::INTEGER, 'foreign_priority'
        FROM '{PV}/pg_foreign_priority.parquet' p JOIN out_docs o ON o.pub_tok = 'U:' || p.pgpub_id
        WHERE try_cast(p.filing_date AS DATE) BETWEEN DATE '1836-01-01' AND DATE '2026-12-31' AND o.fam_ext IS NOT NULL""")
    c.execute(f"""COPY (SELECT fam_ext, min(d) AS priority_date, arg_min(src, d) AS priority_src,
                          min(d) FILTER (WHERE src = 'us_filing') AS earliest_us_filing
                        FROM dates GROUP BY 1) TO '{OUT}/family_dates.parquet' (FORMAT parquet, COMPRESSION zstd)""")

    s = {
        "docs": c.execute("SELECT count(*) FROM out_docs").fetchone()[0],
        "docs_without_family": c.execute("SELECT count(*) FROM out_docs WHERE fam_ext IS NULL").fetchone()[0],
        "docs_by_class": c.execute("SELECT doc_class, count(*) FROM out_docs GROUP BY 1").fetchall(),
        "families_strict": c.execute("SELECT count(DISTINCT fam_strict) FROM out_docs").fetchone()[0],
        "families_ext": c.execute("SELECT count(DISTINCT fam_ext) FROM out_docs").fetchone()[0],
        "size_quantiles_strict": c.execute("""SELECT quantile_cont(n, [0.5, 0.9, 0.99, 0.999]), max(n) FROM
                                              (SELECT count(*) n FROM out_docs GROUP BY fam_strict)""").fetchone(),
        "size_quantiles_ext": c.execute("""SELECT quantile_cont(n, [0.5, 0.9, 0.99, 0.999]), max(n) FROM
                                           (SELECT count(*) n FROM out_docs GROUP BY fam_ext)""").fetchone(),
        "top10_ext_sizes": c.execute("SELECT fam_ext, count(*) n FROM out_docs GROUP BY 1 ORDER BY 2 DESC LIMIT 10").fetchall(),
        "families_with_priority_date": c.execute(f"SELECT count(*) FROM '{OUT}/family_dates.parquet'").fetchone()[0],
        "priority_src": c.execute(f"SELECT priority_src, count(*) FROM '{OUT}/family_dates.parquet' GROUP BY 1").fetchall(),
    }
    json.dump(s, open(f"{OUT}/family_stats.json", "w"), indent=1, default=str)
    log("DONE", json.dumps(s, default=str))


if __name__ == "__main__":
    main()
