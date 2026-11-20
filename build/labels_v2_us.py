#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
"""US-only metrics v2 (USPTO / PatentsView data only, nothing from EPO or DOCDB).

  PAB_WORK=$WORK python build/labels_v2_us.py --oa-foreign $WORK/v2in/oa_foreign_cites.parquet \
      [--src $WORK/eval_us] [--out $WORK/eval_us_v2]

Rule (benchmark design, labels v2): the world is US publications. A cited
document is relevant if it is a US publication (labels v1, unchanged) or a WO publication whose US national stage is a
granted US patent (PatentsView g_pct_data, pct_doc_type = 'wo_grant'); the relevant unit is that US family. Other foreign
citations (JP, CN, EP, ...) cannot be resolved from USPTO data and stay excluded; they are counted in the report.
Grades and filters as v1: office-action first-action X = 3, Y = 2; instituted PTAB references = 3; the target must have a
US representative document, be prior art by the US family's own dates (published before the query's priority date, or
secret prior art by priority), and not be the query's own strict family. A family cited both ways keeps its highest grade.
Splits, dev/test halves and queries are unchanged (labels v2 change the evaluation only; training data keeps v1).

Output (--out, the layout of eval_us): queries/, labels/{labels (column via:
'us' | 'wo'), known_relevant, splits, devtest}.parquet, family_map/, tacd/ and tac/ = the v1 pool files (hard links)
+ part-100.parquet with the newly relevant families not yet in the pool; labels_v2_report.json (the v1 -> v2 change).
"""
import argparse, glob, json, os, re, shutil, sys, time

import duckdb
import pyarrow as pa, pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wo_cites_us import wo_key

W = os.environ.get("PAB_WORK", "work")
PTAB_JSON = os.environ.get("PAB_SRC", "src") + "/inputs/ptab_eval_full.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oa-foreign", required=True, help="build/wo_cites_us.py output")
    ap.add_argument("--src", default=f"{W}/eval_us")
    ap.add_argument("--out", default=f"{W}/eval_us_v2")
    a = ap.parse_args()
    t0 = time.time()
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    for d in ("queries", "labels", "family_map", "tacd", "tac"):
        os.makedirs(f"{a.out}/{d}")
    c = duckdb.connect()
    c.execute(f"SET threads=4; SET memory_limit='10GB'; SET temp_directory='{W}/tmp'; SET preserve_insertion_order=false")

    # WO publication -> US granted patent token (PatentsView, USPTO-sourced)
    pct = pq.read_table(f"{W}/pv/g_pct_data.parquet", columns=["patent_id", "pct_doc_number", "pct_doc_type"]).to_pylist()
    ys, ns, toks = [], [], []
    for r in pct:
        if r["pct_doc_type"] != "wo_grant":
            continue
        k = wo_key(r["pct_doc_number"])
        if k:
            ys.append(k[0]); ns.append(k[1]); toks.append("G:" + re.sub(r"^0+", "", r["patent_id"]))
    c.register("wo_map_raw", pa.table({"wo_year": pa.array(ys, pa.int32()), "wo_serial": pa.array(ns, pa.int64()), "tok": toks}))
    c.execute(f"CREATE TABLE docs AS SELECT pub_tok, family_id FROM '{W}/family_map/docs.parquet' WHERE pub_tok LIKE 'G:%'")
    c.execute("""CREATE TABLE wo_fam_all AS SELECT DISTINCT w.wo_year, w.wo_serial, d.family_id FROM wo_map_raw w JOIN docs d ON d.pub_tok = w.tok""")
    # a WO publication is one document -> one US family; if PatentsView links it to several, keep the lowest family id
    c.execute("CREATE TABLE wo_fam AS SELECT wo_year, wo_serial, min(family_id) AS family_id, count(*) AS n_fam FROM wo_fam_all GROUP BY 1, 2")

    # WO citations of the evaluation queries: office actions (wo_cites_us.py extract) + PTAB (instituted trials)
    c.execute(f"CREATE TABLE ev AS SELECT query_id, source, split FROM '{a.src}/labels/splits.parquet'")
    c.execute(f"""CREATE TABLE cites AS SELECT query_id, 'oa_' || category AS category, grade, wo_year, wo_serial, ref_country
                  FROM '{a.oa_foreign}' WHERE query_id IN (SELECT query_id FROM ev)""")
    ptab = c.execute(f"""SELECT 'ptab:' || trial, r FROM (SELECT trial, unnest(reference_numbers) AS r
                         FROM read_json_auto('{PTAB_JSON}', format='newline_delimited', maximum_object_size=200000000)
                         WHERE instituted AND len(reference_numbers) > 0) WHERE NOT starts_with(r, 'US')""").fetchall()
    keys = [wo_key(r) if r.startswith("WO") else None for _, r in ptab]
    c.register("ptab_raw", pa.table({"query_id": [q for q, _ in ptab], "category": ["ptab"] * len(ptab), "grade": pa.array([3] * len(ptab), pa.int32()),
                                     "wo_year": pa.array([k[0] if k else None for k in keys], pa.int32()),
                                     "wo_serial": pa.array([k[1] if k else None for k in keys], pa.int64()),
                                     "ref_country": [r[:2] for _, r in ptab]}))
    c.execute("INSERT INTO cites SELECT * FROM ptab_raw WHERE query_id IN (SELECT query_id FROM ev)")

    # resolve and apply the v1 rules
    c.execute(f"CREATE TABLE fam AS SELECT family_id, strict_id, priority_date, earliest_pd FROM '{W}/family_map/families.parquet'")
    c.execute(f"CREATE TABLE scope AS SELECT DISTINCT family_id FROM '{W}/reps/reps_us.parquet'")
    c.execute(f"CREATE TABLE qk AS SELECT query_id, source, family_id, strict_id, priority_date FROM '{a.src}/queries/queries.parquet'")
    c.execute("""CREATE TABLE lab2 AS
        SELECT x.query_id, w.family_id AS target_family_id, max(x.grade) AS grade, string_agg(DISTINCT x.category, '|' ORDER BY x.category) AS category
        FROM cites x JOIN wo_fam w USING (wo_year, wo_serial) GROUP BY ALL""")
    c.execute("""CREATE TABLE v2 AS
        SELECT l.query_id, l.target_family_id, l.grade, l.category, NULL::VARCHAR AS rel_claims, qk.source, qk.family_id AS query_family_id,
               qk.strict_id AS query_strict_id, tf.strict_id AS target_strict_id,
               (l.target_family_id IN (SELECT family_id FROM scope)) AS target_in_scope, tf.earliest_pd AS target_earliest_pd,
               qk.priority_date AS query_priority_date, tf.priority_date AS target_priority_date,
               (tf.earliest_pd IS NOT NULL AND qk.priority_date IS NOT NULL AND tf.earliest_pd < qk.priority_date) AS date_ok
        FROM lab2 l JOIN qk USING (query_id) LEFT JOIN fam tf ON tf.family_id = l.target_family_id""")
    c.execute("""ALTER TABLE v2 ADD COLUMN secret_prior_art BOOLEAN; ALTER TABLE v2 ADD COLUMN keep BOOLEAN;
                 UPDATE v2 SET secret_prior_art = NOT coalesce(date_ok, false) AND target_priority_date IS NOT NULL
                               AND query_priority_date IS NOT NULL AND target_priority_date < query_priority_date;
                 UPDATE v2 SET keep = grade >= 2 AND target_in_scope AND (date_ok OR secret_prior_art)
                                     AND target_strict_id IS DISTINCT FROM query_strict_id""")
    c.execute(f"CREATE TABLE v1 AS SELECT *, 'us' AS via FROM '{a.src}/labels/labels.parquet'")
    cols = [r[0] for r in c.execute("DESCRIBE v1").fetchall()]
    sel = ", ".join(cols[:-1])
    # union; per (query, family) keep the row with the highest grade, v1 first on ties
    c.execute(f"""CREATE TABLE merged AS
        SELECT {sel}, via FROM (
            SELECT *, row_number() OVER (PARTITION BY query_id, target_family_id ORDER BY grade DESC, (via = 'us') DESC) AS rn
            FROM (SELECT * FROM v1 UNION ALL BY NAME SELECT *, 'wo' AS via FROM v2)) WHERE rn = 1""")
    c.execute(f"COPY merged TO '{a.out}/labels/labels.parquet' (FORMAT parquet, COMPRESSION zstd)")
    c.execute(f"""COPY (SELECT DISTINCT query_id, family_id FROM '{a.src}/labels/known_relevant.parquet'
                        UNION SELECT DISTINCT query_id, target_family_id FROM v2) TO '{a.out}/labels/known_relevant.parquet' (FORMAT parquet)""")
    for f in ("splits", "devtest"):
        shutil.copy(f"{a.src}/labels/{f}.parquet", f"{a.out}/labels/{f}.parquet")
    shutil.copy(f"{a.src}/queries/queries.parquet", f"{a.out}/queries/queries.parquet")
    shutil.copy(f"{a.src}/family_map/families.parquet", f"{a.out}/family_map/families.parquet")

    # pool: v1 files (hard links) + one file with the newly relevant families (all v2 targets not in the pool)
    for d in ("tacd", "tac"):
        for f in sorted(glob.glob(f"{a.src}/{d}/part-*.parquet")):
            os.link(f, f"{a.out}/{d}/{os.path.basename(f)}")
    c.execute(f"CREATE TABLE pool AS SELECT DISTINCT family_id FROM '{a.src}/tacd/part-*.parquet'")
    c.execute("""CREATE TABLE add_f AS SELECT DISTINCT v2.target_family_id AS family_id FROM v2
                 ANTI JOIN pool ON pool.family_id = v2.target_family_id WHERE v2.target_in_scope""")
    n_add = c.execute("SELECT count(*) FROM add_f").fetchone()[0]
    nxt = len(glob.glob(f"{a.src}/tacd/part-*.parquet"))
    if n_add:
        # description text with the pool's own builder (build/tacd.py: same passages, sections and 45,000-character cap);
        # only the corpus files holding the added families' representative documents are read
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prep"))
        from tacd import one as tacd_one
        tmp = f"{a.out}/_tacd_tmp"
        os.makedirs(f"{tmp}/parts")
        c.execute(f"COPY add_f TO '{tmp}/need.parquet' (FORMAT parquet)")
        files = [f for (f,) in c.execute(f"SELECT DISTINCT replace(file, '/', '__') FROM '{W}/reps/reps_us.parquet' SEMI JOIN add_f USING (family_id)").fetchall()]
        for f in files:
            tacd_one((f"{W}/corpus/us/{f}", [f"{tmp}/need.parquet"], tmp, 45000))
        c.execute(f"""COPY (SELECT family_id, tacd_text FROM '{tmp}/parts/*.parquet' ORDER BY family_id)
                      TO '{a.out}/tacd/part-{nxt:03d}.parquet' (FORMAT parquet, COMPRESSION zstd)""")
        shutil.rmtree(tmp)
        c.execute(f"""COPY (SELECT t.family_id, t.tac_text FROM '{W}/tac/shard-*.parquet' t SEMI JOIN add_f USING (family_id) ORDER BY family_id)
                      TO '{a.out}/tac/part-{nxt:03d}.parquet' (FORMAT parquet, COMPRESSION zstd)""")
    got_d = c.execute(f"SELECT count(DISTINCT family_id) FROM '{a.out}/tacd/*.parquet'").fetchone()[0]
    got_t = c.execute(f"SELECT count(DISTINCT family_id) FROM '{a.out}/tac/*.parquet'").fetchone()[0]

    # report: v1 -> v2 by source and half (dev/test), reachable (keep AND date_ok) as the metrics count them
    c.execute(f"CREATE TABLE dt AS SELECT query_id, part FROM '{a.out}/labels/devtest.parquet'")
    def per(tbl):
        return c.execute(f"""SELECT e.source, coalesce(d.part, '-') AS part, count(DISTINCT l.query_id) AS queries, count(*) AS relevant
                             FROM {tbl} l JOIN ev e USING (query_id) LEFT JOIN dt d USING (query_id)
                             WHERE l.keep AND l.date_ok AND e.split IN ('gold', 'eval_ptab') GROUP BY 1, 2 ORDER BY 1, 2""").fetchall()
    rep = {
        "rule": "US publications + WO publications resolved to granted US patents (PatentsView wo_grant); other foreign excluded",
        "wo_map": c.execute("SELECT count(*), count_if(n_fam > 1) FROM wo_fam").fetchone(),
        "foreign_citations_by_country_and_source": c.execute(
            "SELECT ref_country, starts_with(query_id, 'ptab:') AS ptab, count(*) FROM cites GROUP BY 1, 2 ORDER BY 3 DESC").fetchall(),
        "wo_citations": c.execute("SELECT starts_with(query_id, 'ptab:'), count(*), count(wo_serial) FROM cites WHERE ref_country = 'WO' GROUP BY 1").fetchall(),
        "wo_resolved_to_us_family": c.execute("SELECT starts_with(x.query_id, 'ptab:'), count(*) FROM cites x JOIN wo_fam USING (wo_year, wo_serial) GROUP BY 1").fetchall(),
        "v2_label_rows": c.execute("SELECT source, count(*), count_if(keep), count_if(keep AND date_ok) FROM v2 GROUP BY 1").fetchall(),
        "v2_rows_already_in_v1": c.execute("SELECT count(*) FROM v2 SEMI JOIN v1 USING (query_id, target_family_id)").fetchone()[0],
        "scored_v1": per("v1"), "scored_v2": per("merged"),
        "pool_added_families": n_add, "pool_families_tacd": got_d, "pool_families_tac": got_t,
        "seconds": round(time.time() - t0)}
    json.dump(rep, open(f"{a.out}/labels_v2_report.json", "w"), indent=1, default=str)
    print(json.dumps(rep, default=str))


if __name__ == "__main__":
    main()
