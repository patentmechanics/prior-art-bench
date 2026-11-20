#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
"""Build a training pack from the data-prep outputs (the work directory of build/, see build/README.md).

  python train/pack.py --data DATA --out PACK [--sources ep_search_report=1,us_oa=1] [--max-queries 0]
                       [--negatives NEG.parquet] [--query-field tac_text] [--split train]

DATA layout: queries/queries.parquet, labels/{labels,splits,known_relevant}.parquet, tac/**/shard-*.parquet
Outputs:
  PACK/queries.parquet  query_id, source, family_id, q_text, pos_ids list<int64>, pos_grades list<int>, neg_ids list<int64>,
                        known_ids list<int64> (every cited family + the query family: never used as a negative)
  PACK/docs.parquet     family_id, text   (title+abstract+claims of every positive and listed negative)
  PACK/meta.json        counts, arguments
--max-queries N samples N queries in total, split across sources by weight, in a fixed md5 order (reproducible subsets:
the 1/3-data run is --max-queries ~800000).
"""
import argparse, glob, json, os, time

import duckdb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sources", default="ep_search_report=1,us_oa=1")
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--negatives", default=None, help="parquet: query_id, neg_ids list<int64> (e.g. from train/mine.py)")
    ap.add_argument("--max-negs", type=int, default=32)
    ap.add_argument("--query-field", default="tac_text", choices=["tac_text", "independent_claims", "claim1"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--min-grade", type=int, default=2)
    ap.add_argument("--y-indep-only", action="store_true", help="drop Y positives cited only against dependent claims (see below)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--docs-glob", default=None, help="document shards (default DATA/tac/**/shard-*.parquet)")
    ap.add_argument("--text-col", default="tac_text", help="document text column (descriptions ablation: tacd_text)")
    ap.add_argument("--ids-only", action="store_true", help="write queries.parquet + needed family ids, no document text")
    ap.add_argument("--source-caps", default=None, help="absolute caps per source, e.g. ep_search_report=100000,us_oa=100000 "
                    "(sources not listed: uncapped); overrides --max-queries")
    ap.add_argument("--memory", default="16GB", help="DuckDB memory limit")
    ap.add_argument("--query-ids", default=None, help="parquet with query_id: keep only these queries (e.g. the queries that were mined)")
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    weights = {k: float(v) for k, v in (x.split("=") for x in a.sources.split(","))}
    c = duckdb.connect()
    c.execute(f"SET threads={a.threads}; SET memory_limit='{a.memory}'")
    D = a.data
    qf = {"tac_text": "q.tac_text", "independent_claims": "concat_ws(E'\\n\\n', q.title, array_to_string(q.independent_claims, E'\\n'))",
          "claim1": "concat_ws(E'\\n\\n', q.title, q.claim1)"}[a.query_field]
    src_list = ", ".join(f"'{s}'" for s in weights)
    c.execute(f"""CREATE TABLE q AS
        SELECT q.query_id, q.source, q.family_id, {qf} AS q_text, hash(md5(q.query_id || 'pack')) AS h
        FROM '{D}/queries/queries.parquet' q JOIN '{D}/labels/splits.parquet' s USING (query_id)
        WHERE s.split = '{a.split}' AND q.source IN ({src_list}) AND {qf} IS NOT NULL""")
    if a.query_ids:
        c.execute(f"DELETE FROM q WHERE query_id NOT IN (SELECT query_id FROM '{a.query_ids}')")
    if a.source_caps:
        caps = {k: int(v) for k, v in (x.split("=") for x in a.source_caps.split(","))}
        c.execute("CREATE TABLE cap(source VARCHAR, n BIGINT)")
        c.executemany("INSERT INTO cap VALUES (?, ?)", list(caps.items()))
        c.execute("""DELETE FROM q WHERE query_id IN (
            SELECT query_id FROM (SELECT query_id, source, row_number() OVER (PARTITION BY source ORDER BY h, query_id) AS r FROM q)
            JOIN cap USING (source) WHERE r > cap.n)""")
    elif a.max_queries:
        tot = sum(weights.values())
        caps = {s: int(round(a.max_queries * w / tot)) for s, w in weights.items()}
        c.execute("CREATE TABLE cap(source VARCHAR, n BIGINT)")
        c.executemany("INSERT INTO cap VALUES (?, ?)", list(caps.items()))
        c.execute("""DELETE FROM q WHERE query_id IN (
            SELECT query_id FROM (SELECT query_id, source, row_number() OVER (PARTITION BY source ORDER BY h, query_id) AS r FROM q)
            JOIN cap USING (source) WHERE r > cap.n)""")
    lab = f"(SELECT * FROM '{D}/labels/labels.parquet' WHERE keep AND grade >= {a.min_grade} AND query_id IN (SELECT query_id FROM q))"
    if a.y_indep_only:
        # Y-cleaning (release recipe): drop Y (grade 2) positives cited only against dependent claims. Office-action
        # claim numbers often differ from the published claims (amendments: only ~72% of X citations hit a published
        # independent claim), so the rule applies only to queries whose numbering is shown consistent (>= 1 X citation hits an
        # independent claim); elsewhere every positive is kept. Dropped Y citations stay in known_relevant (never negatives).
        c.execute(f"""CREATE TABLE lab AS
            WITH qi AS (SELECT query_id, list_transform(list_filter(claims, lambda x: x.is_independent), lambda x: x.num) AS ind
                        FROM '{D}/queries/queries.parquet' WHERE query_id IN (SELECT query_id FROM q)),
                 l AS (SELECT l.*, len(list_intersect(qi.ind, list_transform(string_split(l.rel_claims, ','),
                                   lambda x: TRY_CAST(trim(x) AS INTEGER)))) > 0 AS hits_ind
                       FROM {lab} l LEFT JOIN qi USING (query_id)),
                 ok AS (SELECT query_id FROM l WHERE grade >= 3 AND hits_ind GROUP BY 1)
            SELECT * FROM l WHERE grade >= 3 OR hits_ind OR query_id NOT IN (SELECT query_id FROM ok)""")
        n0, n1 = c.execute(f"SELECT (SELECT count(*) FROM {lab}), (SELECT count(*) FROM lab)").fetchone()
        print(json.dumps({"y_indep_only": {"positives_before": n0, "positives_after": n1}}), flush=True)
        lab = "lab"
    c.execute(f"""CREATE TABLE pos AS
        SELECT query_id, list(target_family_id ORDER BY grade DESC, target_family_id) AS pos_ids,
               list(grade ORDER BY grade DESC, target_family_id) AS pos_grades
        FROM {lab} GROUP BY 1""")
    c.execute(f"""CREATE TABLE known AS
        SELECT query_id, list(DISTINCT family_id) AS known_ids FROM '{D}/labels/known_relevant.parquet'
        WHERE query_id IN (SELECT query_id FROM q) GROUP BY 1""")
    if a.negatives:
        c.execute(f"""CREATE TABLE neg AS SELECT query_id, neg_ids[1:{a.max_negs}] AS neg_ids FROM '{a.negatives}'
                      WHERE query_id IN (SELECT query_id FROM q)""")
    else:
        c.execute("CREATE TABLE neg(query_id VARCHAR, neg_ids BIGINT[])")
    c.execute("""CREATE TABLE out AS
        SELECT q.query_id, q.source, q.family_id, q.q_text, p.pos_ids, p.pos_grades, coalesce(n.neg_ids, []::BIGINT[]) AS neg_ids,
               list_distinct(list_concat(coalesce(k.known_ids, []::BIGINT[]), [q.family_id])) AS known_ids
        FROM q JOIN pos p USING (query_id) LEFT JOIN neg n USING (query_id) LEFT JOIN known k USING (query_id)
        ORDER BY q.h""")
    c.execute(f"COPY out TO '{a.out}/queries.parquet' (FORMAT parquet, COMPRESSION zstd)")
    c.execute("""CREATE TABLE need AS SELECT DISTINCT unnest(list_concat(pos_ids, neg_ids)) AS family_id FROM out""")
    if a.ids_only:
        c.execute(f"COPY (SELECT family_id FROM need ORDER BY 1) TO '{a.out}/need.parquet' (FORMAT parquet)")
        print(json.dumps({"queries": c.execute("SELECT count(*) FROM out").fetchone()[0], "need": c.execute("SELECT count(*) FROM need").fetchone()[0]}))
        return
    shards = sorted(glob.glob(a.docs_glob or f"{D}/tac/**/shard-*.parquet", recursive=True))
    # stream the document shards row group by row group (a DuckDB join over ~18 GB of description text ran out of memory)
    import pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
    need = pa.array([r[0] for r in c.execute("SELECT family_id FROM need").fetchall()], pa.int64())
    w = None
    for sh in shards:
        pf = pq.ParquetFile(sh)
        for i in range(pf.num_row_groups):
            t = pf.read_row_group(i, columns=["family_id", a.text_col])
            t = t.filter(pc.is_in(t.column("family_id"), value_set=need))
            if t.num_rows:
                t = pa.table({"family_id": t.column("family_id"), "text": t.column(a.text_col)})
                w = w or pq.ParquetWriter(f"{a.out}/docs.parquet", t.schema, compression="zstd")
                w.write_table(t)
    if w:
        w.close()
    stats = {
        "args": vars(a),
        "queries_by_source": c.execute("SELECT source, count(*), round(avg(len(pos_ids)), 2), round(avg(len(neg_ids)), 1) FROM out GROUP BY 1 ORDER BY 1").fetchall(),
        "docs": c.execute(f"SELECT count(*) FROM '{a.out}/docs.parquet'").fetchone()[0],
        "needed_docs": c.execute("SELECT count(*) FROM need").fetchone()[0],
        "seconds": round(time.time() - t0, 1),
    }
    json.dump(stats, open(f"{a.out}/meta.json", "w"), indent=1)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
