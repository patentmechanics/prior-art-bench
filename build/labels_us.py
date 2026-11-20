#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
"""US-only step 3: queries, labels and splits (USPTO data only; no DOCDB, EPO or EP-derived input).

Inputs (all read only): family_map/{docs,families}.parquet (select_us.py), reps/reps_us.parquet, oa/oa_us.parquet (oa_slim.py),
  queries_oa/oa_a1.parquet (published-application claims of every office-action application, from the USPTO bulk files),
  queries/ptab_docs.parquet + inputs/ptab_eval_full.jsonl (granted claims of PTAB-challenged patents; PTAB API references).
Outputs under $OUT, in the layout train/pack.py and train/evaluate.py already read:
  queries/queries.parquet  query_id, source (us_oa | ptab), family_id (fam_ext, dense id), strict_id, priority_date, title, abstract, claims,
                           independent_claims, claim1, tac_text, ...
  labels/labels.parquet, labels/known_relevant.parquet, labels/splits.parquet, labels/gold_families.parquet, labels/stats.json
Label rules (same as the US+EP pipeline, re-implemented on USPTO identifiers):
  first-action X = grade 3, Y = grade 2; later actions and category A = grade 0 (known-relevant only, never a negative);
  PTAB references of instituted trials = grade 3. keep = grade >= 2, target has a US representative document, target published
  before the query's priority date or filed before it (US 102(e) secret prior art, from our own family priority dates), and the
  target is not in the query's own (strict) family.
Splits: new salt, by extended family. Priority date >= 2024-01-01 -> holdout_time; ~5,000 gold and ~5,000 dev office-action queries;
  all else train unless the query's family is an evaluation family. PTAB -> eval_ptab.
"""
import argparse, json, os, time
import duckdb

OUT = os.environ.get("PAB_WORK", "work")
SALT = "sf-us-v1"
CLAIM_STRUCT = 'STRUCT(num INTEGER, "text" VARCHAR, is_independent BOOLEAN, depends_on INTEGER[])[]'
NUM_RE = r"\b(US|EP|WO|JP|DE|GB|FR|CN|KR)[ -]?\d{2,4}[/ ,.]?\d{3,}"
OA_A1 = os.environ.get("PAB_SRC", "src") + "/queries_oa/oa_a1.parquet"          # build/queries_us.py oa_a1
PTAB_DOCS = os.environ.get("PAB_SRC", "src") + "/queries/ptab_docs.parquet"     # build/queries_us.py ptab
PTAB_JSON = os.environ.get("PAB_SRC", "src") + "/inputs/ptab_eval_full.jsonl"   # PTAB trials + references (see build/README.md)


def log(msg):
    os.makedirs(f"{OUT}/logs", exist_ok=True)
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    open(f"{OUT}/logs/labels_us.log", "a").write(line + "\n")


def tok_sql(col):
    """US reference number (DOCDB short form, as in the PTAB and office-action data) -> graph token."""
    return (f"CASE WHEN regexp_matches({col}, '^US\\d{{4}}\\d{{6}}$') THEN 'U:' || substr({col}, 3, 4) || '0' || substr({col}, 7) "
            f"WHEN regexp_matches({col}, '^US\\d{{7,8}}$') THEN 'G:' || regexp_replace(substr({col}, 3), '^0+', '') "
            f"WHEN regexp_matches({col}, '^USRE\\d+$') THEN 'G:RE' || regexp_replace(substr({col}, 5), '^0+', '') ELSE NULL END")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=int, default=5000)
    ap.add_argument("--dev", type=int, default=5000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--mem", default="60GB")
    a = ap.parse_args()
    for d in ("queries", "labels", "tmp"):
        os.makedirs(f"{OUT}/{d}", exist_ok=True)
    c = duckdb.connect()
    c.execute(f"SET threads={a.threads}; SET memory_limit='{a.mem}'; SET temp_directory='{OUT}/tmp'; SET preserve_insertion_order=false")
    c.execute(f"CREATE TABLE docs AS SELECT doc_id, pub_tok, app_tok, family_id, strict_id, pub_date FROM '{OUT}/family_map/docs.parquet'")
    c.execute(f"CREATE TABLE fam AS SELECT family_id, strict_id, priority_date, earliest_pd FROM '{OUT}/family_map/families.parquet'")
    c.execute(f"CREATE TABLE scope AS SELECT DISTINCT family_id FROM '{OUT}/reps/reps_us.parquet'")
    c.execute("CREATE TABLE tokfam AS SELECT pub_tok AS tok, family_id FROM docs WHERE pub_tok IS NOT NULL")
    log("start")

    # ---- queries: office-action applications (published-application claims) and PTAB challenged patents
    c.execute(f"""CREATE TABLE q_oa AS
        SELECT 'us_oa:' || o.app_number AS query_id, 'us_oa' AS source, o.app_number, o.doc_id AS query_doc_id, d.family_id, d.strict_id,
               o.title, o.abstract, o.claims::{CLAIM_STRUCT} AS claims, 'us_published_application' AS claims_source
        FROM '{OA_A1}' o JOIN docs d ON d.doc_id = o.doc_id""")
    c.execute(f"""CREATE TABLE q_ptab AS
        WITH t AS (SELECT trial, regexp_replace(challenged_patent, '^US0*', 'US') AS pn_nokind
                   FROM read_json_auto('{PTAB_JSON}', format='newline_delimited', maximum_object_size=200000000)
                   WHERE instituted AND len(reference_numbers) > 0)
        SELECT 'ptab:' || t.trial AS query_id, 'ptab' AS source, NULL AS app_number, p.doc_id AS query_doc_id, d.family_id, d.strict_id,
               p.title, p.abstract, p.claims::{CLAIM_STRUCT} AS claims, 'ptab_granted' AS claims_source
        FROM t JOIN '{PTAB_DOCS}' p USING (pn_nokind) LEFT JOIN docs d ON d.doc_id = p.doc_id""")
    c.execute("CREATE TABLE q AS SELECT * FROM q_oa UNION ALL BY NAME SELECT * FROM q_ptab")
    c.execute(f"""CREATE TABLE qout AS
        SELECT q.*, f.priority_date,
               list_transform(list_filter(claims, lambda x: x.is_independent), lambda x: x.text) AS independent_claims,
               list_filter(claims, lambda x: x.num = 1)[1].text AS claim1,
               concat_ws(E'\\n\\n', title, abstract, array_to_string(list_transform(claims, lambda x: x.num || '. ' || x.text), E'\\n')) AS tac_text,
               regexp_matches(coalesce(array_to_string(list_transform(claims, lambda x: x.text), ' '), ''), '{NUM_RE}') AS claims_mention_patent_number,
               len(claims) AS n_claims
        FROM q LEFT JOIN fam f USING (family_id) WHERE len(claims) > 0""")
    c.execute(f"COPY qout TO '{OUT}/queries/queries.parquet' (FORMAT parquet, COMPRESSION zstd)")
    log("queries " + json.dumps(c.execute("SELECT source, count(*), count(family_id), count(priority_date) FROM qout GROUP BY 1").fetchall()))
    c.execute("CREATE TABLE qk AS SELECT query_id, source, family_id, strict_id, priority_date FROM qout")

    # ---- raw labels
    c.execute(f"""CREATE TABLE raw AS
        SELECT 'us_oa:' || o.app_number AS query_id, o.tok,
               CASE WHEN o.is_first_action THEN 'oa_' ELSE 'oa_later_' END || o.category AS category, o.related_claims AS rel_claims,
               CASE WHEN NOT o.is_first_action THEN 0 WHEN o.category = 'X' THEN 3 WHEN o.category = 'Y' THEN 2 ELSE 0 END AS grade
        FROM '{OUT}/oa/oa_us.parquet' o WHERE o.tok IS NOT NULL""")
    c.execute(f"""INSERT INTO raw
        SELECT 'ptab:' || j.trial, {tok_sql('r')}, 'ptab', NULL, 3
        FROM (SELECT trial, unnest(reference_numbers) AS r
              FROM read_json_auto('{PTAB_JSON}', format='newline_delimited', maximum_object_size=200000000)
              WHERE instituted AND len(reference_numbers) > 0) j
        WHERE {tok_sql('r')} IS NOT NULL""")
    c.execute("""CREATE TABLE lab AS
        SELECT r.query_id, t.family_id AS target_family_id, max(r.grade) AS grade, string_agg(DISTINCT r.category, '|' ORDER BY r.category) AS category,
               string_agg(DISTINCT r.rel_claims, '|' ORDER BY r.rel_claims) AS rel_claims
        FROM raw r JOIN tokfam t ON t.tok = r.tok JOIN qk ON qk.query_id = r.query_id GROUP BY ALL""")
    c.execute("""CREATE TABLE labels AS
        SELECT l.*, qk.source, qk.family_id AS query_family_id, qk.strict_id AS query_strict_id, tf.strict_id AS target_strict_id,
               (l.target_family_id IN (SELECT family_id FROM scope)) AS target_in_scope,
               tf.earliest_pd AS target_earliest_pd, qk.priority_date AS query_priority_date, tf.priority_date AS target_priority_date,
               (tf.earliest_pd IS NOT NULL AND qk.priority_date IS NOT NULL AND tf.earliest_pd < qk.priority_date) AS date_ok
        FROM lab l JOIN qk USING (query_id) LEFT JOIN fam tf ON tf.family_id = l.target_family_id""")
    c.execute("""ALTER TABLE labels ADD COLUMN secret_prior_art BOOLEAN; ALTER TABLE labels ADD COLUMN keep BOOLEAN;
                 UPDATE labels SET secret_prior_art = NOT coalesce(date_ok, false) AND target_priority_date IS NOT NULL
                                   AND query_priority_date IS NOT NULL AND target_priority_date < query_priority_date;
                 UPDATE labels SET keep = grade >= 2 AND target_in_scope AND (date_ok OR secret_prior_art)
                                         AND target_strict_id IS DISTINCT FROM query_strict_id""")
    c.execute(f"COPY labels TO '{OUT}/labels/labels.parquet' (FORMAT parquet, COMPRESSION zstd)")
    c.execute(f"COPY (SELECT DISTINCT query_id, target_family_id AS family_id FROM labels) TO '{OUT}/labels/known_relevant.parquet' (FORMAT parquet)")
    log("labels " + json.dumps(c.execute("SELECT source, grade, count(*), count_if(keep) FROM labels GROUP BY ALL ORDER BY ALL").fetchall()))

    # ---- splits
    c.execute("CREATE TABLE npos AS SELECT query_id, count_if(keep) AS n_pos FROM labels GROUP BY 1")
    c.execute(f"""CREATE TABLE sp AS
        SELECT q.query_id, q.source, q.family_id, coalesce(n.n_pos, 0) AS n_pos, q.priority_date, hash(md5(q.family_id::VARCHAR || '{SALT}')) AS h
        FROM qk q LEFT JOIN npos n USING (query_id)""")
    c.execute("""ALTER TABLE sp ADD COLUMN split VARCHAR;
        UPDATE sp SET split = 'eval_ptab' WHERE source = 'ptab';
        UPDATE sp SET split = 'excluded_no_positive' WHERE source = 'us_oa' AND n_pos = 0;
        UPDATE sp SET split = 'excluded_no_family_or_date' WHERE split IS NULL AND (family_id IS NULL OR priority_date IS NULL);
        UPDATE sp SET split = 'holdout_time' WHERE split IS NULL AND priority_date >= 20240101""")
    c.execute("CREATE TABLE ordfam AS SELECT source, family_id, min(h) AS h, count(*) AS n FROM sp WHERE split IS NULL GROUP BY 1, 2")
    c.execute(f"""CREATE TABLE pick AS
        SELECT source, family_id, CASE WHEN cum <= {a.gold} THEN 'gold' WHEN cum <= {a.gold + a.dev} THEN 'dev' END AS split
        FROM (SELECT source, family_id, sum(n) OVER (PARTITION BY source ORDER BY h, family_id) AS cum FROM ordfam)""")
    c.execute("""UPDATE sp SET split = pick.split FROM pick WHERE sp.split IS NULL AND sp.source = pick.source
                 AND sp.family_id = pick.family_id AND pick.split IS NOT NULL""")
    c.execute("UPDATE sp SET split = 'train' WHERE split IS NULL")
    c.execute("""CREATE TABLE gold_fam AS SELECT DISTINCT family_id FROM sp
                 WHERE family_id IS NOT NULL AND (split LIKE 'eval_%' OR split IN ('gold', 'dev', 'holdout_time'))""")
    c.execute("UPDATE sp SET split = 'excluded_eval_family' WHERE split = 'train' AND family_id IN (SELECT family_id FROM gold_fam)")
    c.execute(f"COPY (SELECT query_id, source, family_id, split, n_pos, priority_date FROM sp) TO '{OUT}/labels/splits.parquet' (FORMAT parquet)")
    c.execute(f"COPY gold_fam TO '{OUT}/labels/gold_families.parquet' (FORMAT parquet)")
    stats = {
        "labels_by_source_grade": c.execute("SELECT source, grade, count(*), count_if(keep) FROM labels GROUP BY ALL ORDER BY ALL").fetchall(),
        "drop_reasons_grade_ge2": c.execute("""SELECT source, count(*), count_if(NOT target_in_scope), count_if(target_in_scope AND NOT coalesce(date_ok, false) AND NOT coalesce(secret_prior_art, false)),
                                                  count_if(target_strict_id = query_strict_id), count_if(keep AND secret_prior_art)
                                               FROM labels WHERE grade >= 2 GROUP BY 1 ORDER BY 1""").fetchall(),
        "splits": c.execute("SELECT source, split, count(*), sum(n_pos), round(avg(n_pos), 2) FROM sp GROUP BY 1, 2 ORDER BY 1, 2").fetchall(),
        "train_eval_family_overlap": c.execute("SELECT count(*) FROM sp WHERE split = 'train' AND family_id IN (SELECT family_id FROM gold_fam)").fetchone()[0],
        "ptab_trials_with_query": c.execute("SELECT count(*) FROM qk WHERE source = 'ptab'").fetchone()[0],
        "ptab_refs_total_us_resolved": c.execute("SELECT count(*), count_if(keep) FROM labels WHERE source = 'ptab'").fetchone(),
    }
    json.dump(stats, open(f"{OUT}/labels/stats.json", "w"), indent=1, default=str)
    log("DONE " + json.dumps(stats, default=str))


if __name__ == "__main__":
    main()
