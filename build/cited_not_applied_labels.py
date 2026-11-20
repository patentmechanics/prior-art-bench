# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
"""Cited-but-never-applied labels for office-action queries of one half: PAB_WORK=$WORK python build/cited_not_applied_labels.py [dev|test]. Applied = every target family in the v2 labels
for the query (any keep/date state) + the query's own family. Cited = front page 'cited by examiner' (US patents and
applications in our corpus) of the patent granted on the same application."""
import duckdb, json, sys
PART = sys.argv[1] if len(sys.argv) > 1 else "dev"
import os
W = os.environ.get("PAB_WORK", "work"); T = f"{W}/tmp/cited_not_applied"
c = duckdb.connect(); c.execute("SET memory_limit='6GB'; SET threads=2")
# queries of this half whose application was granted under the same application number
c.execute(f"""CREATE TABLE q AS SELECT q.query_id, q.app_number, q.family_id FROM '{W}/queries/queries.parquet' q
  JOIN '{W}/labels/devtest.parquet' d USING (query_id) WHERE d.part = '{PART}' AND d.source = 'us_oa'""")
c.execute(f"""CREATE TABLE g AS SELECT q.*, d.doc_id AS grant_id, d.pub_date FROM q
  JOIN '{W}/family_map/docs.parquet' d ON d.app_number = q.app_number AND d.doc_class = 'grant'""")
c.execute(f"COPY g TO '{T}/granted_{PART}_oa.parquet'")
c.execute(f"""CREATE TABLE ex AS SELECT g.query_id, x.cited_corpus_publication_number AS doc_id, x.ref_type
  FROM read_parquet('{T}/cites/20*.parquet') x JOIN g ON x.citing_publication_number = g.grant_id
  WHERE x.cited_by = 'examiner' AND x.ref_type IN ('patent', 'application')""")
c.execute(f"CREATE TABLE cited_any AS SELECT DISTINCT citing_publication_number FROM read_parquet('{T}/cites/20*.parquet') WHERE citing_publication_number IN (SELECT grant_id FROM g)")
c.execute(f"""CREATE TABLE exf AS SELECT DISTINCT ex.query_id, d.family_id AS fam FROM ex JOIN '{W}/family_map/docs.parquet' d USING (doc_id)""")
c.execute(f"CREATE TABLE lab AS SELECT * FROM '{W}/eval_us_v2/labels/labels.parquet' WHERE query_id IN (SELECT query_id FROM g)")
c.execute(f"CREATE TABLE fam AS SELECT * FROM '{W}/eval_us_v2/family_map/families.parquet'")
c.execute("CREATE TABLE qp AS SELECT DISTINCT query_id, query_priority_date AS qpd FROM lab")
c.execute(f"CREATE TABLE pool AS SELECT family_id FROM '{W}/tac_subset_250k/families.parquet'")
c.execute("""CREATE TABLE cna AS SELECT e.query_id, e.fam, f.earliest_pd, f.in_scope, (e.fam IN (SELECT family_id FROM pool)) AS in_pool
  FROM exf e JOIN g ON g.query_id = e.query_id JOIN qp ON qp.query_id = e.query_id JOIN fam f ON f.family_id = e.fam
  WHERE e.fam <> g.family_id AND e.fam NOT IN (SELECT target_family_id FROM lab l WHERE l.query_id = e.query_id)
    AND f.earliest_pd < qp.qpd""")
c.execute(f"COPY cna TO '{T}/cited_not_applied_{PART}.parquet'")
r = {}
r["part"] = PART; r["oa_queries"] = c.execute("SELECT count(*) FROM q").fetchone()[0]
r["granted_oa_queries"] = c.execute("SELECT count(DISTINCT query_id) FROM g").fetchone()[0]
r["grants_with_citation_rows"] = c.execute("SELECT count(*) FROM cited_any").fetchone()[0]
r["queries_with_examiner_cites"] = c.execute("SELECT count(DISTINCT query_id) FROM exf").fetchone()[0]
r["examiner_cited_families"] = c.execute("SELECT count(*) FROM exf").fetchone()[0]
r["examiner_cited_that_are_applied"] = c.execute("SELECT count(*) FROM exf e WHERE e.fam IN (SELECT target_family_id FROM lab l WHERE l.query_id = e.query_id)").fetchone()[0]
r["applied_labels_scored_on_these_queries"] = c.execute("SELECT count(*) FROM lab WHERE keep AND date_ok").fetchone()[0]
r["applied_scored_also_examiner_cited"] = c.execute("SELECT count(*) FROM lab l WHERE keep AND date_ok AND target_family_id IN (SELECT fam FROM exf e WHERE e.query_id = l.query_id)").fetchone()[0]
r["cna_pairs"], r["cna_queries"], r["cna_in_scope"], r["cna_in_pool"], r["cna_distinct_families_not_in_pool"] = c.execute(
  "SELECT count(*), count(DISTINCT query_id), sum(in_scope::INT), sum(in_pool::INT), count(DISTINCT fam) FILTER (WHERE NOT in_pool) FROM cna").fetchone()
r["cna_per_query_median"] = c.execute("SELECT median(n) FROM (SELECT count(*) n FROM cna GROUP BY query_id)").fetchone()[0]
r["oa_first_action_present"] = c.execute(f"SELECT count(DISTINCT app_number) FROM '{W}/oa/oa_us.parquet' WHERE is_first_action AND app_number IN (SELECT app_number FROM g)").fetchone()[0]
print(json.dumps(r, indent=1, default=str))
