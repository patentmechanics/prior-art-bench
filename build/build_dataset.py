#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
"""Build the Hugging Face dataset Patent Mechanics Prior-Art Bench v1 (patentmechanics/prior-art-bench-v1) from the work
directory of the earlier build steps. Text and labels only; run files and vectors were added separately.

  PAB_WORK=$WORK python build/build_dataset.py --out $WORK/release_v1/dataset

Configs written (parquet, zstd), all keyed by our family_id (USPTO-identifier families, see the dataset card):
  queries/            11,968 evaluation queries: id, source, dev/test, publication, dates, title, abstract, claims, text, CPC
  qrels/              labels v2 with grades and the flags the scorer uses (keep AND date_ok)
  corpus-pool-v2/     250,367 families: representative publication, dates, title, abstract, claims, description
                      paragraphs (with section labels), the exact `text` the model encoded, printed and current CPC
  corpus-pool-v2-chunks/  1,908,369 description chunks in the order of the shipped chunk vectors (family_id, chunk_index)
  cpc-members/        CPC of every family member published (used by the CPC-strict stage), 589,887 rows
  families/           9.44M families: dates, size, representative publication
  family-members/     17.47M US publications -> family_id (the full corpus as identifiers)
  train-qrels/        positives of the release model's training queries (IDs and grades; no negatives, A37)
  sample/             1,000 dev queries, their qrels, and a 10,000-family collection (relevant + random distractors)
  canary/             sha256 of every test query publication / application and of every publication in a test query family
"""
import argparse, glob, hashlib, json, os, resource, time
from concurrent.futures import ProcessPoolExecutor

import duckdb

WORK = os.environ.get("PAB_WORK", "work")
EV = f"{WORK}/eval_us_v2"
SALT = "pm-prior-art-bench-v1"


def con(mem="6GB", threads=4):
    c = duckdb.connect()
    c.execute(f"SET threads={threads}; SET memory_limit='{mem}'; SET preserve_insertion_order=true; SET temp_directory='{WORK}/tmp'")
    return c


def copy(c, sql, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    n = c.execute(f"COPY ({sql}) TO '{dst}.tmp' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 50000)").fetchone()[0]
    os.replace(f"{dst}.tmp", dst)
    return n


def corpus_part(args):
    """Representative-document rows of the pool families found in one corpus source file."""
    part, need, dst = args
    if os.path.exists(dst):
        return 0
    c = con("3GB", 1)
    n = c.execute(f"""COPY (SELECT family_id, doc_id, kind, doc_class, pub_date::INTEGER AS publication_date,
                                    filing_date::INTEGER AS filing_date, app_number, title, abstract, claims,
                                    paragraphs AS description
                             FROM '{part}' WHERE family_id IN (SELECT family_id FROM '{need}'))
                      TO '{dst}.tmp' (FORMAT parquet, COMPRESSION zstd)""").fetchone()[0]
    os.replace(f"{dst}.tmp", dst)
    return n


def main():
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)   # thousands of corpus part files (2026-11-15: "Too many open files")
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, 65536), hard))
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    O, T = a.out, f"{a.out}/../tmp"
    os.makedirs(T, exist_ok=True)
    t0 = time.time()
    c = con("8GB", 4)
    st = {}
    # pool families = the evaluated collection (the files the model encoded)
    c.execute(f"CREATE VIEW pool AS SELECT family_id, tacd_text AS text FROM '{EV}/tacd/*.parquet'")
    c.execute(f"COPY (SELECT DISTINCT family_id FROM pool) TO '{T}/pool_ids.parquet' (FORMAT parquet)")
    st["pool_families"] = c.execute("SELECT count(DISTINCT family_id) FROM pool").fetchone()[0]

    # --- queries (with the dev/test half and the query's printed / current CPC)
    st["queries"] = copy(c, f"""
        SELECT q.query_id, q.source, d.part AS split, q.family_id, q.query_doc_id AS publication_number, q.app_number,
               q.priority_date, q.title, q.abstract, q.claims, q.tac_text AS text,
               cs.cpc_printed, cs.cpc_current
        FROM '{EV}/queries/queries.parquet' q JOIN '{EV}/labels/devtest.parquet' d USING (query_id)
        LEFT JOIN '{WORK}/eval_us_v2_cpc/query_cpc_strict.parquet' cs USING (query_id)
        ORDER BY q.query_id""", f"{O}/queries/queries.parquet")

    # --- qrels: every label row with the flags the scorer applies (keep AND date_ok)
    st["qrels"] = copy(c, f"""
        SELECT query_id, target_family_id AS family_id, grade, category, source, rel_claims, via,
               target_earliest_pd AS family_earliest_publication_date, date_ok, secret_prior_art, keep
        FROM '{EV}/labels/labels.parquet' ORDER BY query_id, family_id""", f"{O}/qrels/qrels.parquet")

    # --- corpus: representative rows from the parsed full text (parallel over source files that hold pool families)
    files = [r[0] for r in c.execute(f"""SELECT DISTINCT file FROM '{WORK}/reps/reps_us.parquet'
                                         WHERE family_id IN (SELECT family_id FROM '{T}/pool_ids.parquet')""").fetchall()]
    parts = sorted(set(f"{WORK}/corpus/us/{os.path.basename(f).replace('.parquet', '')}.parquet" for f in files))
    parts = [p for p in parts if os.path.exists(p)] or sorted(glob.glob(f"{WORK}/corpus/us/*.parquet"))
    os.makedirs(f"{T}/corpus_parts", exist_ok=True)
    with ProcessPoolExecutor(a.workers) as ex:
        got = sum(ex.map(corpus_part, [(p, f"{T}/pool_ids.parquet", f"{T}/corpus_parts/{os.path.basename(p)}") for p in parts]))
    st["corpus_rows_found"] = got
    import pyarrow.parquet as pq
    nonempty = [p for p in sorted(glob.glob(f"{T}/corpus_parts/*.parquet")) if pq.ParquetFile(p).metadata.num_rows > 0]
    st["corpus_part_files"] = len(nonempty)
    fam = f"{WORK}/family_map/families.parquet"
    # 25 family-hash shards: one query over all parts with nested descriptions ran out of memory (2026-11-15)
    c.execute("SET threads=2; SET memory_limit='12GB'; SET preserve_insertion_order=false")
    parts_sql = f"read_parquet({json.dumps(nonempty)})"
    st["corpus"], nsh = 0, 25
    os.makedirs(f"{O}/corpus-pool-v2", exist_ok=True)
    for i in range(nsh):
        st["corpus"] += copy(c, f"""
            SELECT r.family_id, r.doc_id AS publication_number, r.kind, r.doc_class, r.publication_date, r.filing_date, r.app_number,
                   f.earliest_pd AS family_earliest_publication_date, f.priority_date AS family_priority_date,
                   r.title, r.abstract, r.claims, r.description, p.text, dc.cpc_printed, dc.cpc_current
            FROM (SELECT * FROM pool WHERE family_id % {nsh} = {i}) p
            JOIN (SELECT * FROM {parts_sql} WHERE family_id % {nsh} = {i}) r USING (family_id)
            LEFT JOIN '{fam}' f USING (family_id)
            LEFT JOIN '{WORK}/eval_us_v2_cpc/doc_cpc.parquet' dc ON dc.family_id = r.family_id AND dc.doc_id = r.doc_id
            ORDER BY r.family_id""", f"{O}/corpus-pool-v2/part-{i:03d}.parquet")
    c.execute("SET threads=4; SET memory_limit='8GB'; SET preserve_insertion_order=true")

    # --- chunks, in the order of the chunk files (= the order of the shipped chunk vectors)
    pas = sorted(glob.glob(f"{WORK}/eval_us_v2_pas/part-*.parquet"))
    n = 0
    for p in pas:
        n += copy(c, f"""SELECT family_id, (row_number() OVER (PARTITION BY family_id ORDER BY file_row_number) - 1)::INTEGER AS chunk_index,
                                pas_text AS text, '{os.path.basename(p)[:-8]}' AS vector_file, file_row_number::INTEGER AS vector_row
                         FROM read_parquet('{p}', file_row_number=true) ORDER BY file_row_number""",
                  f"{O}/corpus-pool-v2-chunks/{os.path.basename(p)}")
    st["chunks"] = n
    st["chunk_families_split_across_files"] = c.execute(f"""SELECT count(*) FROM (SELECT family_id FROM read_parquet('{WORK}/eval_us_v2_pas/part-*.parquet', filename=true)
                                                            GROUP BY 1 HAVING count(DISTINCT filename) > 1)""").fetchone()[0]

    # --- CPC of family members (CPC-strict stage)
    st["cpc_members"] = copy(c, f"SELECT * FROM '{WORK}/eval_us_v2_cpc/doc_cpc.parquet' ORDER BY family_id, doc_id",
                             f"{O}/cpc-members/cpc_members.parquet")

    # --- families and members (full corpus as identifiers)
    c.execute(f"CREATE TABLE fam AS SELECT f.family_id, f.earliest_pd AS earliest_publication_date, f.priority_date, f.n_docs, "
              f"r.doc_id AS representative_publication_number FROM '{fam}' f LEFT JOIN '{WORK}/reps/reps_us.parquet' r USING (family_id)")
    st["families"] = 0
    for i in range(4):
        st["families"] += copy(c, f"SELECT * FROM fam WHERE family_id % 4 = {i} ORDER BY family_id", f"{O}/families/part-{i}.parquet")
    c.execute("DROP TABLE fam")
    st["family_members"] = 0
    for i in range(8):
        st["family_members"] += copy(c, f"""SELECT doc_id AS publication_number, family_id, kind, doc_class, pub_date AS publication_date,
                                             app_number, filing_date FROM '{WORK}/family_map/docs.parquet' WHERE family_id % 8 = {i}
                                             ORDER BY family_id, doc_id""", f"{O}/family-members/part-{i}.parquet")

    # --- training positives of the release model (IDs and grades only)
    st["train_qrels"] = copy(c, f"""SELECT query_id, family_id AS query_family_id, unnest(pos_ids) AS family_id, unnest(pos_grades) AS grade
                                    FROM '{WORK}/packs/pack_us_ha/queries.parquet' ORDER BY query_id""", f"{O}/train-qrels/train_qrels.parquet")

    # --- sample: 1,000 dev queries (salted hash), their scored qrels, collection = their relevant pool families + distractors
    c.execute(f"""CREATE TABLE sq AS SELECT query_id FROM '{O}/queries/queries.parquet' WHERE split = 'dev'
                  ORDER BY md5('{SALT}' || query_id) LIMIT 1000""")
    copy(c, f"SELECT * FROM '{O}/queries/queries.parquet' WHERE query_id IN (SELECT query_id FROM sq) ORDER BY query_id",
         f"{O}/sample/queries.parquet")
    copy(c, f"SELECT * FROM '{O}/qrels/qrels.parquet' WHERE query_id IN (SELECT query_id FROM sq) ORDER BY query_id, family_id",
         f"{O}/sample/qrels.parquet")
    c.execute(f"""CREATE TABLE sf AS SELECT DISTINCT family_id FROM '{O}/qrels/qrels.parquet'
                  WHERE query_id IN (SELECT query_id FROM sq) AND family_id IN (SELECT family_id FROM '{T}/pool_ids.parquet')""")
    nrel = c.execute("SELECT count(*) FROM sf").fetchone()[0]
    c.execute(f"""INSERT INTO sf SELECT family_id FROM '{T}/pool_ids.parquet' WHERE family_id NOT IN (SELECT family_id FROM sf)
                  ORDER BY md5('{SALT}' || family_id::VARCHAR) LIMIT {max(0, 10000 - nrel)}""")
    st["sample"] = {"queries": 1000, "relevant_families": nrel,
                    "collection": copy(c, f"SELECT * FROM '{O}/corpus-pool-v2/*.parquet' WHERE family_id IN (SELECT family_id FROM sf) ORDER BY family_id",
                                       f"{O}/sample/corpus.parquet")}

    # --- canary: sha256 of upper-case publication / application numbers, test half only
    os.makedirs(f"{O}/canary", exist_ok=True)
    rows = c.execute(f"""SELECT DISTINCT upper(x) FROM (
            SELECT publication_number AS x FROM '{O}/queries/queries.parquet' WHERE split = 'test'
            UNION ALL SELECT app_number FROM '{O}/queries/queries.parquet' WHERE split = 'test' AND app_number IS NOT NULL
            UNION ALL SELECT m.doc_id FROM '{WORK}/family_map/docs.parquet' m
                      WHERE m.family_id IN (SELECT family_id FROM '{O}/queries/queries.parquet' WHERE split = 'test')) WHERE x IS NOT NULL""").fetchall()
    hs = sorted(hashlib.sha256(r[0].encode()).hexdigest() for r in rows)
    open(f"{O}/canary/test_publications.sha256.txt", "w").write("\n".join(hs) + "\n")
    st["canary_hashes"] = len(hs)

    # --- provenance: sha256 of every file written
    man = {}
    for p in sorted(glob.glob(f"{O}/**/*", recursive=True)):
        if os.path.isfile(p) and not p.endswith("manifest.json"):
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for b in iter(lambda: fh.read(1 << 22), b""):
                    h.update(b)
            man[os.path.relpath(p, O)] = {"bytes": os.path.getsize(p), "sha256": h.hexdigest()}
    st["seconds"] = round(time.time() - t0)
    os.makedirs(f"{O}/provenance", exist_ok=True)
    json.dump({"stats": st, "files": man}, open(f"{O}/provenance/manifest.json", "w"), indent=1)
    print(json.dumps(st))


if __name__ == "__main__":
    main()
