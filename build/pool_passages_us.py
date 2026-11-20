#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
"""Passage chunks of the US-only evaluation pool v2 (local files only, USPTO data only). Passage-level test of the
release plan (path 6, 2026-11-13): a document also scores by its best description chunk, so text past the 8,192-token
window counts.

  PAB_WORK=$WORK python build/pool_passages_us.py [--out $WORK/eval_us_v2_pas] [--chunk-chars 5000]

For every pool family (eval_us_v2/tacd) its representative document (the doc_id whose text is in the pool, from
tacd/subset-us-*.parquet), that document's description paragraphs (passages/us, in order; sections "related" and
"drawings" dropped, as build/tacd.py), packed into chunks of >= --chunk-chars characters, each prefixed with the
document's title and abstract. Output: part-NNN.parquet (family_id, pas_text), --per-file chunks per file, so the
evaluation shards by file (evaluate.py encode --shard/--nshards). Streams one passage file at a time (a document's
paragraphs are all in its own publication file), so memory stays small. Writes summary.json.
"""
import argparse, glob, json, os, time

import duckdb
import pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq

W = os.environ.get("PAB_WORK", "work")
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=f"{W}/eval_us_v2_pas")
ap.add_argument("--chunk-chars", type=int, default=5000)
ap.add_argument("--per-file", type=int, default=20000)
a = ap.parse_args()
t0 = time.time()
os.makedirs(a.out, exist_ok=True)
assert not glob.glob(f"{a.out}/part-*.parquet"), f"{a.out} already has part files"
c = duckdb.connect()
c.execute("SET threads=4; SET memory_limit='8GB'")
pool = f"{W}/eval_us_v2/tacd/*.parquet"
# prefix = title + abstract: the first two blocks of the pool's own text (computed in DuckDB; the full texts are ~11 GB)
rows = c.execute(f"""
    WITH p AS (SELECT family_id::BIGINT AS family_id,
                      array_to_string(list_slice(string_split(tacd_text, E'\\n\\n'), 1, 2), E'\\n\\n') AS prefix FROM '{pool}'),
         d AS (SELECT DISTINCT family_id::BIGINT AS family_id, doc_id FROM read_parquet('{W}/tacd/subset-us-*.parquet', union_by_name=true)
               WHERE family_id::BIGINT IN (SELECT family_id::BIGINT FROM '{pool}'))
    SELECT p.family_id, d.doc_id, p.prefix FROM p LEFT JOIN d USING (family_id)""").fetchall()
n_pool = len({r[0] for r in rows})
doc2fam, prefix, no_doc = {}, {}, 0
for fam, doc, pre in rows:
    if doc is None:
        no_doc += 1
        continue
    assert doc not in doc2fam, f"document {doc} in two pool families"
    doc2fam[doc] = fam
    prefix[doc] = pre
assert len(doc2fam) == n_pool - no_doc, f"{len(doc2fam)} documents for {n_pool - no_doc} families (a family with two documents?)"
del rows
docs = pa.array(list(doc2fam))
schema = pa.schema([("family_id", pa.int64()), ("pas_text", pa.string())])
part, n_chunks, chars, seen, buf = 0, 0, 0, set(), ([], [])


def flush(force=False):
    """write a full file (--per-file chunks) or, at the end, the remainder"""
    global part
    while len(buf[0]) >= a.per_file or (force and buf[0]):
        n = min(a.per_file, len(buf[0]))
        pq.write_table(pa.table({"family_id": pa.array(buf[0][:n], pa.int64()), "pas_text": buf[1][:n]}, schema=schema),
                       f"{a.out}/part-{part:03d}.parquet", compression="zstd", row_group_size=2500)
        del buf[0][:n], buf[1][:n]
        part += 1


def emit(fam, text):
    global n_chunks, chars
    buf[0].append(fam)
    buf[1].append(text)
    n_chunks, chars = n_chunks + 1, chars + len(text)


files = sorted(glob.glob(f"{W}/passages/us/*.parquet"))
for i, f in enumerate(files):
    t = pq.read_table(f, columns=["doc_id", "p_idx", "section", "text"])
    t = t.filter(pc.and_(pc.is_in(t.column("doc_id"), value_set=docs),
                         pc.invert(pc.is_in(t.column("section"), value_set=pa.array(["related", "drawings"])))))
    paras = {}
    for d, i_, x in zip(t.column("doc_id").to_pylist(), t.column("p_idx").to_pylist(), t.column("text").to_pylist()):
        if x:
            paras.setdefault(d, []).append((i_, x))
    for d, ps in paras.items():
        if d in seen:   # a document in two passage files would be chunked twice
            continue
        seen.add(d)
        ps.sort()
        cur = ""
        for _, x in ps:
            cur = (cur + "\n\n" + x) if cur else x
            if len(cur) >= a.chunk_chars:
                emit(doc2fam[d], prefix[d] + "\n\nDescription (part):\n" + cur)
                cur = ""
        if cur:
            emit(doc2fam[d], prefix[d] + "\n\nDescription (part):\n" + cur)
    flush()
    if i % 200 == 0:
        print(json.dumps({"files": i + 1, "of": len(files), "documents": len(seen), "chunks": n_chunks,
                          "minutes": round((time.time() - t0) / 60, 1)}), flush=True)
flush(force=True)
summary = {"pool_families": n_pool, "families_without_doc": no_doc, "documents_with_passages": len(seen),
           "documents_without_passages": len(doc2fam) - len(seen), "chunks": n_chunks, "files": part,
           "avg_chunk_chars": round(chars / max(1, n_chunks)), "chunk_chars": a.chunk_chars, "minutes": round((time.time() - t0) / 60, 1)}
json.dump(summary, open(f"{a.out}/summary.json", "w"), indent=1)
print(json.dumps(summary), flush=True)
