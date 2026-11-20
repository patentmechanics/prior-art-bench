#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
"""BM25 baseline on the benchmark collection (CPU only; 16-24 GB of RAM for the full 250,367-family pool).

  python baselines/bm25.py --part test --out runs/bm25.test.parquet [--data DATA] [--workers 6]

Corpus = corpus-pool-v2 `text` (title + abstract + claims + description, the exact text the dense models encode), one
row per family. Queries = queries/queries.parquet `text` (title + abstract + claims). Same filters as
`evaluate.py search`: the query's own family and every family whose earliest publication is on or after the query's
priority date are excluded; only documents with a positive score are listed (top --k).
BM25 variant "lucene" as in bm25s: k1 1.2, b 0.75, idf = log(1 + (N - df + 0.5) / (df + 0.5)),
tf / (tf + k1 (1 - b + b dl / avgdl)); lower-cased text, token pattern \\b\\w\\w+\\b, English stopwords from bm25s
(removed before stemming), Snowball English stemmer (PyStemmer), query terms weighted by their count in the query.
Weights are float32, as in the run that produced the published BM25 numbers.
"""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from multiprocessing import get_context

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pab_data as D  # noqa: E402

TOK = re.compile(r"(?u)\b\w\w+\b")
_STEM = None
STOP = None


def _init():
    global _STEM, STOP
    if _STEM is None:
        import Stemmer
        from bm25s.stopwords import STOPWORDS_EN
        _STEM, STOP = Stemmer.Stemmer("english"), frozenset(STOPWORDS_EN)


def index_texts(fams, texts):
    """documents -> (family ids, stem vocabulary, CSR pieces with local term ids, document lengths)"""
    _init()
    fams = np.asarray(fams, dtype=np.int64)
    raw, doc_ids, tok_ids = {}, [], []
    for d, txt in enumerate(texts):
        ids = [raw.setdefault(w, len(raw)) for w in TOK.findall((txt or "").lower()) if w not in STOP]
        tok_ids.append(np.asarray(ids, dtype=np.int32))
        doc_ids.append(np.full(len(ids), d, dtype=np.int32))
    nd = len(fams)
    stems, sv = _STEM.stemWords(list(raw)), {}
    raw2stem = np.asarray([sv.setdefault(s, len(sv)) for s in stems], dtype=np.int32)
    allt = raw2stem[np.concatenate(tok_ids)] if tok_ids and len(raw2stem) else np.zeros(0, np.int32)
    alld = np.concatenate(doc_ids) if doc_ids else np.zeros(0, np.int32)
    nv = max(1, len(sv))
    key = alld.astype(np.int64) * nv + allt
    uk, cnt = np.unique(key, return_counts=True)
    doc, term = (uk // nv).astype(np.int32), (uk % nv).astype(np.int32)
    indptr = np.concatenate([[0], np.cumsum(np.bincount(doc, minlength=nd))]).astype(np.int64)
    return fams, list(sv), indptr, term, cnt.astype(np.float32), np.bincount(alld, minlength=nd).astype(np.float32)


def index_part(args):
    path, rg, keep = args
    import pyarrow.parquet as pq
    t = pq.ParquetFile(path).read_row_group(rg, columns=["family_id", "text"])
    fams, texts = t.column("family_id").to_pylist(), t.column("text").to_pylist()
    return index_texts([fams[i] for i in keep], [texts[i] for i in keep])


def build_index(parts_iter, k1=1.2, b=0.75):
    """parts_iter yields index_texts() outputs in corpus order -> dict with the term-major weight matrix"""
    from scipy.sparse import csr_matrix
    vg, fam_l, ip_l, ind_l, dat_l, dl_l, off = {}, [], [np.zeros(1, np.int64)], [], [], [], 0
    for fams, stems, indptr, term, cnt, dl in parts_iter:
        mp = np.asarray([vg.setdefault(s, len(vg)) for s in stems], dtype=np.int32)
        fam_l.append(fams); ind_l.append(mp[term] if len(term) else np.zeros(0, np.int32)); dat_l.append(cnt); dl_l.append(dl)
        ip_l.append(indptr[1:] + off); off += int(indptr[-1])
    fam, dl = np.concatenate(fam_l), np.concatenate(dl_l)
    indptr, indices, data = np.concatenate(ip_l), np.concatenate(ind_l), np.concatenate(dat_l)
    N, V = len(fam), len(vg)
    df = np.bincount(indices, minlength=V).astype(np.float64)
    idf = np.log(1.0 + (N - df + 0.5) / (df + 0.5)).astype(np.float32)
    avgdl = float(dl.mean())
    CH = 50_000_000                                  # weights in place, in blocks (keeps temporaries small)
    for s in range(0, len(data), CH):
        e = min(len(data), s + CH)
        row = np.searchsorted(indptr, np.arange(s, e), side="right") - 1
        tf = data[s:e].copy()
        data[s:e] = idf[indices[s:e]] * tf / (tf + k1 * (1.0 - b + b * dl[row] / avgdl))
    W = csr_matrix((data, indices, indptr), shape=(N, V)).tocsc()    # term-major postings
    return {"N": N, "ip": W.indptr, "rows": W.indices, "data": W.data, "vg": vg, "fam": fam, "nnz": int(W.nnz),
            "avgdl": avgdl}


# globals shared with forked search workers (copy-on-write)
G = {}


def search_one(q):
    _init()
    qid, fam, pd, text = q
    toks = [w for w in TOK.findall((text or "").lower()) if w not in STOP]
    N = G["N"]
    scores = np.zeros(N, dtype=np.float32)
    ip, rows, data, vg = G["ip"], G["rows"], G["data"], G["vg"]
    for s, c in Counter(_STEM.stemWords(toks)).items():
        t = vg.get(s)
        if t is not None:
            a, b = ip[t], ip[t + 1]
            scores[rows[a:b]] += data[a:b] * c
    scores[G["fam"] == fam] = -np.inf
    scores[G["pd"] >= pd] = -np.inf
    k = min(G["k"], N)
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx], kind="stable")]
    idx = idx[np.isfinite(scores[idx]) & (scores[idx] > 0)]
    return qid, G["fam"][idx].tolist(), scores[idx].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=D.HF_DATASET)
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", default="all", choices=["dev", "test", "all"], help="queries to search (scores do not depend on it)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--k1", type=float, default=1.2)
    ap.add_argument("--b", type=float, default=0.75)
    ap.add_argument("--limit-queries", type=int, default=0)
    a = ap.parse_args()
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq
    from evaluate import family_dates
    t0 = time.time()
    files = D.files(a.data, "corpus-pool-v2/part-*.parquet")
    # one row per family (the first one, in file order)
    seen, jobs, dups = set(), [], 0
    for f in files:
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            fam = pf.read_row_group(rg, columns=["family_id"]).column(0).to_pylist()
            keep = []
            for i, x in enumerate(fam):
                if x in seen:
                    dups += 1
                else:
                    seen.add(x); keep.append(i)
            jobs.append((f, rg, keep))
    if dups:
        print(json.dumps({"note": f"{dups} repeated family rows in corpus-pool-v2 skipped (one document per family)"}), flush=True)
    ctx = get_context("fork")
    with ctx.Pool(a.workers) as pool:
        idx = build_index(pool.imap(index_part, jobs), a.k1, a.b)
    print(f"index built: {idx['N']} docs, {len(idx['vg'])} terms, nnz {idx['nnz']}, avgdl {idx['avgdl']:.0f}, "
          f"{time.time() - t0:.0f}s", flush=True)
    c = duckdb.connect()
    qs = c.execute(f"""SELECT query_id, family_id, coalesce(priority_date, 99999999), text, split
                       FROM read_parquet('{D.data_file(a.data, "queries/queries.parquet")}')
                       WHERE text IS NOT NULL ORDER BY query_id""").fetchall()
    qs = [(q, -1 if f is None else f, p, t) for q, f, p, t, s in qs if a.part == "all" or s == a.part]
    if a.limit_queries:
        qs = qs[:a.limit_queries]
    G.update(idx, pd=family_dates(a.data, idx["fam"]).astype(np.int32), k=a.k)
    out_q, out_f, out_s = [], [], []
    with ctx.Pool(a.workers) as pool:
        for i, (qid, fl, sl) in enumerate(pool.imap(search_one, qs, chunksize=16)):
            out_q.append(qid); out_f.append(fl); out_s.append(sl)
            if (i + 1) % 1000 == 0:
                print(f"searched {i + 1}/{len(qs)} queries, {time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    pq.write_table(pa.table({"query_id": out_q, "family_ids": pa.array(out_f, pa.list_(pa.int64())),
                             "scores": pa.array(out_s, pa.list_(pa.float32()))}), a.out)
    st = {"docs": idx["N"], "terms": len(idx["vg"]), "nnz": idx["nnz"], "queries": len(qs), "k1": a.k1, "b": a.b,
          "seconds": round(time.time() - t0)}
    json.dump(st, open(a.out + ".stats.json", "w"))
    print(json.dumps(st), flush=True)


if __name__ == "__main__":
    main()
