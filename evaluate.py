#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
"""Scoring, exact search and encoding for Patent Mechanics Prior-Art Bench v1.

  # score a run file (local or hf://), overall and by source
  python evaluate.py metrics --run RUN.parquet --part test [--data DATA] [--hit] [--queries IDS.txt]
  # paired bootstrap 95% intervals, B minus A (2,000 resamples over queries, seed 0)
  python evaluate.py compare --a A.parquet --b B.parquet --part test [--metrics recall@10,recall@100,ndcg@10]
  # exact search over vectors (CPU; numpy), top --k families per query with the benchmark's filters
  python evaluate.py search --vectors VEC --part test --out RUN.parquet [--collection documents|chunks] [--k 1000]
  # encode queries, documents (and description chunks) with any model (GPU recommended)
  python evaluate.py encode --model M --config configs/M.json --out VEC [--chunks] [--backend st|vllm]

DATA = a local copy of the dataset or hf://datasets/patentmechanics/prior-art-bench-v1 (default; files are downloaded
into the Hugging Face cache on first use). VEC = a directory with documents/, chunks/ and queries/ in the layout of
vectors/anticipate-0.6b/ of the dataset (also accepted as hf://...).

Scoring rules (identical to the code that produced every published number):
  - unit = patent family on both sides; a family listed twice in a run keeps its best rank;
  - gold = qrels rows with keep AND date_ok (the family's earliest publication is before the query's priority date);
  - only queries with at least one such gold family are scored; --part selects the dev or test half;
  - recall@10/100/1000 over the gold families, nDCG@10 with gain = grade (3 or 2), MRR@10;
  - hit@10/100/1000 (--hit): share of queries with at least one gold family in the top k (the launch blog's "found at
    least one" rate; post hoc, not one of the paper's pre-registered metrics).
--queries scores only the query ids listed in a text file, one per line (e.g. runs/all-us.test-subset.query_ids.txt).
Search filters: the query's own family is never a candidate; a family is a candidate only if its earliest publication
date is before the query's priority date (--date-filter pub, default); a query without a priority date gets none of
this filter (99999999).
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pab_data as D  # noqa: E402

METRICS = ("recall@10", "recall@100", "recall@1000", "ndcg@10", "mrr@10")


def _con():
    import duckdb
    c = duckdb.connect()
    if os.environ.get("PAB_DUCKDB_MEM"):   # e.g. 4GB on a small laptop
        c.execute(f"SET memory_limit='{os.environ['PAB_DUCKDB_MEM']}'")
    return c


# ---------------------------------------------------------------------------------------------------------- metrics
def score_query(ranked, gold):
    """ranked: family ids in rank order (may repeat); gold: {family_id: grade}. -> {metric: value}"""
    g = gold
    seen, dedup = set(), []
    for f in ranked:              # a family listed twice (duplicate corpus rows, passage runs) keeps its best rank
        if f not in seen:
            seen.add(f); dedup.append(f)
    pos = {f: i for i, f in enumerate(dedup) if f in g}
    m = {}
    for kk in (10, 100, 1000):
        m[f"recall@{kk}"] = sum(1 for f in g if pos.get(f, 1 << 30) < kk) / len(g)
        m[f"hit@{kk}"] = float(any(i < kk for i in pos.values()))
    dcg = sum(g[f] / math.log2(i + 2) for f, i in pos.items() if i < 10)
    ideal = sum(v / math.log2(i + 2) for i, v in enumerate(sorted(g.values(), reverse=True)[:10]))
    m["ndcg@10"] = dcg / ideal if ideal else 0.0
    first = min(pos.values()) if pos else None
    m["mrr@10"] = 1.0 / (first + 1) if first is not None and first < 10 else 0.0
    return m


def per_query(run, data=D.HF_DATASET, part="dev", date_filter="pub", only=None):
    """{query_id: ("<source>/<split>", {metric: value})} for every scored query of a run file (only the ids in the
    text file `only`, one per line, when given)."""
    run = D.fetch(run)
    qrels, queries = D.data_file(data, "qrels/qrels.parquet"), D.data_file(data, "queries/queries.parquet")
    c = _con()
    date_cond = "date_ok" if date_filter == "pub" else "TRUE"
    c.execute(f"""CREATE TABLE rel AS SELECT query_id, family_id, grade FROM read_parquet('{qrels}')
                  WHERE keep AND {date_cond}""")
    c.execute(f"CREATE TABLE run AS SELECT * FROM read_parquet('{run}')")
    c.execute(f"CREATE TABLE sp AS SELECT query_id, source, split FROM read_parquet('{queries}')")
    if part != "all":
        c.execute("DELETE FROM sp WHERE split <> ?", [part])
    rows = c.execute("""SELECT r.query_id, sp.source, sp.split, r.family_ids, list(rel.family_id), list(rel.grade)
                        FROM run r JOIN sp USING (query_id) JOIN rel USING (query_id) GROUP BY ALL""").fetchall()
    out = {}
    for qid, src, split, ranked, rel_f, rel_g in rows:
        out[qid] = (f"{src}/{split}", score_query(ranked, dict(zip(rel_f, rel_g))))
    if only:
        keep = {x.strip() for x in open(only) if x.strip()}
        out = {q: v for q, v in out.items() if q in keep}
    return out


def summary(pq_):
    """mean of every metric per source and over all scored queries (4 decimals)"""
    agg = {}
    for _, (key, m) in pq_.items():
        for k in (key.split("/")[0], "all"):
            d = agg.setdefault(k, {"queries": 0})
            d["queries"] += 1
            for mk, v in m.items():
                d[mk] = d.get(mk, 0.0) + v
    return {k: {mk: (round(v / d["queries"], 4) if mk != "queries" else v) for mk, v in d.items()} for k, d in agg.items()}


def paired(A, B, metric="recall@10", n_boot=2000, seed=0):
    """B minus A per group (all, then each source in order of first appearance among the sorted common query ids):
    mean difference and 95% percentile interval of a paired bootstrap over the queries scored in both runs. One
    generator (seed 0) per call, drawn group by group in that order: this is the exact procedure of the published
    intervals."""
    out = {}
    common = sorted(set(A) & set(B))
    groups = {"all": common}
    for q in common:
        groups.setdefault(A[q][0].split("/")[0], []).append(q)
    rng = np.random.default_rng(seed)
    for g, qs in groups.items():
        d = np.array([B[q][1][metric] - A[q][1][metric] for q in qs])
        boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
        out[g] = {"n": len(qs), "diff": round(float(d.mean()), 4),
                  "ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]}
    return out


def _order(keys):
    return [k for k in ("all", "us_oa", "ptab") if k in keys] + sorted(k for k in keys if k not in ("all", "us_oa", "ptab"))


def cmd_metrics(a):
    s = summary(per_query(a.run, a.data, a.part, a.date_filter, a.queries))
    for k in _order(s):
        d = s[k]
        print(f"{k:8s} n={d['queries']:6d}  R@10={d['recall@10']:.4f}  R@100={d['recall@100']:.4f}  "
              f"R@1000={d['recall@1000']:.4f}  nDCG@10={d['ndcg@10']:.4f}  MRR@10={d['mrr@10']:.4f}"
              + (f"  hit@10={d['hit@10']:.4f}  hit@100={d['hit@100']:.4f}  hit@1000={d['hit@1000']:.4f}" if a.hit else ""))
    if a.out:
        json.dump({"run": a.run, "part": a.part, "date_filter": a.date_filter, "metrics": s}, open(a.out, "w"), indent=1)


def cmd_compare(a):
    A = per_query(a.a, a.data, a.part, a.date_filter, a.queries)
    B = per_query(a.b, a.data, a.part, a.date_filter, a.queries)
    common = set(A) & set(B)
    sa = summary({q: A[q] for q in common})
    sb = summary({q: B[q] for q in common})
    res = {m: paired(A, B, m, a.n_boot, a.seed) for m in a.metrics.split(",")}
    for m, r in res.items():
        for g in _order(r):
            x = r[g]
            print(f"{m:11s} {g:6s} n={x['n']:6d}  {sa[g][m]:.4f} -> {sb[g][m]:.4f}  diff {100 * x['diff']:+.1f} "
                  f"[{100 * x['ci95'][0]:+.1f}, {100 * x['ci95'][1]:+.1f}] points  ({x['diff']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}])")
    if a.out:
        json.dump({"a": a.a, "b": a.b, "part": a.part, "n_boot": a.n_boot, "seed": a.seed, "paired": res,
                   "a_metrics": sa, "b_metrics": sb}, open(a.out, "w"), indent=1)


# ---------------------------------------------------------------------------------------------------------- search
def load_vectors(vec_dir):
    """<name>.npy + <name>.family_ids.npy pairs, files in sorted order -> ([memory-mapped arrays], family ids)"""
    embs, ids = [], []
    for f in sorted(glob.glob(f"{vec_dir}/*.npy")):
        if f.endswith(".family_ids.npy") or f.endswith(".ids.npy"):
            continue
        idf = f[:-4] + ".family_ids.npy"
        if not os.path.exists(idf):
            idf = f[:-4] + ".ids.npy"
        embs.append(np.load(f, mmap_mode="r"))
        ids.append(np.load(idf))
        if len(ids[-1]) != embs[-1].shape[0]:
            raise SystemExit(f"{f}: {embs[-1].shape[0]} vectors but {len(ids[-1])} family ids")
    if not embs:
        raise SystemExit(f"no vector files in {vec_dir}")
    ids = np.concatenate(ids).astype(np.int64)
    _, first = np.unique(ids, return_index=True)
    if len(first) < len(ids):
        print(json.dumps({"note": f"{len(ids) - len(first)} repeated family ids (expected for chunks); ranking keeps every row, "
                                  "scoring keeps each family's best rank"}), flush=True)
    return embs, ids


def exact_search(q_emb, q_fam, q_pd, embs, ids, doc_pd, k=1000, date_filter="pub", corpus_chunk=65536, q_chunk=512):
    """Exact inner-product top-k with the benchmark's filters, in float32 on the CPU.
    q_emb [nq, dim]; q_fam [nq] (own family, -1 if none); q_pd [nq] (priority date yyyymmdd); embs: list of [n_i, dim]
    arrays (memory-mapped is fine); ids / doc_pd: per corpus row. A row is masked if its family is the query's own
    family or (date_filter "pub") its family's earliest publication date >= the query's priority date.
    Returns (family ids, scores) per query, best first; masked rows never appear."""
    nq = q_emb.shape[0]
    q = np.ascontiguousarray(q_emb, dtype=np.float32)
    best_s = np.full((nq, k), -np.inf, dtype=np.float32)
    best_i = np.full((nq, k), -1, dtype=np.int64)
    base = 0
    for e in embs:
        for s0 in range(0, e.shape[0], corpus_chunk):
            ce = np.asarray(e[s0:s0 + corpus_chunk], dtype=np.float32)
            n = ce.shape[0]
            dpd = doc_pd[base + s0: base + s0 + n]
            dfm = ids[base + s0: base + s0 + n]
            for q0 in range(0, nq, q_chunk):
                q1 = min(nq, q0 + q_chunk)
                sc = q[q0:q1] @ ce.T
                bad = dfm[None, :] == q_fam[q0:q1, None]
                if date_filter == "pub":
                    bad |= dpd[None, :] >= q_pd[q0:q1, None]
                sc[bad] = -np.inf
                kk = min(k, n)
                ti = np.argpartition(-sc, kk - 1, axis=1)[:, :kk]
                ts = np.take_along_axis(sc, ti, 1)
                ms = np.concatenate([best_s[q0:q1], ts], 1)
                mi = np.concatenate([best_i[q0:q1], ti + base + s0], 1)
                sel = np.argpartition(-ms, k - 1, axis=1)[:, :k]
                best_s[q0:q1] = np.take_along_axis(ms, sel, 1)
                best_i[q0:q1] = np.take_along_axis(mi, sel, 1)
        base += e.shape[0]
    fams, scores = [], []
    for r in range(nq):
        o = np.argsort(-best_s[r], kind="stable")
        bi, bs = best_i[r][o], best_s[r][o]
        ok = (bi >= 0) & np.isfinite(bs)
        fams.append(ids[bi[ok]].tolist())
        scores.append(bs[ok].astype(np.float32).tolist())
    return fams, scores


def family_dates(data, need):
    """earliest publication date (yyyymmdd) per family id in `need`; 0 if unknown (= always a candidate)"""
    c = _con()
    parts = D.files(data, "families/part-*.parquet")
    t = c.execute(f"""SELECT family_id, earliest_publication_date::INTEGER AS pd FROM read_parquet({json.dumps(parts)})
                      WHERE earliest_publication_date IS NOT NULL""").fetchnumpy()
    return lookup_dates(np.asarray(t["family_id"], dtype=np.int64), np.asarray(t["pd"], dtype=np.int64), need)


def lookup_dates(fam, pd_, need):
    """vectorised dict.get(family, 0)"""
    need = np.asarray(need, dtype=np.int64)
    if len(fam) == 0:
        return np.zeros(len(need), dtype=np.int64)
    o = np.argsort(fam, kind="stable")
    fam, pd_ = fam[o], pd_[o]
    i = np.clip(np.searchsorted(fam, need), 0, len(fam) - 1)
    return np.where(fam[i] == need, pd_[i], 0).astype(np.int64)


def write_run(path, qids, fams, scores):
    import pyarrow as pa
    import pyarrow.parquet as pq
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pq.write_table(pa.table({"query_id": pa.array(qids, pa.string()), "family_ids": pa.array(fams, pa.list_(pa.int64())),
                             "scores": pa.array(scores, pa.list_(pa.float32()))}), path)


def read_run(path):
    """run file -> [{query_id, family_ids, scores}] in file order"""
    import pyarrow.parquet as pq
    return pq.read_table(D.fetch(path), columns=["query_id", "family_ids", "scores"]).to_pylist()


def cmd_search(a):
    import pyarrow.parquet as pq
    vec = D.fetch(a.vectors, [f"{a.collection}/*", "queries/*"]) if D.is_remote(a.vectors) else a.vectors
    qdir = a.query_vectors or f"{vec}/queries"
    embs, ids = load_vectors(f"{vec}/{a.collection}")
    qv = np.load(f"{qdir}/queries.npy", mmap_mode="r")
    qv_ids = pq.read_table(f"{qdir}/queries.query_ids.parquet").column("query_id").to_pylist()
    row = {q: i for i, q in enumerate(qv_ids)}
    c = _con()
    qs = c.execute(f"""SELECT query_id, family_id, coalesce(priority_date, 99999999)::INTEGER, split
                       FROM read_parquet('{D.data_file(a.data, "queries/queries.parquet")}') ORDER BY query_id""").fetchall()
    qs = [q for q in qs if q[0] in row and (a.part == "all" or q[3] == a.part)]
    if not qs:
        raise SystemExit("no query vectors for the selected part")
    k = a.k or (4000 if a.collection == "chunks" else 1000)
    doc_pd = family_dates(a.data, ids)
    print(json.dumps({"queries": len(qs), "corpus_rows": int(len(ids)), "k": k, "collection": a.collection}), flush=True)
    t0 = time.time()
    q_emb = np.stack([np.asarray(qv[row[q[0]]], dtype=np.float32) for q in qs])
    q_fam = np.asarray([q[1] if q[1] is not None else -1 for q in qs], dtype=np.int64)
    q_pd = np.asarray([q[2] for q in qs], dtype=np.int64)
    fams, scores = exact_search(q_emb, q_fam, q_pd, embs, ids, doc_pd, k, a.date_filter, a.corpus_chunk, a.q_chunk)
    write_run(a.out, [q[0] for q in qs], fams, scores)
    json.dump({k_: v for k_, v in vars(a).items() if k_ != "func"}, open(a.out + ".args.json", "w"), indent=1)
    print(json.dumps({"written": a.out, "seconds": round(time.time() - t0, 1)}), flush=True)


# ---------------------------------------------------------------------------------------------------------- encode
def cmd_encode(a):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pab_encoding import Encoder, load_config, query_prefix
    cfg = load_config(a.config)
    model = a.model or cfg.get("model")
    if not model:
        raise SystemExit("--model or a config with 'model' is required")
    rev = a.revision or cfg.get("revision")
    qlen, dlen = a.q_len or cfg["q_len"], a.d_len or cfg["d_len"]
    enc = Encoder(model, a.backend, revision=rev, max_len=dlen, pooling=cfg.get("pooling", "LAST"), batch=a.batch,
                  dtype=a.dtype, device=a.device)
    what = set(a.what.split(","))
    if a.chunks:
        what.add("chunks")
    sample = a.collection == "sample"

    def save(dst, vecs, fam_ids):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        np.save(dst[:-4] + ".family_ids.npy", np.asarray(fam_ids, dtype=np.int64))
        np.save(dst + ".tmp.npy", vecs)
        os.replace(dst + ".tmp.npy", dst)

    if "queries" in what and a.shard == 0:
        dst = f"{a.out}/queries/queries.npy"
        if not os.path.exists(dst):
            t = pq.read_table(D.data_file(a.data, "sample/queries.parquet" if sample else "queries/queries.parquet"),
                              columns=["query_id", "source", "split", "text"]).to_pylist()
            t = [r for r in t if r["text"] and (a.part == "all" or r["split"] == a.part)]
            enc.set_max_len(qlen)
            vec = [None] * len(t)
            for src in sorted({r["source"] for r in t}):
                idx = [i for i, r in enumerate(t) if r["source"] == src]
                pre = query_prefix(src, cfg["query_prefix"])
                for i, v in zip(idx, enc([pre + t[i]["text"] for i in idx])):
                    vec[i] = v
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            pq.write_table(pa.table({"query_id": [r["query_id"] for r in t]}), f"{a.out}/queries/queries.query_ids.parquet")
            np.save(dst, np.stack(vec).astype(np.float16))
            print(json.dumps({"queries": len(t), "q_len": qlen}), flush=True)
    enc.set_max_len(dlen)
    jobs = []
    if "documents" in what:
        src = [D.data_file(a.data, "sample/corpus.parquet")] if sample else D.files(a.data, "corpus-pool-v2/part-*.parquet")
        jobs += [("documents", f) for f in src]
    if "chunks" in what:
        jobs += [("chunks", f) for f in D.files(a.data, "corpus-pool-v2-chunks/part-*.parquet")]
    for kind, f in jobs[a.shard::a.nshards]:
        name = os.path.basename(f)[:-len(".parquet")]
        dst = f"{a.out}/{kind}/{name}.npy"
        if os.path.exists(dst):
            continue
        cols = ["family_id", "text"] + (["vector_file", "vector_row"] if kind == "chunks" else [])
        t = pq.read_table(f, columns=cols).to_pylist()
        if kind == "documents":   # one vector per family (the text is the family's representative document)
            seen, keep = set(), []
            for r in t:
                if r["family_id"] not in seen:
                    seen.add(r["family_id"]); keep.append(r)
            t = keep
        else:                     # rows in vector order; the shipped chunk vectors use the vector_file name
            t.sort(key=lambda r: r["vector_row"])
            name = t[0]["vector_file"] if t else name
            dst = f"{a.out}/{kind}/{name}.npy"
        t0 = time.time()
        e = enc([cfg["doc_prefix"] + (r["text"] or "") for r in t])
        save(dst, e, [r["family_id"] for r in t])
        print(json.dumps({"file": f"{kind}/{name}", "rows": len(t), "seconds": round(time.time() - t0, 1)}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    pm = sub.add_parser("metrics", help="score a run file")
    pm.add_argument("--run", required=True)
    pc = sub.add_parser("compare", help="paired bootstrap, B minus A")
    pc.add_argument("--a", required=True)
    pc.add_argument("--b", required=True)
    pc.add_argument("--metrics", default="recall@10,recall@100,ndcg@10", help="any of recall@k, hit@k (k = 10, 100, 1000), ndcg@10, mrr@10")
    pc.add_argument("--n-boot", type=int, default=2000)
    pc.add_argument("--seed", type=int, default=0)
    for p in (pm, pc):
        p.add_argument("--part", default="dev", choices=["dev", "test", "all"], help="dev for decisions; test once, for the number you report")
        p.add_argument("--out", default=None, help="also write the result as JSON")
        p.add_argument("--queries", default=None, help="score only the query ids in this text file (one per line)")
    pm.add_argument("--hit", action="store_true", help="also print hit@10/100/1000")
    ps = sub.add_parser("search", help="exact search over vectors (CPU)")
    ps.add_argument("--vectors", "--emb", dest="vectors", default=f"{D.HF_DATASET}/{D.VECTORS}")
    ps.add_argument("--collection", default="documents", choices=["documents", "chunks"])
    ps.add_argument("--query-vectors", default=None, help="directory with queries.npy + queries.query_ids.parquet (default VEC/queries)")
    ps.add_argument("--part", default="dev", choices=["dev", "test", "all"])
    ps.add_argument("--out", required=True)
    ps.add_argument("--k", type=int, default=0, help="default 1000 (documents) / 4000 (chunks: one row per chunk)")
    ps.add_argument("--corpus-chunk", type=int, default=65536)
    ps.add_argument("--q-chunk", type=int, default=512)
    pe = sub.add_parser("encode", help="encode queries / documents / chunks")
    pe.add_argument("--model", default=None)
    pe.add_argument("--revision", default=None)
    pe.add_argument("--config", default=None, help="configs/<system>.json: prefixes, lengths, pooling")
    pe.add_argument("--backend", default="st", choices=["st", "vllm"])
    pe.add_argument("--out", required=True)
    pe.add_argument("--what", default="queries,documents")
    pe.add_argument("--chunks", action="store_true", help="also the description chunks (passage stage)")
    pe.add_argument("--collection", default="corpus-pool-v2", choices=["corpus-pool-v2", "sample"])
    pe.add_argument("--part", default="all", choices=["dev", "test", "all"], help="queries to encode")
    pe.add_argument("--q-len", type=int, default=0)
    pe.add_argument("--d-len", type=int, default=0)
    pe.add_argument("--batch", type=int, default=8, help="sentence-transformers batch size")
    pe.add_argument("--dtype", default="bfloat16")
    pe.add_argument("--device", default=None)
    pe.add_argument("--shard", type=int, default=0)
    pe.add_argument("--nshards", type=int, default=1)
    for p in (pm, pc, ps, pe):
        p.add_argument("--data", default=D.HF_DATASET, help="dataset root (local or hf://)")
    for p in (pm, pc, ps):
        p.add_argument("--date-filter", default="pub", choices=["pub", "none"])
    a = ap.parse_args()
    {"metrics": cmd_metrics, "compare": cmd_compare, "search": cmd_search, "encode": cmd_encode}[a.cmd](a)


if __name__ == "__main__":
    main()
