#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
"""Encoding and retrieval in the build/ work-directory layout (DATA = queries/, labels/, family_map/families.parquet),
used for hard-negative mining over TRAINING queries (then train/mine.py). For evaluation on the published benchmark use
the top-level evaluate.py, which reads the Hugging Face dataset layout; the metric code here is the same.

  # 1. encode the mining pool (document shards with family_id + text column); one process per GPU
  python train/retrieve.py encode --backend vllm --model M --tac 'POOL/part-*.parquet' --text-col tacd_text --d-len 8192 --out EMB
  # 2. top-k for the training queries to mine (a parquet with query_id), with the publication-date filter
  python train/retrieve.py search --backend vllm --model M --emb EMB --data DATA --queries Q.parquet --q-len 4096 --k 100 --out RUN.parquet
  # 3. negatives: python train/mine.py --run RUN.parquet --data DATA --out NEG.parquet --lo 30 --hi 100 --pos-cap 0.95 --cocited ...

Date filter (--date-filter pub, default): a document is a candidate only if its family's earliest publication is before the
query's priority date; the query's own family is never a candidate. Environment: PAB_DOC_PREFIX / PAB_QUERY_PREFIX
override the document / query prefix (baselines), PAB_DUCKDB_MEM caps DuckDB memory.
"""
import argparse, glob, json, math, os, sys, time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import encode, format_query, instruction_for, load_model


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def cmd_encode(a):
    files = sorted(glob.glob(a.tac, recursive=True))
    mine = files[a.shard::a.nshards]
    os.makedirs(a.out, exist_ok=True)
    if a.backend == "vllm":
        from transformers import AutoTokenizer
        from common import VllmEncoder
        tok, venc = AutoTokenizer.from_pretrained(a.model), VllmEncoder(a.model, a.d_len, gpu_mem=a.gpu_mem or 0.90)
    else:
        tok, model = load_model(a.model, adapter=a.adapter, device=device(), attn=a.attn)
    for f in mine:
        name = os.path.relpath(f, os.path.commonpath(files) if len(files) > 1 else os.path.dirname(f)).replace("/", "__").replace(".parquet", "")
        dst = f"{a.out}/{name}.npy"
        if os.path.exists(dst):
            continue
        t = pq.read_table(f, columns=["family_id", a.text_col])
        if a.families:
            import pyarrow.compute as pc
            t = t.filter(pc.is_in(t.column("family_id"), value_set=pq.read_table(a.families).column("family_id")))
            if t.num_rows == 0:
                continue
        texts = t.column(a.text_col).to_pylist()
        if os.environ.get("PAB_DOC_PREFIX"):   # e.g. "passage: " for embedders trained with document prefixes
            texts = [os.environ["PAB_DOC_PREFIX"] + x for x in texts]
        t0 = time.time()
        if a.backend == "vllm":
            e = venc(tok, texts, a.dim)
        else:
            e = encode(tok, model, texts, a.d_len, a.pooling, batch_size=a.batch, device=device(), dim=a.dim, max_tokens=a.max_tokens)
        np.save(f"{a.out}/{name}.ids.npy", np.asarray(t.column("family_id").to_pylist(), dtype=np.int64))
        np.save(dst + ".tmp.npy", e)
        os.replace(dst + ".tmp.npy", dst)
        print(json.dumps({"file": name, "docs": len(texts), "seconds": round(time.time() - t0, 1),
                          "docs_per_s": round(len(texts) / max(1e-9, time.time() - t0), 1)}), flush=True)


def load_corpus(emb_dir):
    embs, ids = [], []
    for f in sorted(glob.glob(f"{emb_dir}/*.npy")):
        if f.endswith(".ids.npy"):
            continue
        embs.append(np.load(f, mmap_mode="r"))
        ids.append(np.load(f[:-4] + ".ids.npy"))
    ids = np.concatenate(ids)
    _, first = np.unique(ids, return_index=True)
    if len(first) < len(ids):
        print(json.dumps({"warning": f"{len(ids) - len(first)} duplicate family rows in the corpus embeddings; ranking keeps the best"}), flush=True)
    return embs, ids


def claim_set_texts(title, abstract, claims):
    """[(independent claim num, query text)]: title, abstract and that independent claim with every claim depending on it
    (directly or through other dependents), in the same layout as the full query text (title, blank line, abstract, blank
    line, '<num>. <text>' one per line). No claim flagged independent -> one text with all claims."""
    claims = sorted(claims or [], key=lambda x: x["num"])
    by = {x["num"]: x for x in claims}
    indep = [x["num"] for x in claims if x["is_independent"]]

    def root(x):
        n0 = x["num"]
        for _ in range(100):
            if x["is_independent"]:
                return x["num"]
            if not x["depends_on"] or x["depends_on"][0] not in by:   # broken link: nearest independent claim before it
                return max([n for n in indep if n < n0], default=indep[0] if indep else n0)
            x = by[x["depends_on"][0]]
        return n0
    groups = {}
    for x in claims:
        groups.setdefault(root(x), []).append(x)
    roots = [r for r in groups if by[r]["is_independent"]] or ([claims[0]["num"]] if claims else [])
    if not any(by[r]["is_independent"] for r in roots):
        groups = {roots[0]: claims} if roots else {}
    return [(r, "\n\n".join(t for t in (title, abstract, "\n".join(f"{x['num']}. {x['text']}" for x in groups[r])) if t))
            for r in roots]


def cmd_search(a):
    import duckdb
    dev = device()
    embs, ids = load_corpus(a.emb)
    c = duckdb.connect()
    if os.environ.get("PAB_DUCKDB_MEM"):   # cap DuckDB memory (default: 80% of RAM)
        c.execute(f"SET memory_limit='{os.environ['PAB_DUCKDB_MEM']}'; SET threads=2")
    splits = ", ".join(f"'{s}'" for s in a.splits.split(","))
    each = a.query_field == "each_independent"
    field = {"tac_text": "q.tac_text", "independent_claims": "concat_ws(E'\\n\\n', q.title, array_to_string(q.independent_claims, E'\\n'))",
             "claim1": "concat_ws(E'\\n\\n', q.title, q.claim1)", "each_independent": "q.claims"}[a.query_field]
    cols = "q.title, q.abstract, q.claims" if each else f"{field} AS text"
    qs = c.execute(f"""SELECT q.query_id, q.family_id, coalesce(q.priority_date, 99999999) AS pd, {cols}, q.source
                       FROM '{a.data}/queries/queries.parquet' q JOIN '{a.data}/labels/splits.parquet' s USING (query_id)
                       WHERE {"TRUE" if a.queries else f"s.split IN ({splits})"} AND {field} IS NOT NULL
                       {f"AND q.query_id IN (SELECT query_id FROM '{a.queries}')" if a.queries else ''}
                       {'AND hash(q.query_id) % ' + str(a.sample_mod) + ' = 0' if a.sample_mod > 1 else ''}
                       ORDER BY q.query_id""").fetchall()
    if each:   # one sub-query per independent claim: "<query_id>#<claim num>"; fuse per query (max or RRF) yourself
        qs = [(f"{qid}#{root}", fam, pd, text, src) for qid, fam, pd, title, abstract, claims, src in qs
              for root, text in claim_set_texts(title, abstract, claims)]
    if a.nqshards > 1:   # stable across machines: md5, not DuckDB's hash
        import hashlib
        qs = [q for q in qs if int(hashlib.md5(q[0].encode()).hexdigest()[:8], 16) % a.nqshards == a.qshard]
    if a.max_queries:
        qs = qs[:a.max_queries]
    print(json.dumps({"queries": len(qs), "corpus": int(len(ids))}), flush=True)
    # family earliest publication date per corpus row
    fam = c.execute(f"SELECT family_id, earliest_pd::INTEGER FROM '{a.data}/family_map/families.parquet' WHERE in_scope").fetchall()
    pd_map = dict(fam)
    doc_pd = torch.tensor([pd_map.get(int(f), 0) for f in ids], dtype=torch.int32, device=dev)
    doc_fam = torch.tensor(ids, dtype=torch.int64, device=dev)
    if os.environ.get("PAB_QUERY_PREFIX") is not None:   # plain prefix (e.g. "query: ") or "" (no instruction) instead of ours
        q_texts = [os.environ["PAB_QUERY_PREFIX"] + t for _, _, _, t, _ in qs]
    else:
        q_texts = [format_query(t, a.instruction if a.instruction is not None else instruction_for(src)) for _, _, _, t, src in qs]
    if a.backend == "vllm":
        from transformers import AutoTokenizer
        from common import VllmEncoder
        tok = AutoTokenizer.from_pretrained(a.model)
        model = VllmEncoder(a.model, a.q_len, gpu_mem=a.gpu_mem or 0.5)
        q_emb = torch.from_numpy(model(tok, q_texts, a.dim)).to(dev)
    else:
        tok, model = load_model(a.model, adapter=a.adapter, device=dev, attn=a.attn)
        q_emb = torch.from_numpy(encode(tok, model, q_texts, a.q_len, a.pooling, batch_size=a.batch, device=dev, dim=a.dim)).to(dev)
    np.save(a.out + ".qemb.npy", q_emb.cpu().numpy().astype("float16"))
    pq.write_table(pa.table({"query_id": [q for q, _, _, _, _ in qs]}), a.out + ".qids.parquet")
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()
    q_pd = torch.tensor([p for _, _, p, _, _ in qs], dtype=torch.int32, device=dev)
    q_fam = torch.tensor([f if f is not None else -1 for _, f, _, _, _ in qs], dtype=torch.int64, device=dev)
    k = a.k
    best_s = torch.full((len(qs), k), float("-inf"), device=dev)
    best_i = torch.full((len(qs), k), -1, dtype=torch.int64, device=dev)
    dt = torch.float16 if dev == "cuda" else torch.float32
    base = 0
    for e in embs:
        for s0 in range(0, e.shape[0], a.corpus_chunk):
            ce = torch.from_numpy(np.ascontiguousarray(e[s0:s0 + a.corpus_chunk, :a.dim] if a.dim else e[s0:s0 + a.corpus_chunk])).to(dev, dt)
            n = ce.shape[0]
            dpd = doc_pd[base + s0: base + s0 + n]
            dfm = doc_fam[base + s0: base + s0 + n]
            for q0 in range(0, len(qs), a.q_chunk):
                qe = q_emb[q0:q0 + a.q_chunk].to(dt)
                sc = (qe @ ce.T).float()
                bad = dfm[None, :] == q_fam[q0:q0 + a.q_chunk, None]
                if a.date_filter == "pub":
                    bad |= dpd[None, :] >= q_pd[q0:q0 + a.q_chunk, None]
                sc.masked_fill_(bad, float("-inf"))
                kk = min(k, n)
                ts, ti = sc.topk(kk, dim=1)
                ms = torch.cat([best_s[q0:q0 + a.q_chunk], ts], 1)
                mi = torch.cat([best_i[q0:q0 + a.q_chunk], ti + base + s0], 1)
                o = ms.topk(k, dim=1)
                best_s[q0:q0 + a.q_chunk] = o.values
                best_i[q0:q0 + a.q_chunk] = mi.gather(1, o.indices)
        base += e.shape[0]
    bi, bs = best_i.cpu().numpy(), best_s.cpu().numpy()
    fams, scores = [], []
    for r in range(len(qs)):
        ok = bi[r] >= 0
        ok &= np.isfinite(bs[r])
        fams.append(ids[bi[r][ok]].tolist())
        scores.append(bs[r][ok].astype(np.float32).tolist())
    pq.write_table(pa.table({"query_id": [q for q, _, _, _, _ in qs], "family_ids": fams, "scores": scores}), a.out)
    json.dump(vars(a), open(a.out + ".args.json", "w"), indent=1)
    print(json.dumps({"written": a.out}), flush=True)


def per_query(run, data, date_filter, labels_dir="labels", part="dev"):
    """{query_id: (source/split, {metric: value})} for every scored query of a run.
    part: dev (default) / test / all - the evaluation-query half from DATA/labels/devtest.parquet (build/devtest.py).
    Decisions use dev; test is for the final numbers only. Splits without dev/test assignment (e.g. holdout_time) are kept."""
    import duckdb
    c = duckdb.connect()
    if os.environ.get("PAB_DUCKDB_MEM"):   # cap DuckDB memory (default: 80% of RAM)
        c.execute(f"SET memory_limit='{os.environ['PAB_DUCKDB_MEM']}'; SET threads=2")
    date_cond = "date_ok" if date_filter == "pub" else "TRUE"
    c.execute(f"""CREATE TABLE rel AS SELECT query_id, target_family_id AS family_id, grade FROM '{data}/{labels_dir}/labels.parquet'
                  WHERE keep AND {date_cond}""")
    c.execute(f"CREATE TABLE run AS SELECT * FROM '{run}'")
    c.execute(f"CREATE TABLE sp AS SELECT query_id, source, split FROM '{data}/{labels_dir}/splits.parquet'")
    if part != "all":
        dt = f"{data}/labels/devtest.parquet"
        if not os.path.exists(dt):
            raise SystemExit(f"{dt} missing (build/devtest.py); use --part all to score every evaluation query")
        c.execute(f"""DELETE FROM sp WHERE query_id IN (SELECT query_id FROM '{dt}' WHERE part <> '{part}')""")
    rows = c.execute("""SELECT r.query_id, sp.source, sp.split, r.family_ids, list(rel.family_id), list(rel.grade)
                        FROM run r JOIN sp USING (query_id) JOIN rel USING (query_id) GROUP BY ALL""").fetchall()
    out = {}
    for qid, src, split, ranked, rel_f, rel_g in rows:
        g = dict(zip(rel_f, rel_g))
        seen, dedup = set(), []
        for f in ranked:              # a family listed twice (duplicate corpus rows) keeps its best rank
            if f not in seen:
                seen.add(f); dedup.append(f)
        pos = {f: i for i, f in enumerate(dedup) if f in g}
        m = {}
        for kk in (10, 100, 1000):
            m[f"recall@{kk}"] = sum(1 for f in g if pos.get(f, 1 << 30) < kk) / len(g)
        dcg = sum(g[f] / math.log2(i + 2) for f, i in pos.items() if i < 10)
        ideal = sum(v / math.log2(i + 2) for i, v in enumerate(sorted(g.values(), reverse=True)[:10]))
        m["ndcg@10"] = dcg / ideal if ideal else 0.0
        first = min(pos.values()) if pos else None
        m["mrr@10"] = 1.0 / (first + 1) if first is not None and first < 10 else 0.0
        out[qid] = (f"{src}/{split}", m)
    return out


def cmd_metrics(a):
    pq_ = per_query(a.run, a.data, a.date_filter, a.labels_dir, a.part)
    agg = {}
    for qid, (key0, m) in pq_.items():
        for key in (key0, "all"):
            d = agg.setdefault(key, {"queries": 0})
            d["queries"] += 1
            for mk, v in m.items():
                d[mk] = d.get(mk, 0.0) + v
    out = {key: {mk: (round(v / d["queries"], 4) if mk != "queries" else v) for mk, v in d.items()} for key, d in sorted(agg.items())}
    out["_args"] = vars(a)
    json.dump(out, open(a.out, "w"), indent=1)
    for key, d in out.items():
        if key != "_args":
            print(f"{key:40s} n={d['queries']:6d}  R@10={d['recall@10']:.4f}  R@100={d['recall@100']:.4f}  "
                  f"R@1000={d['recall@1000']:.4f}  nDCG@10={d['ndcg@10']:.4f}  MRR@10={d['mrr@10']:.4f}")


def cmd_compare(a):
    """Paired bootstrap over queries scored in both runs: mean difference (B - A) with a 95% interval, per set."""
    A = per_query(a.run_a, a.data, a.date_filter, a.labels_dir, a.part)
    Bq = per_query(a.run_b, a.data, a.date_filter, a.labels_dir, a.part)
    common = sorted(set(A) & set(Bq))
    groups = {}
    for q in common:
        groups.setdefault(A[q][0], []).append(q)
        groups.setdefault("all", []).append(q)
    rng = np.random.default_rng(0)
    out = {}
    for key, qs in sorted(groups.items()):
        res = {"queries": len(qs)}
        for mk in a.metrics.split(","):
            d = np.array([Bq[q][1][mk] - A[q][1][mk] for q in qs])
            boots = d[rng.integers(0, len(d), size=(a.n_boot, len(d)))].mean(1)
            res[mk] = {"a": round(float(np.mean([A[q][1][mk] for q in qs])), 4), "b": round(float(np.mean([Bq[q][1][mk] for q in qs])), 4),
                       "diff": round(float(d.mean()), 4), "ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]}
        out[key] = res
    json.dump({"a": a.run_a, "b": a.run_b, "labels_dir": a.labels_dir, "part": a.part, "results": out}, open(a.out, "w"), indent=1)
    for key, r in out.items():
        cells = "  ".join(f"{mk} {r[mk]['a']:.4f}->{r[mk]['b']:.4f} ({r[mk]['diff']:+.4f} [{r[mk]['ci95'][0]:+.4f},{r[mk]['ci95'][1]:+.4f}])"
                          for mk in a.metrics.split(","))
        print(f"{key:36s} n={r['queries']:6d}  {cells}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("encode", "search"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True)
        p.add_argument("--adapter", default=None)
        p.add_argument("--pooling", default="last", choices=["last", "mean", "cls"])
        p.add_argument("--dim", type=int, default=None, help="truncate (Matryoshka) and re-normalize")
        p.add_argument("--batch", type=int, default=32)
        p.add_argument("--attn", default=None)
        p.add_argument("--gpu-mem", type=float, default=None, help="vLLM GPU memory fraction (share a GPU with training: ~0.3)")
        p.add_argument("--backend", default="hf", choices=["hf", "vllm"],
                       help="vllm: --model must be an exported checkpoint (train/export.py) or a stock model; no --adapter")
    pe = sub.choices["encode"]
    pe.add_argument("--tac", required=True, help="glob of document shards (title+abstract+claims, or with description)")
    pe.add_argument("--text-col", default="tac_text", help="document text column (descriptions ablation: tacd_text)")
    pe.add_argument("--families", default=None, help="parquet with family_id: encode only these (repairs)")
    pe.add_argument("--out", required=True)
    pe.add_argument("--d-len", type=int, default=4096)
    pe.add_argument("--max-tokens", type=int, default=None, help="token budget per batch (shards are length-sorted)")
    pe.add_argument("--shard", type=int, default=0)
    pe.add_argument("--nshards", type=int, default=1)
    ps = sub.choices["search"]
    ps.add_argument("--emb", required=True)
    ps.add_argument("--data", required=True)
    ps.add_argument("--splits", default="gold")
    ps.add_argument("--out", required=True)
    ps.add_argument("--k", type=int, default=1000)
    ps.add_argument("--q-len", type=int, default=2048)
    ps.add_argument("--instruction", default=None, help="override; default: per source (common.instruction_for)")
    ps.add_argument("--query-field", default="tac_text", choices=["tac_text", "independent_claims", "claim1", "each_independent"])
    ps.add_argument("--date-filter", default="pub", choices=["pub", "none"])
    ps.add_argument("--max-queries", type=int, default=0)
    ps.add_argument("--sample-mod", type=int, default=1, help="keep queries with hash %% N == 0 (subsample training queries)")
    ps.add_argument("--queries", default=None, help="parquet with query_id: search exactly these queries (any split; e.g. a training "
                    "pack's queries for mining) instead of --splits")
    ps.add_argument("--qshard", type=int, default=0, help="with --nqshards: this machine's share of the queries (md5 of query_id)")
    ps.add_argument("--nqshards", type=int, default=1)
    ps.add_argument("--corpus-chunk", type=int, default=500000)
    ps.add_argument("--q-chunk", type=int, default=1024)
    pm = sub.add_parser("metrics")
    pm.add_argument("--run", required=True)
    pm.add_argument("--data", required=True)
    pm.add_argument("--out", required=True)
    pm.add_argument("--date-filter", default="pub", choices=["pub", "none"])
    pm.add_argument("--part", default="dev", choices=["dev", "test", "all"], help="evaluation-query half (build/devtest.py); test only for final numbers")
    pm.add_argument("--labels-dir", default="labels", help="labels/splits directory under DATA (ablation B: labels_b)")
    pc = sub.add_parser("compare")
    pc.add_argument("--run-a", required=True)
    pc.add_argument("--run-b", required=True)
    pc.add_argument("--data", required=True)
    pc.add_argument("--out", required=True)
    pc.add_argument("--date-filter", default="pub", choices=["pub", "none"])
    pc.add_argument("--part", default="dev", choices=["dev", "test", "all"], help="evaluation-query half (build/devtest.py); test only for final numbers")
    pc.add_argument("--labels-dir", default="labels")
    pc.add_argument("--metrics", default="recall@10,recall@100,ndcg@10")
    pc.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    {"encode": cmd_encode, "search": cmd_search, "metrics": cmd_metrics, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
