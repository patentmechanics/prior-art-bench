#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
"""Search-time stages of the recommended system, driven by fusion-v1.json:

  1. passages:   family score = max(whole-document score, best description-chunk score), same model, same cosine scale
  2. cpc_strict: fused = z(score) + w x level / 3, level = 3 same full CPC symbol, 2 same main group, 1 same subclass,
                 0 none, between the CPC printed on the query's document and the current CPC of the candidate family's
                 members published before the query's priority date
  3. bm25:       fused = z(previous) + w x z(BM25) over the union of both top-1000 lists (missing = that list's minimum)

  python evaluate.py search --collection documents --part test --out runs/model.test.parquet
  python evaluate.py search --collection chunks    --part test --out runs/chunks.test.parquet      # k 4,000 chunks
  python baselines/bm25.py --part test --out runs/bm25.test.parquet        # or the shipped runs/bm25.test.parquet
  python fusion.py --dense runs/model.test.parquet --chunks runs/chunks.test.parquet --bm25 runs/bm25.test.parquet \
      --config fusion-v1.json --out runs/system.test.parquet [--save-stages]
  python evaluate.py metrics --run runs/system.test.parquet --part test

Every stage is a port of the code that produced the published runs, including the order of operations and the float32
run files written between stages (--save-stages writes them: <out>.passages.parquet, <out>.passages_cpc.parquet).
The CPC stage reads cpc-members/cpc_members.parquet and queries/queries.parquet (priority_date, cpc_printed) from --data.
"""
import argparse
import datetime
import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pab_data as D  # noqa: E402
from evaluate import read_run, write_run  # noqa: E402


def _f32(xs):
    return [float(np.float32(x)) for x in xs]


def _rows(ranked, k):
    """{query_id: [(family, score)] best first} -> run rows (query ids sorted, top k, float32 scores as written)"""
    return [{"query_id": q, "family_ids": [f for f, _ in ranked[q][:k]], "scores": _f32([s for _, s in ranked[q][:k]])}
            for q in sorted(ranked)]


def _best(rows):
    out = {}
    for r in rows:
        d = {}
        for f, s in zip(r["family_ids"], r["scores"]):
            if f not in d or s > d[f]:
                d[f] = s
        out[r["query_id"]] = d
    return out


def passages(doc_rows, chunk_rows, k=1000, missing=-9.0):
    """max of the document score and the family's best chunk score, for queries present in both runs"""
    Dm, C = _best(doc_rows), _best(chunk_rows)
    mx = {}
    for q in set(Dm) & set(C):
        d, c = Dm[q], C[q]
        mx[q] = sorted(((f, max(d.get(f, missing), c.get(f, missing))) for f in set(d) | set(c)), key=lambda x: -x[1])
    return _rows(mx, k)


def cpc_levels(syms):
    """(full symbols, main groups, subclasses) of CPC symbols written like 'B61C11/04'"""
    return ({x for x in syms}, {x.split("/")[0] for x in syms}, {x[:4] for x in syms})


def cpc_agree(Q, F):
    return 3 if Q[0] & F[0] else 2 if Q[1] & F[1] else 1 if Q[2] & F[2] else 0


def _yyyymmdd(v):
    if v is None:
        return 0
    if isinstance(v, (datetime.date, datetime.datetime)):
        return v.year * 10000 + v.month * 100 + v.day
    return int(v)


def cpc_strict(base_rows, member_rows, query_rows, w):
    """base_rows: run rows; member_rows: {family_id, pub_date, cpc_current, cpc_printed}; query_rows: {query_id,
    priority_date, cpc_printed}. Re-orders each query's list by z(score) + w * level / 3 (stable sort)."""
    docs = {}
    for r in member_rows:
        docs.setdefault(r["family_id"], []).append((_yyyymmdd(r["pub_date"]), r["cpc_current"] or [], r["cpc_printed"] or []))
    qdate = {r["query_id"]: _yyyymmdd(r["priority_date"]) for r in query_rows}
    qp = {r["query_id"]: cpc_levels(r["cpc_printed"]) for r in query_rows if r["cpc_printed"]}

    def fam_before(f, q):   # current CPC of the family's members published before the query's priority date
        syms = {x for d in docs.get(f, ()) if d[0] < qdate[q] for x in d[1]}
        return cpc_levels(syms) if syms else None
    out = []
    for r in base_rows:
        q = r["query_id"]
        sc = np.asarray(r["scores"], dtype=np.float64)
        z = (sc - sc.mean()) / (sc.std() + 1e-9)
        lv = np.zeros(len(sc))
        if q in qp:
            Q = qp[q]
            for i, f in enumerate(r["family_ids"]):
                F = fam_before(f, q)
                if F:
                    lv[i] = cpc_agree(Q, F)
        lv = lv / 3.0
        fams = r["family_ids"]
        s_ = z + w * lv
        o = np.argsort(-s_, kind="stable")
        out.append({"query_id": q, "family_ids": [fams[i] for i in o], "scores": _f32(sorted(s_.tolist(), reverse=True))})
    return out


def _zs(v):
    v = np.asarray(v, dtype=np.float64)
    return (v - v.mean()) / (v.std() + 1e-9)


def zscore_fusion(base_rows, other_runs, weights, k=1000):
    """fused = z(base) + sum_i w_i z(list_i) over the union of the lists; a family missing from a list gets that list's
    minimum z. other_runs: [run rows]; output covers every query of the base run."""
    base = {r["query_id"]: (r["family_ids"], r["scores"]) for r in base_rows}
    others = [{r["query_id"]: (r["family_ids"], r["scores"]) for r in rows} for rows in other_runs]

    def zmap(fams, scr):
        x = _zs(np.asarray(scr, dtype=np.float64))
        return dict(zip(fams, x)), (float(x.min()) if len(x) else 0.0)
    ranked = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # a query missing from a list: mean of an empty list
        for q in sorted(base):
            parts = [zmap(*base[q])] + [zmap(*o.get(q, ([], []))) for o in others]
            U = set().union(*(p[0] for p in parts))
            s = {f: parts[0][0].get(f, parts[0][1]) + sum(w * p[0].get(f, p[1]) for w, p in zip(weights, parts[1:])) for f in U}
            ranked[q] = sorted(s.items(), key=lambda x: -x[1])[:k]
    return _rows(ranked, k)


def load_cpc(data):
    import pyarrow.parquet as pq
    members = pq.read_table(D.data_file(data, "cpc-members/cpc_members.parquet"),
                            columns=["family_id", "pub_date", "cpc_current", "cpc_printed"]).to_pylist()
    queries = pq.read_table(D.data_file(data, "queries/queries.parquet"),
                            columns=["query_id", "priority_date", "cpc_printed"]).to_pylist()
    return members, queries


def main():
    ap = argparse.ArgumentParser(description="passages + CPC strict + BM25 fusion (fusion-v1.json)")
    ap.add_argument("--dense", required=True, help="document run of the model (evaluate.py search)")
    ap.add_argument("--chunks", default=None, help="chunk run of the same model (evaluate.py search --collection chunks)")
    ap.add_argument("--bm25", default=None, help="BM25 run (baselines/bm25.py, or runs/bm25.<part>.parquet of the dataset)")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "fusion-v1.json"))
    ap.add_argument("--stages", default=None, help="comma list (default: the config's stages)")
    ap.add_argument("--data", default=D.HF_DATASET)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-stages", action="store_true")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    stages = a.stages.split(",") if a.stages else cfg["stages"]
    k = cfg.get("k", 1000)
    rows = read_run(a.dense)
    done = []
    for st in stages:
        if st == "passages":
            if not a.chunks:
                raise SystemExit("passages stage: --chunks is required")
            rows = passages(rows, read_run(a.chunks), k, cfg["passages"].get("missing_score", -9.0))
        elif st == "cpc_strict":
            members, queries = load_cpc(a.data)
            rows = cpc_strict(rows, members, queries, cfg["cpc_strict"]["weight"])
        elif st == "bm25":
            if not a.bm25:
                raise SystemExit("bm25 stage: --bm25 is required")
            rows = zscore_fusion(rows, [read_run(a.bm25)], [cfg["bm25"]["weight"]], k)
        else:
            raise SystemExit(f"unknown stage {st}")
        done.append(st)
        print(json.dumps({"stage": st, "queries": len(rows)}), flush=True)
        if a.save_stages and st != stages[-1]:
            p = f"{a.out[:-len('.parquet')] if a.out.endswith('.parquet') else a.out}.{'_'.join(x.replace('_strict', '') for x in done)}.parquet"
            write_run(p, [r["query_id"] for r in rows], [r["family_ids"] for r in rows], [r["scores"] for r in rows])
    write_run(a.out, [r["query_id"] for r in rows], [r["family_ids"] for r in rows], [r["scores"] for r in rows])
    print(json.dumps({"written": a.out, "stages": done}))


if __name__ == "__main__":
    main()
