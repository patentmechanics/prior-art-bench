#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
"""Check the shipped vectors against the published model on your own machine (no GPU needed).

Draws a random sample of evaluation queries and collection documents, encodes them with sentence-transformers and
compares each vector with the shipped one. Pass = every cosine >= --min-cos (default 0.99; bf16 arithmetic on different
hardware typically gives 0.999+).

  pip install -r requirements.txt
  python verify_sample.py --model patentmechanics/anticipate-0.6b --n-queries 100 --n-docs 300 --seed <your own>
  python verify_sample.py --data /path/to/prior-art-bench-v1 ...          # a local copy of the dataset

DATA (default: the Hugging Face dataset; files are downloaded into the Hugging Face cache on first use):
  queries/queries.parquet                      query_id, source, text (the exact query text that was encoded)
  corpus-pool-v2/part-*.parquet                family_id, text (the exact document text that was encoded)
                                               (--docs-from sample: sample/corpus.parquet, 10,000 families, smaller download)
  vectors/anticipate-0.6b/ (or --vectors)      documents/<file>.npy + <file>.family_ids.npy, float16 [n, 1024];
                                               queries/queries.npy + queries.query_ids.parquet
Encoding settings are the ones used for the published numbers: queries with the `query` prompt (office actions) or the
`invalidity_query` prompt (PTAB), truncated at 4,096 tokens; documents with no prompt, truncated at 8,192 tokens.
"""
import argparse, glob, json, os, platform, random, sys, time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pab_data as D  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="patentmechanics/anticipate-0.6b")
ap.add_argument("--revision", default=None)
ap.add_argument("--data", default=D.HF_DATASET)
ap.add_argument("--vectors", default=None, help=f"default DATA/{D.VECTORS}")
ap.add_argument("--docs-from", default="pool", choices=["pool", "sample"])
ap.add_argument("--query-text-col", default="text")
ap.add_argument("--doc-text-col", default="text")
ap.add_argument("--n-queries", type=int, default=100)
ap.add_argument("--n-docs", type=int, default=300)
ap.add_argument("--seed", type=int, default=0, help="choose your own, so the sample is not ours")
ap.add_argument("--min-cos", type=float, default=0.99)
ap.add_argument("--device", default=None, help="cpu | mps | cuda (default: best available)")
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--out", default=None)
a = ap.parse_args()

from sentence_transformers import SentenceTransformer
import torch

dev = a.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model = SentenceTransformer(a.model, revision=a.revision, device=dev, model_kwargs={"torch_dtype": torch.bfloat16})
rng = random.Random(a.seed)
vec = D.fetch(a.vectors or f"{a.data.rstrip('/')}/{D.VECTORS}", ["documents/*", "queries/*"])

def cosine(v, ref):
    # Both sides re-normalised in float32: the model's bf16 output and the float16 shipped vector are only near unit length.
    v = np.asarray(v, dtype=np.float32)
    return float(v @ ref / (np.linalg.norm(v) * np.linalg.norm(ref)))


# queries
qv = np.load(f"{vec}/queries/queries.npy", mmap_mode="r")
qids = pq.read_table(f"{vec}/queries/queries.query_ids.parquet").column("query_id").to_pylist()
qrow = {q: i for i, q in enumerate(qids)}
qt = pq.read_table(D.data_file(a.data, "queries/queries.parquet"), columns=["query_id", "source", a.query_text_col])
rows = [r for r in qt.to_pylist() if r["query_id"] in qrow and r[a.query_text_col]]
qs = rng.sample(rows, min(a.n_queries, len(rows)))

# documents
files = sorted(f for f in glob.glob(f"{vec}/documents/*.npy") if not f.endswith(".family_ids.npy"))
dvec, drow = [], {}
for f in files:
    ids = np.load(f[:-4] + ".family_ids.npy")
    for j, fam in enumerate(ids.tolist()):
        drow[fam] = (len(dvec), j)
    dvec.append(np.load(f, mmap_mode="r"))
doc_files = ([D.data_file(a.data, "sample/corpus.parquet")] if a.docs_from == "sample"
             else D.files(a.data, "corpus-pool-v2/part-*.parquet"))
if a.docs_from == "sample":   # draw from the families of the sample collection only
    have = set(pq.read_table(doc_files[0], columns=["family_id"]).column(0).to_pylist())
    cand = sorted(f for f in drow if f in have)
else:
    cand = sorted(drow)
pick = set(rng.sample(cand, min(a.n_docs, len(cand))))
docs, got = [], set()
for f in doc_files:
    t = pq.read_table(f, columns=["family_id", a.doc_text_col])
    t = t.filter(pc.is_in(t.column("family_id"), value_set=pa.array(sorted(pick), type=t.schema.field("family_id").type)))
    for r in t.to_pylist():
        if r["family_id"] not in got:   # one text per family
            got.add(r["family_id"]); docs.append(r)

t0 = time.time()
res = {"queries": [], "documents": []}
for src in ("query", "invalidity_query"):
    sel = [r for r in qs if (r["source"] == "ptab") == (src == "invalidity_query")]
    if not sel:
        continue
    model.max_seq_length = 4096
    e = model.encode([r[a.query_text_col] for r in sel], prompt_name=src, batch_size=a.batch, normalize_embeddings=True)
    for r, v in zip(sel, e):
        ref = np.asarray(qv[qrow[r["query_id"]]], dtype=np.float32)
        res["queries"].append(cosine(v, ref))
model.max_seq_length = 8192
e = model.encode([d[a.doc_text_col] for d in docs], batch_size=a.batch, normalize_embeddings=True)
for d, v in zip(docs, e):
    fi, j = drow[d["family_id"]]
    ref = np.asarray(dvec[fi][j], dtype=np.float32)
    res["documents"].append(cosine(v, ref))

out = {"model": a.model, "device": dev, "platform": platform.platform(), "seed": a.seed, "seconds": round(time.time() - t0, 1)}
for k, v in res.items():
    v = np.asarray(v)
    out[k] = {"n": int(len(v)), "min_cos": float(v.min()) if len(v) else None, "mean_cos": float(v.mean()) if len(v) else None,
              "below_threshold": int((v < a.min_cos).sum())}
out["pass"] = all(out[k]["n"] > 0 and out[k]["below_threshold"] == 0 for k in res)
print(json.dumps(out, indent=1))
if a.out:
    json.dump(out, open(a.out, "w"), indent=1)
raise SystemExit(0 if out["pass"] else 1)
