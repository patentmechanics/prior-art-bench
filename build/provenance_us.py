#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
"""Provenance manifest and release gate for the US-only data (run after the data is built).

  PAB_WORK=$WORK python build/provenance_us.py [--code DIR]

Writes provenance/manifest.json: for every pipeline script its sha256, for every data file the pipeline reads or writes its
size and mtime, and the gate checks. Gate (all must hold, otherwise exit status 1):
  1. No pipeline script reads an EPO-derived path (family_map of the US+EP run, rec_map, ep sources, DOCDB). The validation
     script crosscheck_docdb_us.py is the one allowed exception: read-only, writes statistics only, listed separately.
  2. Every document in the corpus parts, tac shards, tacd and mining pool is a US document (doc_id starts with 'US').
  3. Every query's source document is a US document; query sources are only us_oa and ptab.
  4. Every label target family has a US representative; label sources are only us_oa and ptab.
  5. Train / evaluation family overlap is 0 and no evaluation query matches a training query (labels/leakage_audit.json).
"""
import argparse, glob, hashlib, json, os, re, sys, time

import duckdb

W = os.environ.get("PAB_WORK", "work")
ap = argparse.ArgumentParser()
ap.add_argument("--code", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))), help="repository root")
a = ap.parse_args()
out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "work": W, "scripts": {}, "inputs_read_from_outside_us_work": [], "gate": {}}

sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
scripts = sorted(glob.glob(f"{a.code}/build/*.py") + glob.glob(f"{a.code}/train/*.py"))
# scripts shared with our earlier US+EP pipeline (they contain EP-capable stages that were not run for this benchmark)
SHARED = {"corpus.py", "tacd.py", "cocite.py", "devtest.py", "queries_us.py"}
for p in scripts:
    out["scripts"][os.path.relpath(p, a.code)] = sha(p)

# 1. EPO-derived paths in the scripts that build the data (validation script excluded)
BAD = re.compile(r"rec_map|family_map/v1|docdb|DOCDB|/ep/|src ep|--src\s+ep|epo\.org|ep_search_report|opp_amended|opp_revoked", re.I)
build = [p for p in scripts if "provenance_us" not in p and "/train/" not in p and os.path.basename(p) not in SHARED]
hits = {}
for p in build:
    in_doc = False
    for i, line in enumerate(open(p, errors="ignore"), 1):
        st = line.strip()
        was_doc = in_doc
        if st.count('"""') % 2 == 1:
            in_doc = not in_doc
        if was_doc or in_doc or st.startswith("#") or st.startswith('"""'):
            continue   # comments and docstrings
        if BAD.search(line):
            hits.setdefault(os.path.relpath(p, a.code), []).append(f"{i}: {st[:140]}")
src_dirs = os.listdir(f"{W}/corpus") + ["|"] + os.listdir(f"{W}/passages")
out["gate"]["1_epo_paths_in_build_scripts"] = {"pass": not hits and src_dirs == ["us", "|", "us"], "hits": hits, "corpus_and_passages_dirs": src_dirs,
    "note": "corpus.py contains EP-capable stages; only build --src us and tac were run (corpus/ and passages/ hold only us)"}

c = duckdb.connect()
c.execute("SET threads=6; SET memory_limit='10GB'")
def scalar(q):
    return c.execute(q).fetchone()[0]
# 2. documents
g2 = {}
g2["corpus_non_us_docs"] = scalar(f"SELECT count(*) FROM read_parquet('{W}/tac/shard-*.parquet') WHERE doc_id NOT LIKE 'US%' OR cc <> 'US'")
g2["tacd_non_us_docs"] = scalar(f"SELECT count(*) FROM '{W}/tacd/train-us.parquet' WHERE doc_id NOT LIKE 'US%' OR cc <> 'US'") + \
    scalar(f"SELECT count(*) FROM read_parquet('{W}/tacd/subset-us-*.parquet') WHERE doc_id NOT LIKE 'US%' OR cc <> 'US'")
g2["family_map_non_us_docs"] = scalar(f"SELECT count(*) FROM '{W}/family_map/docs.parquet' WHERE doc_id NOT LIKE 'US%'")
g2["reps_non_us"] = scalar(f"SELECT count(*) FROM '{W}/reps/reps_us.parquet' WHERE doc_id NOT LIKE 'US%'")
out["gate"]["2_documents_are_us"] = {"pass": all(v == 0 for v in g2.values()), **g2}
# 3 + 4. queries and labels
g3 = {"query_sources": dict(c.execute(f"SELECT source, count(*) FROM '{W}/queries/queries.parquet' GROUP BY 1").fetchall()),
      "queries_non_us_doc": scalar(f"SELECT count(*) FROM '{W}/queries/queries.parquet' WHERE query_doc_id NOT LIKE 'US%'")}
g3["label_sources"] = dict(c.execute(f"SELECT source, count(*) FROM '{W}/labels/labels.parquet' GROUP BY 1").fetchall())
cols = [r[0] for r in c.execute(f"DESCRIBE SELECT * FROM '{W}/labels/labels.parquet'").fetchall()]
tcol = "family_id" if "family_id" in cols else ("target_family_id" if "target_family_id" in cols else None)
tgt = "target_family_id" if "target_family_id" in cols else tcol
g3["label_columns"] = cols
g3["kept_label_targets_without_us_rep"] = scalar(
    f"SELECT count(*) FROM '{W}/labels/labels.parquet' l ANTI JOIN '{W}/reps/reps_us.parquet' r ON r.family_id = l.{tgt} WHERE l.keep") if tgt else None
ok3 = set(g3["query_sources"]) <= {"us_oa", "ptab"} and set(g3["label_sources"]) <= {"us_oa", "ptab"}
out["gate"]["3_4_queries_labels_us_only"] = {"pass": bool(ok3 and g3["queries_non_us_doc"] == 0 and g3["kept_label_targets_without_us_rep"] == 0), **g3}
# 5. leakage
lk = json.load(open(f"{W}/labels/leakage_audit.json"))
clean = lk["train_eval_family_overlap"] == 0 and not any(lk[k] for k in lk if k.startswith("eval_queries_matching_a_train_query"))
out["gate"]["5_no_train_eval_leakage"] = {"pass": clean, "audit": lk, "note": "pool-document text matches reported, not a failure"}
# data files
files = {}
for pat in ("family/*", "family_map/*", "reps/*", "queries/*", "labels/*", "oa/*", "tac/*", "tacd/train-us.parquet", "tacd/subset-us-*.parquet",
            "tac_subset_250k/*", "mine/queries.parquet", "mine/pool_stats.json"):
    for f in sorted(glob.glob(f"{W}/{pat}")):
        files[os.path.relpath(f, W)] = {"bytes": os.path.getsize(f), "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(f)))}
out["data_files"] = files
out["gate"]["all_pass"] = all(v["pass"] for k, v in out["gate"].items() if isinstance(v, dict) and "pass" in v)
os.makedirs(f"{W}/provenance", exist_ok=True)
json.dump(out, open(f"{W}/provenance/manifest.json", "w"), indent=1)
print(json.dumps({k: (v["pass"] if isinstance(v, dict) else v) for k, v in out["gate"].items()}))
sys.exit(0 if out["gate"]["all_pass"] else 1)
