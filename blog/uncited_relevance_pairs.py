#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
"""Uncited-relevance check: build the blind (query, document) pairs for the judge.
Strata (the judge never sees them):
  top10_uncited   our full system's top 10 (query's own family removed) minus the examiner's scored X/Y families
  examiner_xy     every scored examiner X/Y family of the query (calibration: how often the judge agrees with examiners)
  decoy           2 per query from our ranks 151-200, not examiner-cited, picked in md5 order (calibration: false positives)
Pair IDs and order are md5-based, so anyone rebuilding from the same inputs gets the same list.
  fams    QUERIES.json FULL_TOP200.json FAMS.json            -> family ids whose texts are needed (store 1 step)
  build   QUERIES.json FULL_TOP200.json TEXTS.jsonl PAIRS.jsonl"""
import hashlib, json, sys
md5 = lambda s: hashlib.md5(s.encode()).hexdigest()
SALT = "uncited-relevance-v1"


def select(qfile, topfile):
    Q = [q for q in json.load(open(qfile)) if q.get("set") == "main 30"]
    top = {x["query_id"]: x["family_ids"] for x in json.load(open(topfile))}
    out = []
    for q in Q:
        ranked = [f for f in top[q["query_id"]] if f != q["family_id"]]; pos = set(q["positives"])
        for r, f in enumerate(ranked[:10], 1):
            if f not in pos: out.append((q, f, "top10_uncited", r))
        for f in sorted(pos): out.append((q, f, "examiner_xy", ranked.index(f) + 1 if f in ranked else None))
        band = [(r, f) for r, f in enumerate(ranked, 1) if 151 <= r <= 200 and f not in pos]
        for r, f in sorted(band, key=lambda x: md5(f"{SALT}|{q['query_id']}|{x[1]}"))[:2]: out.append((q, f, "decoy", r))
    return out


if __name__ == "__main__":
    cmd, a = sys.argv[1], sys.argv[2:]
    S = select(a[0], a[1])
    if cmd == "fams":
        json.dump(sorted({f for _, f, _, _ in S}), open(a[2], "w")); print(len(S), "pairs,", len({f for _, f, _, _ in S}), "families")
    else:
        T = {r["family_id"]: r for r in map(json.loads, open(a[2]))}
        rows = []
        for q, f, stratum, rank in S:
            t = T.get(f)
            rows.append({"pair_id": md5(f"{SALT}|{q['query_id']}|{f}")[:12], "query_id": q["query_id"], "family_id": f, "stratum": stratum, "rank": rank,
                         "priority_date": q["priority_date"], "first_pub": t["first_pub"] if t else None,
                         "prior_art": bool(t and t["first_pub"] < q["priority_date"]), "has_text": bool(t)})
        rows.sort(key=lambda r: r["pair_id"])
        with open(a[3], "w") as o:
            for r in rows: o.write(json.dumps(r) + "\n")
        from collections import Counter
        print(len(rows), "pairs", Counter(r["stratum"] for r in rows), "no text", sum(not r["has_text"] for r in rows),
              "not prior art", Counter(r["stratum"] for r in rows if not r["prior_art"]))
