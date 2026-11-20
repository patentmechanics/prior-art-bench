#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
"""The blog post's other chat-app figures, from the files score_chat_apps.py reads and writes.
  named   QUERIES.json PARSED.json                     answers that name the application itself (its publication or
                                                       application number), per app, main 30
  paired  QUERIES.json PER_QUERY.json A B [--n 2000]   B minus A in questions with a relied-on reference in the top 10
                                                       (hit@10), paired bootstrap over the 30 queries, 95% range
  repeat  QUERIES.json PARSED.json MAP.json            same question asked twice in new chats: families in both top-10 lists
  repeat-ours QUERIES.json REPEAT.json                 the same for our full system (ours/full_system_repeat.json: its top
                                                       10 with the reranker run twice)
PER_QUERY.json is the *.per_query_metrics.json that `score_chat_apps.py report` writes next to PARSED.json."""
import argparse, json, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_chat_apps import metrics  # noqa: E402


def main30(qfile):
    Q = json.load(open(qfile))
    return {q["query_id"]: q for q in Q if q.get("set") == "main 30"}


def named(a):
    Q, P = main30(a.queries), json.load(open(a.parsed))
    apps = sorted({k.split("|")[0] for k in P})
    for app in apps:
        rows = [(qid, P[f"{app}|{qid}"]["own_named_in_answer"]) for qid in Q if f"{app}|{qid}" in P]
        print(f"{app:28s} named the application in {sum(v for _, v in rows)} of {len(rows)} answers")


def paired(a):
    Q, M = main30(a.queries), json.load(open(a.per_query))
    qs = [q for q in Q if q in M[a.a] and q in M[a.b]]
    d = [M[a.b][q]["hit10"] - M[a.a][q]["hit10"] for q in qs]
    rng = random.Random(a.seed)
    bs = sorted(sum(rng.choice(d) for _ in d) for _ in range(a.n))
    lo, hi = bs[int(0.025 * a.n)], bs[int(0.975 * a.n) - 1]
    print(f"{a.b} minus {a.a}: {sum(d):+.0f} questions of {len(qs)} (95% range {lo:+.0f} to {hi:+.0f}; {a.n} resamples, seed {a.seed})")


def repeat(a):
    Q, P, M = main30(a.queries), json.load(open(a.parsed)), json.load(open(a.map))
    nn = {q: i for i, q in enumerate(Q, 1)}

    def top(v, q):   # top 10 as families (a document outside our corpus is kept under its own number), own family dropped
        out = []
        for k in v["keys"]:
            if not k:
                continue
            f = M.get(k, [None])[0]
            f = f if f is not None else k
            if f != Q[q]["family_id"] and f not in out:
                out.append(f)
        return out[:10]

    apps = sorted({k.split("|")[0].split("@")[0] for k in P})
    for app in apps:
        rows, flips = [], 0
        for q in sorted(nn, key=nn.get):
            x, y = P.get(f"{app}@first|{q}"), P.get(f"{app}@second|{q}")
            if not (x and y):
                continue
            ta, tb = top(x, q), top(y, q)
            fam = lambda t: [v if isinstance(v, int) else None for v in t]
            ha, hb = metrics(fam(ta), Q[q])["hit10"], metrics(fam(tb), Q[q])["hit10"]
            flips += ha != hb
            rows.append(len(set(ta) & set(tb)))
            print(f"{app:18s} query {nn[q]:02d}: {rows[-1]} of 10 in both lists; hit@10 first {int(ha)}, second {int(hb)}")
        if rows:
            print(f"{app:18s} mean {sum(rows) / len(rows):.1f} of 10 in both lists over {len(rows)} queries; "
                  f"hit@10 changed on {flips} of {len(rows)}\n")


def repeat_ours(a):
    Q, R = main30(a.queries), json.load(open(a.repeat))
    nn = {q: i for i, q in enumerate(Q, 1)}
    rows = []
    for q in sorted(R["first"], key=lambda x: nn.get(x, 99)):
        x, y = R["first"][q], R["second"][q]
        rows.append(len(set(x) & set(y)))
        print(f"full-system        query {nn.get(q, 0):02d}: {rows[-1]} of 10 in both lists; same order {x == y}")
    print(f"full-system        mean {sum(rows) / len(rows):.1f} of 10 in both lists over {len(rows)} queries")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("named"); p.add_argument("queries"); p.add_argument("parsed")
    p = sp.add_parser("paired"); p.add_argument("queries"); p.add_argument("per_query"); p.add_argument("a"); p.add_argument("b")
    p.add_argument("--n", type=int, default=2000); p.add_argument("--seed", type=int, default=0)
    p = sp.add_parser("repeat"); p.add_argument("queries"); p.add_argument("parsed"); p.add_argument("map")
    p = sp.add_parser("repeat-ours"); p.add_argument("queries"); p.add_argument("repeat")
    a = ap.parse_args()
    {"named": named, "paired": paired, "repeat": repeat, "repeat-ours": repeat_ours}[a.cmd](a)
