#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
"""Uncited-relevance check: blind LLM judge of (query claims, document) pairs, and the summary.
Judge = Claude Opus 5.5 through headless Claude Code (`claude -p`; needs the Claude Code CLI and an account), no tools, no MCP,
no skills, no session saved, run from an empty directory. It sees only the claims under examination and one document's
text; never the stratum, our rank, or whether the examiner cited it. Every document is dated before the query's priority date.
  judge    PAIRS.jsonl QUERIES.json TEXTS.jsonl OUTDIR --run 1 [--limit 20] [--workers 4]
  summary  PAIRS.jsonl QUERIES.json TEXTS.jsonl OUTDIR
Ratings land in OUTDIR/run<N>/<pair_id>.json (resumable: done pairs are skipped)."""
import argparse, collections, concurrent.futures as cf, json, os, random, re, subprocess, sys, tempfile, time

MODEL, EFFORT = "claude-opus-5-5", "medium"
PROMPT = """You are a senior patent examiner. Judge one candidate prior-art document against the claims of a patent application.
The candidate was published before the application's priority date, so it is prior art; ignore dates.

<claims_under_examination>
{q}
</claims_under_examination>

<candidate_document>
{d}
</candidate_document>

Method:
1. Identify the independent claims. Choose the one independent claim that the candidate document comes closest to meeting.
2. Split that claim into its elements (the preamble's essential features and each limitation of the body).
3. For each element decide whether the candidate document discloses it: "yes" (disclosed, quote the passage), "partly"
   (disclosed in part or only by a broad generic statement, quote the passage), or "no".
4. Give one category, as in a search report:
   X = the document alone discloses every element of the chosen claim (anticipation), or every element except differences
       that a person skilled in the art would find obvious without another document.
   Y = the document discloses most elements of the chosen claim, including its central inventive feature or most of the
       claim, and an examiner would reject the claim by combining it with one other document for the missing elements.
   A = same technical field or same problem, but the central features of the claim are missing; background only.
   N = not relevant.
   Be strict: an element is "yes" only if the quoted passage actually discloses it.

Reply with one JSON object only, no other text:
{{"claim": <number>, "elements": [{{"element": "<short>", "disclosed": "yes|partly|no", "evidence": "<short quote or empty>"}}], "category": "X|Y|A|N", "reason": "<one sentence>"}}"""


def judge(a):
    P = [json.loads(l) for l in open(a.pairs)]
    Q = {q["query_id"]: q for q in json.load(open(a.queries))}
    T = {r["family_id"]: r["text"] for r in map(json.loads, open(a.texts))}
    if a.stratum: P = [p for p in P if p["stratum"] == a.stratum]   # e.g. a repeat run on the uncited top-10 results only
    if a.limit:   # pilot: a fixed random sample, every stratum present
        P = random.Random(a.run).sample(P, a.limit)
    out = f"{a.outdir}/run{a.run}"; os.makedirs(out, exist_ok=True)
    wd = tempfile.mkdtemp(prefix="jd-"); nomcp = f"{wd}.mcp.json"; open(nomcp, "w").write('{"mcpServers": {}}')
    base = ["claude", "-p", "--model", MODEL, "--effort", EFFORT, "--tools", "", "--strict-mcp-config", "--mcp-config", nomcp,
            "--disable-slash-commands", "--no-session-persistence", "--exclude-dynamic-system-prompt-sections", "--output-format", "json"]

    def one(p):
        f = f"{out}/{p['pair_id']}.json"
        if os.path.exists(f): return "skip"
        prompt = PROMPT.format(q=Q[p["query_id"]]["text"], d=T[p["family_id"]])
        for attempt in range(4):
            t0 = time.time()
            r = subprocess.run(base, input=prompt, cwd=wd, capture_output=True, text=True, timeout=1800)
            try:
                res = json.loads(r.stdout); txt = res["result"]
                j = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
                assert j["category"] in ("X", "Y", "A", "N")
                json.dump({"pair_id": p["pair_id"], "rating": j, "seconds": round(time.time() - t0), "usage": res.get("usage"),
                           "model": MODEL, "effort": EFFORT}, open(f, "w"), indent=1)
                return j["category"]
            except Exception:
                err = (r.stderr or r.stdout or "")[-300:]
                if "limit" in err.lower(): time.sleep(600)   # account usage limit: wait, then retry
                else: time.sleep(20 * (attempt + 1))
        print("FAILED", p["pair_id"], err, flush=True); return "fail"

    with cf.ThreadPoolExecutor(a.workers) as ex:
        for i, (p, c) in enumerate(zip(P, ex.map(one, P)), 1):
            print(i, p["pair_id"], c, flush=True)


def summary(pairs, queries, texts, outdir):
    """Per run and over runs: X/Y/A/N per stratum; examiner references split by the examiner's own category (X = cited as X
    in the action, Y = cited only as Y); top 10 by rank band; run-to-run agreement; verbatim check of the evidence quotes;
    95% range for the headline (uncited top-10 rated X or Y, mean of runs) from 10,000 resamples of queries (seed 0)."""
    P = {p["pair_id"]: p for p in map(json.loads, open(pairs))}
    Q = {q["query_id"]: q for q in json.load(open(queries))}
    T = {r["family_id"]: r["text"] for r in map(json.loads, open(texts))}
    xfam = {(qid, r["family_id"]) for qid, q in Q.items() for r in q["examiner_references"] if "X" in r["categories"]}
    sub = lambda p: ("examiner_cited_X" if (p["query_id"], p["family_id"]) in xfam else "examiner_cited_Y_only") if p["stratum"] == "examiner_xy" else \
        ("top10_uncited r1-5" if p["rank"] <= 5 else "top10_uncited r6-10") if p["stratum"] == "top10_uncited" else p["stratum"]
    runs = sorted(d[3:] for d in os.listdir(outdir) if re.fullmatch(r"run\d+", d))
    R = {}
    for run in runs:
        for fn in os.listdir(f"{outdir}/run{run}"):
            if fn.endswith(".json"): R.setdefault(run, {})[fn[:-5]] = json.load(open(f"{outdir}/run{run}/{fn}"))["rating"]
    xy = lambda c: c in ("X", "Y")
    groups = ["top10_uncited", "top10_uncited r1-5", "top10_uncited r6-10", "examiner_xy", "examiner_cited_X", "examiner_cited_Y_only", "decoy"]
    member = lambda p, g: p["stratum"] == g or sub(p) == g
    for run in runs:
        print(f"\nrun {run}: {len(R[run])} of {len(P)} pairs rated")
        for g in groups:
            c = collections.Counter(R[run][k]["category"] for k in R[run] if member(P[k], g)); n = sum(c.values())
            if n: print(f"  {g:22s} n={n:3d}  X {c['X']:3d}  Y {c['Y']:3d}  A {c['A']:3d}  N {c['N']:3d}  X+Y {(c['X'] + c['Y']) / n:.1%}")
    full = [k for k in P if all(k in R[r] for r in runs)]
    if len(runs) > 1:
        print(f"\nover {len(runs)} runs ({len(full)} pairs rated in every run): mean X+Y share per group (min-max over runs)")
        for g in groups:
            ks = [k for k in full if member(P[k], g)]
            if ks:
                v = [sum(xy(R[r][k]["category"]) for k in ks) / len(ks) for r in runs]
                print(f"  {g:22s} n={len(ks):3d}  {sum(v) / len(v):.1%}  ({min(v):.1%}-{max(v):.1%})")
        same = sum(len({R[r][k]["category"] for r in runs}) == 1 for k in full); samexy = sum(len({xy(R[r][k]["category"]) for r in runs}) == 1 for k in full)
        print(f"  same category in every run: {same / len(full):.1%}; same X+Y vs not: {samexy / len(full):.1%}")
    quotes = [(e["evidence"], T[P[k]["family_id"]]) for r in runs for k in R[r] for e in R[r][k].get("elements", []) if e.get("evidence")]
    norm = lambda s: re.sub(r"\W+", " ", s).lower().strip()
    first = lambda e: norm(re.split(r"\.\.\.|…|\[\.\.\.\]", e)[0])[:60]   # quotes elided with "..." are checked on their first part
    found = sum(first(e) in norm(t) for e, t in quotes)
    print(f"\nevidence quotes found verbatim (first 60 normalised characters before any ellipsis): {found} of {len(quotes)} ({found / max(1, len(quotes)):.1%})")
    if full:
        uq = sorted({P[k]["query_id"] for k in full if P[k]["stratum"] == "top10_uncited"})
        by_q = {q: [k for k in full if P[k]["query_id"] == q and P[k]["stratum"] == "top10_uncited"] for q in uq}
        val = lambda k: sum(xy(R[r][k]["category"]) for r in runs) / len(runs)
        head = sum(val(k) for q in uq for k in by_q[q]) / sum(len(by_q[q]) for q in uq)
        rng = random.Random(0); bs = []
        for _ in range(10000):
            s = [rng.choice(uq) for _ in uq]; n = sum(len(by_q[q]) for q in s)
            bs.append(sum(val(k) for q in s for k in by_q[q]) / n)
        bs.sort()
        print(f"headline: uncited top-10 results rated X or Y = {head:.1%} (95% range over queries {bs[250]:.1%}-{bs[9750]:.1%}), "
              f"{sum(len(v) for v in by_q.values())} results, {len(uq)} queries")
        perq = {q: sum(val(k) for k in by_q[q]) for q in uq}
        print(f"queries with at least one uncited top-10 result rated X or Y in a majority of runs: "
              f"{sum(any(val(k) > 0.5 for k in by_q[q]) for q in uq)} of {len(uq)}")


if __name__ == "__main__":
    if sys.argv[1] == "summary":
        summary(*sys.argv[2:6])
    else:
        ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("pairs"); ap.add_argument("queries"); ap.add_argument("texts"); ap.add_argument("outdir")
        ap.add_argument("--run", type=int, default=1); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--workers", type=int, default=4)
        ap.add_argument("--stratum", default="")
        judge(ap.parse_args())
