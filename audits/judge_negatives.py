#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
"""LLM false-negative audit: would an examiner cite this uncited candidate against the query's claims? (GLM 5 on Bedrock)

  python audits/judge_negatives.py SAMPLE.jsonl OUT.jsonl [--workers 8] [--max-usd 5]   (AWS credentials with Bedrock access, us-east-1)

SAMPLE.jsonl: one of the samples in audits-negatives-v1.tar (query and candidate texts included). The judge does not see the stratum, rank or score. Categories follow EP
search reports: X (alone anticipates / renders obvious an independent claim), Y (relevant in combination), A (background /
same field), N (not relevant). X or Y on an uncited candidate = a false negative if it were used as a hard negative.
Calibration stratum positive_xy (real examiner X/Y citations) shows how often the judge agrees with examiners.
Resumable: rows already in OUT are skipped. Prints the summary table at the end.
"""
import argparse, json, os, re, sys, threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore.config import Config

MODEL = "zai.glm-5"
PROMPT = """You are a senior patent examiner doing a prior-art search for the claims below.

<claims_under_examination>
{q}
</claims_under_examination>

<candidate_document>
{d}
</candidate_document>

Classify how you would cite the candidate document against these claims, using search-report categories:
X = on its own it discloses all essential features of at least one independent claim (anticipation, or obviousness from this document alone)
Y = it discloses a substantial part of the claimed features and you would cite it in combination with another document
A = same technical field or general background only; it does not disclose the characterising features
N = not relevant
Ignore publication dates. Reply with JSON only: {{"category": "X|Y|A|N", "reason": "<one sentence>"}}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sample")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--q-chars", type=int, default=8000)
    ap.add_argument("--d-chars", type=int, default=12000)
    ap.add_argument("--max-usd", type=float, default=5.0, help="stop sending new requests once the estimated spend reaches this")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.sample)]
    if os.path.exists(a.out):   # keep judged rows, retry failed ones
        prev = [r for r in map(json.loads, open(a.out)) if r.get("category")]
        with open(a.out, "w") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in prev)
    else:
        prev = []
    done = {(r["query_id"], r["family_id"], r["stratum"]) for r in prev}
    todo = [r for r in rows if (r["query_id"], r["family_id"], r["stratum"]) not in done]
    br = boto3.session.Session(region_name="us-east-1").client("bedrock-runtime", config=Config(read_timeout=300, retries={"max_attempts": 6, "mode": "adaptive"}))
    lock, usage = threading.Lock(), Counter()

    def one(r):
        with lock:
            if usage["in"] / 1e6 * 1.0 + usage["out"] / 1e6 * 3.2 >= a.max_usd:
                usage["skipped"] += 1
                return
        msg = PROMPT.format(q=r["query_text"][: a.q_chars], d=r["doc_text"][: a.d_chars])
        cat, reason, raw, u = None, None, "", {}
        for attempt in range(3):
            try:
                resp = br.converse(modelId=MODEL, messages=[{"role": "user", "content": [{"text": msg}]}],
                                   inferenceConfig={"maxTokens": 4000, "temperature": 0})
                raw = "".join(c.get("text", "") for c in resp["output"]["message"]["content"])
                u = resp.get("usage", {})
                m = re.findall(r"\{[^{}]*\"category\"[^{}]*\}", raw)
                if m:
                    j = json.loads(m[-1]); cat, reason = j.get("category", "").strip().upper()[:1], j.get("reason")
                    if cat in "XYAN":
                        break
            except Exception as e:
                raw = f"error: {e}"
        with lock:
            usage["in"] += u.get("inputTokens", 0)
            usage["out"] += u.get("outputTokens", 0)
            out = {k: v for k, v in r.items() if k not in ("query_text", "doc_text")}
            out.update(category=cat, reason=reason, raw=None if cat else raw[-500:])
            with open(a.out, "a") as fh:
                fh.write(json.dumps(out) + "\n")
            usage["n"] += 1
            if usage["n"] % 25 == 0:
                print(f"{usage['n']}/{len(todo)} judged", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, todo))
    res = [json.loads(l) for l in open(a.out)]
    by = defaultdict(Counter)
    for r in res:
        by[r["stratum"]][r["category"] or "?"] += 1
    order = ["top10", "r11_30", "r31_100_kept", "r31_100_capped", "r31_100_capped_nopos", "cocited_strict", "cocited_weak_only", "positive_xy"]
    print(f"\n{'stratum':24s} {'n':>4s} {'X':>4s} {'Y':>4s} {'A':>4s} {'N':>4s} {'?':>3s}  X+Y share")
    for s in order + sorted(k for k in by if k not in order):
        c = by.get(s)
        if not c:
            continue
        n = sum(c.values())
        print(f"{s:24s} {n:4d} {c['X']:4d} {c['Y']:4d} {c['A']:4d} {c['N']:4d} {c['?']:3d}  {(c['X'] + c['Y']) / max(1, n - c['?']):.2f}")
    print(json.dumps({"tokens_in_this_run": usage["in"], "tokens_out_this_run": usage["out"], "skipped_budget": usage["skipped"],
                      "est_cost_usd": round(usage["in"] / 1e6 * 1.0 + usage["out"] / 1e6 * 3.2, 2)}))


if __name__ == "__main__":
    main()
