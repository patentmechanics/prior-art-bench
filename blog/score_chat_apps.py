#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
"""Score the chat apps' answers and our system's lists on the 30 fresh office-action queries (blog post).
Copied from the code that scored the blog post; only the import and the map step's input path changed.
  parse  QUERIES.json OUT.json name=RUNDIR ...   read each answer (RUNDIR/NN.txt, NN = position of the query in the main 30)
                                                 -> 10 numbers per answer, whether it named the application; OUT.keys.json
  map    KEYS.json MAP.json DOCS.parquet         number -> [family_id, first publication date] (the shipped map.json is ours)
  report QUERIES.json PARSED.json MAP.json [ours=RUN.json ...]   R@10, hit@10, X hit@10 per system
Scoring as the benchmark: families, the query's own family never counts, positives = examiner X/Y references whose family
was first published before the query's priority date. A listed document that is not in our corpus or is dated on or after
the priority date simply cannot match."""
import glob, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from answer_numbers import NUM, key, numbers


def keys_in(text):
    return [key(m.group(1)) for m in NUM.finditer(text or "")]


def load_run(path):
    """-> answer text, [(kind, text)] calls, [page texts], seconds, usage"""
    if path.endswith(".txt"):   # answer pasted by hand from a chat app; optional first line "seconds: 150"
        t = open(path).read().translate({0x201c: 34, 0x201d: 34, 0x2018: 39, 0x2019: 39})
        m = re.match(r"\s*seconds\s*[:=]\s*(\d+)", t)
        body = re.sub(r"\A\s*(?:(?:seconds|note)\s*[:=][^\n]*\n)+", "", t, flags=re.I)   # header lines are ours, not the app's
        return body, [], [], int(m.group(1)) if m else None, {}
    if path.endswith(".jsonl"):
        ev = [json.loads(l) for l in open(path) if l.startswith("{")]
        res = [e for e in ev if e.get("type") == "result"][-1]
        calls = [(b["name"], b["input"].get("query") or b["input"].get("url") or "") for e in ev if e.get("type") == "assistant"
                 for b in e["message"]["content"] if b.get("type") == "tool_use"]
        pages = [json.dumps(b.get("content")) for e in ev if e.get("type") == "user" and isinstance(e["message"]["content"], list)
                 for b in e["message"]["content"] if b.get("type") == "tool_result"]
        return res["result"], calls, pages, round(res.get("duration_ms", 0) / 1000), {"api_equivalent_usd": res.get("total_cost_usd")}
    r = json.load(open(path))
    if "output" in r:   # OpenAI Responses
        text = "".join(c.get("text", "") for o in r["output"] if o["type"] == "message" for c in o["content"])
        calls = []
        for o in r["output"]:
            if o["type"] == "web_search_call":
                a = o.get("action", {})
                calls.append((a.get("type", "search"), a.get("query") or " | ".join(a.get("queries") or []) or a.get("url") or ""))
                calls += [("source", s.get("url", "")) for s in a.get("sources") or []]
        u = r.get("usage", {})
        return text, calls, [], r.get("_seconds"), {"input_tokens": u.get("input_tokens"), "output_tokens": u.get("output_tokens"), "web_calls": sum(o["type"] == "web_search_call" for o in r["output"])}
    c = r["candidates"][0]; g = c.get("groundingMetadata", {})   # Vertex generateContent
    text = "".join(p.get("text", "") for p in c.get("content", {}).get("parts", []))
    calls = [("search", q) for q in g.get("webSearchQueries", [])] + [("source", (x.get("web", {}).get("title") or "") + " " + x.get("web", {}).get("uri", "")) for x in g.get("groundingChunks", [])]
    um = c.get("urlContextMetadata") or c.get("url_context_metadata") or {}
    calls += [("open_page", x.get("retrievedUrl") or x.get("retrieved_url") or "") for x in um.get("urlMetadata") or um.get("url_metadata") or []]
    return text, calls, [], r.get("_seconds"), r.get("usageMetadata", {})


def parse(qfile, out, runs):
    Q = json.load(open(qfile)); P = {}; allkeys = set()
    hand = {q["query_id"]: i for i, q in enumerate([q for q in Q if q.get("set") == "main 30"], 1)}   # by-hand files: NN.txt, NN = position in the main 30
    for name, d in runs:
        for q in Q:
            qid = q["query_id"].replace("/", "_").replace(":", "_")
            f = [p for p in (f"{d}/raw/{qid}.orig.jsonl", f"{d}/raw/{qid}.json", f"{d}/{hand.get(q['query_id'], 0):02d}.txt") if os.path.exists(p)]
            if not f: continue
            text, calls, pages, secs, usage = load_run(f[0])
            nums, how = numbers(text); ks = []
            own = {key(re.match(r"US(\d+)", q["query_doc_id"]).group(1)), f"A:{q['app_number']}"}
            own_named = bool(set(keys_in(text)) & own) or q["app_number"] in re.sub(r"[,/ ]", "", text)
            if how == "regex":   # no JSON line: the answer's table in order of first mention, repeats and the application itself dropped
                kept = []
                for n in nums:
                    k = key(NUM.search(n).group(1))
                    if k not in own and k not in [key(NUM.search(x).group(1)) for x in kept]: kept.append(n)
                nums = kept
            for n in nums[:10]:
                m = NUM.search(n if str(n).upper().lstrip().startswith("US") else "US" + str(n)); ks.append(key(m.group(1)) if m else None)
            refs = {key(r["number"]) for r in q["examiner_references"]}
            seen = lambda t: set(keys_in(t)) | ({f"A:{q['app_number']}"} if q["app_number"] in re.sub(r"[,/ ]", "", t or "") else set())
            ctext = [c for _, c in calls]
            P[f"{name}|{q['query_id']}"] = {"numbers": nums[:10], "keys": ks, "how": how, "seconds": secs, "usage": usage, "n_calls": len(calls),
                "own_in_calls": any(seen(c) & own for c in ctext), "own_in_answer": bool(set(ks) & own), "own_named_in_answer": own_named,
                "answer_seen": any(len(set(keys_in(t)) & refs) >= 2 for t in ctext + pages),
                "refs_in_any_page": len(refs & {k for t in pages for k in keys_in(t)}) if pages else None}
            allkeys |= {k for k in ks if k}
    json.dump(P, open(out, "w"), indent=1); json.dump(sorted(allkeys), open(out.replace(".json", ".keys.json"), "w"))
    print(len(P), "runs parsed;", sum(v["how"] == "regex" for v in P.values()), "without the JSON line;", len(allkeys), "numbers to map")


def map_(kfile, out, docs):
    """DOCS = a parquet table (pub_tok, family_id, pub_date) of every US publication; ours is not shipped (the mapping it
    produced is, as map.json). pub_tok is key() of the publication number: G:<grant number> or U:<11-digit pre-grant>."""
    import duckdb
    K = json.load(open(kfile)); c = duckdb.connect(); c.execute("SET memory_limit='3GB'; SET threads=2")
    c.execute("CREATE TABLE k(pub_tok VARCHAR)"); c.executemany("INSERT INTO k VALUES (?)", [(k,) for k in K])
    D = docs
    m = {t: [f, str(d)] for t, f, d in c.execute(f"SELECT pub_tok, min(family_id), min(pub_date) FROM '{D}' JOIN k USING (pub_tok) GROUP BY ALL").fetchall()}
    json.dump(m, open(out, "w")); print(len(K), "numbers,", len(m), "in the corpus")


def metrics(fams, q):
    pos = set(q["positives"]); xpos = {r["family_id"] for r in q["examiner_references"] if r["date_ok"] and "X" in r["categories"] and r["family_id"] != q["family_id"]}
    top = []
    for f in fams:
        if f is not None and f != q["family_id"] and f not in top: top.append(f)
    top = top[:10]
    return {"recall10": len(pos & set(top)) / len(pos), "hit10": float(bool(pos & set(top))), "xhit10": float(bool(xpos & set(top))) if xpos else None}


def report(qfile, pfile, mfile, ours):
    Q = [q for q in json.load(open(qfile))]; P = json.load(open(pfile)); M = json.load(open(mfile))
    systems = {}
    for k, v in P.items():
        name, qid = k.split("|", 1); systems.setdefault(name, {})[qid] = [M.get(x, [None])[0] if x else None for x in v["keys"]]
    for name, f in ours:
        j = json.load(open(f))   # {query_id: [family ids]} or [{"query_id":, "family_ids": [...]}]
        systems[name] = {x["query_id"]: x["family_ids"] for x in j} if isinstance(j, list) else j
    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None
    for sset in ("main 30", None):
        print("\n==", sset or "all queries with a run")
        for name, runs in systems.items():
            qs = [q for q in Q if q["query_id"] in runs and (sset is None or q.get("set") == sset)]
            ms = [metrics(runs[q["query_id"]], q) for q in qs]
            line = f"{name:28s} n={len(qs):2d}  R@10 {mean([m['recall10'] for m in ms])}  hit@10 {mean([m['hit10'] for m in ms])}  X hit@10 {mean([m['xhit10'] for m in ms if m['xhit10'] is not None])} (n={sum(m['xhit10'] is not None for m in ms)})"
            au = [P.get(f"{name}|{q['query_id']}") for q in qs]; au = [x for x in au if x]
            if au:
                line += f"  | own looked up {sum(x['own_in_calls'] for x in au)}/{len(au)}, answer seen {sum(x['answer_seen'] for x in au)}, not in corpus {sum(1 for q in qs for kk in P[name + '|' + q['query_id']]['keys'] if kk and kk not in M)}, median s {sorted(x['seconds'] or 0 for x in au)[len(au) // 2]}"
            print(line)
    json.dump({n: {q["query_id"]: metrics(r[q["query_id"]], q) for q in Q if q["query_id"] in r} for n, r in systems.items()}, open(pfile.replace(".json", ".per_query_metrics.json"), "w"), indent=1)


if __name__ == "__main__":
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == "parse": parse(a[0], a[1], [x.split("=", 1) for x in a[2:]])
    elif cmd == "map": map_(a[0], a[1], a[2])
    else: report(a[0], a[1], a[2], [x.split("=", 1) for x in a[3:]])
