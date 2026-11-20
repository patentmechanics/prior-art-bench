#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
"""Step 2: the corpus, one row per in-scope DOCDB family (English full text, US + EP).

Stages (each logs to $WORK/logs/<stage>_<src>.log; safe to re-run, finished outputs are skipped):
  meta   --src us|ep   per source file, small columns only            -> $WORK/meta/<src>.parquet
  select               representative publication per in-scope family -> $WORK/reps/reps.parquet (+ per-source splits)
  build  --src us|ep   full rows for the representatives, per source file, in parallel -> $WORK/corpus/<src>/<file>.parquet
  tac                  title+abstract+claims sorted by token length, ~1 GB shards  -> $WORK/tac/shard-*.parquet

Representative choice (deterministic): has parsed claims, then grant before application, then longer description,
then later publication, then doc_id. Candidates: US utility/reissue publications (not design or plant) dated 1976 on,
and EP publications whose own language (DOCDB) is English.

Source rows: parsed USPTO full text under $PAB_WORK/us/ (one parquet per bulk file, 37-column normalized schema; the
parser is not part of this repository, see build/README.md). Only `build --src us` and `tac` were run for the US benchmark;
the EP paths are unused.
Family map: build/select_us.py (family_map/).
"""
import argparse, glob, hashlib, json, math, os, re, shutil, time
from concurrent.futures import ProcessPoolExecutor

import duckdb

WORK = os.environ.get("PAB_WORK", "work")
TOKENIZER = os.environ.get("PAB_TOKENIZER", "tokenizer.json")   # Qwen3 tokenizer.json (token-length sort of tac shards)
SRC_DIR = {"us": f"{WORK}/us", "ep": f"{WORK}/ep"}
DESC_SAMPLE_PCT = 1  # exact description token counts for ~1% of documents (to calibrate chars -> tokens)

# DOCDB publication number for a normalized/docs doc_id (US applications use year + 6 digits in DOCDB).
PN_SQL = r"""CASE
  WHEN regexp_matches(doc_id, '^US\d{4}0\d{6}[A-Z]\d?$') THEN 'US' || substr(doc_id, 3, 4) || substr(doc_id, 8)
  WHEN regexp_matches(doc_id, '^US[A-Z]{0,2}0\d+[A-Z]\d?$') THEN regexp_replace(doc_id, '^US([A-Z]{0,2})0+(\d+[A-Z]\d?)$', 'US\1\2')
  ELSE doc_id END"""


def log(stage, msg):
    os.makedirs(f"{WORK}/logs", exist_ok=True)
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(f"{WORK}/logs/{stage}.log", "a") as f:
        f.write(line + "\n")


def con(threads=4, mem="16GB"):
    c = duckdb.connect()
    c.execute(f"SET threads={threads}; SET memory_limit='{mem}'; SET preserve_insertion_order=false;"
              f"SET temp_directory='{WORK}/tmp'")
    return c


# ---------------------------------------------------------------- meta
def meta(a):
    stage = f"meta_{a.src}"
    files = sorted(glob.glob(f"{SRC_DIR[a.src]}/**/*.parquet", recursive=True))
    log(stage, f"start: {len(files)} files")
    os.makedirs(f"{WORK}/meta", exist_ok=True)
    c = con(threads=a.threads, mem="24GB")
    c.execute(f"""COPY (
        SELECT doc_id, {PN_SQL} AS pn, substr(doc_id, 1, 2) AS cc, kind, doc_class, patent_type, pub_date, filing_date,
               coalesce(desc_chars, 0) AS desc_chars, coalesce(n_claims_parsed, 0) AS n_claims,
               coalesce(n_independent, 0) AS n_independent, length(coalesce(abstract, '')) AS abstract_chars,
               replace(filename, '{SRC_DIR[a.src]}/', '') AS file
        FROM read_parquet({json.dumps(files)}, filename = true)
      ) TO '{WORK}/meta/{a.src}.parquet' (FORMAT parquet, COMPRESSION zstd)""")
    n = c.execute(f"SELECT count(*), count(DISTINCT doc_id) FROM '{WORK}/meta/{a.src}.parquet'").fetchone()
    log(stage, f"DONE rows={n[0]} distinct_doc_ids={n[1]}")


# ---------------------------------------------------------------- select
def select(a):
    stage = "select"
    fm = a.family_map
    os.makedirs(f"{WORK}/reps", exist_ok=True)
    c = con(threads=a.threads, mem="40GB")
    c.execute(f"CREATE VIEW meta AS SELECT * FROM read_parquet(['{WORK}/meta/us.parquet', '{WORK}/meta/ep.parquet'])")
    c.execute(f"CREATE VIEW mem AS SELECT * FROM '{fm}/members.parquet'")
    c.execute(f"CREATE VIEW fam AS SELECT * FROM '{fm}/families.parquet' WHERE in_scope")
    c.execute(f"""CREATE TABLE cand AS
        SELECT mem.family_id, m.*
        FROM meta m JOIN mem ON mem.pn = m.pn
        WHERE (m.cc = 'US' AND coalesce(m.patent_type, '') NOT IN ('design', 'plant') AND m.pub_date >= 19760101)
           OR (m.cc = 'EP' AND mem.lang = 'en' AND m.kind IN ('A1', 'A2', 'B1', 'B2', 'B3'))""")
    c.execute("""CREATE TABLE reps AS
        SELECT * FROM cand
        QUALIFY row_number() OVER (PARTITION BY family_id
            ORDER BY (n_claims > 0) DESC, (doc_class = 'grant') DESC, desc_chars DESC, pub_date DESC, doc_id) = 1""")
    c.execute(f"COPY reps TO '{WORK}/reps/reps.parquet' (FORMAT parquet, COMPRESSION zstd)")
    for src in ("us", "ep"):
        c.execute(f"COPY (SELECT * FROM reps WHERE cc = '{src.upper()}') TO '{WORK}/reps/reps_{src}.parquet' (FORMAT parquet)")
    s = {
        "meta_rows": c.execute("SELECT count(*) FROM meta").fetchone()[0],
        "meta_rows_matched_to_in_scope_family": c.execute("SELECT count(*) FROM cand").fetchone()[0],
        "in_scope_families": c.execute("SELECT count(*) FROM fam").fetchone()[0],
        "families_with_rep": c.execute("SELECT count(*) FROM reps").fetchone()[0],
        "reps_by_cc_class": c.execute("SELECT cc, doc_class, count(*) FROM reps GROUP BY ALL ORDER BY ALL").fetchall(),
        "reps_without_claims": c.execute("SELECT count(*) FROM reps WHERE n_claims = 0").fetchone()[0],
        "us_meta_unmatched_sample": c.execute("""SELECT doc_id FROM meta ANTI JOIN mem USING (pn)
                                                 WHERE cc = 'US' AND pub_date >= 19760101 USING SAMPLE 10""").fetchall(),
    }
    s["in_scope_families_without_text"] = s["in_scope_families"] - s["families_with_rep"]
    json.dump(s, open(f"{WORK}/reps/select_stats.json", "w"), indent=1, default=str)
    log(stage, "DONE " + json.dumps(s, default=str))


# ---------------------------------------------------------------- build
SECTION_RULES = [
    ("related", r"cross[- ]reference|related applications?|priority|incorporation by reference|continuation"),
    ("field", r"^(technical )?field|field of (the )?(invention|disclosure)"),
    ("background", r"background|prior art|related art|description of (the )?related|state of the art"),
    ("summary", r"summary|disclosure of (the )?invention|objects?( of the invention)?$|problem|solution to"),
    ("drawings", r"(brief )?description of (the )?(several views of the )?(drawings?|figures?)"),
    ("detailed", r"detailed description|embodiments?|examples?|best mode|mode\(?s?\)? (for|of) (carrying|the invention)|description of (the )?(preferred|illustrative)|^description( of (the )?invention)?$"),
]
SECTION_RE = [(name, re.compile(rx, re.I)) for name, rx in SECTION_RULES]
_tok = None


def _init_worker():
    global _tok
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from tokenizers import Tokenizer
    _tok = Tokenizer.from_file(TOKENIZER)


def heading_section(line):
    s = line.strip()
    if not s or len(s) > 120 or s.endswith((".", ";", ",")) or len(s.split()) > 14:
        return None
    letters = [ch for ch in s if ch.isalpha()]
    if not letters:
        return None
    looks = s.isupper() or any(rx.search(s) for _, rx in SECTION_RE)
    if not looks:
        return None
    for name, rx in SECTION_RE:
        if rx.search(s):
            return name
    return "other" if s.isupper() else None


def paragraphs(desc):
    out, section, heading = [], "preamble", None
    for line in (desc or "").split("\n"):
        t = line.strip()
        if not t:
            continue
        sec = heading_section(t)
        if sec:
            section, heading = sec, t
            continue
        out.append({"section": section, "heading": heading, "text": t})
    return out


def render_tac(title, abstract, claims):
    cl = sorted(claims or [], key=lambda c: (c["num"] if c["num"] is not None else 10**6))
    parts = [title or "", abstract or "", "\n".join(f"{c['num']}. {c['text']}" for c in cl if c.get("text"))]
    return "\n\n".join(p for p in parts if p)


PASSAGE_CHARS = 3600      # ~800 Qwen3 tokens at the measured ~4.5 characters per token
PASSAGE_MAX_CHARS = 4800
_SENT = re.compile(r"(?<=[.;:])\s+(?=[A-Z0-9(\[])")


def make_passages(paras):
    """Description passages for multi-vector indexing and reranker input: paragraphs are packed in order up to
    PASSAGE_CHARS, never across a section change; a paragraph longer than PASSAGE_MAX_CHARS is split at sentence ends."""
    out, buf, sec, head = [], [], None, None

    def flush():
        if buf:
            t = "\n".join(buf)
            out.append({"section": sec, "heading": head, "text": t, "n_chars": len(t)})
            buf.clear()

    for p in paras:
        pieces = [p["text"]]
        if len(p["text"]) > PASSAGE_MAX_CHARS:
            pieces, cur = [], ""
            for s in _SENT.split(p["text"]):
                if cur and len(cur) + len(s) + 1 > PASSAGE_CHARS:
                    pieces.append(cur); cur = s
                else:
                    cur = f"{cur} {s}" if cur else s
            pieces.append(cur)
            pieces = [x[i:i + PASSAGE_MAX_CHARS] for x in pieces for i in range(0, len(x), PASSAGE_MAX_CHARS)]
        for piece in pieces:
            if p["section"] != sec or (buf and sum(len(b) for b in buf) + len(piece) > PASSAGE_CHARS):
                flush()
                sec, head = p["section"], p["heading"]
            buf.append(piece)
    flush()
    return out


def _build_file(args):
    src, rel, ids_path, out, fmap = args
    t0 = time.time()
    c = con(threads=1, mem="3GB")
    tab = c.sql(f"""
        SELECT r.family_id, f.earliest_pd, d.doc_id, substr(d.doc_id, 1, 2) AS cc, d.kind, d.doc_class, d.pub_date,
               d.filing_date, d.app_number, d.cpc, d.title, d.abstract, d.claims, d.description, d.desc_chars
        FROM read_parquet('{SRC_DIR[src]}/{rel}') d
        JOIN (SELECT family_id, doc_id FROM '{ids_path}' WHERE file = '{rel}') r USING (doc_id)
        LEFT JOIN (SELECT family_id, earliest_pd FROM '{fmap}/families.parquet') f USING (family_id)
    """)
    tab = tab.to_arrow_table() if hasattr(tab, "to_arrow_table") else tab.arrow()
    rows = tab.to_pylist()
    tacs = [render_tac(r["title"], r["abstract"], r["claims"]) for r in rows]
    enc = _tok.encode_batch(tacs, add_special_tokens=False) if rows else []
    out_rows, pass_rows = [], []
    for r, tac, e in zip(rows, tacs, enc):
        paras = paragraphs(r.pop("description"))
        for i, ps in enumerate(make_passages(paras)):
            pass_rows.append({"family_id": r["family_id"], "doc_id": r["doc_id"], "p_idx": i, **ps})
        sample = int(hashlib.md5(r["doc_id"].encode()).hexdigest()[:4], 16) % 100 < DESC_SAMPLE_PCT
        desc_tok = len(_tok.encode("\n".join(p["text"] for p in paras), add_special_tokens=False).ids) if sample else None
        r.update(tac_text=tac, tac_tokens=len(e.ids), paragraphs=paras, n_paragraphs=len(paras),
                 n_background_paragraphs=sum(p["section"] == "background" for p in paras),
                 desc_tokens_exact=desc_tok)
        out_rows.append(r)
    import pyarrow as pa, pyarrow.parquet as pq
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if out_rows:
        pq.write_table(pa.Table.from_pylist(out_rows), out + ".tmp", compression="zstd")
    else:
        pq.write_table(pa.table({"family_id": pa.array([], pa.int64())}), out + ".tmp")
    pout = out.replace(f"{WORK}/corpus/", f"{WORK}/passages/")
    os.makedirs(os.path.dirname(pout), exist_ok=True)
    if pass_rows:
        pq.write_table(pa.Table.from_pylist(pass_rows), pout + ".tmp", compression="zstd")
    else:
        pq.write_table(pa.table({"family_id": pa.array([], pa.int64())}), pout + ".tmp")
    os.replace(pout + ".tmp", pout)
    os.replace(out + ".tmp", out)  # corpus part last: its presence marks the file as done
    return rel, len(out_rows), time.time() - t0


def build(a):
    stage = f"build_{a.src}"
    ids_path = f"{WORK}/reps/reps_{a.src}.parquet"
    files = [r[0] for r in con().execute(f"SELECT DISTINCT file FROM '{ids_path}' ORDER BY 1").fetchall()]
    todo = []
    for rel in files:
        out = f"{WORK}/corpus/{a.src}/{rel.replace('/', '__')}"
        if not os.path.exists(out):
            todo.append((a.src, rel, ids_path, out, a.family_map))
    log(stage, f"start: {len(files)} files with representatives, {len(todo)} to do, {a.workers} workers")
    done = len(files) - len(todo)
    n_rows = 0
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, initializer=_init_worker) as ex:
        for rel, n, s in ex.map(_build_file, todo, chunksize=1):
            done += 1
            n_rows += n
            if done % 20 == 0 or done == len(files):
                el = time.time() - t0
                rate = (done - (len(files) - len(todo))) / max(el, 1)
                eta = (len(files) - done) / max(rate, 1e-9)
                log(stage, f"{done}/{len(files)} files, {n_rows} docs this run, {el / 60:.0f} min elapsed, ETA {eta / 60:.0f} min")
    log(stage, f"DONE {len(files)} files")


# ---------------------------------------------------------------- tac shards
def tac(a):
    stage = "tac"
    os.makedirs(f"{WORK}/tac", exist_ok=True)
    c = con(threads=a.threads, mem="30GB")
    parts = sorted(glob.glob(f"{WORK}/corpus/*/*.parquet"))
    log(stage, f"start: {len(parts)} corpus parts")
    src = f"read_parquet({json.dumps(parts)}, union_by_name = true)"
    # A single global ORDER BY over all text runs out of memory on US (~8.5M docs). Instead: token-length buckets with about
    # equal tokens (pass 1 streams rows into bucket files, no sort), then sort each bucket on its own (pass 2) -> one shard
    # per bucket, globally ordered by tac_tokens.
    hist = c.execute(f"SELECT tac_tokens, count(*) FROM {src} WHERE tac_text IS NOT NULL GROUP BY 1 ORDER BY 1").fetchall()
    total = sum(t * n for t, n in hist)
    nb = max(1, math.ceil(total / 0.75e9))
    bounds, cum, nxt = [], 0, 1
    for t, n in hist:            # bucket k holds tac_tokens in [bounds[k-1], bounds[k])
        if nxt < nb and cum >= total * nxt / nb:
            bounds.append(t); nxt += 1
        cum += t * n
    log(stage, f"{total / 1e9:.2f}B tokens -> {len(bounds) + 1} buckets, bounds {bounds}")
    tmp = f"{WORK}/tmp/tac_buckets"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(f"{WORK}/tac", ignore_errors=True)
    os.makedirs(f"{WORK}/tac")
    c.execute(f"""COPY (
        SELECT family_id, doc_id, cc, tac_tokens, tac_text, len(list_filter({bounds}::INTEGER[], lambda b: b <= tac_tokens)) AS bucket
        FROM {src} WHERE tac_text IS NOT NULL
      ) TO '{tmp}' (FORMAT parquet, PARTITION_BY (bucket), COMPRESSION zstd)""")
    log(stage, "pass 1 done")
    c.execute("SET preserve_insertion_order=true")
    for b in range(len(bounds) + 1):
        c.execute(f"""COPY (SELECT family_id, doc_id, cc, tac_tokens, tac_text FROM '{tmp}/bucket={b}/*.parquet'
                            ORDER BY tac_tokens, family_id)
                      TO '{WORK}/tac/shard-{b:03d}.parquet' (FORMAT parquet, COMPRESSION zstd)""")
    shutil.rmtree(tmp, ignore_errors=True)
    shards = sorted(glob.glob(f"{WORK}/tac/shard-*.parquet"))
    man = []
    for s in shards:
        r = c.execute(f"SELECT count(*), sum(tac_tokens), min(tac_tokens), max(tac_tokens) FROM '{s}'").fetchone()
        man.append({"file": os.path.basename(s), "rows": r[0], "tokens": int(r[1] or 0), "min_tokens": r[2], "max_tokens": r[3],
                    "md5": hashlib.md5(open(s, "rb").read()).hexdigest()})
    json.dump(man, open(f"{WORK}/tac/manifest.json", "w"), indent=1)
    log(stage, f"DONE {len(shards)} shards, {sum(m['rows'] for m in man)} rows, {sum(m['tokens'] for m in man) / 1e9:.2f}B tokens")


def raise_nofile():
    # tac reads every corpus part at once (~3,800 files on US); default soft limit is 1,024
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    want = 65536 if hard == resource.RLIM_INFINITY else min(65536, hard)
    if soft < want:
        resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))


def main():
    raise_nofile()
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["meta", "select", "build", "tac"])
    ap.add_argument("--src", choices=["us", "ep"])
    ap.add_argument("--family-map", default=f"{WORK}/family_map")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(f"{WORK}/tmp", exist_ok=True)
    {"meta": meta, "select": select, "build": build, "tac": tac}[a.stage](a)


if __name__ == "__main__":
    main()
