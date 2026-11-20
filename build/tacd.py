#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
"""Descriptions ablation (C): document text = title + abstract + claims, then the description, for a chosen set of families.

  python build/tacd.py --src us --families F1.parquet [F2.parquet ...] --subset SUBSET_FAMILIES.parquet --out DIR

The description is rebuilt from the passages (corpus.py build): passages in order, sections 'related' (cross-references
to other applications) and 'drawings' (list of figures) left out. Claims come before the description so that truncation
at the model's document length only ever cuts description. Text is capped at --max-chars (8,192 tokens is ~37K chars).
Outputs in DIR:
  parts/<part>.parquet       family_id, doc_id, cc, tacd_text, tacd_chars, desc_chars_used  (all requested families)
  subset-<src>-NNN.parquet   the --subset families only (for evaluate.py encode; vLLM batches by tokens itself)
  train-<src>.parquet        all requested families (for pack.py --docs-glob ... --text-col tacd_text)
"""
import argparse, glob, json, os, time
from concurrent.futures import ProcessPoolExecutor

import duckdb

WORK = os.environ.get("PAB_WORK", "work")
SKIP_SECTIONS = ("related", "drawings")


def one(args):
    part, need, out, max_chars = args
    dst = f"{out}/parts/{os.path.basename(part)}"
    if os.path.exists(dst):
        return 0
    c = duckdb.connect()
    c.execute("SET threads=1; SET memory_limit='3GB'")
    pas = part.replace("/corpus/", "/passages/")
    skip = ", ".join(f"'{x}'" for x in SKIP_SECTIONS)
    c.execute(f"CREATE TABLE need AS SELECT family_id FROM read_parquet({json.dumps(need)})")
    c.execute(f"""CREATE TABLE d AS SELECT family_id, string_agg(text, E'\\n\\n' ORDER BY p_idx) AS desc_text
                  FROM '{pas}' WHERE section NOT IN ({skip}) AND family_id IN (SELECT family_id FROM need) GROUP BY 1""")
    n = c.execute(f"""COPY (
        SELECT t.family_id, t.doc_id, t.cc,
               left(t.tac_text || coalesce(E'\\n\\nDescription:\\n' || d.desc_text, ''), {max_chars}) AS tacd_text,
               length(left(t.tac_text || coalesce(E'\\n\\nDescription:\\n' || d.desc_text, ''), {max_chars})) AS tacd_chars,
               coalesce(length(d.desc_text), 0) AS desc_chars_used
        FROM '{part}' t LEFT JOIN d USING (family_id)
        WHERE t.family_id IN (SELECT family_id FROM need) AND t.tac_text IS NOT NULL
      ) TO '{dst}.tmp' (FORMAT parquet, COMPRESSION zstd)""").fetchone()[0]
    os.replace(f"{dst}.tmp", dst)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, choices=["us", "ep"])
    ap.add_argument("--families", nargs="+", required=True, help="parquet files with a family_id column (union)")
    ap.add_argument("--subset", required=True, help="parquet with the evaluation subset's family_id")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-chars", type=int, default=45000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rows-per-shard", type=int, default=150000)
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(f"{a.out}/parts", exist_ok=True)
    parts = sorted(glob.glob(f"{WORK}/corpus/{a.src}/*.parquet"))
    with ProcessPoolExecutor(a.workers) as ex:
        n = sum(ex.map(one, [(p, a.families + [a.subset], a.out, a.max_chars) for p in parts]))
    # combine by streaming the parts one at a time (a single query over ~4,000 files of long text ran out of memory)
    import pyarrow as pa, pyarrow.parquet as pq
    need = set(pq.read_table(a.families[0], columns=["family_id"]).column("family_id").to_pylist())
    for f in a.families[1:]:
        need |= set(pq.read_table(f, columns=["family_id"]).column("family_id").to_pylist())
    sub = set(pq.read_table(a.subset, columns=["family_id"]).column("family_id").to_pylist())
    tw, sw, si, srows = None, None, 0, 0
    for part in sorted(glob.glob(f"{a.out}/parts/*.parquet")):
        t = pq.read_table(part)
        fam = t.column("family_id").to_pylist()
        tr = t.filter(pa.array([f in need for f in fam], type=pa.bool_()))
        sb = t.filter(pa.array([f in sub for f in fam], type=pa.bool_()))
        if tr.num_rows:
            tw = tw or pq.ParquetWriter(f"{a.out}/train-{a.src}.parquet", tr.schema, compression="zstd")
            tw.write_table(tr)
        if sb.num_rows:
            if sw is None or srows >= a.rows_per_shard:
                if sw:
                    sw.close()
                sw = pq.ParquetWriter(f"{a.out}/subset-{a.src}-{si:03d}.parquet", sb.schema, compression="zstd")
                si, srows = si + 1, 0
            sw.write_table(sb)
            srows += sb.num_rows
    for w in (tw, sw):
        if w:
            w.close()
    c = duckdb.connect()
    c.execute("SET threads=4; SET memory_limit='8GB'")
    c.execute(f"CREATE VIEW t AS SELECT tacd_chars, desc_chars_used FROM '{a.out}/train-{a.src}.parquet' UNION ALL "
              f"SELECT tacd_chars, desc_chars_used FROM '{a.out}/subset-{a.src}-*.parquet'")
    ns = c.execute(f"SELECT count(*) FROM '{a.out}/subset-{a.src}-*.parquet'").fetchone()[0]
    st = c.execute(f"""SELECT count(*), round(avg(tacd_chars)), round(median(tacd_chars)), count_if(desc_chars_used = 0),
                             count_if(tacd_chars >= {a.max_chars}) FROM t""").fetchone()
    print(json.dumps({"src": a.src, "families_written": st[0], "subset": ns, "avg_chars": st[1], "median_chars": st[2],
                      "without_description": st[3], "capped": st[4], "seconds": round(time.time() - t0)}))


if __name__ == "__main__":
    main()
