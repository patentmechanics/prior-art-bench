# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
"""Tiny synthetic checks of the scoring rules, the search filters and the fusion stages.

  python -m pytest tests/        or        python tests/test_metrics.py

Needs numpy, pyarrow and duckdb only (no model, no network, no benchmark data).
"""
import ast
import math
import os
import sys
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import evaluate as E  # noqa: E402
import fusion as FU  # noqa: E402


def write_dataset(d):
    """queries: q1 (us_oa, dev), q2 (ptab, dev), q3 (us_oa, test), q4 (us_oa, dev; its only gold is not date_ok)"""
    os.makedirs(f"{d}/queries"); os.makedirs(f"{d}/qrels"); os.makedirs(f"{d}/families")
    pq.write_table(pa.table({"query_id": ["q1", "q2", "q3", "q4"], "source": ["us_oa", "ptab", "us_oa", "us_oa"],
                             "split": ["dev", "dev", "test", "dev"], "family_id": pa.array([100, 200, 300, 400], pa.int64()),
                             "priority_date": pa.array([20100101, 20150101, 20120101, None], pa.int32()),
                             "text": ["a", "b", "c", "d"],
                             "cpc_printed": pa.array([["B61C11/04"], None, None, None], pa.list_(pa.string()))}),
                   f"{d}/queries/queries.parquet")
    rows = [  # query_id, family_id, grade, keep, date_ok
        ("q1", 1, 3, True, True), ("q1", 2, 2, True, True), ("q1", 3, 3, False, True), ("q1", 4, 3, True, False),
        ("q2", 5, 3, True, True),
        ("q3", 6, 2, True, True),
        ("q4", 7, 3, True, False)]
    pq.write_table(pa.table({"query_id": [r[0] for r in rows], "family_id": pa.array([r[1] for r in rows], pa.int64()),
                             "grade": pa.array([r[2] for r in rows], pa.int32()), "keep": [r[3] for r in rows],
                             "date_ok": [r[4] for r in rows]}), f"{d}/qrels/qrels.parquet")
    pq.write_table(pa.table({"family_id": pa.array([1, 2, 3, 9], pa.int64()),
                             "earliest_publication_date": pa.array([20000101, 20090101, 20100101, 20200101], pa.int32())}),
                   f"{d}/families/part-0.parquet")


def write_run(path, rows):
    E.write_run(path, [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows])


class ScoreQuery(unittest.TestCase):
    def test_recall_ndcg_mrr(self):
        gold = {1: 3, 2: 2}
        m = E.score_query([9, 1, 8, 2], gold)
        self.assertEqual(m["recall@10"], 1.0)
        dcg = 3 / math.log2(3) + 2 / math.log2(5)
        ideal = 3 / math.log2(2) + 2 / math.log2(3)
        self.assertAlmostEqual(m["ndcg@10"], dcg / ideal)
        self.assertAlmostEqual(m["mrr@10"], 0.5)

    def test_dedup_keeps_best_rank(self):
        # family 9 listed 11 times: after dedup family 1 is at rank 1, inside the top 10
        m = E.score_query([9] * 11 + [1], {1: 3})
        self.assertEqual(m["recall@10"], 1.0)
        self.assertAlmostEqual(m["mrr@10"], 0.5)

    def test_cutoffs(self):
        ranked = list(range(1000, 1010)) + [1] + list(range(2000, 2100)) + [2]
        m = E.score_query(ranked, {1: 3, 2: 3, 3: 2})
        self.assertAlmostEqual(m["recall@10"], 0.0)
        self.assertAlmostEqual(m["recall@100"], 1 / 3)
        self.assertAlmostEqual(m["recall@1000"], 2 / 3)
        self.assertEqual(m["ndcg@10"], 0.0)
        self.assertEqual(m["mrr@10"], 0.0)
        self.assertEqual((m["hit@10"], m["hit@100"], m["hit@1000"]), (0.0, 1.0, 1.0))


class PerQuery(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name
        write_dataset(self.d)
        self.run = f"{self.d}/run.parquet"
        write_run(self.run, [("q1", [3, 1, 4, 2], [0.9, 0.8, 0.7, 0.6]), ("q2", [5], [0.5]), ("q3", [6], [0.4]),
                             ("q4", [7], [0.3])])

    def tearDown(self):
        self.tmp.cleanup()

    def test_gold_is_keep_and_date_ok(self):
        P = E.per_query(self.run, self.d, "dev")
        # q1 gold = {1: 3, 2: 2} (3 is not keep, 4 is not date_ok); dedup ranks: 3,1,4,2 -> 1 at 1, 2 at 3
        m = P["q1"][1]
        self.assertEqual(m["recall@10"], 1.0)
        dcg = 3 / math.log2(3) + 2 / math.log2(5)
        self.assertAlmostEqual(m["ndcg@10"], dcg / (3 + 2 / math.log2(3)))

    def test_part_and_unscorable_queries(self):
        P = E.per_query(self.run, self.d, "dev")
        self.assertEqual(sorted(P), ["q1", "q2"])          # q3 is test; q4 has no gold left
        self.assertEqual(P["q2"][0], "ptab/dev")
        self.assertEqual(sorted(E.per_query(self.run, self.d, "test")), ["q3"])
        self.assertEqual(sorted(E.per_query(self.run, self.d, "all")), ["q1", "q2", "q3"])
        self.assertIn("q4", E.per_query(self.run, self.d, "dev", date_filter="none"))

    def test_summary_and_paired(self):
        P = E.per_query(self.run, self.d, "dev")
        s = E.summary(P)
        self.assertEqual(s["all"]["queries"], 2)
        self.assertEqual(set(s), {"all", "us_oa", "ptab"})
        run_b = f"{self.d}/b.parquet"
        write_run(run_b, [("q1", [8] * 1 + list(range(20, 40)) + [1, 2], [0.0] * 23), ("q2", [5], [0.5])])
        B = E.per_query(run_b, self.d, "dev")
        r = E.paired(P, B, "recall@10")
        self.assertEqual(list(r), ["all", "us_oa", "ptab"])   # all first, then sources in order of first query id
        d = np.array([B[q][1]["recall@10"] - P[q][1]["recall@10"] for q in ["q1", "q2"]])
        rng = np.random.default_rng(0)
        boots = d[rng.integers(0, 2, size=(2000, 2))].mean(1)
        self.assertEqual(r["all"]["diff"], round(float(d.mean()), 4))
        self.assertEqual(r["all"]["ci95"], [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)])


class Search(unittest.TestCase):
    def test_filters_and_chunking(self):
        rng = np.random.default_rng(1)
        docs = rng.normal(size=(50, 8)).astype(np.float16)
        ids = np.arange(50, dtype=np.int64) // 2              # two rows per family (like description chunks)
        doc_pd = np.where(ids % 5 == 0, 20200101, 20000101).astype(np.int64)
        doc_pd[ids == 7] = 20100101                            # equal to q0's priority date -> not a candidate
        doc_pd[ids == 8] = 0                                   # unknown date -> always a candidate
        q = rng.normal(size=(3, 8)).astype(np.float32)
        q_fam = np.array([3, -1, 11], dtype=np.int64)
        q_pd = np.array([20100101, 99999999, 20100101], dtype=np.int64)
        f1, s1 = E.exact_search(q, q_fam, q_pd, [docs], ids, doc_pd, k=30, corpus_chunk=7, q_chunk=2)
        f2, s2 = E.exact_search(q, q_fam, q_pd, [docs[:20], docs[20:]], ids, doc_pd, k=30, corpus_chunk=1000, q_chunk=100)
        self.assertEqual(f1, f2)
        np.testing.assert_allclose(s1, s2, atol=1e-5)   # float32 BLAS rounding depends on block shapes
        for r in range(3):
            self.assertNotIn(int(q_fam[r]), f1[r])                       # own family never listed
            self.assertTrue(all(s1[r][i] >= s1[r][i + 1] for i in range(len(s1[r]) - 1)))
        self.assertNotIn(7, f1[0]); self.assertNotIn(10, f1[0])          # published on / after the priority date
        self.assertIn(8, f1[0])                                          # date unknown (0)
        self.assertIn(10, f1[1]); self.assertIn(7, f1[1])                # no priority date: no date filter
        # brute force for query 0
        sc = docs.astype(np.float32) @ q[0]
        ok = (ids != 3) & (doc_pd < 20100101)
        want = ids[np.argsort(-np.where(ok, sc, -np.inf), kind="stable")][:ok.sum()]
        self.assertEqual(f1[0], want[:30].tolist())

    def test_lookup_dates(self):
        got = E.lookup_dates(np.array([5, 1, 3]), np.array([50, 10, 30]), np.array([1, 2, 3, 5, 6]))
        self.assertEqual(got.tolist(), [10, 0, 30, 50, 0])


class Fusion(unittest.TestCase):
    def test_passages_max(self):
        doc = [{"query_id": "q", "family_ids": [1, 2], "scores": [0.5, 0.4]}]
        ch = [{"query_id": "q", "family_ids": [2, 2, 3], "scores": [0.45, 0.6, 0.3]}, {"query_id": "x", "family_ids": [1], "scores": [1.0]}]
        out = FU.passages(doc, ch, k=10)
        self.assertEqual([r["query_id"] for r in out], ["q"])           # queries in both runs only
        self.assertEqual(out[0]["family_ids"], [2, 1, 3])
        self.assertAlmostEqual(out[0]["scores"][0], 0.6, places=6)

    def test_cpc_strict_levels_and_dates(self):
        F = FU.cpc_levels(["B61C11/04"])
        self.assertEqual(FU.cpc_agree(F, FU.cpc_levels(["B61C11/04"])), 3)
        self.assertEqual(FU.cpc_agree(F, FU.cpc_levels(["B61C11/99"])), 2)
        self.assertEqual(FU.cpc_agree(F, FU.cpc_levels(["B61C99/01"])), 1)
        self.assertEqual(FU.cpc_agree(F, FU.cpc_levels(["H04L9/00"])), 0)
        base = [{"query_id": "q1", "family_ids": [1, 2, 3], "scores": [0.30, 0.2999, 0.28]}]
        members = [{"family_id": 2, "pub_date": 20050101, "cpc_current": ["B61C11/04"], "cpc_printed": None},
                   {"family_id": 3, "pub_date": 20150101, "cpc_current": ["B61C11/04"], "cpc_printed": None},   # after the query date
                   {"family_id": 1, "pub_date": 20050101, "cpc_current": ["H04L9/00"], "cpc_printed": ["B61C11/04"]}]  # printed CPC unused
        queries = [{"query_id": "q1", "priority_date": 20100101, "cpc_printed": ["B61C11/04"]}]
        out = FU.cpc_strict(base, members, queries, 0.75)
        self.assertEqual(out[0]["family_ids"], [2, 1, 3])
        sc = np.array([0.30, 0.2999, 0.28]); z = (sc - sc.mean()) / (sc.std() + 1e-9)
        self.assertAlmostEqual(out[0]["scores"][0], float(np.float32(z[1] + 0.75 * 1.0)), places=6)
        out = FU.cpc_strict(base, members, [{"query_id": "q1", "priority_date": 20100101, "cpc_printed": None}], 0.75)
        self.assertEqual(out[0]["family_ids"], [1, 2, 3])                # no printed CPC on the query: order unchanged

    def test_zscore_fusion_missing_gets_min(self):
        base = [{"query_id": "q", "family_ids": [1, 2, 3], "scores": [3.0, 2.0, 1.0]}]
        bm = [{"query_id": "q", "family_ids": [4, 3], "scores": [10.0, 9.0]}]
        out = FU.zscore_fusion(base, [bm], [0.2])
        zb = E_z([3.0, 2.0, 1.0]); zm = E_z([10.0, 9.0])
        want = {1: zb[0] + 0.2 * zm.min(), 2: zb[1] + 0.2 * zm.min(), 3: zb[2] + 0.2 * zm[1], 4: zb.min() + 0.2 * zm[0]}
        self.assertEqual(out[0]["family_ids"], sorted(want, key=lambda f: -want[f]))
        out = FU.zscore_fusion(base, [[]], [0.2])                        # query missing from BM25: base order kept
        self.assertEqual(out[0]["family_ids"], [1, 2, 3])


def E_z(v):
    v = np.asarray(v, dtype=np.float64)
    return (v - v.mean()) / (v.std() + 1e-9)


class Instructions(unittest.TestCase):
    def test_same_strings_as_training(self):
        from pab_encoding import DEFAULT_INSTRUCTION, GRANTED_INSTRUCTION, INVALIDITY_PREFIX, QUERY_PREFIX, query_prefix
        with open(os.path.join(ROOT, "train", "common.py")) as fh:
            tree = ast.parse(fh.read())
        consts = {t.id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                  for t in n.targets if isinstance(t, ast.Name) and t.id.endswith("_INSTRUCTION")}
        self.assertEqual(consts["DEFAULT_INSTRUCTION"], DEFAULT_INSTRUCTION)
        self.assertEqual(consts["GRANTED_INSTRUCTION"], GRANTED_INSTRUCTION)
        self.assertEqual(QUERY_PREFIX, f"Instruct: {DEFAULT_INSTRUCTION}\nQuery: ")
        self.assertEqual(query_prefix("ptab", "instruction"), INVALIDITY_PREFIX)
        self.assertEqual(query_prefix("us_oa", "instruction"), QUERY_PREFIX)
        self.assertEqual(query_prefix("us_oa", ""), "")


if __name__ == "__main__":
    unittest.main()
