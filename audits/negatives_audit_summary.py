#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
"""The hard-negatives audit rates of the paper (Section 4.3) from the judged files in audits-negatives-v1.tar.

  python audits/negatives_audit_summary.py audits-negatives-v1/judged

Share of sampled pairs the judge (GLM 5) graded X or Y, with Wilson 95% intervals:
  base recipe     audit_sample + audit_sample2: kept negatives (ranks 31-100) and real examiner X/Y citations
                  (audit_sample also has ranks 1-10, 11-30, capped and co-cited strata)
  tuned miner     audit_rr2: candidates ranked by a tuned model, ranks 1-10 / 11-30 / 31-50, and real citations
  release pack    audit_ha: the release negatives (ranked by the harrier-oss-v1-0.6b fine-tune), ranks 31-50 / 51-75 / 76-100, and real citations"""
import json, math, sys
from collections import Counter, defaultdict


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{k:3d}/{n:3d} = {100 * p:4.1f}%  (95% CI {100 * (c - h):.1f}-{100 * (c + h):.1f}%)"


def load(d, name):
    by = defaultdict(Counter)
    for l in open(f"{d}/{name}.judged.jsonl"):
        r = json.loads(l); by[r["stratum"]][r["category"] or "?"] += 1
    return by


def line(label, cs):
    n = sum(sum(c.values()) - c["?"] for c in cs); k = sum(c["X"] + c["Y"] for c in cs); x = sum(c["X"] for c in cs)
    print(f"  {label:40s} {wilson(k, n)}  of which X: {x}")


d = sys.argv[1]
a, a2 = load(d, "audit_sample"), load(d, "audit_sample2")
print("base recipe")
line("kept negatives, ranks 31-100", [a["r31_100_kept"], a2["r31_100_kept"]])
line("real examiner X/Y citations", [a["positive_xy"], a2["positive_xy"]])
for s, lab in (("top10", "ranks 1-10"), ("r11_30", "ranks 11-30"), ("r31_100_capped", "ranks 31-100, above the 0.95 cap"),
               ("cocited_strict", "co-cited (strict)"), ("cocited_weak_only", "co-cited (weak only)")):
    line(lab, [a[s]])
r = load(d, "audit_rr2")
print("tuned miner")
for s, lab in (("r1_10_kept", "ranks 1-10"), ("r11_30_kept", "ranks 11-30"), ("r31_50_kept", "ranks 31-50"), ("positive_xy", "real examiner X/Y citations")):
    line(lab, [r[s]])
h = load(d, "audit_ha")
print("release pack")
line("all negatives, ranks 31-100", [h["r31_50_kept"], h["r51_75_kept"], h["r76_100_kept"]])
for s, lab in (("r31_50_kept", "ranks 31-50"), ("r51_75_kept", "ranks 51-75"), ("r76_100_kept", "ranks 76-100"), ("positive_xy", "real examiner X/Y citations")):
    line(lab, [h[s]])
