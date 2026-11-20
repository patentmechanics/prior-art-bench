#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
"""Reading US patent numbers out of a chatbot answer (used by score_chat_apps.py). Copied unchanged from the code
that scored the blog post."""
import json, re

NUM = re.compile(r"US\s*(?:RE\s*)?(\d{4}\s*/\s*\d{5,7}|\d{11}(?!\d)|\d{1,2}[,.]?\d{3}[,.]?\d{3}(?!\d))\s*(?:[AB][12]?|E)?", re.I)


def key(m):
    s = re.sub(r"[\s,.]", "", m)
    if "/" in s:
        y, n = s.split("/"); return f"U:{y}{int(n):07d}"
    if len(s) == 11 and s.startswith("20"): return f"U:{s}"
    return f"G:{int(s)}"


def numbers(text):
    """the final {"results": [...]} line if present, else every US number in order of first mention"""
    for line in reversed(text.strip().splitlines()):
        m = re.search(r'\{\s*"results"\s*:\s*\[.*?\]\s*\}', line)
        if m:
            try: return [x for x in json.loads(m.group(0))["results"]], "json"
            except Exception: pass
    return [x.group(0) for x in NUM.finditer(text)], "regex"
