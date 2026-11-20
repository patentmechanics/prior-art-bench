#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
"""Export a trained checkpoint for vLLM / serving: merge LoRA and write the base model's own layout.

  python train/export.py --base Qwen/Qwen3-Embedding-0.6B --trained RUN/step-N --out EXPORT        # full fine-tune
  python train/export.py --base Qwen/Qwen3-Embedding-8B --adapter RUN/step-N --out EXPORT          # LoRA
(--trained pointing at a LoRA checkpoint, i.e. with adapter_config.json, is treated as --adapter)
"""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import export_for_serving

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--trained", default=None)
ap.add_argument("--adapter", default=None)
ap.add_argument("--out", required=True)
a = ap.parse_args()
assert bool(a.trained) != bool(a.adapter), "give --trained or --adapter"
if a.trained and os.path.exists(os.path.join(a.trained, "adapter_config.json")):   # a LoRA checkpoint given as --trained
    a.trained, a.adapter = None, a.trained
print(export_for_serving(a.base, a.out, trained=a.trained, adapter=a.adapter))
