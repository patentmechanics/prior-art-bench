#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
"""Turn an exported checkpoint (train/export.py output: bf16 transformer weights + tokenizer, base-model layout) into the
Hugging Face model repo of Patent Mechanics Anticipate-0.6B: adds the sentence-transformers files (last-token pooling,
L2 normalisation, our prompts, max_seq_length 8192) and checks the weights are bf16. Model card, LICENSE and NOTICE are
added separately.

  python train/make_model_repo.py --export EXPORT_DIR --base BASE_MODEL_DIR --out REPO_DIR
"""
import argparse, hashlib, json, os, shutil, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DEFAULT_INSTRUCTION, GRANTED_INSTRUCTION, format_query   # the exact strings used in training/evaluation

ap = argparse.ArgumentParser()
ap.add_argument("--export", required=True)
ap.add_argument("--base", required=True, help="local snapshot of microsoft/harrier-oss-v1-0.6b (for its pooling config)")
ap.add_argument("--out", required=True)
a = ap.parse_args()

if os.path.exists(a.out):
    shutil.rmtree(a.out)
shutil.copytree(a.export, a.out, ignore=shutil.ignore_patterns("chat_template.jinja", "generation_config.json"))

cfg = json.load(open(f"{a.out}/config.json"))
assert cfg.get("architectures") == ["Qwen3Model"], cfg.get("architectures")
assert cfg.get("dtype", cfg.get("torch_dtype")) == "bfloat16", "export must be bf16"

# sentence-transformers: Transformer -> Pooling (last token, as in training and evaluation) -> Normalize
json.dump([{"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
           {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
           {"idx": 2, "name": "2", "path": "2_Normalize", "type": "sentence_transformers.models.Normalize"}],
          open(f"{a.out}/modules.json", "w"), indent=2)
os.makedirs(f"{a.out}/1_Pooling", exist_ok=True)
os.makedirs(f"{a.out}/2_Normalize", exist_ok=True)
pool = json.load(open(f"{a.base}/1_Pooling/config.json"))
assert pool["pooling_mode_lasttoken"] and pool["word_embedding_dimension"] == 1024, pool
json.dump(pool, open(f"{a.out}/1_Pooling/config.json", "w"), indent=2)
json.dump({"max_seq_length": 8192, "do_lower_case": False}, open(f"{a.out}/sentence_bert_config.json", "w"), indent=2)
qp = format_query("", DEFAULT_INSTRUCTION)
ip = format_query("", GRANTED_INSTRUCTION)
json.dump({"prompts": {"query": qp, "invalidity_query": ip, "document": ""}, "default_prompt_name": None,
           "similarity_fn_name": "cosine"}, open(f"{a.out}/config_sentence_transformers.json", "w"), indent=2)

manifest = {}
for root, _, files in os.walk(a.out):
    for f in sorted(files):
        p = os.path.join(root, f)
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
        manifest[os.path.relpath(p, a.out)] = {"bytes": os.path.getsize(p), "sha256": h.hexdigest()}
print(json.dumps({"repo": a.out, "prompts": {"query": qp, "invalidity_query": ip}, "files": manifest}, indent=1))
