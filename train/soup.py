#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
"""Weight-space average ("model soup") of checkpoints with the same architecture, written as a new checkpoint in the
training format (train/train.py step-N/ directories). The release weights of Anticipate-0.6B are the uniform average of
the three step-75 checkpoints (seeds 0, 1, 2).

  python train/soup.py --ckpts out/seed0/step-75 out/seed1/step-75 out/seed2/step-75 --out out/soup
  python train/soup.py --ckpts A=0.7 B=0.3 --out out/wise               # weighted (WiSE-FT style)

Weights are normalised to sum 1. Averaging is in fp32, one input at a time (memory ~2 x the model size). Tensors of
later inputs are matched to the first input's names with or without a "model." prefix. Config and tokenizer files are
copied from the first input; soup.json records the inputs and weights. Then train/export.py writes the serving layout.
"""
import argparse, json, os, shutil, sys, time

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True, help="checkpoint directories (each with model.safetensors), optional =weight")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ins = [(x.rsplit("=", 1)[0], float(x.rsplit("=", 1)[1])) if "=" in os.path.basename(x) else (x, 1.0) for x in a.ckpts]
    tot = sum(w for _, w in ins)
    ins = [(p, w / tot) for p, w in ins]
    if os.path.exists(a.out):
        sys.exit(f"{a.out} exists")
    acc, ref_keys = None, None
    for i, (d, w) in enumerate(ins):
        t0 = time.time()
        path = os.path.join(d, "model.safetensors")
        assert os.path.exists(path), f"no model.safetensors in {d}"
        with safe_open(path, framework="pt") as f:
            keys = list(f.keys())
            if acc is None:
                ref_keys = keys
                acc = {k: f.get_tensor(k).to(torch.float32) * w for k in keys}
                shutil.copytree(d, a.out, ignore=shutil.ignore_patterns("model.safetensors", "*.pt"))
            else:
                have = set(keys)
                for k in ref_keys:
                    src = k if k in have else (f"model.{k}" if f"model.{k}" in have else
                                               (k[len("model."):] if k.startswith("model.") and k[len("model."):] in have else None))
                    assert src is not None, f"{d}: no tensor for {k}"
                    t = f.get_tensor(src).to(torch.float32)
                    assert t.shape == acc[k].shape, f"{d}: shape {tuple(t.shape)} != {tuple(acc[k].shape)} for {k}"
                    acc[k] += t * w
        print(json.dumps({"input": d, "weight": round(w, 4), "tensors": len(keys), "seconds": round(time.time() - t0, 1)}), flush=True)
    save_file({k: v.contiguous() for k, v in acc.items()}, os.path.join(a.out, "model.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(a.out, "soup.json"), "w") as fh:
        json.dump({"inputs": [{"path": p, "weight": w} for p, w in ins], "time": time.time()}, fh, indent=1)
    print(json.dumps({"out": a.out}))


if __name__ == "__main__":
    main()
