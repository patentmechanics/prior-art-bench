#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
"""Contrastive fine-tuning of an embedding model on a training pack (train/pack.py).

  torchrun --nproc_per_node 8 train/train.py --pack PACK --model Qwen/Qwen3-Embedding-0.6B --out RUN \
      --batch 256 --hard-negs 7 --q-len 2048 --d-len 4096 --lr 2e-5 --steps 5000 [--lora 32]

Loss: InfoNCE over the query's positive vs every document in the global batch (all ranks): in-batch positives and hard
negatives of other queries, plus the query's own hard negatives. False-negative masking: any document whose family is in the
query's known set (every family cited for it, and its own family) is removed from its denominator, except its positive.
Matryoshka (--mrl 1024,512,256): the loss is averaged over truncated, re-normalized dimensions.
GradCache (--chunk N): representations are computed without gradients in chunks of N sequences, the loss gradient w.r.t.
the representations is taken, then each chunk is re-encoded with gradients and back-propagated with that cached gradient.
Memory therefore scales with the chunk size, not the batch size.
Each rank draws its own --batch/world queries per step. Sources are drawn with --mix weights (default: as packed).
Checkpoints: RUN/step-N/ (adapter or full model + tokenizer) and RUN/state.pt (resume with the same command).
--keep-every K deletes local intermediate checkpoints that are not multiples of K once a newer one is saved.
--batch is split as evenly as possible over any number of GPUs (sizes may differ by one); the batch drawn at a step
depends only on (seed, step, rank).
LoRA: the frozen base weights are kept in bf16 (adapter weights and optimizer in fp32) under bf16-mixed.
The release recipe (Anticipate-0.6B) is in train/run_anticipate_0.6b.sh.
(Our runs also had an option to keep the checkpoint of record in object storage for multi-machine restarts; it is left
out here and does not change the computation.)
"""
import argparse, json, os, shutil, sys, time
from contextlib import nullcontext
from datetime import timedelta

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (cosine_lr, instruction_for, embed, env_rank, format_query, load_model, normalize, pool, tokenize)


class Pack:
    def __init__(self, path):
        q = pq.read_table(f"{path}/queries.parquet")
        self.qid = q.column("query_id").to_pylist()
        self.source = q.column("source").to_pylist()
        self.q_text = q.column("q_text")
        self.pos = q.column("pos_ids").to_pylist()
        self.grades = q.column("pos_grades").to_pylist()
        self.negs = q.column("neg_ids").to_pylist()
        self.known = [set(k) for k in q.column("known_ids").to_pylist()]
        # documents as an uncompressed Arrow file, memory-mapped: long texts (descriptions ablation, ~18 GB) stay in the
        # page cache instead of process memory
        arrow = f"{path}/docs.arrow"
        if not os.path.exists(arrow):
            import pyarrow as pa
            pf = pq.ParquetFile(f"{path}/docs.parquet")
            with pa.OSFile(arrow + ".tmp", "wb") as sink, pa.ipc.new_file(sink, pf.schema_arrow) as w:
                for b in pf.iter_batches(batch_size=20000):
                    w.write_batch(b)
            os.replace(arrow + ".tmp", arrow)
        import pyarrow as pa
        d = pa.ipc.open_file(pa.memory_map(arrow, "r")).read_all()
        self.doc_text = d.column("text")
        self.doc_row = {f: i for i, f in enumerate(d.column("family_id").to_pylist())}
        self.doc_ids = list(self.doc_row)
        self.by_source = {}
        for i, s in enumerate(self.source):
            self.by_source.setdefault(s, []).append(i)

    def text(self, fam):
        return self.doc_text[self.doc_row[fam]].as_py()


def sample_batch(pack, rng, n, hard_negs, mix, grade_weight):
    """n queries -> (query texts, doc texts, doc families, positive index per query, known sets, query sources)."""
    srcs = list(mix)
    w = np.array([mix[s] for s in srcs], dtype=float)
    w /= w.sum()
    qs, docs, fams, pos_idx, known, q_src = [], [], [], [], [], []
    for _ in range(n):
        s = srcs[rng.choice(len(srcs), p=w)]
        i = pack.by_source[s][rng.integers(len(pack.by_source[s]))]
        cands = [(f, g) for f, g in zip(pack.pos[i], pack.grades[i]) if f in pack.doc_row]
        if not cands:
            continue
        gw = np.array([grade_weight.get(g, 1.0) for _, g in cands])
        f_pos = cands[rng.choice(len(cands), p=gw / gw.sum())][0]
        negs = [f for f in pack.negs[i] if f in pack.doc_row and f not in pack.known[i]]
        rng.shuffle(negs)
        negs = negs[:hard_negs]
        while len(negs) < hard_negs:   # top up with random corpus documents
            f = pack.doc_ids[rng.integers(len(pack.doc_ids))]
            if f not in pack.known[i] and f not in negs:
                negs.append(f)
        qs.append(pack.q_text[i].as_py())
        q_src.append(s)
        pos_idx.append(len(docs))
        for f in [f_pos] + negs:
            docs.append(pack.text(f))
            fams.append(f)
        known.append(pack.known[i] | {f_pos})
    return qs, docs, fams, pos_idx, known, q_src


def gather(x, world, rank):
    """all-gather [n_r, d] tensors whose n_r may differ by rank (padded to the largest); the local slice keeps its graph."""
    if world == 1:
        return x
    n = torch.tensor([x.shape[0]], device=x.device)
    ns = [torch.zeros_like(n) for _ in range(world)]
    dist.all_gather(ns, n)
    ns = [int(v) for v in ns]
    pad = x.new_zeros((max(ns),) + tuple(x.shape[1:]))
    pad[:x.shape[0]] = x.detach()
    parts = [torch.empty_like(pad) for _ in range(world)]
    dist.all_gather(parts, pad)
    parts = [t[:k] for t, k in zip(parts, ns)]
    parts[rank] = x
    return torch.cat(parts)


def branch_fetch(src, n, out):
    """a cool-down branch starts from the main run's kept state at step n: state-n.pt and its step-n checkpoint"""
    os.makedirs(out, exist_ok=True)
    shutil.copy(f"{src}/state-{n}.pt", f"{out}/state.pt.branch")
    shutil.copytree(f"{src}/step-{n}", f"{out}/step-{n}", dirs_exist_ok=True)
    st = torch.load(f"{out}/state.pt.branch", map_location="cpu", weights_only=False)
    assert st["step"] == n, (st["step"], n)
    st["ckpt"] = f"{out}/step-{n}"
    torch.save(st, f"{out}/state.pt")
    os.remove(f"{out}/state.pt.branch")


def gather_obj(o, world):
    if world == 1:
        return [o]
    out = [None] * world
    dist.all_gather_object(out, o)
    return out


def contrastive_loss(q, d, pos_idx, mask, temp, mrl):
    """q [B, H], d [D, H] (global), pos_idx [B] (global doc index), mask [B, D] True = exclude."""
    losses = []
    for dim in (mrl or [None]):
        s = normalize(q, dim) @ normalize(d, dim).T / temp
        s = s.masked_fill(mask, float("-inf"))
        losses.append(F.cross_entropy(s, pos_idx))
    return torch.stack(losses).mean()


def plan_chunks(tok, texts, max_len, pooling, chunk, chunk_tokens):
    """Tokenize once and group sequences into GPU mini-batches. With chunk_tokens > 0: sort by length and fill each
    mini-batch up to chunk_tokens padded tokens (short texts many per forward, long ones few; almost no padding).
    Otherwise fixed groups of `chunk` in the given order. Returns (input_ids lists, list of index lists)."""
    ids = tok(texts, truncation=True, max_length=max_len)["input_ids"]
    if not chunk_tokens:
        return ids, [list(range(i, min(len(ids), i + chunk))) for i in range(0, len(ids), chunk)]
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    groups, cur = [], []
    for i in order:
        if cur and (len(cur) + 1) * len(ids[i]) > chunk_tokens:
            groups.append(cur); cur = []
        cur.append(i)
    if cur:
        groups.append(cur)
    return ids, groups


def batch_of(tok, ids, idx, pooling, device):
    tok.padding_side = "left" if pooling == "last" else "right"
    b = tok.pad({"input_ids": [ids[i] for i in idx]}, padding=True, return_tensors="pt")
    return {k: v.to(device) for k, v in b.items()}


def encode_chunks(model, tok, ids, groups, pooling, device):
    """representations (no grad) in original order"""
    reps = [None] * len(ids)
    with torch.no_grad():
        for idx in groups:
            r = embed(model, batch_of(tok, ids, idx, pooling, device), pooling).float()
            for j, i in enumerate(idx):
                reps[i] = r[j]
    return torch.stack(reps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pooling", default="last", choices=["last", "mean", "cls"])
    ap.add_argument("--instruction", default=None, help="override; default: per source (common.instruction_for)")
    ap.add_argument("--batch", type=int, default=64, help="global queries per step")
    ap.add_argument("--hard-negs", type=int, default=3)
    ap.add_argument("--q-len", type=int, default=1024)
    ap.add_argument("--d-len", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=8, help="GradCache chunk (sequences per forward) when --chunk-tokens is 0")
    ap.add_argument("--chunk-tokens", type=int, default=0,
                    help="length-grouped GPU mini-batches of at most this many padded tokens (0 = fixed --chunk groups)")
    ap.add_argument("--temp", type=float, default=0.02)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--mix", default="", help="source=weight,... (default: proportional to the pack)")
    ap.add_argument("--grade-weight", default="3=1.0,2=0.7")
    ap.add_argument("--mrl", default="", help="Matryoshka dims, e.g. 1024,512,256 (empty = full only)")
    ap.add_argument("--lora", type=int, default=0, help="LoRA rank (0 = full fine-tuning)")
    ap.add_argument("--lora-alpha", type=int, default=0)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--attn", default=None, help="e.g. flash_attention_2 / sdpa")
    ap.add_argument("--precision", default="auto", choices=["auto", "bf16-mixed", "bf16", "fp32"],
                    help="auto: bf16-mixed on GPU (fp32 weights + optimizer, bf16 compute), fp32 on CPU")
    ap.add_argument("--prefetch", type=int, default=1, help="prepare the next batch on a CPU thread (1) or not (0)")
    ap.add_argument("--stop-at", type=int, default=0, help="stop after this step while keeping the --steps schedule "
                    "(e.g. compare with another run's step-N checkpoint)")
    ap.add_argument("--save-every", type=int, default=500)
    # cool-down branches (warmup-stable-decay): the main run keeps optimizer states at the branch points,
    # a branch resumes from one of them and decays the LR linearly to ~0 over --decay-steps (stop it with --stop-at)
    ap.add_argument("--save-state-at", default="", help="steps whose full state (weights + optimizer) is kept as state-N.pt")
    ap.add_argument("--branch-from", default="", help="directory of a main run: start from its state-N.pt")
    ap.add_argument("--branch-step", type=int, default=0, help="N of --branch-from")
    ap.add_argument("--decay-from", type=int, default=0, help="from this step, LR falls linearly from its scheduled value to ~0")
    ap.add_argument("--decay-steps", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-every", type=int, default=0, help="delete intermediate checkpoints not divisible by this")
    ap.add_argument("--pg-timeout-min", type=int, default=10, help="collective timeout (minutes) for multi-GPU runs")
    a = ap.parse_args()

    rank, world, local = env_rank()
    device = f"cuda:{local}" if torch.cuda.is_available() else "cpu"
    if device != "cpu":
        torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", timeout=timedelta(minutes=a.pg_timeout_min))
    os.makedirs(a.out, exist_ok=True)
    lora = {"r": a.lora, "alpha": a.lora_alpha or 2 * a.lora} if a.lora else None
    state_path = f"{a.out}/state.pt"
    if a.branch_from and not os.path.exists(state_path):   # a branch's first start: no state of its own yet
        if local == 0:
            branch_fetch(a.branch_from.rstrip("/"), a.branch_step, a.out)
        if world > 1:
            dist.barrier()
    resume = torch.load(state_path, map_location="cpu", weights_only=False) if os.path.exists(state_path) else None
    adapter = resume["ckpt"] if (resume and a.lora) else None
    model_path = resume["ckpt"] if (resume and not a.lora) else a.model
    precision = a.precision if a.precision != "auto" else ("bf16-mixed" if device != "cpu" else "fp32")
    # bf16-mixed: fp32 weights and optimizer state, bf16 compute (autocast). Pure bf16 rounds away small updates.
    # LoRA: the frozen base needs no fp32 master copy (PEFT keeps the adapter in fp32): bf16 base halves memory and traffic
    w_dtype = torch.bfloat16 if precision == "bf16" or (a.lora and precision == "bf16-mixed") else torch.float32
    amp = (lambda: torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16)) if precision == "bf16-mixed" else nullcontext
    tok, model = load_model(model_path, adapter=adapter, lora=lora, grad_ckpt=a.grad_ckpt, attn=a.attn, device=device, dtype=w_dtype)
    model.train()
    ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local] if device != "cpu" else None) if world > 1 else model
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd, fused=(device != "cpu"))
    step = 0
    if resume:
        opt.load_state_dict(resume["opt"])
        step = resume["step"]
    pack = Pack(a.pack)
    mix = ({k: float(v) for k, v in (x.split("=") for x in a.mix.split(","))} if a.mix
           else {s: len(v) for s, v in pack.by_source.items()})
    mix = {s: w for s, w in mix.items() if s in pack.by_source and w > 0}
    gw = {int(k): float(v) for k, v in (x.split("=") for x in a.grade_weight.split(","))}
    mrl = [int(x) for x in a.mrl.split(",") if x] or None
    if mrl:
        mrl = [None] + mrl
    per_rank = a.batch // world + (1 if rank < a.batch % world else 0)
    if rank == 0:
        n_train = sum(p.numel() for p in params)
        print(json.dumps({"start": step, "world": world, "per_rank": per_rank, "trainable_params": n_train, "mix": mix, "precision": precision,
                          "queries": len(pack.qid), "docs": len(pack.doc_ids)}), flush=True)
        json.dump(vars(a), open(f"{a.out}/args.json", "w"), indent=1)
    logf = open(f"{a.out}/log.jsonl", "a") if rank == 0 else None
    if logf:   # restarts and resumes are visible in the log
        logf.write(json.dumps({"start": step, "world": world, "per_rank": per_rank, "time": round(time.time())}) + "\n"); logf.flush()
    t0, seen, start_step = time.time(), 0, step
    import copy
    tok_w = copy.deepcopy(tok) if a.prefetch else tok   # the fast tokenizer must not be shared across threads

    def make_batch(st):
        rng = np.random.default_rng([a.seed, st, rank])   # depends only on (seed, step, rank): prefetch changes nothing
        qs, docs, fams, pos_idx, known, srcs = sample_batch(pack, rng, per_rank, a.hard_negs, mix, gw)
        qs = [format_query(t, a.instruction if a.instruction is not None else instruction_for(sr)) for t, sr in zip(qs, srcs)]
        q_ids, q_groups = plan_chunks(tok_w, qs, a.q_len, a.pooling, a.chunk, a.chunk_tokens)
        d_ids, d_groups = plan_chunks(tok_w, docs, a.d_len, a.pooling, a.chunk, a.chunk_tokens)
        return qs, fams, pos_idx, known, q_ids, q_groups, d_ids, d_groups
    from concurrent.futures import ThreadPoolExecutor
    pool_ex = ThreadPoolExecutor(1) if a.prefetch else None
    nxt = pool_ex.submit(make_batch, step) if pool_ex else None
    end = min(a.steps, a.stop_at) if a.stop_at else a.steps
    keep_state = {int(x) for x in a.save_state_at.split(",") if x}
    while step < end:
        if pool_ex:
            batch = nxt.result()
            if step + 1 < end:
                nxt = pool_ex.submit(make_batch, step + 1)
        else:
            batch = make_batch(step)
        qs, fams, pos_idx, known, q_ids, q_groups, d_ids, d_groups = batch
        docs = d_ids
        # 1. representations without gradients
        with amp():
            q_rep = encode_chunks(model, tok, q_ids, q_groups, a.pooling, device)
            d_rep = encode_chunks(model, tok, d_ids, d_groups, a.pooling, device)
        # 2. loss over the global batch; gradients w.r.t. the local representations
        q_leaf = q_rep.detach().requires_grad_(True)
        d_leaf = d_rep.detach().requires_grad_(True)
        all_fams = sum(gather_obj(fams, world), [])
        n_docs = gather_obj(len(docs), world)
        offset = sum(n_docs[:rank])
        q_all = gather(q_leaf, world, rank)
        d_all = gather(d_leaf, world, rank)
        all_known = sum(gather_obj(known, world), [])
        all_pos = []
        off = 0
        for r, pl in enumerate(gather_obj(pos_idx, world)):
            all_pos += [p + off for p in pl]
            off += n_docs[r]
        fam_t = np.asarray(all_fams, dtype=np.int64)
        mask_np = np.stack([np.isin(fam_t, np.fromiter(kn, dtype=np.int64)) for kn in all_known])
        mask_np[np.arange(len(all_pos)), all_pos] = False
        mask = torch.from_numpy(mask_np).to(device)
        pos_t = torch.tensor(all_pos, device=device)
        loss = contrastive_loss(q_all, d_all, pos_t, mask, a.temp, mrl)
        # each rank computes the same global loss; scale so that DDP's gradient averaging gives the true gradient
        (loss * world).backward()
        q_grad, d_grad = q_leaf.grad, d_leaf.grad
        # 3. re-encode with gradients, chunk by chunk, back-propagating the cached gradients
        jobs = [(q_ids, q_groups, q_grad), (d_ids, d_groups, d_grad)]
        n_chunks = sum(len(gr) for _, gr, _ in jobs)
        k = 0
        for ids, groups, g in jobs:
            for idx in groups:
                k += 1
                ctx = ddp.no_sync() if (world > 1 and k < n_chunks) else nullcontext()
                with ctx, amp():
                    b = batch_of(tok, ids, idx, a.pooling, device)
                    out = ddp(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
                    r = pool(out.last_hidden_state, b["attention_mask"], a.pooling).float()
                    (r * g[torch.tensor(idx, device=g.device)]).sum().backward()
        lr = cosine_lr(step, a.steps, a.warmup, a.lr)
        if a.decay_steps and step >= a.decay_from:   # cool-down: linear from the scheduled LR at decay_from to ~0
            lr = cosine_lr(a.decay_from, a.steps, a.warmup, a.lr) * max(0.0, (a.decay_from + a.decay_steps - step) / a.decay_steps)
        for pg in opt.param_groups:
            pg["lr"] = lr
        gnorm = torch.nn.utils.clip_grad_norm_(params, a.clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        seen += a.batch
        if rank == 0 and (step % a.log_every == 0 or step == 1):
            rec = {"step": step, "loss": round(loss.item(), 4), "lr": lr, "gnorm": round(float(gnorm), 3),
                   "q_per_s": round(seen / (time.time() - t0), 2), "docs_per_step": len(fam_t),
                   "s_per_step": round((time.time() - t0) / (step - start_step), 2),
                   "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1) if device != "cpu" else 0}
            print(json.dumps(rec), flush=True)
            logf.write(json.dumps(rec) + "\n"); logf.flush()
        if rank == 0 and (step % a.save_every == 0 or step == end or step in keep_state):
            ck = f"{a.out}/step-{step}"
            model.save_pretrained(ck)
            tok.save_pretrained(ck)
            torch.save({"step": step, "opt": opt.state_dict(), "ckpt": ck}, state_path + ".tmp")
            os.replace(state_path + ".tmp", state_path)
            if step in keep_state:   # branch point: this state is kept (state.pt itself moves on)
                shutil.copy(state_path, f"{a.out}/state-{step}.pt")
            if a.keep_every:
                for d in os.listdir(a.out):
                    if d.startswith("step-") and d != f"step-{step}" and int(d[5:]) % a.keep_every:
                        shutil.rmtree(f"{a.out}/{d}", ignore_errors=True)
            print(json.dumps({"saved": ck}), flush=True)
        if world > 1:
            dist.barrier()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
