# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
"""Shared model code for training and evaluation: load a (decoder or encoder) embedding model, optional LoRA adapter,
pooling, instruction formatting, batched encoding.

Qwen3-Embedding / harrier-oss-v1-0.6b / Nemotron-style decoders: last-token pooling with left padding and an instruction prefix on the
query side only. Encoder models (ModernBERT etc.): mean pooling. Both are selected by --pooling.
"""
import math
import os

import torch
import torch.nn.functional as F

DEFAULT_INSTRUCTION = ("Given the claims of a patent application, retrieve prior-art patent documents that anticipate the "
                       "claims or render them obvious")
# granted claims attacked after grant (PTAB trials, EP oppositions): invalidity search
GRANTED_INSTRUCTION = ("Given the claims of a granted patent, retrieve prior art that anticipates them or renders them obvious "
                       "(invalidity search)")
GRANTED_SOURCES = {"ptab", "opp_amended", "opp_revoked"}


def instruction_for(source):
    return GRANTED_INSTRUCTION if source in GRANTED_SOURCES else DEFAULT_INSTRUCTION


def format_query(text, instruction=DEFAULT_INSTRUCTION):
    if not instruction:
        return text
    return f"Instruct: {instruction}\nQuery: {text}"


def load_model(name_or_path, adapter=None, lora=None, dtype="auto", grad_ckpt=False, attn=None, device="cpu"):
    """lora: dict(r=, alpha=, dropout=, targets=[...]) to wrap a fresh LoRA adapter for training; adapter: path to a
    trained adapter to load (evaluation or resume)."""
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name_or_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if dtype == "auto":
        dtype = torch.bfloat16 if device != "cpu" and torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    elif isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    kw = {"torch_dtype": dtype}
    if attn:
        kw["attn_implementation"] = attn
    model = AutoModel.from_pretrained(name_or_path, **kw)
    if grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=lora is not None)
    elif lora:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora.get("dropout", 0.0),
                         target_modules=lora.get("targets") or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(model, cfg)
    if grad_ckpt and (lora or adapter):
        model.enable_input_require_grads()
    return tok, model.to(device)


def pool(last_hidden, attention_mask, pooling):
    if pooling == "last":
        # works for left and right padding: index of the last non-pad token in each row
        if attention_mask[:, -1].all():
            return last_hidden[:, -1]
        idx = attention_mask.sum(1) - 1
        return last_hidden[torch.arange(last_hidden.size(0), device=last_hidden.device), idx]
    if pooling == "mean":
        m = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        return (last_hidden * m).sum(1) / m.sum(1).clamp(min=1)
    if pooling == "cls":
        return last_hidden[:, 0]
    raise ValueError(pooling)


def tokenize(tok, texts, max_len, pooling):
    tok.padding_side = "left" if pooling == "last" else "right"
    return tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")


def embed(model, batch, pooling):
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    return pool(out.last_hidden_state, batch["attention_mask"], pooling)


def normalize(x, dim=None):
    if dim:
        x = x[..., :dim]
    return F.normalize(x.float(), dim=-1)


@torch.no_grad()
def encode(tok, model, texts, max_len, pooling, batch_size=32, device="cpu", dim=None, max_tokens=None):
    """Encode texts in the given order; returns float16 numpy [n, dim]. If max_tokens is set, batches are built by token
    budget (texts should be roughly sorted by length, as the tac shards are)."""
    import numpy as np
    model.eval()
    outs = []
    i = 0
    while i < len(texts):
        j = min(len(texts), i + batch_size)
        if max_tokens:
            est = max(1, min(max_len, len(texts[j - 1]) // 4))
            j = min(len(texts), i + max(1, max_tokens // est))
        b = tokenize(tok, texts[i:j], max_len, pooling)
        b = {k: v.to(device) for k, v in b.items()}
        e = normalize(embed(model, b, pooling), dim)
        outs.append(e.cpu().to(torch.float16).numpy())
        i = j
    return np.concatenate(outs) if outs else np.zeros((0, dim or 0), dtype="float16")


def env_rank():
    return int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("LOCAL_RANK", 0))


def cosine_lr(step, total, warmup, base):
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


def export_for_serving(base, out, trained=None, adapter=None):
    """Write a checkpoint in the base model's own layout (e.g. Qwen3ForCausalLM for Qwen3-Embedding), so vLLM and other
    servers load it exactly like the stock model: LoRA merged, trained transformer weights copied into the base."""
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
    cfg = AutoConfig.from_pretrained(base)
    arch = (cfg.architectures or [""])[0]
    tok = AutoTokenizer.from_pretrained(trained or adapter or base)
    if adapter:
        from peft import PeftModel
        inner = PeftModel.from_pretrained(AutoModel.from_pretrained(base, torch_dtype=torch.bfloat16), adapter).merge_and_unload()
    else:
        inner = AutoModel.from_pretrained(trained, torch_dtype=torch.bfloat16)
    if arch.endswith("ForCausalLM"):
        full = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
        full.model.load_state_dict(inner.state_dict(), strict=True)
        full.save_pretrained(out)
    else:
        inner.save_pretrained(out)
    tok.save_pretrained(out)
    return out


class VllmEncoder:
    """Offline batch embedding with vLLM (pooling runner). Token ids come from the same Hugging Face tokenizer as
    training (identical truncation); last-token pooling, then our own normalization / Matryoshka truncation."""

    def __init__(self, path, max_len, gpu_mem=0.90, tp=1):
        from vllm import LLM
        # PAB_VLLM_DTYPE=float16 on GPUs without bf16 (T4); compare only runs encoded with the same dtype
        kw = dict(model=path, dtype=os.environ.get("PAB_VLLM_DTYPE", "bfloat16"), max_model_len=max_len, gpu_memory_utilization=gpu_mem,
                  tensor_parallel_size=tp, enforce_eager=False)
        try:
            from vllm.config import PoolerConfig
            # PAB_POOLING=MEAN for mean-pooling embedders; default LAST
            # PAB_POOLING=NATIVE: no override, the model's own sentence-transformers pooling
            pt = os.environ.get("PAB_POOLING", "LAST")
            pc = None if pt == "NATIVE" else PoolerConfig(pooling_type=pt, normalize=False)
        except Exception:
            pc = None
        for extra in ({"runner": "pooling", "pooler_config": pc}, {"runner": "pooling", "override_pooler_config": pc},
                      {"task": "embed", "override_pooler_config": pc}):
            try:
                self.llm = LLM(**kw, **{k: v for k, v in extra.items() if v is not None})
                break
            except TypeError:
                continue
        self.max_len = max_len

    def __call__(self, tok, texts, dim=None, batch=None):
        """Encode in batches: token ids and vLLM's per-text Python float lists are large (a 250K-document call
        used > 60 GB of RAM and hung the trial box), so each batch is converted to float16 before the next one.
        Batch size is a token budget (8M token ids, ~1-2 GB of Python lists): 8192 documents at max_len 8192
        (descriptions) used 11 GB and was OOM-killed in the ablation C early read."""
        import numpy as np
        batch = batch or max(256, min(8192, 8_000_000 // self.max_len))
        tok.padding_side = "right"
        parts = []
        for i in range(0, len(texts), batch):
            ids = tok(texts[i:i + batch], truncation=True, max_length=self.max_len)["input_ids"]
            outs = self.llm.embed([{"prompt_token_ids": x} for x in ids], use_tqdm=False)
            e = torch.tensor(np.asarray([o.outputs.embedding for o in outs], dtype=np.float32))
            parts.append(normalize(e, dim).to(torch.float16).numpy())
            del ids, outs, e
        return np.concatenate(parts) if parts else np.zeros((0, dim or 0), dtype="float16")
