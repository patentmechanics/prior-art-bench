# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
"""Query instructions and text encoders (sentence-transformers or vLLM) used by evaluate.py encode and verify_sample.py.

The instruction strings and the "Instruct: ...\\nQuery: " layout are the exact ones used in training and in every
published evaluation (train/common.py has the same strings; tests/test_metrics.py checks they agree). Anticipate-0.6B
ships them as its sentence-transformers prompts `query` (office-action queries) and `invalidity_query` (PTAB queries),
so `prompt_name="query"` and prepending QUERY_PREFIX give identical input text.
"""
import numpy as np

DEFAULT_INSTRUCTION = ("Given the claims of a patent application, retrieve prior-art patent documents that anticipate the "
                       "claims or render them obvious")
# granted claims attacked after grant (PTAB trials): invalidity search
GRANTED_INSTRUCTION = ("Given the claims of a granted patent, retrieve prior art that anticipates them or renders them obvious "
                       "(invalidity search)")
GRANTED_SOURCES = {"ptab", "opp_amended", "opp_revoked"}


def instruction_for(source):
    return GRANTED_INSTRUCTION if source in GRANTED_SOURCES else DEFAULT_INSTRUCTION


def format_query(text, instruction=DEFAULT_INSTRUCTION):
    if not instruction:
        return text
    return f"Instruct: {instruction}\nQuery: {text}"


QUERY_PREFIX = format_query("", DEFAULT_INSTRUCTION)          # = prompt `query` of Anticipate-0.6B
INVALIDITY_PREFIX = format_query("", GRANTED_INSTRUCTION)     # = prompt `invalidity_query`


def query_prefix(source, setting):
    """setting: "instruction" (ours, per source), "" (none) or a literal prefix (e.g. a baseline's own query prompt)."""
    if setting == "instruction":
        return format_query("", instruction_for(source))
    return setting or ""


def best_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class Encoder:
    """texts -> L2-normalised float16 [n, dim]. backend "st": sentence-transformers with the model's own modules
    (pooling, normalisation); backend "vllm": offline vLLM pooling runner with token ids from the Hugging Face tokenizer
    (as in our evaluations), pooling LAST / MEAN / NATIVE (the model's own sentence-transformers pooling)."""

    def __init__(self, model, backend="st", revision=None, max_len=8192, pooling="LAST", device=None, batch=8,
                 dtype="bfloat16", gpu_mem=0.90):
        self.backend, self.max_len, self.batch = backend, max_len, batch
        if backend == "st":
            import torch
            from sentence_transformers import SentenceTransformer
            self.device = device or best_device()
            self.st = SentenceTransformer(model, revision=revision, device=self.device,
                                          model_kwargs={"torch_dtype": getattr(torch, dtype)})
            self.st.max_seq_length = max_len
        elif backend == "vllm":
            from transformers import AutoTokenizer
            from vllm import LLM
            self.tok = AutoTokenizer.from_pretrained(model, revision=revision)
            kw = dict(model=model, revision=revision, dtype=dtype, max_model_len=max_len, gpu_memory_utilization=gpu_mem,
                      enforce_eager=False)
            try:
                from vllm.config import PoolerConfig
                pc = None if pooling.upper() == "NATIVE" else PoolerConfig(pooling_type=pooling.upper(), normalize=False)
            except Exception:
                pc = None
            self.llm = None
            for extra in ({"runner": "pooling", "pooler_config": pc}, {"runner": "pooling", "override_pooler_config": pc},
                          {"task": "embed", "override_pooler_config": pc}):
                try:
                    self.llm = LLM(**kw, **{k: v for k, v in extra.items() if v is not None})
                    break
                except TypeError:
                    continue
            if self.llm is None:
                raise SystemExit("this vLLM version accepts none of the known pooling-runner arguments")
        else:
            raise ValueError(backend)

    def set_max_len(self, n):
        self.max_len = n
        if self.backend == "st":
            self.st.max_seq_length = n

    def __call__(self, texts):
        if not texts:
            return np.zeros((0, 0), dtype=np.float16)
        if self.backend == "st":
            e = self.st.encode(texts, batch_size=self.batch, normalize_embeddings=True, convert_to_numpy=True,
                               show_progress_bar=False)
            return np.asarray(e, dtype=np.float32).astype(np.float16)
        # vLLM: batches by token budget, converted to float16 batch by batch (memory)
        batch = max(256, min(8192, 8_000_000 // self.max_len))
        self.tok.padding_side = "right"
        parts = []
        for i in range(0, len(texts), batch):
            ids = self.tok(texts[i:i + batch], truncation=True, max_length=self.max_len)["input_ids"]
            outs = self.llm.embed([{"prompt_token_ids": x} for x in ids], use_tqdm=False)
            e = np.asarray([o.outputs.embedding for o in outs], dtype=np.float32)
            e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
            parts.append(e.astype(np.float16))
        return np.concatenate(parts)


def load_config(path):
    """configs/<system>.json -> dict with model, revision, query_prefix, doc_prefix, q_len, d_len, pooling."""
    import json
    cfg = {"revision": None, "query_prefix": "instruction", "doc_prefix": "", "q_len": 4096, "d_len": 8192, "pooling": "LAST"}
    if path:
        cfg.update(json.load(open(path)))
    return cfg
