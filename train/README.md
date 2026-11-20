# Training

The code that trained Anticipate-0.6B, with our infrastructure-specific parts (object-storage checkpointing, job
queue hooks) removed. The computation is unchanged.

> **Mined hard negatives are not shipped.** The benchmark ships the training positives as IDs and grades
> (`train-qrels/` in the dataset: Y-cleaned office-action X/Y citations), but not our mined negatives and not a
> ready-made training pack. To train, build the training queries and labels with `build/` (level 5 of
> [REPRODUCING.md](../REPRODUCING.md)), mine your own negatives with `retrieve.py` + `mine.py`, and pack them with
> `pack.py`. Your pack will differ from ours, so expect a model close to, but not identical with, the release.

| Script | What |
|---|---|
| `pack.py` | training pack: training queries + positives (`keep`, grade >= 2; `--y-indep-only` = Y-cleaning) + mined negatives + known set; document text |
| `retrieve.py` | encode a document pool and retrieve the top-k for training queries (build/ layout); used for mining |
| `mine.py` | hard negatives from a retrieval run: rank window, known and co-cited families excluded, positive-aware score cap |
| `train.py` | InfoNCE over the global batch (in-batch positives + hard negatives), false-negative masking, GradCache, DDP, full or LoRA |
| `soup.py` | uniform (or weighted) average of checkpoints |
| `export.py` | write a checkpoint in the base model's own layout (vLLM loads it like the stock model) |
| `make_model_repo.py` | add the sentence-transformers files (last-token pooling, normalisation, the `query` / `invalidity_query` prompts) |
| `common.py` | model loading, pooling, query instructions, batched encoding, vLLM encoder |
| `run_anticipate_0.6b.sh` | the release recipe end to end (3 seeds, soup, export) |

## Recipe (Anticipate-0.6B)

Full fine-tuning of `microsoft/harrier-oss-v1-0.6b` @ `f9b9dc8d367d443f2479d27aa5d8d2850c0774ee`; InfoNCE, temperature
0.02; 64 queries per step with in-batch negatives and 3 mined hard negatives per query; AdamW, learning rate 1e-5,
50 warm-up steps, cosine schedule over 900 steps, stopped at step 75; query 4,096 / document 8,192 tokens; seeds 0, 1,
2; release weights = uniform average of the three step-75 checkpoints. `run_anticipate_0.6b.sh PACK OUT [NPROC]` runs
exactly these settings.

## Making a pack like ours

Our pack was built in this order (inputs from `build/`, see `build/README.md`):

1. Mining queries: the first 100,000 office-action training queries in `pack.py`'s md5 order (`build/mine_pool_us.py ids`),
   and the mining pool = every positive family of those queries (`build/mine_pool_us.py parts`).
2. Retrieval with a base-recipe fine-tune of harrier-oss-v1-0.6b (three seeds averaged), documents = title + abstract + claims
   (4,096 tokens): `retrieve.py encode` over the pool, `retrieve.py search --queries ... --k 100`.
3. Negatives: `mine.py --lo 30 --hi 100 --n 32 --pos-cap 0.95 --cocited <cocited_w1.parquet>` (ranks 31-100; families
   cited for the query in any source, its own family and families co-cited with a positive excluded
   (`build/cocite.py`); anything scoring above 0.95 x the best positive dropped).
4. Pack: `pack.py --sources us_oa=1 --negatives NEG --max-negs 32 --query-field tac_text --split train --min-grade 2
   --text-col tacd_text --query-ids <the 100,000 queries> --y-indep-only` (`build/make_pack_us.sh` has the full call).

Our three seeds gave dev R@10 0.514 / 0.515 / 0.519 and 0.529 for the average (pool v2, dev half).

## Hardware

Our runs: NVIDIA RTX PRO 6000 Blackwell (96 GB), bf16-mixed precision, PyTorch SDPA attention; `requirements-gpu.txt`
pins the environment. Multi-GPU: `torchrun --standalone --nproc_per_node N train/train.py ...`; the global batch stays
64 for any number of GPUs.
