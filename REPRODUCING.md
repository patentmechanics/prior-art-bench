# Reproducing the published results

Every number we publish (model card, dataset card, paper, blog post) can be checked by someone outside Patent
Mechanics, using only public artefacts. This page says, for each number, what to download, what to run, what it
costs, and what you should get.

Public artefacts used below:

| Artefact | Where | Licence |
|---|---|---|
| Model `anticipate-0.6b` (bf16 safetensors) | huggingface.co/patentmechanics/anticipate-0.6b | Apache 2.0 |
| Benchmark `prior-art-bench-v1`: queries, pool v2 full text, description chunks, qrels, splits, families, run files, reference vectors, canary lists | huggingface.co/datasets/patentmechanics/prior-art-bench-v1 (pinned to the revision tagged `v1.0`; Zenodo DOI added at publication) | Text: USPTO publications; the rest CC BY 4.0 |
| Code: `evaluate.py`, `fusion.py`, `verify_sample.py`, `baselines/`, `build/`, `blog/`, `train/`, `audits/` | this repo, tag `v1.0` | Apache 2.0 |
| Blog-post files: chat-app scored lists, our lists, grader output (`blog-v1.tar`) | GitHub release `v1.0` of this repo | CC BY 4.0 (patent text: USPTO publications) |
| Hard-negatives audit: sampled pairs and the judge's grades (`audits-negatives-v1.tar`) | GitHub release `v1.0` of this repo | CC BY 4.0 (patent text: USPTO publications) |
| Base model `microsoft/harrier-oss-v1-0.6b` @ `f9b9dc8d367d443f2479d27aa5d8d2850c0774ee` | Hugging Face | MIT |
| Raw USPTO data (only for level 5) | USPTO bulk data, Open Data Portal, CPC Master Classification Files, PatentsView | Public domain |

## Levels at a glance

| Level | What it checks | Hardware | Time |
|---|---|---|---|
| 0. Re-score our run files | That our metrics, splits and intervals are computed as stated | Laptop | minutes |
| 1. Search from the shipped vectors | That the run files follow from the vectors; the search-time system (passages, CPC, BM25) | Laptop (16 GB RAM recommended) | under an hour |
| 2. Check the vectors against the model | That the shipped vectors come from the published weights | Laptop (16 GB RAM recommended; small samples run on 8 GB) | under an hour |
| 3. Re-encode everything | That the published weights give the published numbers end to end, on your GPU | One 80-96 GB GPU | ~2-3 GPU-hours per model |
| 4. Re-train the model | That the recipe gives the published weights (within seed noise) | 1-8 GPUs with 96 GB | hours per seed |
| 5. Rebuild the benchmark from USPTO data | That queries, labels, splits and the pool follow from public USPTO data | CPU machine, ~64 GB RAM, ~1 TB disk | ~1-2 days |
| Blog post | Our full system's numbers, the all-US-families numbers, the chat-app comparison, repeat test and uncited-results grading | Laptop (re-grading: a Claude account) | minutes |

Levels 0-2 are the ones we expect most readers to run. Each level is independent: you can start at any of them.

Install once:

```bash
git clone https://github.com/patentmechanics/prior-art-bench && cd prior-art-bench && git checkout v1.0
pip install -r requirements.txt          # CPU levels 0-2
pip install -r requirements-gpu.txt      # levels 3-4 (pinned: vLLM 0.33.0, torch, transformers; exact versions in the file)
```

Every command reads the dataset from the Hugging Face Hub unless you pass `--data /path/to/a/local/copy`; files are
downloaded into the Hugging Face cache once. Below, `HF=hf://datasets/patentmechanics/prior-art-bench-v1`.

---

## Level 0: re-score the published run files (laptop, minutes)

The dataset ships `runs/<system>.<split>.parquet` for every system in the results tables: top 1,000 families per
query, with scores.

| System | Run file name |
|---|---|
| BM25 | `bm25` |
| EmbeddingGemma 2 | `embeddinggemma-2` |
| PaECTER | `paecter` |
| Qwen3-Embedding-0.6B | `qwen3-embedding-0.6b` |
| Octen-Embedding-0.6B | `octen-embedding-0.6b` |
| harrier-oss-v1-0.6b | `harrier-0.6b` |
| Qwen3-Embedding-0.6B fine-tuned (3-seed average) | `qwen3-embedding-0.6b-finetuned-soup` |
| harrier-oss-v1-0.6b fine-tuned, base recipe (3-seed average) | `harrier-0.6b-finetuned-soup` |
| Anticipate-0.6B | `anticipate-0.6b` |
| + passages | `anticipate-0.6b-passages` |
| + passages + CPC strict | `anticipate-0.6b-passages-cpc` |
| + passages + CPC strict + BM25 (recommended system) | `anticipate-0.6b-passages-cpc-bm25` |
| PatentSBERTa † | `patentsberta` |
| patembed-large † (paper text only) | `patembed-large` |

The launch blog's extra run files (our full system, and searches of all 9.44M US families) are listed under
[The blog post](#the-blog-post).

```bash
HF=hf://datasets/patentmechanics/prior-art-bench-v1
python evaluate.py metrics --run $HF/runs/anticipate-0.6b.test.parquet --part test
python evaluate.py compare --a $HF/runs/harrier-0.6b.test.parquet --b $HF/runs/anticipate-0.6b.test.parquet --part test
```

`metrics` applies the scoring rules (family level; own family removed; candidates and gold restricted to families
published before the query's priority date; only queries with at least one remaining gold family are scored) and
prints R@10, R@100, R@1000, nDCG@10 and MRR@10, overall and by source. `compare` prints paired bootstrap 95% intervals
(2,000 resamples over queries, seed 0; one generator per metric, drawn for all queries first and then per source),
B minus A, for R@10, R@100 and nDCG@10 (`--metrics` to change).

**Expected (test split, 4,837 scored queries: 2,368 office actions, 2,469 PTAB):**

| System | R@10 | R@100 | R@1000 | nDCG@10 |
|---|---|---|---|---|
| PatentSBERTa † | 0.2393 | 0.4882 | 0.7402 | 0.1818 |
| BM25 | 0.3262 | 0.5844 | 0.8197 | 0.2488 |
| EmbeddingGemma 2 | 0.3991 | 0.6916 | 0.8893 | 0.2946 |
| PaECTER | 0.3995 | 0.6999 | 0.8943 | 0.3031 |
| Qwen3-Embedding-0.6B | 0.4083 | 0.6902 | 0.8821 | 0.3098 |
| Octen-Embedding-0.6B | 0.4226 | 0.6999 | 0.8866 | 0.3243 |
| harrier-oss-v1-0.6b | 0.4506 | 0.7366 | 0.9056 | 0.3460 |
| Qwen3-Embedding-0.6B fine-tuned (3-seed average) | 0.4988 | 0.7714 | 0.9220 | 0.3863 |
| harrier-oss-v1-0.6b fine-tuned, base recipe (3-seed average) | 0.5132 | 0.7824 | 0.9275 | 0.3987 |
| **Anticipate-0.6B** | **0.5380** | **0.7959** | **0.9318** | **0.4155** |
| + passages | 0.5472 | 0.8141 | 0.9395 | 0.4209 |
| + passages + CPC strict | 0.5523 | 0.8196 | 0.9395 | 0.4254 |
| + passages + CPC strict + BM25 (recommended system) | 0.5622 | 0.8205 | 0.9385 | 0.4364 |
| patembed-large † | 0.4423 | 0.7464 | 0.9172 | 0.3334 |

† Added after the test half was scored, with nothing tuned (not in the pre-registered list). patembed-large appears
only in the paper's text, not in its table or in the blog post.

Paired R@10 differences for Anticipate-0.6B: vs harrier-oss-v1-0.6b fine-tuned +2.5 [+1.8, +3.2]; vs harrier-oss-v1-0.6b
+8.7 [+7.9, +9.6]; vs Qwen3-Embedding-0.6B +13.0 [+12.0, +14.0]; vs Qwen3 fine-tuned +3.9 [+3.0, +4.8]; vs BM25
+21.2 [+20.0, +22.3]. Recommended system vs the model alone +2.4 [+1.7, +3.2].

Queries whose first result is a cited family (paper, Section 5.2; not pre-registered): MRR@10 is 1 exactly when the
first result is relevant, so count those queries from the per-query scores:

```bash
python -c "
import evaluate as E
for r in ('harrier-0.6b', 'anticipate-0.6b'):
    pq = E.per_query('$HF/runs/' + r + '.test.parquet', '$HF', 'test')
    n = sum(m['mrr@10'] == 1 for _, m in pq.values())
    print(r, n, 'of', len(pq), f'= {n / len(pq):.1%}')
"
# harrier-0.6b 1385 of 4837 = 28.6%
# anticipate-0.6b 1672 of 4837 = 34.6%
```

Discussion, Table 7 (each step paired against the one before it; R@1000 for the model over its stock base):

```bash
R=$HF/runs
python evaluate.py compare --a $R/bm25.test.parquet --b $R/harrier-0.6b.test.parquet --part test --metrics recall@10
python evaluate.py compare --a $R/harrier-0.6b.test.parquet --b $R/harrier-0.6b-finetuned-soup.test.parquet --part test --metrics recall@10
python evaluate.py compare --a $R/harrier-0.6b.test.parquet --b $R/anticipate-0.6b-passages-cpc-bm25.test.parquet --part test --metrics recall@10
python evaluate.py compare --a $R/harrier-0.6b.test.parquet --b $R/anticipate-0.6b.test.parquet --part test --metrics recall@1000
# recall@10   all  0.3262 -> 0.4506  diff +12.4 [+11.3, +13.6]   (stock harrier-oss-v1-0.6b over BM25)
# recall@10   all  0.4506 -> 0.5132  diff +6.3 [+5.2, +7.3]      (base-recipe fine-tune over stock)
# recall@10   all  0.4506 -> 0.5622  diff +11.2 [+10.3, +12.1]   (full search-time system over stock)
# recall@1000 all  0.9056 -> 0.9318  diff +2.6 [+2.2, +3.0]      (Anticipate-0.6B over stock)
```
The other two steps of Table 7 (release recipe over the base recipe, +2.5; search-time stages over the model, +2.4)
are the paired differences listed above.

Level 0 must match to the last digit: it is deterministic arithmetic on fixed files. The same command with
`--part dev` reproduces the development tables in the paper.

## Level 1: search from the shipped vectors (laptop, no GPU)

The dataset ships the reference model's vectors (1,024 dimensions, float16) for every pool document, every
description chunk and every dev and test query, under `vectors/anticipate-0.6b/`. Exact search over them
recreates the run file, and `fusion.py` adds the three search-time stages with the weights fixed on dev
(`fusion-v1.json`: passages = max of document and best chunk score, chunk search depth 4,000; CPC strict weight 0.75;
BM25 z-score fusion weight 0.2).

```bash
python evaluate.py search --vectors $HF/vectors/anticipate-0.6b --part test --out runs/model.test.parquet
python evaluate.py metrics --run runs/model.test.parquet --part test                      # expect R@10 0.5389 (published run 0.5380; see below)
python evaluate.py search --vectors $HF/vectors/anticipate-0.6b --collection chunks --part test \
    --out runs/chunks.test.parquet                                                        # top 4,000 chunks per query
python baselines/bm25.py --part test --out runs/bm25.test.parquet                         # expect R@10 0.3262 (16-24 GB RAM)
python fusion.py --dense runs/model.test.parquet --chunks runs/chunks.test.parquet \
    --bm25 runs/bm25.test.parquet --config fusion-v1.json --out runs/system.test.parquet --save-stages
python evaluate.py metrics --run runs/system.test.parquet --part test                     # expect R@10 0.5622
```

`--save-stages` also writes `runs/system.test.passages.parquet` and `runs/system.test.passages_cpc.parquet` (the
"+ passages" and "+ passages + CPC strict" rows). Instead of re-running BM25 you can pass the shipped
`$HF/runs/bm25.test.parquet` to `--bm25`. `search` runs in float32 on the CPU in blocks (a few GB of RAM); the
vectors are memory-mapped.

Expected: the run-file numbers of level 0 within ±0.001. The document and chunk vectors are the ones the published
runs used; the query vectors were re-encoded from the released weights on a different GPU, and bf16 arithmetic moves a
few ranks. Our own run of exactly these commands gave test R@10 0.5389 / R@100 0.7966 / nDCG@10 0.4166 (published
0.5380 / 0.7959 / 0.4155) and dev R@10 0.5286 (published 0.5289); the top 10 matched the published run for 82% of
queries and overlapped 99.6% overall.

## Level 2: check the vectors against the published weights (laptop)

```bash
python verify_sample.py --model patentmechanics/anticipate-0.6b --n-queries 100 --n-docs 300 --seed 0
```

It draws a random sample of queries and documents, encodes them on your machine with the published weights
(sentence-transformers; MPS on Apple Silicon, CPU otherwise), and compares each vector with the shipped one.
Pass = every cosine >= 0.99 (typical 0.999+; differences come from bf16 arithmetic on different hardware). Choose
your own `--seed` so the sample is not ours. `--docs-from sample` draws the documents from the 10,000-family sample
collection instead of the whole pool (a much smaller download). Together with level 1, this shows the published
numbers follow from the published weights without anyone needing a GPU.
Tested on a small sample on an 8 GB M1 (10 queries, 20 documents: pass, minimum cosine 0.998, about 5 minutes); we
recommend 16 GB for the full check.

## Level 3: re-encode from scratch (one GPU)

For the release model, any baseline in the table, or your own model:

```bash
python evaluate.py encode --model patentmechanics/anticipate-0.6b --config configs/anticipate-0.6b.json \
    --collection corpus-pool-v2 --chunks --backend vllm --out emb/pm/
python evaluate.py search --vectors emb/pm/ --out runs/pm.test.parquet --part test
python evaluate.py metrics --run runs/pm.test.parquet --part test
```

(`--chunks` also encodes the description chunks, for `search --collection chunks` and the passage stage.)

Settings we used (`configs/<system>.json`, one file per stock model of the table; the fine-tuned baselines were not
released):

| | |
|---|---|
| Engine | vLLM 0.33.0 pooling runner, bf16 (`--backend vllm`; the default `--backend st` uses sentence-transformers) |
| Pooling | last token, left padding, L2-normalised (harrier-oss-v1-0.6b, Qwen3, Octen, ours); mean (PaECTER); model card setting (EmbeddingGemma 2) |
| Lengths | query 4,096, document 8,192 tokens (PaECTER: its 512-token limit) |
| Query prefix | per source: office-action queries use the trained `query` instruction; PTAB queries use the `invalidity_query` instruction (both in the model card); PaECTER none; EmbeddingGemma 2 its card prefix |
| Document prefix | none |
| Query text | title, abstract, claims (`queries/queries.parquet` column `text`) |
| Document text | title + abstract + claims + description, cut at the model's length (`corpus-pool-v2` column `text`) |

Time: the pool is about 2 billion tokens at 8,192 tokens per document; one 96 GB GPU does it in about 2-3 hours.
`--shard i --nshards n` splits it over machines; `--collection sample` (1,000 queries,
10,000 documents) is a ten-minute smoke test.

Expected: within about ±0.001 of level 0 on any metric. Different GPUs, drivers and kernels change bf16 rounding;
in our own runs this never moved a reported conclusion or a system's rank.

## Level 4: re-train the model

The recipe (model card, "Training procedure"; paper section 4): full fine-tuning of harrier-oss-v1-0.6b @ `f9b9dc8d`, InfoNCE,
temperature 0.02, 64 queries per step with in-batch negatives and 3 mined hard negatives per query, learning rate
1e-5 with 50 warm-up steps on a cosine schedule over 900 steps, stopped at step 75; query 4,096 / document 8,192
tokens; seeds 0, 1, 2; the release weights are the uniform average of the three step-75 checkpoints.

```bash
bash train/run_anticipate_0.6b.sh <training pack> out/ [number of GPUs]
# = train/train.py for seeds 0, 1, 2, then
#   python train/soup.py --ckpts out/seed0/step-75 out/seed1/step-75 out/seed2/step-75 --out out/soup
#   python train/export.py ... and python train/make_model_repo.py ... -> out/model
python evaluate.py encode --model out/model --config configs/anticipate-0.6b.json --out emb/mine/   # then level 3
```

The benchmark ships the training positives as IDs and grades (`train-qrels`: Y-cleaned X/Y citations), but not our
mined negatives or the training pack. The training queries' text and their known-relevant families come from the
level 5 build (`build/labels_us.py`). Mine your own negatives with `train/retrieve.py` (encode the mining pool, search
the training queries) and `train/mine.py` (ranks 30-100 from a retriever, excluding known citations, families
co-cited with a positive, and anything scoring above 0.95 x the best positive; we mined with a base-recipe fine-tune
of harrier-oss-v1-0.6b, three seeds averaged), then pack with `train/pack.py` (`train/README.md` lists every setting). Your
pack will differ from ours, so expect a model close to, but not identical with, the release: our seeds gave dev R@10
0.514 / 0.515 / 0.519 and 0.529 for the average.

Compute: RTX PRO 6000 Blackwell 96 GB GPUs.

## Level 5: rebuild the benchmark from USPTO data (CPU)

`build/` turns public USPTO data into the benchmark: family grouping, office-action and PTAB labels (v2, including WO
citations resolved through PatentsView PCT data), salted-hash dev/test split, leakage audit, near-duplicate guard, pool
v2 selection, description chunks, CPC tables, and the dataset layout itself (`build/build_dataset.py`). Inputs and
order are in `build/README.md`, which also lists the steps whose scripts are not in this repository yet. The
dataset's `provenance/manifest.json` lists every released file with its sha256 and the build statistics, so you can
check each stage's counts against ours. A (free) USPTO Open Data Portal API key is needed for the office-action
datasets.

Expected: identical query, qrels, split and pool contents (byte-identical parquet is not guaranteed). Office-action
data published after our cut-off (2026-10-30) must be excluded: `build/oa_fetch.py` takes the cut-off as
`PAB_OA_CUTOFF=2026-10-30`.

---

## The blog post

The launch post ("We are opening our patent search model. Here is how it compares.") reports four kinds of number.
They are not equally reproducible, and this section says which is which.

| Blog number | What backs it | How to check it |
|---|---|---|
| Open-model table and chart (PatentSBERTa 0.239 ... Anticipate-0.6B 0.538, + passages + CPC + BM25 0.562; +8.7 points [7.9, 9.6] over harrier-oss-v1-0.6b) | Shipped run files | Level 0 (exact) or Levels 1-3 |
| Our full system on the benchmark, R@10 0.618 | Our search system plus a re-ranking model that we do **not** release; its ranked lists ship as `runs/full-system.test.parquet` | Re-score exactly (below). It cannot be regenerated without the re-ranking model |
| All 9.44 million US families: keyword search 19% (dev, 4,561 questions); Anticipate-0.6B alone 31% and our full system 42% (1,212 test questions); top 100: 62% and 69% | Ranked lists over every US family, not only the 250,367-family collection: `runs/*.all-us.*.parquet` | Re-score exactly (below). Regenerating the Anticipate-0.6B and keyword-search lists needs the full corpus (Level 5, then Level 3 / `baselines/bm25.py`); our full system's cannot be regenerated |
| Chat apps on 30 fresh office actions: Claude app 13, ChatGPT app 16, our full system 17 of 30; ChatGPT named the application in 27 of 30; repeat test 4.6 and 5.0 of 10 (our full system 10 of 10) | The 10 patent numbers we read from each app answer, our full system's lists, and the scoring code | Re-score exactly from the release asset below. The apps cannot be re-run identically: they change, and they answer differently each time (that is the repeat test) |
| Uncited results graded by Claude Opus 5.5: 36% vs 36% for the examiner's own references and 6% for our ranks 151-200; 94% run-to-run agreement | The grader's instructions, document texts, every grade with its quotes | Re-summarise exactly from the release asset; re-grade with `blog/uncited_relevance.py` (a rerun should land within a point or two, not on the same grades) |

**Release asset `blog-v1.tar`** (GitHub release `v1.0`; unpacks to `blog-v1/`):

| Path | Contents |
|---|---|
| `queries.json` | The 40 published applications whose first office action was mailed on 10 November 2026 (30 scored, `set` = `main 30`, plus 10 spares): application and publication numbers, priority date, the claims text pasted into the apps, and the examiner's references with their X/Y categories and families |
| `chat-apps/prompts/` | The exact prompt pasted for each of the 30 questions |
| `chat-apps/scored/parsed.json` | For each app and question: the 10 patent numbers read from the answer, in order, as written and as lookup keys; whether the answer had the requested final JSON line (`how`: `json` or `regex`); whether it named the application itself. The answers themselves are not published |
| `chat-apps/scored/repeat-parsed.json` | The same for the repeat test: first and second attempt (a new chat) on 5 questions |
| `chat-apps/map.json` | Every US number the apps listed -> [patent family, first publication date] in our family table |
| `ours/full_system_top200.json` | Our full system's top 200 families for each of the 40 queries |
| `ours/full_system_repeat.json` | Our full system's top 10 on the 5 repeat questions, with the re-ranking model run twice |
| `uncited/pairs.jsonl`, `uncited/texts.jsonl` | The 431 blind (query, document) pairs and the document texts the grader read |
| `uncited/run1/`, `uncited/run2/` | Every grade: chosen claim, element-by-element mapping with quotes, category, reason |

```bash
# our full system and the all-US-families lists (dataset run files; expected values in the comments)
python evaluate.py metrics --run $HF/runs/full-system.test.parquet --part test --hit                  # R@10 0.6182, hit@10 0.7943
python evaluate.py metrics --run $HF/runs/anticipate-0.6b.all-us.test.parquet --part test        # all 4,837: R@10 0.1737, R@100 0.4302, R@1000 0.6955 (paper, Section 5.1)
python evaluate.py metrics --run $HF/runs/anticipate-0.6b.all-us.test.parquet --part test --hit \
    --queries runs/all-us.test-subset.query_ids.txt                                                    # hit@10 0.3135, hit@100 0.6155
python evaluate.py metrics --run $HF/runs/full-system.all-us.test.parquet --part test --hit            # hit@10 0.4249, hit@100 0.6914
python evaluate.py metrics --run $HF/runs/bm25.all-us.dev.parquet --part dev --hit                    # hit@10 0.1934
# (--queries takes a local file: download runs/all-us.test-subset.query_ids.txt from the dataset first)

B=blog-v1; X=$B/chat-apps; W=work; mkdir -p $W; cp $X/scored/*.json $W/
# chat apps vs our full system, main 30 (expect hit@10 0.433 / 0.533 / 0.567, i.e. 13, 16, 17 of 30)
python blog/score_chat_apps.py report $B/queries.json $W/parsed.json $X/map.json full-system=$B/ours/full_system_top200.json
python blog/chat_apps_extra.py named $B/queries.json $W/parsed.json                               # ChatGPT 27 of 30, Claude 6 of 30
python blog/chat_apps_extra.py paired $B/queries.json $W/parsed.per_query_metrics.json chatgpt-app full-system   # +1 (95% range -3 to +5)

# repeat test: first and second attempts side by side (expect 4.6 and 5.0 of 10; hit@10 changed on 2 + 1 of the 10 repeats)
python blog/chat_apps_extra.py repeat $B/queries.json $W/repeat-parsed.json $X/map.json
python blog/chat_apps_extra.py repeat-ours $B/queries.json $B/ours/full_system_repeat.json         # 10 of 10 on all 5

# uncited results (expect 35.9%, 95% range 26.9%-45.7%; examiner references 36.2% in run 1; ranks 151-200 5.8%; 93.9% agreement)
python blog/uncited_relevance.py summary $B/uncited/pairs.jsonl $B/queries.json $B/uncited/texts.jsonl $B/uncited
# re-grade (needs the Claude Code CLI and an account; about 18K input tokens per pair):
# python blog/uncited_relevance.py judge $B/uncited/pairs.jsonl $B/queries.json $B/uncited/texts.jsonl mygrades --run 1
```

How the numbers were read: a listed number counts if it maps to a family that the examiner relied on and that was first
published before the application's priority date; the application's own family never counts. When an answer has no
final JSON line, its numbers are read in order of first mention, dropping repeats and the application itself
(`blog/score_chat_apps.py parse` does this; it is the code that produced `parsed.json` from the answers). Protocol: one
new chat per question, pasted by hand on 19 November 2026; Claude app with Opus 5.5 (effort medium) and ChatGPT app with
GPT-6.2 Sol (medium); web search on, deep research and memory off (memory was on, and empty, for ChatGPT's first
question only). Data quality: two Claude answers were repaired after a damaged paste (a final JSON line cut off, completed
from the answer's own list); 19 ChatGPT answers had no JSON line and were read from their table.

How our lists were made: `full-system.test.parquet` is the launch system (the `anticipate-0.6b-passages-cpc-bm25` run)
with its top 50 re-ordered by z(launch score) + 1.5 z(re-ranker score); ranks 51-1,000 are the launch system's. The
all-US-families runs search all 9,441,930 US families; family ids that are not in `families/` are US families outside
the benchmark collection, which no query has as gold (so every score there is a lower bound: those families are
unjudged). `full-system.all-us.test.parquet` is a learned ranking stage with its top 100 re-ordered by the same blend,
for the 1,212 test questions the re-ranker was run on (`runs/all-us.test-subset.query_ids.txt`). In both full-system
files, scores below the re-ordered top k only keep the order; they are not on the blend's scale. hit@k (share of
questions with at least one relied-on family in the top k) was added after the paper's pre-registration, for the blog.

**Not published:** the chat apps' answer texts (only the numbers read from them, above).

---

## What can differ, and what cannot

- **Must match exactly:** level 0; query, label and split counts; the set of scored queries.
- **May differ by about ±0.001 per metric:** levels 1-3 (float16 / bf16 arithmetic, GPU kernels, ties).
  Paired intervals may move by the same amount at their ends.
- **May differ by seed noise (about ±0.005 dev R@10):** level 4.
- If you get something outside these ranges, open an issue with your command, environment (`pip freeze`, GPU, driver)
  and output. We will publish confirmed discrepancies and corrections in the dataset's changelog.

## Things we could not make reproducible

- **The base model's training data** is not public, so nobody, including us, can rule out that harrier-oss-v1-0.6b saw some
  benchmark patents during pre-training. Our fine-tuning data excludes every evaluation family.
- **Our full system** (the search system plus a re-ranking model) is not released, so its numbers can be re-scored from
  its ranked lists but not regenerated.
- **The chat apps** change over time and answer differently each time. We publish the numbers read from their answers,
  not the answer texts.
- **The LLM-judge audit of hard negatives** (an aid to a training decision; no label came from it) used GLM 5 on
  Amazon Bedrock. `audits-negatives-v1.tar` ships the sampled pairs (with the texts the judge read) and a grade for each
  pair; `python audits/negatives_audit_summary.py audits-negatives-v1/judged` prints the rates. Those grades come from a
  re-run of the original prompt on the original samples (20 November 2026): the first runs' grades were not kept.
  Re-grading (`audits/judge_negatives.py`) needs Bedrock access and can move counts by a
  few pairs even at temperature 0.
- **The 50,000-family development collection** we used for fast learning-rate checks is a subset of pool v2 (all
  dev-relevant families plus hashed-random distractors). Its build script ships in `build/small_pool.py`; it plays no
  part in any headline number.
