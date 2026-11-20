# prior-art-bench

Evaluation code for **Patent Mechanics Prior-Art Bench v1 (US)** and the **Patent Mechanics Anticipate-0.6B** model: prior-art retrieval over
US patent full text, scored against USPTO office-action citations and PTAB trial references.

- Benchmark: https://huggingface.co/datasets/patentmechanics/prior-art-bench-v1
- Model: [Anticipate-0.6B](https://huggingface.co/patentmechanics/anticipate-0.6b)
- Paper: https://doi.org/10.5281/zenodo.24394100

## Results

Test split, scored once, 250,367-family collection, 4,837 queries (2,368 office actions, 2,469 PTAB). Systems, metrics
and comparisons were fixed before any test score; all choices and weights were made on dev.

| System | R@10 | R@100 | nDCG@10 | Office actions R@10 | PTAB R@10 |
|---|---|---|---|---|---|
| PatentSBERTa † | 0.239 | 0.488 | 0.182 | 0.211 | 0.266 |
| BM25 | 0.326 | 0.584 | 0.249 | 0.306 | 0.346 |
| EmbeddingGemma 2 | 0.399 | 0.692 | 0.295 | 0.352 | 0.445 |
| PaECTER | 0.400 | 0.700 | 0.303 | 0.360 | 0.437 |
| Qwen3-Embedding-0.6B | 0.408 | 0.690 | 0.310 | 0.369 | 0.446 |
| Octen-Embedding-0.6B | 0.423 | 0.700 | 0.324 | 0.386 | 0.458 |
| harrier-oss-v1-0.6b | 0.451 | 0.737 | 0.346 | 0.410 | 0.490 |
| Qwen3-Embedding-0.6B fine-tuned (3-seed average) | 0.499 | 0.771 | 0.386 | 0.465 | 0.531 |
| harrier-oss-v1-0.6b fine-tuned (base recipe, 3-seed average) | 0.513 | 0.782 | 0.399 | 0.478 | 0.547 |
| Anticipate-0.6B | 0.538 | 0.796 | 0.416 | 0.488 | 0.586 |
| Anticipate-0.6B + passages + CPC strict + BM25 | 0.562 | 0.821 | 0.436 | 0.521 | 0.602 |

† Added after the test half was scored, with nothing tuned (not in the pre-registered list).

Paired R@10 differences (95% bootstrap CI): Anticipate-0.6B vs harrier-oss-v1-0.6b fine-tuned +2.5 (+1.8..+3.2), vs
harrier-oss-v1-0.6b +8.7 (+7.9..+9.6), vs BM25 +21.2 (+20.0..+22.3); the three fusion stages together +2.4 (+1.7..+3.2).

**Reproducing our numbers:** [REPRODUCING.md](REPRODUCING.md) covers every published number (results tables, paper,
blog post) at five levels, from re-scoring our run files on a laptop to re-training the model and rebuilding the
benchmark from USPTO data, with expected values and tolerances. It also says which blog numbers can be re-scored
but not regenerated (those of our full system, which uses a re-ranking model we do not release; its ranked lists ship).

## Quick start on a laptop (no GPU)

The benchmark ships the model's vectors for every document, description chunk and query of the collection. The
model itself is about 1.2 GB in bf16 and runs on any Apple Silicon Mac. Level 2 (checking the shipped vectors against the weights) was
tested on a small sample on an 8 GB M1; we recommend 16 GB for the full check.
Every command reads the dataset from the Hugging Face Hub by default (`--data`; files are cached after the first
download) or from a local copy (`--data /path/to/prior-art-bench-v1`).

```bash
pip install -r requirements.txt

# 1. check the shipped vectors against the model on your machine
#    (re-encodes ~100 random queries and ~300 documents; passes at cosine >= 0.99)
python verify_sample.py --model patentmechanics/anticipate-0.6b --seed 1234

# 2. search and score from the shipped vectors
python evaluate.py search --part test --out runs/model.test.parquet
python evaluate.py metrics --run runs/model.test.parquet --part test

# 3. the recommended system: passages, CPC strict, BM25 (fixed dev weights in fusion-v1.json)
python evaluate.py search --collection chunks --part test --out runs/chunks.test.parquet     # top 4,000 chunks
python fusion.py --dense runs/model.test.parquet --chunks runs/chunks.test.parquet \
    --bm25 hf://datasets/patentmechanics/prior-art-bench-v1/runs/bm25.test.parquet \
    --config fusion-v1.json --out runs/system.test.parquet
python evaluate.py metrics --run runs/system.test.parquet --part test
```

Or re-score the published run files directly:

```bash
python evaluate.py metrics --run hf://datasets/patentmechanics/prior-art-bench-v1/runs/anticipate-0.6b.test.parquet --part test
python evaluate.py compare --a hf://datasets/patentmechanics/prior-art-bench-v1/runs/harrier-0.6b.test.parquet \
    --b hf://datasets/patentmechanics/prior-art-bench-v1/runs/anticipate-0.6b.test.parquet --part test
```

`metrics` prints R@10, R@100, R@1000, nDCG@10 and MRR@10, overall and by source. `compare` prints paired bootstrap
95% intervals (2,000 resamples over queries, seed 0).

## Evaluate your own model (GPU)

```bash
# 1. embed queries and documents (sentence-transformers, or --backend vllm)
python evaluate.py encode --model <your-model> --config configs/<your-model>.json --out emb/

# 2. exact search with the benchmark's filters, top 1000 per query
python evaluate.py search --vectors emb/ --out runs/my-model.dev.parquet --part dev

# 3. score
python evaluate.py metrics --run runs/my-model.dev.parquet --part dev
```

A config sets the query prefix (`"instruction"` = our per-source instruction, `""` = none, or a literal prefix), the
document prefix, the query and document lengths and the pooling used by the vLLM backend; `configs/` has one for
every stock model in the table. Encoding the 250,367-family collection is about 2 billion tokens at 8,192 tokens per
document. Use `--shard i --nshards n` to split it over machines. Start with `--collection sample` (1,000 queries,
10,000 documents).

Use `--part test` once, for the number you report.

## Scoring rules (what `search` and `metrics` enforce)

1. Unit = patent family, on both sides.
2. The query's own family is never a candidate.
3. A family is a candidate only if its earliest publication is before the query's priority date; later families are also dropped from the query's gold set.
4. Only queries with at least one remaining gold family are scored.
5. nDCG gain = grade (3 or 2).

## Baselines and fusion

`baselines/bm25.py` (k1 1.2, b 0.75, English stopwords, stemming, full text). `fusion.py` implements the three stages
of the recommended system: passages (family score = max of whole-document and best description-chunk score), CPC
strict (boost by shared subclass / main group / full symbol, using only the CPC printed on the query and the CPC of
family members published before its priority date), and z-score BM25 fusion. `fusion-v1.json` holds the weights
chosen on dev half A (CPC strict 0.75, BM25 0.2). Encoding configs for the other systems in the results table are in
`configs/`.

## Rebuild the benchmark from USPTO data

`build/` holds the scripts that produced the benchmark from USPTO bulk files, Open Data Portal datasets, the CPC
Master Classification Files and PatentsView data: family grouping, labels (v2) and splits, leakage audit,
near-duplicate guard, pool selection, description chunks, and the provenance check. Order, inputs and the parts not
included yet: `build/README.md`. The dataset's `provenance/manifest.json` lists every released file with its sha256.
A USPTO Open Data Portal API key is needed for the office-action datasets.

## Training

`train/` holds the training code and the release recipe (`train/run_anticipate_0.6b.sh`). Mined hard negatives are
not shipped: mine your own (`train/README.md`).

## Reporting results

For a result to be comparable with ours:

- no use of the query patent's prosecution history, citations or PTAB filings, or of anything published after its
  priority date;
- training data must exclude the published evaluation families (filter with the hashed canary lists in the dataset
  repo);
- tune on dev only and score test once.

Score your run file with `evaluate.py` and publish it with your results so others can re-score it. To share a result
with us, open an issue with a link to the run file and a short description of the system.

## Tests

`python -m pytest tests/` (or `python tests/test_metrics.py`): synthetic checks of the scoring rules, the search
filters and the fusion stages; no data or model needed.

## Licence

Code: Apache 2.0. Benchmark: patent text from USPTO publications, no rights claimed by us; qrels, splits, family table, run files and
vectors CC BY 4.0. Model: Apache 2.0 (base model MIT; see the model's NOTICE).

Source: United States Patent and Trademark Office, www.uspto.gov. About 0.6% of documents carry a notice under 37 CFR
1.71(d)-(e) reserving copyright or mask-work rights in part of the disclosure; we include their text, with the notice,
for research use. Rights holders can request removal at contact@patentmechanics.com.

## Citation

```bibtex
@misc{patentmechanics2026priorart,
  title     = {Learning from Examiners: An Open Embedding Model and a Benchmark for US Patent Prior-Art Retrieval},
  author    = {Venkatraj, Rahul},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.24394100},
  url       = {https://doi.org/10.5281/zenodo.24394100}
}
```
