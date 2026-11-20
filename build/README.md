# Building the benchmark from USPTO data

These are **the scripts as we ran them**, not a polished pipeline. Each one is a stage we ran by hand, in the order
below, on one CPU machine (about 64 GB of RAM, about 1 TB of disk). Internal paths were replaced by environment
variables; the logic is unchanged. Expect to read the docstring at the top of each script before running it.

Two working directories, as in our runs:

| Variable | Holds |
|---|---|
| `PAB_SRC` | inputs and shared intermediate files: parsed USPTO full text (`us/`), per-document metadata (`meta/us.parquet`), office-action tables (`uspto_oa/`), PTAB input (`inputs/`), query texts (`queries/`, `queries_oa/`) |
| `PAB_WORK` | everything the US benchmark is built from: families, representative documents, corpus, queries, labels, splits, pools, CPC tables, packs |

## Inputs (all public)

| Input | Where it goes | Used by |
|---|---|---|
| USPTO bulk full text: applications (APPXML) and grants (PTGRXML, PTGRSGM, PTGRAPS), parsed to one parquet per bulk file | `$PAB_SRC/us/<product>/<year>/*.parquet` (also readable as `PAB_US_TEXT`) | `family_us.py`, `corpus.py`, `queries_us.py` |
| PatentsView tables `g_application`, `g_us_rel_doc`, `pg_us_rel_doc`, `g_pct_data`, `pg_pct_data`, `g_foreign_priority`, `pg_foreign_priority` (as parquet) | `$PAB_WORK/pv/` | `family_us.py`, `labels_v2_us.py` |
| USPTO Open Data Portal office-action datasets (`oa_rejections`, `oa_citations`, `enriched_cited_reference_metadata`); free API key in `USPTO_ODP_API_KEY` | `$PAB_SRC/uspto_oa/` | `oa_fetch.py`, `oa_build.py` |
| PTAB trials (instituted trials, challenged patent, references), one JSON object per line | `$PAB_SRC/inputs/ptab_eval_full.jsonl` | `queries_us.py ptab`, `labels_us.py`, `labels_v2_us.py` |
| CPC Master Classification Files for grants and pre-grant publications (ODP products CPCMCPT / CPCMCAPP, `US_*_CPC_MCF_XML_*.zip`) | `$PAB_WORK/cpc/` | `cpc_us.py` |
| Grant front-page citations with `cited_by` (chart 2 only) | `$PAB_WORK/tmp/cited_not_applied/cites/20*.parquet` | `cited_not_applied_labels.py` |

**Not included in this repository** (open items, see the list at the end): the parser that turns the USPTO bulk XML /
SGML / APS files into the parsed full-text parquet (37-column schema: `doc_id`, `kind`, `doc_class`, `pub_date`,
`app_number`, `filing_date`, `title`, `abstract`, `claims`, `paragraphs`, `cpc`, ...), the script that wrote the PTAB
JSONL from the PTAB API, the extraction of front-page citations for chart 2, and the selection of the 250,000-family
pool v1 (`tac_subset_250k/`, `tacd/subset-us-*.parquet`: every family relevant to an evaluation query plus random
distractors).

## Order

| # | Command | Output |
|---|---|---|
| 1 | `PAB_WORK=$PAB_SRC python build/corpus.py meta --src us` | `$PAB_SRC/meta/us.parquet` (small columns of every US document) |
| 2 | `python build/oa_fetch.py run` (with `PAB_OA_CUTOFF`), then `python build/oa_build.py` | `$PAB_SRC/uspto_oa/v1/` office-action tables (`oa_actions`, `oa_citations`, ...) |
| 3 | `python build/family_us.py` | `$PAB_WORK/family/` US families from continuity, PCT and priority links |
| 4 | `python build/select_us.py` | `family_map/{families,docs}.parquet`, `reps/reps_us.parquet` (one representative per family) |
| 5 | `ln -s $PAB_SRC/us $PAB_WORK/us`; `python build/corpus.py build --src us`, then `python build/corpus.py tac` (needs a Qwen3 `tokenizer.json` in `PAB_TOKENIZER` for the token-length sort) | `corpus/us/`, `passages/us/` (description paragraphs), `tac/shard-*.parquet` |
| 6 | `python build/queries_us.py ptab`, `oa_apps`, `oa_a1` | `$PAB_SRC/queries/ptab_docs.parquet`, `$PAB_SRC/queries_oa/oa_a1.parquet` |
| 7 | `python build/oa_slim.py` | `oa/oa_us.parquet` (citations to US documents, as family-graph tokens) |
| 8 | `python build/labels_us.py` | `queries/queries.parquet`, `labels/{labels,known_relevant,splits}.parquet` (labels v1) |
| 9 | `python build/devtest.py --labels $PAB_WORK/labels --salt devtest-us-v1` | `labels/devtest.parquet` (dev / test halves by salted hash of the query family) |
| 10 | `python build/leakage_audit_us.py`, `python build/dedup_splits_us.py` | `labels/leakage_audit.json`; near-duplicate training queries moved out of training |
| 11 | `python build/tacd.py --src us --families ... --subset ... --out $PAB_WORK/tacd` | document text with descriptions (`tacd_text`) for the pool and the training documents |
| 12 | `python build/eval_pool_us.py` | `eval_us/` evaluation data on pool v1 (250,000 families) |
| 13 | `python build/wo_cites_us.py ...`, then `python build/labels_v2_us.py --oa-foreign ...` | `eval_us_v2/`: labels v2 (WO citations resolved to US families through PatentsView PCT data) and pool v2 (+367 families = 250,367) |
| 14 | `python build/pool_passages_us.py` | `eval_us_v2_pas/` description chunks (>= 5,000 characters, title and abstract prepended) |
| 15 | `python build/cpc_us.py`, then `python build/cpc_strict_us.py` | `eval_us_v2_cpc/{mcf,doc_cpc,query_cpc_strict}.parquet` (CPC-strict stage) |
| 16 | `python build/provenance_us.py` | `provenance/manifest.json` (script hashes, data files) and the release gate (US-only inputs, no leakage) |
| 17 | `python build/build_dataset.py --out DIR` | the Hugging Face dataset layout (queries, qrels, corpus-pool-v2, chunks, cpc-members, families, family-members, train-qrels, sample, canary, provenance) |
| - | `python build/cited_not_applied_labels.py dev` (and `test`) | chart 2 labels |
| - | `python build/small_pool.py`, `python build/fallback_collection_us.py` | development-only collections (50,000 / 1,000,000 families); no headline number uses them |
| - | `python build/cocite.py ...`, `python build/mine_pool_us.py ids` / `parts`, `bash build/make_pack_us.sh ha RUN 30 100 tacd --y-indep-only` | training: co-citation edges, mining queries and pool, the training pack (see `train/README.md`) |

`corpus.py`, `tacd.py`, `cocite.py` and `devtest.py` are shared with an earlier pipeline of ours that also handled
EP documents; only their US paths (`--src us`) were used for this benchmark, and `provenance_us.py` checks that no
build script reads non-US data.

## Checking your build against ours

The dataset ships `provenance/manifest.json` (sha256 and size of every released file, plus the build statistics:
query, label, pool and chunk counts). Byte-identical parquet is not guaranteed (row-group layout, compression), so
compare contents: query ids and splits, qrels rows, pool family ids, chunk counts per family.

Office-action data published after our cut-off must be excluded: `oa_fetch.py` reads the last date to fetch from
`PAB_OA_CUTOFF` (YYYY-MM-DD).

## Open items

- USPTO bulk-file parser (XML / SGML / APS to the parsed full-text parquet) is not in this repository.
- PTAB input builder (`ptab_eval_full.jsonl` from the PTAB API) is not in this repository.
- Pool v1 selection (`tac_subset_250k/`, `tacd/subset-us-*.parquet`) is not in this repository.
- Front-page citation extraction for chart 2 is not in this repository.
