# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
"""Where the benchmark files live: a local copy of the dataset, or the Hugging Face dataset repo.

Every command takes --data (default: the Hugging Face dataset) and file arguments that may be local paths or
hf://datasets/<org>/<name>[@<revision>]/<path inside the repo>. Remote files are downloaded once into the Hugging Face
cache (huggingface_hub), so later runs are offline.
"""
import glob
import os

HF_DATASET = "hf://datasets/patentmechanics/prior-art-bench-v1"
VECTORS = "vectors/anticipate-0.6b"


def _hf(path):
    """hf://datasets/org/name[@rev]/inner -> (repo, revision, inner)"""
    if not path.startswith("hf://datasets/"):
        raise SystemExit(f"only hf://datasets/... paths are supported: {path}")
    parts = path[len("hf://datasets/"):].split("/")
    repo, inner = "/".join(parts[:2]), "/".join(parts[2:]).strip("/")
    rev = None
    if "@" in repo:
        repo, rev = repo.split("@", 1)
    return repo, rev, inner


def is_remote(path):
    return str(path).startswith("hf://")


def fetch(path, patterns=None):
    """Local path for a file or directory (a remote one is downloaded; `patterns` limits a directory download)."""
    if not is_remote(path):
        return path
    from huggingface_hub import hf_hub_download, snapshot_download
    repo, rev, inner = _hf(path)
    if os.path.splitext(inner)[1] in (".parquet", ".npy", ".json", ".txt"):
        return hf_hub_download(repo, inner, repo_type="dataset", revision=rev)
    pats = [f"{inner}/{p}" if inner else p for p in (patterns or ["*", "**/*"])]
    root = snapshot_download(repo, repo_type="dataset", revision=rev, allow_patterns=pats)
    return os.path.join(root, inner) if inner else root


def data_file(data, rel):
    """One file (or a glob such as corpus-pool-v2/part-*.parquet) of the dataset, as a local path / glob."""
    if not is_remote(data):
        return os.path.join(data, rel)
    from huggingface_hub import snapshot_download
    repo, rev, inner = _hf(data)
    full = f"{inner}/{rel}" if inner else rel
    root = snapshot_download(repo, repo_type="dataset", revision=rev, allow_patterns=[full])
    return os.path.join(root, full)


def files(data, rel_glob):
    """Sorted local files matching a glob inside the dataset."""
    out = sorted(glob.glob(data_file(data, rel_glob)))
    if not out:
        raise SystemExit(f"no files match {rel_glob} under {data}")
    return out
