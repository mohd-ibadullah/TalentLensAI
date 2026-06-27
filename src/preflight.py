"""
Preflight checks before ranking — ensures models are cached and embeddings exist.
Judges run this automatically via run_pipeline_full.py to avoid cold-start timeouts.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def ensure_models_cached(verbose: bool = True) -> None:
    """Download HF models if not already in local cache."""
    try:
        from transformers import AutoTokenizer, AutoModel
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError("Missing dependencies. Run: pip install -r requirements.txt") from exc

    models = [
        ("BAAI/bge-base-en-v1.5", "embedding"),
        ("cross-encoder/ms-marco-MiniLM-L6-v2", "cross-encoder"),
    ]

    for model_id, label in models:
        if verbose:
            print(f"Preflight: verifying {label} model '{model_id}'...")
        try:
            if label == "cross-encoder":
                CrossEncoder(model_id)
            else:
                AutoTokenizer.from_pretrained(model_id)
                AutoModel.from_pretrained(model_id)
            if verbose:
                print(f"  [OK] {model_id} ready (cached locally)")
        except Exception as exc:
            if verbose:
                print(f"  [WARN] Could not cache {model_id}: {exc}")
            raise


def embeddings_paths() -> tuple[Path, Path]:
    root = _project_root()
    return root / "data" / "candidate_embeddings.npy", root / "data" / "candidate_ids.json"


def ensure_embeddings_exist(verbose: bool = True) -> bool:
    """Return True if precomputed embeddings are available for full-dataset ranking."""
    npy_path, ids_path = embeddings_paths()
    ok = npy_path.exists() and ids_path.exists()
    if verbose:
        if ok:
            print(f"Preflight: precomputed embeddings found at {npy_path.parent}")
        else:
            print("Preflight: precomputed embeddings NOT found.")
            print("  Run once (with network + candidates.jsonl):")
            print("    python src/download_models.py")
            print("    python src/precompute_embeddings.py")
            print("  Without embeddings the pipeline falls back to BM25-only retrieval (lower quality).")
    return ok


def run_preflight(require_embeddings: bool = False, verbose: bool = True) -> None:
    """Run all preflight checks. Raises on missing models; warns on missing embeddings."""
    if verbose:
        print("=" * 60)
        print("TalentLens AI — Preflight Checks")
        print("=" * 60)
    ensure_models_cached(verbose=verbose)
    has_embeddings = ensure_embeddings_exist(verbose=verbose)
    if require_embeddings and not has_embeddings:
        raise RuntimeError(
            "Precomputed embeddings required but missing. "
            "Run: python src/precompute_embeddings.py"
        )
    if verbose:
        print("Preflight complete — ready to rank offline.\n")
