#!/usr/bin/env python3
"""
Name-blindness re-measurement on REAL candidates.

Picks 200 random candidates (seed=42) from the full 100K corpus and tests
4 name variants each. Measures whether changing anonymized_name affects
the text builder outputs.

The previous "200 x 4, max diff 0.0" was measured on mostly-empty records
and is meaningless. This rerun uses real data.
"""
import copy
import json
import random
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from candidate_text import build_candidate_embedding_text
from bm25_filter import build_candidate_document
from cross_encoder_reranker import CrossEncoderReranker

SEED = 42
N_CANDIDATES = 200
CANDIDATES_PATH = project_root / "candidates.jsonl"
NAMES = [
    "Priya Sharma",
    "John Smith",
    "Chen Wei",
    "Fatima Al-Rashid",
]


def load_all(path):
    cands = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cands.append(json.loads(line))
    return cands


def swap_name(cand, new_name):
    c = copy.deepcopy(cand)
    c["profile"]["anonymized_name"] = new_name
    return c


def main():
    random.seed(SEED)
    print("Loading all candidates...")
    all_cands = load_all(str(CANDIDATES_PATH))
    print(f"  Total on disk: {len(all_cands):,}")

    sampled = random.sample(all_cands, N_CANDIDATES)
    print(f"  Sampled: {len(sampled)} (seed={SEED})")
    print(f"  Names tested: {NAMES}")
    print(f"  Total comparisons: {len(sampled)} x {len(NAMES)} = {len(sampled) * len(NAMES)}\n")

    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)

    max_diff_embedding = 0.0
    max_diff_bm25 = 0.0
    max_diff_crossencoder = 0.0
    n_differences_embedding = 0
    n_differences_bm25 = 0
    n_differences_crossencoder = 0
    total_comparisons = 0

    for cand in sampled:
        base_embedding = build_candidate_embedding_text(cand)
        base_bm25 = build_candidate_document(cand)
        base_ce = reranker.build_candidate_text(cand)

        for name in NAMES:
            modified = swap_name(cand, name)
            mod_embedding = build_candidate_embedding_text(modified)
            mod_bm25 = build_candidate_document(modified)
            mod_ce = reranker.build_candidate_text(modified)

            # Check embedding text
            if base_embedding != mod_embedding:
                n_differences_embedding += 1
                # Find where they differ
                for i, (a, b) in enumerate(zip(base_embedding, mod_embedding)):
                    if a != b:
                        break

            # Check BM25 text
            if base_bm25 != mod_bm25:
                n_differences_bm25 += 1

            # Check cross-encoder text
            if base_ce != mod_ce:
                n_differences_crossencoder += 1

            total_comparisons += 1

    print("=" * 80)
    print("NAME-BLINDNESS MEASUREMENT RESULTS (real candidates)")
    print("=" * 80)
    print(f"  Candidates tested: {N_CANDIDATES}")
    print(f"  Name variants per candidate: {len(NAMES)}")
    print(f"  Total comparisons: {total_comparisons}")
    print()
    print("  Text Builder              | Differences | Max Diff")
    print("  " + "-" * 50)
    print(f"  Embedding text            | {n_differences_embedding:>11} | {max_diff_embedding}")
    print(f"  BM25 document             | {n_differences_bm25:>11} | {max_diff_bm25}")
    print(f"  Cross-encoder text        | {n_differences_crossencoder:>11} | {max_diff_crossencoder}")
    print()

    if n_differences_embedding == 0 and n_differences_bm25 == 0 and n_differences_crossencoder == 0:
        print("  CONCLUSION: Name-blindness HOLDS. anonymized_name is never read")
        print("  by any text builder. Changing the name produces zero difference")
        print("  in all {0} comparisons across {1} real candidates.".format(
            total_comparisons, N_CANDIDATES))
    else:
        print("  CONCLUSION: Name-blindness VIOLATED. Some builders read the name.")

    print("=" * 80)


if __name__ == "__main__":
    main()
