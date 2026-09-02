import json
import os

from typing import Iterator


def _run_corpus_integrity_check(file_path: str) -> None:
    """Run the corpus integrity guard on every load. Prints a one-line summary;
    prints a loud multi-line warning if populated_fraction < 0.95.
    Does NOT raise — some workflows legitimately use small samples.
    """
    try:
        from src.data_integrity import print_corpus_summary
        print_corpus_summary(file_path)
    except Exception as exc:
        # Integrity check is best-effort — never block the caller
        print(f"[data_integrity] WARNING: could not run integrity check: {exc}")


def load_sample_candidates(file_path: str) -> list[dict]:
    """
    Load candidate profiles from a JSON array file (development sample).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Sample candidates file not found at: {file_path}")
    _run_corpus_integrity_check(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def stream_candidates(file_path: str) -> Iterator[dict]:
    """
    A generator that streams candidate profiles line-by-line from a JSONL file.
    This prevents high memory usage when loading the full 100K candidates.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Candidates JSONL file not found at: {file_path}")
    _run_corpus_integrity_check(file_path)
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

