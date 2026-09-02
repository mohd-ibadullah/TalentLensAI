"""
Corpus integrity checker.

Validates a candidates.jsonl before anything downstream trusts it.
Catches the failure class where an artifact is trusted without verification
(e.g., a file with mostly empty shell records).
"""
from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path


def check_corpus(path: str | Path, min_populated_fraction: float = 0.95) -> dict:
    """Validate a candidates.jsonl before anything downstream trusts it.

    Returns a dict with: n_lines, n_populated, populated_fraction,
    sha256, median_line_bytes, ok (bool), problems (list[str]).

    Flags as a problem:
      - populated_fraction < min_populated_fraction
      - median line length under 500 bytes (shell records)
      - any line that fails to parse as JSON
    """
    path = Path(path)
    problems: list[str] = []

    # --- read lines ----------------------------------------------------------
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {
            "n_lines": 0,
            "n_populated": 0,
            "populated_fraction": 0.0,
            "sha256": "",
            "median_line_bytes": 0,
            "ok": False,
            "problems": [f"File not found: {path}"],
        }

    lines = [l for l in raw.splitlines() if l.strip()]
    n_lines = len(lines)

    # --- sha256 --------------------------------------------------------------
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # --- parse + measure -----------------------------------------------------
    line_lengths: list[int] = []
    n_populated = 0
    n_parse_errors = 0

    for line in lines:
        line_lengths.append(len(line))
        try:
            rec = json.loads(line)
            skills = rec.get("skills") or []
            if len(skills) > 0:
                n_populated += 1
        except (json.JSONDecodeError, TypeError):
            n_parse_errors += 1

    populated_fraction = n_populated / n_lines if n_lines else 0.0
    median_line_bytes = int(statistics.median(line_lengths)) if line_lengths else 0

    # --- flag problems -------------------------------------------------------
    if n_parse_errors:
        problems.append(f"{n_parse_errors} line(s) failed to parse as JSON")

    if populated_fraction < min_populated_fraction:
        problems.append(
            f"populated_fraction={populated_fraction:.4f} "
            f"< {min_populated_fraction}  ({n_populated}/{n_lines} have skills)"
        )

    if median_line_bytes < 500:
        problems.append(
            f"median_line_bytes={median_line_bytes} < 500 "
            "(indicates mostly empty shell records)"
        )

    return {
        "n_lines": n_lines,
        "n_populated": n_populated,
        "populated_fraction": populated_fraction,
        "sha256": h,
        "median_line_bytes": median_line_bytes,
        "ok": len(problems) == 0,
        "problems": problems,
    }


def print_corpus_summary(path: str | Path, min_populated_fraction: float = 0.95) -> dict:
    """Run check_corpus and print the one-line summary + loud warning if needed.

    Returns the check result dict.
    """
    result = check_corpus(path, min_populated_fraction=min_populated_fraction)
    name = Path(path).name

    # One-line summary
    sha_short = result["sha256"][:8] if result["sha256"] else "N/A"
    print(
        f"corpus: {name} | "
        f"{result['n_lines']:,} lines | "
        f"{result['n_populated']:,} populated "
        f"({result['populated_fraction'] * 100:.1f}%) | "
        f"sha256 {sha_short}..."
    )

    # Loud warning if corpus is unhealthy
    if not result["ok"]:
        print()
        print("=" * 70)
        print("⚠  CORPUS INTEGRITY WARNING")
        print("=" * 70)
        for p in result["problems"]:
            print(f"  • {p}")
        print()
        print(f"  File: {path}")
        print(f"  sha256: {result['sha256']}")
        print(f"  Total lines: {result['n_lines']:,}")
        print(f"  Populated:   {result['n_populated']:,} "
              f"({result['populated_fraction'] * 100:.1f}%)")
        print(f"  Median line bytes: {result['median_line_bytes']:,}")
        print()
        print("  Downstream results measured against this corpus are INVALID.")
        print("=" * 70)
        print()

    return result
