import json
import os

def load_sample_candidates(file_path):
    """
    Load candidate profiles from a JSON array file (development sample).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Sample candidates file not found at: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def stream_candidates(file_path):
    """
    A generator that streams candidate profiles line-by-line from a JSONL file.
    This prevents high memory usage when loading the full 100K candidates.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Candidates JSONL file not found at: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
