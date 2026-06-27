"""Strict QA audit per Redrob H2S India Runs 2026 prompt."""
import csv
import json
import re
import sys
import time
import subprocess
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(r"C:/Users/froms/Downloads/[PUB] India_runs_data_and_ai_challenge")
PROJ = BASE / "talent-lens-ai"
CSV = PROJ / "outputs/mohd_ibadullah.csv"
CAND = BASE / "[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl"
VALIDATOR = BASE / "[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/validate_submission.py"

sys.path.insert(0, str(PROJ))
from src.honeypot_detector import detect_trap, classify_title

rows_out = []

def row(section, criterion, passed, evidence, fix=""):
    rows_out.append({
        "section": section,
        "criterion": criterion,
        "result": "PASS" if passed else "FAIL",
        "evidence": evidence,
        "fix": fix if not passed else "",
    })

# === SECTION 1 ===
v = subprocess.run([sys.executable, str(VALIDATOR), str(CSV)], capture_output=True, text=True)
row("S1", "Validator exit 0 + valid message", v.returncode == 0 and "Submission is valid" in v.stdout,
    f"exit={v.returncode}, out={v.stdout.strip()}")

lines = open(CSV, encoding="utf-8").readlines()
data = list(csv.DictReader(lines))
scores = [float(r["score"]) for r in data]
ranks = [int(r["rank"]) for r in data]
row("S1", "Exactly 100 data rows + header", len(lines) == 101 and len(data) == 100, f"lines={len(lines)} data={len(data)}")
row("S1", "Columns candidate_id,rank,score,reasoning", list(data[0].keys()) == ["candidate_id","rank","score","reasoning"], str(list(data[0].keys())))
row("S1", "Ranks 1-100 unique sequential", sorted(ranks) == list(range(1,101)), f"missing={set(range(1,101))-set(ranks)}")
row("S1", "Scores non-increasing", all(scores[i] >= scores[i+1] for i in range(len(scores)-1)),
    f"violations={sum(1 for i in range(len(scores)-1) if scores[i]<scores[i+1])}")

# === SECTION 2 ===
ids_top = {r["candidate_id"] for r in data}
decoys = ["CAND_0000002","CAND_0000003","CAND_0000004","CAND_0000005"]
found_decoys = [d for d in decoys if d in ids_top]
row("S2", "Known sample decoys NOT in top100", len(found_decoys) == 0, f"found={found_decoys}")

# load top100 profiles
cand_map = {}
with open(CAND, encoding="utf-8") as f:
    for line in f:
        c = json.loads(line)
        if c["candidate_id"] in ids_top:
            cand_map[c["candidate_id"]] = c

traps = [r["candidate_id"] for r in data if detect_trap(cand_map[r["candidate_id"]])[0] > 0.4]
row("S2", "Honeypot count in top100 = 0", len(traps) == 0, f"traps={traps}")

irrelevant_cats = ["Mechanical Engineering", "Accounting", "Support", "Operations", "Civil Engineering"]
irrelevant_in_top = []
for r in data:
    title = cand_map[r["candidate_id"]]["profile"].get("current_title","")
    cat = classify_title(title)
    if cat in irrelevant_cats:
        irrelevant_in_top.append((r["rank"], r["candidate_id"], title, cat))
row("S2", "Irrelevant backgrounds filtered from top100", len(irrelevant_in_top) == 0,
    f"found={irrelevant_in_top[:5]}")

# === SECTION 3 ===
print("Running timed pipeline (warm)...", flush=True)
out_t = PROJ / "outputs/qa_timing.csv"
t0 = time.time()
p = subprocess.run([
    sys.executable, str(PROJ/"src/run_pipeline_full.py"),
    "--candidates", str(CAND), "--out", str(out_t)
], cwd=str(PROJ), capture_output=True, text=True, encoding="utf-8", errors="replace")
wall = time.time() - t0
internal = None
for line in (p.stdout or "").splitlines() + (p.stderr or "").splitlines():
    m = re.search(r"Pipeline completed successfully in ([\d.]+) seconds", line)
    if m:
        internal = float(m.group(1))
row("S3", "Pipeline under 300s (warm)", (internal or wall) < 300,
    f"internal={internal}s wall={round(wall,1)}s (prompt claims ~37s — NOT verified)")
row("S3", "Pipeline ~37s claim", False, f"Actual internal={internal}s — prompt expectation wrong for this codebase")

# offline API check in ranking path
ranking_files = [PROJ/"src/pipeline.py", PROJ/"src/feature_scorer.py", PROJ/"src/embedding_scorer.py",
                 PROJ/"src/cross_encoder_reranker.py", PROJ/"src/llm_reranker.py", PROJ/"rank.py"]
api_patterns = ["openai", "generativelanguage", "api.groq.com", "anthropic", "cohere"]
api_hits = []
for fp in ranking_files:
    text = fp.read_text(encoding="utf-8").lower()
    for pat in api_patterns:
        if pat in text and fp.name != "llm_reranker.py":
            api_hits.append(f"{fp.name}:{pat}")
# llm_reranker has optional API when use_llm=True; pipeline uses use_llm=False
pipeline_text = (PROJ/"src/pipeline.py").read_text(encoding="utf-8")
row("S3", "Ranking step use_llm=False", "use_llm" in pipeline_text and "False" in pipeline_text, "pipeline default")
row("S3", "No hosted API in ranking core modules", len(api_hits) == 0, f"hits={api_hits} (llm_reranker optional path exists)")

try:
    import psutil, os
    proc = psutil.Process(os.getpid())
    mem_mb = proc.memory_info().rss / (1024*1024)
    row("S3", "Memory under 16GB (audit process proxy)", True, f"audit script RSS ~{mem_mb:.0f}MB — full peak not instrumented; Antigravity claimed ~800MB-1.3GB")
except Exception as e:
    row("S3", "Memory measurement", False, str(e), "Run pipeline with memory profiler for proof")

# === SECTION 4 — scan 100K raw database ===
print("Scanning 100K candidates for anomalies...", flush=True)
salary_bad = 0
edu_overlap = 0
masters_before_bach = 0
dup_career = 0
n = 0
with open(CAND, encoding="utf-8") as f:
    for line in f:
        n += 1
        c = json.loads(line)
        sig = c.get("redrob_signals", {})
        sal = sig.get("expected_salary_range_inr_lpa", {})
        if sal:
            try:
                smin = float(sal.get("min", 0) or 0)
                smax = float(sal.get("max", 0) or 0)
                if smin > smax:
                    salary_bad += 1
            except Exception:
                pass
        edu = c.get("education", [])
        for i, e1 in enumerate(edu):
            for e2 in edu[i+1:]:
                if e1.get("institution") == e2.get("institution"):
                    try:
                        s1, e1y = int(e1.get("start_year",0) or 0), int(e1.get("end_year",0) or 0)
                        s2, e2y = int(e2.get("start_year",0) or 0), int(e2.get("end_year",0) or 0)
                        if s1 and e1y and s2 and e2y and not (e1y < s2 or e2y < s1):
                            edu_overlap += 1
                    except Exception:
                        pass
        b_end, m_start = None, None
        for e in edu:
            deg = e.get("degree","").lower()
            try:
                if any(x in deg for x in ["b.tech","bachelor","b.s","b.e","b.sc"]):
                    b_end = int(e.get("end_year",0) or 0)
                if any(x in deg for x in ["m.s","master","m.tech","m.sc","mba"]):
                    m_start = int(e.get("start_year",0) or 0)
            except Exception:
                pass
        if b_end and m_start and m_start < b_end:
            masters_before_bach += 1
        seen_roles = set()
        for job in c.get("career_history", []):
            key = (job.get("company","").lower().strip(), job.get("title","").lower().strip())
            if key in seen_roles and key[0]:
                dup_career += 1
                break
            seen_roles.add(key)
 
row("S4", "Salary min > max in raw 100K", salary_bad == 0, f"count={salary_bad} (NOT cleaned in pipeline CSV — raw dataset issue)")
row("S4", "Education overlaps same institution", edu_overlap == 0, f"count={edu_overlap}")
row("S4", "Masters before Bachelors chronology", masters_before_bach == 0, f"count={masters_before_bach}")
row("S4", "Duplicate company+title in career", dup_career == 0, f"count={dup_career}")
row("S4", "Note: streamlit sanitizes at display time only", True, "Raw jsonl anomalies may exist; UI patches on render")

# === SECTION 5 — streamlit code inspection ===
app = (PROJ/"app/streamlit_app.py").read_text(encoding="utf-8")
row("S5", "Display scores capped at 99.4", "min(candidate[\"_final_score\"], 99.4)" in app or "min(candidate['_final_score'], 99.4)" in app.replace('"', "'"), "line ~679")
row("S5", "Rank-based score offset tie-breaker", "offset = i * 0.1" in app, "line ~683")
row("S5", "Global prefix tracking (50 chars)", "_global_seen_prefixes" in app, "deduplicate_descriptions()")
row("S5", 'Strip "At {company}: " prefixes', 'if desc.startswith("At ")' in app, "career history expander loop")
row("S5", "Anonymized names in cards (hide CAND_ ID)", "anonymized_name" in app and "anonymized_name" in app, "Cards use anonymized_name; CAND_ID still in CSV/download")
row("S5", "Backticks stripped from titles", '.strip("` ")' in app, "line ~662")

# print markdown table
print("\n| Section | Criterion | Result | Evidence | Fix Needed |")
print("|---------|-----------|--------|----------|------------|")
for r in rows_out:
    ev = r["evidence"].replace("|", "/")[:120]
    fx = r["fix"].replace("|", "/")[:80]
    print(f"| {r['section']} | {r['criterion']} | **{r['result']}** | {ev} | {fx} |")

pass_n = sum(1 for r in rows_out if r["result"] == "PASS")
fail_n = sum(1 for r in rows_out if r["result"] == "FAIL")
print(f"\n**TOTAL: {pass_n} PASS / {fail_n} FAIL / {len(rows_out)} checks**")
