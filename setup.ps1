# TalentLens AI — One-Time Judge Setup (Stage 3)
# Run from talent-lens-ai/ with network access BEFORE offline ranking.

Write-Host "=== TalentLens AI Setup ===" -ForegroundColor Cyan

pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { exit 1 }

python src/download_models.py
if ($LASTEXITCODE -ne 0) { exit 1 }

python src/precompute_embeddings.py
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host ""
Write-Host "Setup complete. Verify:" -ForegroundColor Green
Write-Host "  python scripts/verify_submit_ready.py"
Write-Host "Run ranking (offline):" -ForegroundColor Green
Write-Host "  python rank.py --candidates <path-to-candidates.jsonl> --out ./outputs/mohd_ibadullah.csv"
