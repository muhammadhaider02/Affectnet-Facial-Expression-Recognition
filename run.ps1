# Run the full pipeline: train -> evaluate
# Usage: .\run.ps1 [optional args passed to both commands]
# Example: .\run.ps1 --epochs 20 --batch-size 32

param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host " FER-AffectNet  |  Full Pipeline" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

Write-Host ""
Write-Host ">>> STEP 1/2: Training" -ForegroundColor Yellow
Write-Host "--------------------------------------------------"
uv run fer-train @ExtraArgs

Write-Host ""
Write-Host ">>> STEP 2/2: Evaluation" -ForegroundColor Yellow
Write-Host "--------------------------------------------------"
uv run fer-evaluate @ExtraArgs

Write-Host ""
Write-Host "==================================================" -ForegroundColor Green
Write-Host " Done. Outputs saved to: outputs/" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green
