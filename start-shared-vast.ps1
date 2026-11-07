$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Create .venv and install requirements-models.txt first; see docs/SHARING_TRACE.md.'
}
$env:HF_HUB_CACHE = Join-Path $PSScriptRoot 'model-cache\hf-hub'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:TRACE_REMOTE_BACKEND = 'vast'
if (-not $env:TRACE_REMOTE_URL) { $env:TRACE_REMOTE_URL = 'http://127.0.0.1:18701' }
Write-Host 'Keep your separately opened Vast SSH tunnel running.'
Write-Host 'Paste the existing Vast TRACE API token when prompted; input is hidden.'
Write-Host 'This launcher does not read another Windows user''s encrypted credentials.'
& $python -m traceguard serve --run artifacts/trace-robustness/run-001 `
    --data artifacts/trace-robustness/run-001/dashboard.jsonl --port 8769 --live-llama2-remote
if ($LASTEXITCODE -ne 0) { throw 'TRACE startup failed. See the preceding error.' }
