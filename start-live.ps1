$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
try { $live = Invoke-RestMethod 'http://127.0.0.1:8768/api/live/status' -TimeoutSec 2 } catch { $live = $null }
if ($live) {
    if ($live.mode -ne 'llama2_sleeper_live') { throw 'Port 8768 is occupied by a different live model. Stop it before starting Llama 2.' }
    Write-Host "TRACE Llama 2 dashboard already running ($($live.state)): http://127.0.0.1:8768"
    if ($live.state -eq 'failed') { Write-Host 'Free memory, then click Retry model startup in the dashboard.' }
    exit 0
}
Write-Host 'Starting the Llama 2 dashboard. Full-signal mode needs 8.5 GiB free; low-RAM text-only demo needs 6.0 GiB.'
Write-Host 'Open http://127.0.0.1:8768 and wait for Llama 2 ready. Ctrl+C stops both dashboard and model.'
& .\.venv\Scripts\python.exe -m traceguard serve --run artifacts/trace-robustness/run-001 --data artifacts/trace-robustness/run-001/dashboard.jsonl --port 8768 --live-llama2
