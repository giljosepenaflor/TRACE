$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
$port = 8769
try { $live = Invoke-RestMethod "http://127.0.0.1:$port/api/live/status" -TimeoutSec 2 } catch { $live = $null }
if ($live) {
    if ($live.mode -ne 'llama2_mtba_colab') { throw "Port $port is occupied by a different TRACE service. Stop it before starting the Colab dashboard." }
    Write-Host "TRACE Colab dashboard already running ($($live.state)): http://127.0.0.1:$port"
    exit 0
}
Write-Host 'The Colab model and its Cloudflare tunnel must already be running.'
Write-Host 'The API token prompt is hidden. No token is written to browser code or a project file.'
Write-Host "Starting the local TRACE dashboard at http://127.0.0.1:$port. Ctrl+C stops the local dashboard only."
& .\.venv\Scripts\python.exe -m traceguard serve --run artifacts/trace-robustness/run-001 --data artifacts/trace-robustness/run-001/dashboard.jsonl --port $port --live-llama2-colab
