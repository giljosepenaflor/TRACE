$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
try { $traceExisting = Invoke-RestMethod -Uri 'http://127.0.0.1:8767/api/session' -TimeoutSec 2 } catch { $traceExisting = $null }
if ($traceExisting) {
    $traceExpected = Get-Content -LiteralPath 'artifacts/trace-robustness/run-001/report.json' -Raw | ConvertFrom-Json
    if ($traceExisting.report.frozen_detector_sha256 -eq $traceExpected.frozen_detector_sha256 -and $traceExisting.record_purpose -eq 'engineering_robustness') {
        Write-Host 'TRACE robustness experiment is already running: http://127.0.0.1:8767'
        exit 0
    }
    throw 'Port 8767 serves a different experiment.'
}
Write-Host 'Open http://127.0.0.1:8767 in your browser. F0 remains the original TRACE reference.'
& .\.venv\Scripts\python.exe -m traceguard serve --run artifacts/trace-robustness/run-001 --data artifacts/trace-robustness/run-001/dashboard.jsonl --port 8767
