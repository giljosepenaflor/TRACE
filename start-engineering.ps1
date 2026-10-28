$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
try {
    $traceExisting = Invoke-RestMethod -Uri 'http://127.0.0.1:8766/api/session' -TimeoutSec 2
} catch {
    $traceExisting = $null
}
if ($traceExisting) {
    $traceExpected = Get-Content -LiteralPath 'artifacts/trace-engineering/run-001/report.json' -Raw | ConvertFrom-Json
    if ($traceExisting.report.dataset_sha256 -eq $traceExpected.dataset_sha256 -and $traceExisting.record_purpose -eq 'engineering_pilot') {
        Write-Host 'TRACE engineering pilot is already running: http://127.0.0.1:8766'
        exit 0
    }
    throw 'Port 8766 is serving another experiment. Stop that server before loading this run.'
}
Write-Host 'Open http://127.0.0.1:8766 in your browser.'
& .\.venv\Scripts\python.exe -m traceguard serve --run artifacts/trace-engineering/run-001 --data artifacts/trace-engineering/run-001/interactions.jsonl --port 8766
