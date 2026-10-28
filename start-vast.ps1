$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Security
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
$port = 8769
if (-not $env:TRACE_REMOTE_URL) { $env:TRACE_REMOTE_URL = 'http://127.0.0.1:18701' }
$env:TRACE_REMOTE_BACKEND = 'vast'
try { $live = Invoke-RestMethod "http://127.0.0.1:$port/api/live/status" -TimeoutSec 2 } catch { $live = $null }
if ($live) {
    if ($live.mode -ne 'llama2_mtba_colab' -or $live.backend -ne 'vast') {
        throw "Port $port is occupied by a different TRACE service. Stop it before starting Vast mode."
    }
    Write-Host "TRACE Vast dashboard already running ($($live.state)): http://127.0.0.1:$port"
    exit 0
}
$localApiPath = "$env:USERPROFILE\.ssh\trace_vast_api_token.dpapi"
$loadedToken = $false
if (-not $env:TRACE_API_TOKEN -and (Test-Path -LiteralPath $localApiPath)) {
    $encrypted = [IO.File]::ReadAllBytes($localApiPath)
    $plain = [System.Security.Cryptography.ProtectedData]::Unprotect(
        $encrypted, $null, [System.Security.Cryptography.DataProtectionScope]::CurrentUser)
    $env:TRACE_API_TOKEN = [Text.Encoding]::UTF8.GetString($plain)
    $loadedToken = $true
}
Write-Host 'The Vast inference server and SSH tunnel must already be running.'
if ($env:TRACE_API_TOKEN) {
    Write-Host 'Using the protected TRACE API token for this Windows user.'
} else {
    Write-Host 'The API token prompt is hidden. Use the same TRACE_API_TOKEN as on Vast.'
}
Write-Host "Starting TRACE at http://127.0.0.1:$port. Ctrl+C stops the local dashboard only."
try {
    & .\.venv\Scripts\python.exe -m traceguard serve --run artifacts/trace-robustness/run-001 --data artifacts/trace-robustness/run-001/dashboard.jsonl --port $port --live-llama2-remote
} finally {
    if ($loadedToken) { Remove-Item Env:TRACE_API_TOKEN -ErrorAction SilentlyContinue }
}
