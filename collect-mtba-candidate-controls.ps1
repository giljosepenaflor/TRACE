param([ValidateRange(1,72)][int]$MaxNewRecords = 10)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Security
Set-Location -LiteralPath $PSScriptRoot
$tokenPath = "$env:USERPROFILE\.ssh\trace_vast_api_token.dpapi"
if (-not (Test-Path -LiteralPath $tokenPath)) {
    throw 'The protected TRACE API token is unavailable for this Windows user.'
}
$encrypted = [IO.File]::ReadAllBytes($tokenPath)
$plain = [System.Security.Cryptography.ProtectedData]::Unprotect(
    $encrypted, $null, [System.Security.Cryptography.DataProtectionScope]::CurrentUser)
$env:TRACE_API_TOKEN = [Text.Encoding]::UTF8.GetString($plain)
try {
    & .\.venv\Scripts\python.exe scripts\mtba_collect_candidate_clean_controls.py --max-new-records $MaxNewRecords
    if ($LASTEXITCODE -ne 0) { throw "Candidate clean-control collection failed with exit code $LASTEXITCODE" }
} finally {
    Remove-Item Env:TRACE_API_TOKEN -ErrorAction SilentlyContinue
}
