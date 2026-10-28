$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$tracePython = if (Test-Path -LiteralPath '.venv\Scripts\python.exe') { '.\.venv\Scripts\python.exe' } else { 'python' }
& $tracePython -m traceguard demo
if ($LASTEXITCODE -ne 0) { throw 'TRACE demo preparation failed. See README.md for setup.' }
Write-Host 'Open http://127.0.0.1:8765 in your browser.'
& $tracePython -m traceguard serve
