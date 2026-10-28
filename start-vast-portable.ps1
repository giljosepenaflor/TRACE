param(
    [Parameter(Mandatory=$true)][ValidatePattern('^[A-Za-z0-9.-]+$')][string]$SshHost,
    [Parameter(Mandatory=$true)][ValidateRange(1,65535)][int]$SshPort,
    [string]$KeyPath = "$env:USERPROFILE\.ssh\id_ed25519",
    [string]$KnownHostsPath = "$env:USERPROFILE\.ssh\known_hosts",
    [string]$ApiTokenPath = "$env:USERPROFILE\.ssh\trace_vast_api_token.dpapi",
    [ValidateRange(1024,65535)][int]$LocalPort = 18701,
    [ValidateRange(1024,65535)][int]$DashboardPort = 8769
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Security
Set-Location -LiteralPath $PSScriptRoot
$ssh = "$env:WINDIR\System32\OpenSSH\ssh.exe"
$apiPath = $ApiTokenPath
if (-not (Test-Path -LiteralPath $ssh)) { throw 'Windows OpenSSH is unavailable.' }
if (-not (Test-Path -LiteralPath $KeyPath)) { throw 'SSH private key is unavailable.' }
if (-not (Test-Path -LiteralPath $apiPath)) { throw 'Run setup-vast-portable.ps1 first to install matching API credentials.' }
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) { throw 'The local TRACE .venv is unavailable.' }

function Test-LocalPort([int]$Port) {
    $client = [Net.Sockets.TcpClient]::new()
    try {
        $attempt = $client.BeginConnect('127.0.0.1', $Port, $null, $null)
        return ($attempt.AsyncWaitHandle.WaitOne(750) -and $client.Connected)
    } finally { $client.Close() }
}
if (Test-LocalPort $LocalPort) { throw "Local tunnel port $LocalPort is in use. Close the old TRACE SSH tunnel or choose -LocalPort." }
if (Test-LocalPort $DashboardPort) { throw "Dashboard port $DashboardPort is in use. Close the old dashboard or choose -DashboardPort." }

$sshArgs = @('-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
             '-o',"UserKnownHostsFile=$KnownHostsPath",
             '-o','IdentitiesOnly=yes','-i',$KeyPath,'-p',"$SshPort","root@$SshHost")
$status = & $ssh @sshArgs 'supervisorctl status trace_llama2'
if ($status -notmatch 'trace_llama2\s+(RUNNING|STOPPED|EXITED|FATAL|BACKOFF)') {
    throw 'TRACE service is not registered. Run setup-vast-portable.ps1 first.'
}
if ($status -notmatch '\bRUNNING\b') {
    & $ssh @sshArgs 'supervisorctl start trace_llama2'
    if ($LASTEXITCODE -ne 0) { throw 'The remote Llama 2 service did not start.' }
}

$encrypted = [IO.File]::ReadAllBytes($apiPath)
$plain = [Security.Cryptography.ProtectedData]::Unprotect(
    $encrypted, $null, [Security.Cryptography.DataProtectionScope]::CurrentUser)
$apiToken = [Text.Encoding]::UTF8.GetString($plain)
if ($apiToken.Length -lt 32) { throw 'The protected TRACE API token is invalid.' }
$tunnel = $null
try {
    $tunnelArgs = @('-NoProfile','-ExecutionPolicy','Bypass','-File',
                    (Join-Path $PSScriptRoot 'keep-vast-tunnel.ps1'),
                    '-SshHost',$SshHost,'-SshPort',"$SshPort",'-LocalPort',"$LocalPort",
                    '-KnownHostsPath',$KnownHostsPath,
                    '-KeyPath',$KeyPath)
    $tunnel = Start-Process -FilePath 'powershell.exe' -ArgumentList $tunnelArgs -WindowStyle Hidden -PassThru
    $ready = $false
    for ($attempt = 0; $attempt -lt 120; $attempt++) {
        if ($tunnel.HasExited) { throw 'SSH tunnel supervisor exited before the model became ready.' }
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:$LocalPort/health" `
                -Headers @{Authorization="Bearer $apiToken"} -TimeoutSec 3
            if ($health.status -eq 'ok') { $ready = $true; break }
        } catch { Start-Sleep -Seconds 1 }
    }
    if (-not $ready) { throw 'Timed out waiting for the authenticated Vast model API.' }
    $identity = $health.checkpoint_identity
    if ($identity.model -ne 'meta-llama/Llama-2-7b-chat-hf' -or
        $identity.revision -ne 'f5db02db724555f92da89c216ac04704f23d4590' -or
        $identity.adapter -ne 'BackdoorLLM/Jailbreak_Llama2-7B_MTBA' -or
        $identity.adapter_revision -ne '8652ad60f1605605642c99fd680a15ecccf1b522' -or
        $identity.adapter_loaded -ne $true) {
        throw 'Vast API identity differs from the frozen TRACE base and adapter.'
    }
    $env:TRACE_API_TOKEN = $apiToken
    $env:TRACE_REMOTE_URL = "http://127.0.0.1:$LocalPort"
    $env:TRACE_REMOTE_BACKEND = 'vast'
    $env:PYTHONIOENCODING = 'utf-8'
    Write-Host "Pinned Llama 2 + MTBA is ready. TRACE dashboard: http://127.0.0.1:$DashboardPort"
    Write-Host 'Keep this PowerShell window open while using TRACE. Ctrl+C closes the local dashboard and tunnel.'
    & .\.venv\Scripts\python.exe -m traceguard serve `
        --run artifacts/trace-robustness/run-001 `
        --data artifacts/trace-robustness/run-001/dashboard.jsonl `
        --port $DashboardPort --live-llama2-remote
} finally {
    Remove-Item Env:TRACE_API_TOKEN,Env:TRACE_REMOTE_URL,Env:TRACE_REMOTE_BACKEND -ErrorAction SilentlyContinue
    $apiToken = $null
    $plain = $null
    if ($tunnel -and -not $tunnel.HasExited) { Stop-Process -Id $tunnel.Id -ErrorAction SilentlyContinue }
}
