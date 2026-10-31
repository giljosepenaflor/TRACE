param(
    [Parameter(Mandatory=$true)][ValidatePattern('^[A-Za-z0-9.-]+$')][string]$SshHost,
    [Parameter(Mandatory=$true)][ValidateRange(1,65535)][int]$SshPort,
    [Parameter(Mandatory=$true)][string]$KnownHostsPath,
    [string]$KeyPath = "$env:USERPROFILE\.ssh\id_ed25519",
    [ValidateRange(1024,65535)][int]$LocalPort = 18701
)

# Recover the matching local credential over already trusted SSH. This script
# does not rotate a remote token, restart the GPU service or submit a prompt.
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Security
$workspaceRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$destination = Join-Path $workspaceRoot '.restore-staging\trace-vast-58039092.dpapi'
$ssh = Join-Path $env:WINDIR 'System32\OpenSSH\ssh.exe'
foreach ($required in @($ssh, $KeyPath, $KnownHostsPath)) {
    if (-not (Test-Path -LiteralPath $required)) { throw 'A required SSH executable, key or verified host-key file is missing.' }
}
$sshArgs = @('-T', '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes',
             '-o', 'StrictHostKeyChecking=yes', '-o', "UserKnownHostsFile=$KnownHostsPath",
             '-o', 'ConnectTimeout=10', '-i', $KeyPath, '-p', "$SshPort", "root@$SshHost")
$apiToken = $null
$tokenBytes = $null
try {
    # Capture stdout directly; never print the remote secret or put it in argv.
    $remoteValue = & $ssh @sshArgs 'cat /root/.config/trace/api_token'
    if ($LASTEXITCODE -ne 0) { throw 'Could not read the existing TRACE credential through verified SSH.' }
    $apiToken = ($remoteValue -join "`n").Trim()
    $remoteValue = $null
    if ($apiToken.Length -lt 32 -or $apiToken.Length -gt 512 -or $apiToken -match '\s') {
        throw 'The existing remote API credential has an invalid format.'
    }
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:$LocalPort/health" `
            -Headers @{ Authorization = ('Bearer ' + $apiToken) } -TimeoutSec 10
    } catch {
        throw 'The existing local tunnel did not authenticate with the recovered remote credential. No local credential was changed.'
    }
    $identity = $health.checkpoint_identity
    if ($health.status -ne 'ok' -or
        $identity.model -ne 'meta-llama/Llama-2-7b-chat-hf' -or
        $identity.revision -ne 'f5db02db724555f92da89c216ac04704f23d4590' -or
        $identity.adapter -ne 'BackdoorLLM/Jailbreak_Llama2-7B_MTBA' -or
        $identity.adapter_revision -ne '8652ad60f1605605642c99fd680a15ecccf1b522' -or
        $identity.adapter_loaded -ne $true) {
        throw 'The authenticated API does not match the pinned TRACE model. No credential was changed.'
    }
    $tokenBytes = [Text.Encoding]::UTF8.GetBytes($apiToken)
    $encrypted = [Security.Cryptography.ProtectedData]::Protect(
        $tokenBytes, $null, [Security.Cryptography.DataProtectionScope]::CurrentUser)
    [IO.Directory]::CreateDirectory((Split-Path -Parent $destination)) | Out-Null
    if (Test-Path -LiteralPath $destination) {
        $backup = $destination + '.backup-' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ') + '.dpapi'
        Copy-Item -LiteralPath $destination -Destination $backup -ErrorAction Stop
    }
    [IO.File]::WriteAllBytes($destination, $encrypted)
    [pscustomobject]@{ CredentialFile = $destination; EncryptedWithDPAPI = $true;
        AuthenticatedHealth = $true; PinnedModelVerified = $true;
        RemoteCredentialChanged = $false; ModelPromptsSent = 0 }
} finally {
    $apiToken = $null
    $remoteValue = $null
    if ($tokenBytes) { [Array]::Clear($tokenBytes, 0, $tokenBytes.Length) }
    $tokenBytes = $null
}
