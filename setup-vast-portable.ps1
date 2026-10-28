param(
    [Parameter(Mandatory=$true)][ValidatePattern('^[A-Za-z0-9.-]+$')][string]$SshHost,
    [Parameter(Mandatory=$true)][ValidateRange(1,65535)][int]$SshPort,
    [string]$KeyPath = "$env:USERPROFILE\.ssh\id_ed25519",
    [string]$KnownHostsPath = "$env:USERPROFILE\.ssh\known_hosts",
    [string]$ApiTokenPath = "$env:USERPROFILE\.ssh\trace_vast_api_token.dpapi"
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Security
$ssh = "$env:WINDIR\System32\OpenSSH\ssh.exe"
if (-not (Test-Path -LiteralPath $ssh)) { throw 'Windows OpenSSH is unavailable.' }
if (-not (Test-Path -LiteralPath $KeyPath)) { throw 'SSH private key is unavailable.' }
# The first connection may ask the user to accept a new SSH host key. Keep this
# interactive check inside the one setup command, before sending any credential.
& $ssh -o BatchMode=no -o StrictHostKeyChecking=ask -o "UserKnownHostsFile=$KnownHostsPath" -o IdentitiesOnly=yes `
    -i $KeyPath -p $SshPort "root@$SshHost" 'true'
if ($LASTEXITCODE -ne 0) { throw 'SSH host verification or login failed.' }
$sshArgs = @('-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o',"UserKnownHostsFile=$KnownHostsPath",
             '-o','IdentitiesOnly=yes','-i',$KeyPath,'-p',"$SshPort","root@$SshHost")

Write-Host 'Verifying the copied bundle and CUDA environment. This does not load Llama 2.'
& $ssh @sshArgs 'bash /workspace/trace-portable/app/portable_setup.sh --check-only'
if ($LASTEXITCODE -ne 0) { throw 'Bundle verification failed. No credentials were transferred.' }

# The complete model cache is copied, so no Hugging Face credential is needed.
# Use the existing local DPAPI-protected TRACE API token or create one once.
$apiPath = $ApiTokenPath
if (Test-Path -LiteralPath $apiPath) {
    $encrypted = [IO.File]::ReadAllBytes($apiPath)
    $plain = [Security.Cryptography.ProtectedData]::Unprotect(
        $encrypted, $null, [Security.Cryptography.DataProtectionScope]::CurrentUser)
    $apiToken = [Text.Encoding]::UTF8.GetString($plain)
} else {
    $random = [byte[]]::new(48)
    $generator = [Security.Cryptography.RandomNumberGenerator]::Create()
    try { $generator.GetBytes($random) } finally { $generator.Dispose() }
    $apiToken = [Convert]::ToBase64String($random).TrimEnd('=').Replace('+','-').Replace('/','_')
    $encrypted = [Security.Cryptography.ProtectedData]::Protect(
        [Text.Encoding]::UTF8.GetBytes($apiToken), $null,
        [Security.Cryptography.DataProtectionScope]::CurrentUser)
    [IO.File]::WriteAllBytes($apiPath, $encrypted)
}
if ($apiToken.Length -lt 32 -or $apiToken.Contains("`n")) { throw 'The protected API token is invalid.' }
try {
    $apiToken | & $ssh -T @sshArgs 'umask 077; mkdir -p /root/.config/trace; chmod 700 /root/.config/trace; cat > /root/.config/trace/api_token; chmod 600 /root/.config/trace/api_token'
    if ($LASTEXITCODE -ne 0) { throw 'Secure API credential installation failed.' }
} finally {
    $apiToken = $null
    $plain = $null
}

Write-Host 'Registering the verified TRACE service without loading the GPU model.'
& $ssh @sshArgs 'bash /workspace/trace-portable/app/portable_setup.sh'
if ($LASTEXITCODE -ne 0) { throw 'Remote TRACE setup failed.' }
Write-Host 'Setup complete. Run start-vast-portable.ps1 when you want to use the dashboard.'
