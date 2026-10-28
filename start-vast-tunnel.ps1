param(
    [string]$SshHost = '58.8.185.95',
    [ValidateRange(1,65535)][int]$SshPort = 28797,
    [string]$KeyPath = "$env:USERPROFILE\.ssh\id_ed25519",
    [ValidateRange(1024,65535)][int]$LocalPort = 18701
)
$ErrorActionPreference = 'Stop'
$ssh = "$env:WINDIR\System32\OpenSSH\ssh.exe"
if (-not (Test-Path -LiteralPath $ssh)) { throw 'Windows OpenSSH client was not found.' }
if (-not (Test-Path -LiteralPath $KeyPath)) { throw "SSH private key was not found at $KeyPath" }
$listener = Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort $LocalPort -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($listener) {
    $owner = Get-Process -Id $listener.OwningProcess -ErrorAction SilentlyContinue
    if (-not $owner -or $owner.ProcessName -ne 'ssh') {
        throw "Local port $LocalPort is occupied by another process."
    }
    Write-Host "SSH tunnel already listening on 127.0.0.1:$LocalPort (PID $($owner.Id))."
    exit 0
}
Write-Host "Forwarding 127.0.0.1:$LocalPort to the Vast loopback API. Keep this window open."
Write-Host 'Confirm the host fingerprint against the Vast connection details before accepting it.'
& $ssh -N -L "127.0.0.1:$($LocalPort):127.0.0.1:8000" -i $KeyPath -o IdentitiesOnly=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -p $SshPort "root@$SshHost"
if ($LASTEXITCODE -ne 0) { throw "SSH tunnel exited with code $LASTEXITCODE" }
