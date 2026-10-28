param(
    [string]$SshHost = '154.64.230.50',
    [int]$SshPort = 53576,
    [int]$LocalPort = 18701,
    [string]$KnownHostsPath = "$env:USERPROFILE\.ssh\known_hosts",
    [string]$KeyPath = "$env:USERPROFILE\.ssh\id_ed25519"
)

$ErrorActionPreference = 'Stop'
$ssh = Join-Path $env:WINDIR 'System32\OpenSSH\ssh.exe'
$key = $KeyPath
$knownHosts = $KnownHostsPath

if (-not (Test-Path -LiteralPath $ssh)) { throw 'Windows OpenSSH is unavailable.' }
if (-not (Test-Path -LiteralPath $key)) { throw 'Vast SSH key is unavailable.' }
if (-not (Test-Path -LiteralPath $knownHosts)) { throw 'Pinned Vast host key file is unavailable.' }

while ($true) {
    $sshArgs = @(
        '-N', '-L', "127.0.0.1:$($LocalPort):127.0.0.1:8000",
        '-o', 'BatchMode=yes',
        '-o', 'IdentitiesOnly=yes',
        '-o', 'StrictHostKeyChecking=yes',
        '-o', "UserKnownHostsFile=$($knownHosts.Replace('\', '/'))",
        '-o', 'ExitOnForwardFailure=yes',
        '-o', 'ConnectTimeout=10',
        '-o', 'ServerAliveInterval=10',
        '-o', 'ServerAliveCountMax=6',
        '-i', $key, '-p', $SshPort, "root@$SshHost"
    )
    & $ssh @sshArgs
    Start-Sleep -Seconds 5
}
