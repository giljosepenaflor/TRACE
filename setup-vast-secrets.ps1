param(
    [string]$SshHost = '58.8.185.95',
    [ValidateRange(1,65535)][int]$SshPort = 28797,
    [string]$KeyPath = "$env:USERPROFILE\.ssh\id_ed25519"
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Security
$ssh = "$env:WINDIR\System32\OpenSSH\ssh.exe"
$hfPath = "$env:USERPROFILE\.cache\huggingface\token"
$localApiPath = "$env:USERPROFILE\.ssh\trace_vast_api_token.dpapi"
if (-not (Test-Path -LiteralPath $ssh)) { throw 'Windows OpenSSH client was not found.' }
if (-not (Test-Path -LiteralPath $KeyPath)) { throw "SSH key was not found at $KeyPath" }
if (-not (Test-Path -LiteralPath $hfPath)) { throw 'No existing local Hugging Face login token was found.' }

# The Hugging Face token is read from the user's existing login, never placed in
# a command-line argument or printed. SSH carries it to a root-only remote file.
$hfToken = [IO.File]::ReadAllText($hfPath).Trim()
if ($hfToken.Length -lt 10 -or $hfToken.Contains("`n")) { throw 'The local Hugging Face token file is invalid.' }
$hfToken | & $ssh -T -o BatchMode=yes -i $KeyPath -p $SshPort "root@$SshHost" 'umask 077; mkdir -p /root/.config/trace; chmod 700 /root/.config/trace; cat > /root/.config/trace/hf_token; chmod 600 /root/.config/trace/hf_token'
if ($LASTEXITCODE -ne 0) { throw 'Failed to install the Hugging Face credential on Vast.' }
$hfToken = $null

# Keep the API token stable across launches. Its local copy is encrypted with
# Windows DPAPI for this user; Vast stores a root-only copy for supervisor restarts.
if (Test-Path -LiteralPath $localApiPath) {
    $encrypted = [IO.File]::ReadAllBytes($localApiPath)
    $plain = [System.Security.Cryptography.ProtectedData]::Unprotect(
        $encrypted, $null, [System.Security.Cryptography.DataProtectionScope]::CurrentUser)
    $apiToken = [Text.Encoding]::UTF8.GetString($plain)
} else {
    $random = [byte[]]::new(48)
    $generator = [Security.Cryptography.RandomNumberGenerator]::Create()
    try { $generator.GetBytes($random) } finally { $generator.Dispose() }
    $apiToken = [Convert]::ToBase64String($random).TrimEnd('=').Replace('+','-').Replace('/','_')
    $encrypted = [System.Security.Cryptography.ProtectedData]::Protect(
        [Text.Encoding]::UTF8.GetBytes($apiToken), $null,
        [System.Security.Cryptography.DataProtectionScope]::CurrentUser)
    [IO.File]::WriteAllBytes($localApiPath, $encrypted)
}
if ($apiToken.Length -lt 32 -or $apiToken.Contains("`n")) { throw 'The API credential is invalid.' }
$apiToken | & $ssh -T -o BatchMode=yes -i $KeyPath -p $SshPort "root@$SshHost" 'umask 077; mkdir -p /root/.config/trace; chmod 700 /root/.config/trace; cat > /root/.config/trace/api_token; chmod 600 /root/.config/trace/api_token'
if ($LASTEXITCODE -ne 0) { throw 'Failed to install the TRACE API credential on Vast.' }
$apiToken = $null
Write-Host 'Credentials installed over SSH. Local API token is DPAPI-encrypted; remote files are root-only.'
