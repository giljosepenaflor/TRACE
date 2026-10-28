$ErrorActionPreference = 'Stop'
$workspace = Join-Path $env:USERPROFILE 'Desktop\TRACE'
$launcher = Join-Path $PSScriptRoot 'start-vast-portable.ps1'
$tokenPath = Join-Path $workspace '.restore-staging\trace-vast-58039092.dpapi'
if (Test-Path -LiteralPath $tokenPath) {
    try {
        Add-Type -AssemblyName System.Security
        $encrypted = [IO.File]::ReadAllBytes($tokenPath)
        $plain = [Security.Cryptography.ProtectedData]::Unprotect(
            $encrypted, $null, [Security.Cryptography.DataProtectionScope]::CurrentUser)
        $token = [Text.Encoding]::UTF8.GetString($plain)
        $health = Invoke-RestMethod -Uri 'http://127.0.0.1:18701/health' `
            -Headers @{Authorization="Bearer $token"} -TimeoutSec 5
        $identity = $health.checkpoint_identity
        $status = Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8769/api/live/retry' `
            -Body '{}' -ContentType 'application/json' -TimeoutSec 20
        if ($health.status -eq 'ok' -and $identity.adapter_loaded -eq $true -and
            $identity.model -eq 'meta-llama/Llama-2-7b-chat-hf' -and
            $identity.revision -eq 'f5db02db724555f92da89c216ac04704f23d4590' -and
            $identity.adapter -eq 'BackdoorLLM/Jailbreak_Llama2-7B_MTBA' -and
            $identity.adapter_revision -eq '8652ad60f1605605642c99fd680a15ecccf1b522' -and
            $status.state -eq 'ready') {
            Write-Host 'TRACE is already running: http://127.0.0.1:8769'
            return
        }
    } catch {
        # Let the standard launcher report occupied ports or connection errors.
    } finally {
        $token = $null
        $plain = $null
    }
}
& $launcher -SshHost '122.59.250.166' -SshPort 42147 `
    -KnownHostsPath (Join-Path $workspace '.vast_known_hosts') `
    -ApiTokenPath $tokenPath
