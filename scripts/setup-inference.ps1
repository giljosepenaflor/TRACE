# Creates a separate inference environment; does not install into the prototype .venv.
$ErrorActionPreference = 'Stop'
$traceRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $traceRoot
$traceInstaller = Join-Path $traceRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $traceInstaller)) {
    throw 'The existing TRACE .venv with pip is required by this setup script.'
}
if (-not (Test-Path -LiteralPath '.venv-gpu\Scripts\python.exe')) {
    py -3.13 -m venv --without-pip .venv-gpu
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the isolated Python environment.' }
}
if ((Get-Content -LiteralPath '.venv-gpu\pyvenv.cfg') -match '^include-system-site-packages = true') {
    throw '.venv-gpu must be isolated; include-system-site-packages must be false.'
}
& $traceInstaller -m pip --isolated --python .venv-gpu install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126 --no-cache-dir
if ($LASTEXITCODE -ne 0) { throw 'Official CUDA PyTorch installation failed.' }
& $traceInstaller -m pip --isolated --python .venv-gpu install pip==26.1.2 -r requirements-inference.txt --index-url https://pypi.org/simple --no-cache-dir
if ($LASTEXITCODE -ne 0) { throw 'Inference dependency installation failed.' }
& '.\.venv-gpu\Scripts\python.exe' -B -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Inference dependencies have incompatible requirements.' }
& '.\.venv-gpu\Scripts\python.exe' -B -m traceguard doctor --device cuda --report artifacts/real-smoke/cuda-environment.json
if ($LASTEXITCODE -ne 0) { throw 'CUDA tensor verification failed. No CPU fallback was attempted.' }
