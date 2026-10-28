# Independent file copies preserve both existing environments. No hard links.
$ErrorActionPreference = 'Stop'
$traceRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $traceRoot
$traceSource = Join-Path $traceRoot '.venv-gpu'
$traceDestination = Join-Path $traceRoot '.venv-train'
if (Test-Path -LiteralPath $traceDestination) { throw '.venv-train already exists; refusing to overwrite it.' }
$tracePython = Join-Path $traceSource 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $tracePython)) { throw 'First prepare and verify .venv-gpu.' }
$traceBasePython = & $tracePython -c 'import sys; print(sys._base_executable)'
if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect the inference interpreter.' }
& $traceBasePython -m venv --without-pip $traceDestination
if ($LASTEXITCODE -ne 0) { throw 'Cannot create the independent training environment.' }
robocopy (Join-Path $traceSource 'Lib\site-packages') (Join-Path $traceDestination 'Lib\site-packages') /E /COPY:DAT /DCOPY:DAT /R:0 /W:0 /NFL /NDL /NP /NJH /NJS /XD __pycache__
if ($LASTEXITCODE -ge 8) { throw 'Dependency copy failed. Do not use this incomplete environment.' }
$traceTrainingPython = Join-Path $traceDestination 'Scripts\python.exe'
& $traceTrainingPython -m pip --isolated install -r requirements-training.txt --no-deps --index-url https://pypi.org/simple --no-cache-dir
if ($LASTEXITCODE -ne 0) { throw 'PEFT installation failed.' }
& $traceTrainingPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Training dependency check failed.' }
Write-Output 'Use .venv-train\Scripts\python.exe -m ...; console-script wrappers are intentionally not copied.'
