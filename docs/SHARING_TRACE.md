# Run TRACE on a collaborator's Windows computer

The repository ignores `artifacts/` and `model-cache/`. Cloning it alone cannot
start the current dashboard. The sharing ZIP supplies the existing frozen files,
the pinned MiniLM encoder and matching local application source. Llama 2 stays
on Vast. No retraining or new experimental results are involved.

## Prepare the ZIP on the original computer

From the project root:

```powershell
.\.venv\Scripts\python.exe -B scripts/build_shared_runtime.py
```

Send `outputs/trace-share/TRACE-friend-runtime.zip` privately to your collaborator.
The archive contains historical harmless dashboard records and the frozen MTBA
classifier/protocol, but no MTBA response-review dataset, live conversation
history, API token, SSH private key, DPAPI credential or Llama 2 weights.

## On the collaborator's computer

1. Stop any local TRACE dashboard. Extract the ZIP **into the existing TRACE
   project root**, merging its folders and accepting replacement of bundled
   application files. Do not create a second nested `TRACE` directory. For the
   current collaborator, the result should include:
   `C:\Users\vneil\Documents\GitHub\TRACE\artifacts\trace-robustness\run-001\bundle.json`.
   The ZIP's `START-HERE.txt` repeats these instructions.

2. Open PowerShell in that project root. If `.venv` already exists, use it. If
   not, create a fresh local environment (do not copy somebody else's `.venv`):

   ```powershell
   py -3.13 -m venv .venv
   ```

   Install the dependencies, then verify the local transfer:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements-models.txt
   .\.venv\Scripts\python.exe -B scripts/verify_shared_runtime.py
   ```

   Dependency installation requires internet. The verification uses only local
   files; it checks archive hashes and frozen artifact compatibility, and loads
   the MiniLM encoder on a harmless fixture without running a target LLM.

3. The instance owner must authorize the collaborator's **public** SSH key on
   the existing Vast instance. In a separate PowerShell window, use its current
   SSH host/port. The connection details below were supplied for this instance;
   recheck them in Vast's Connect panel if the instance has moved:

   ```powershell
   ssh -i "$env:USERPROFILE\.ssh\id_ed25519" -o IdentitiesOnly=yes -p 41695 -L 127.0.0.1:18701:127.0.0.1:8000 root@122.59.250.166
   ```

   Enter the SSH key passphrase if prompted. Confirm a new server fingerprint
   with the instance owner before trusting it. Keep this window open. Once at
   the remote Linux prompt, the authorized user can retrieve the existing
   token with `cat /root/.config/trace/api_token`. Copy it privately.

4. Back in the local project PowerShell window, launch:

   ```powershell
   .\start-shared-vast.ps1
   ```

   At `TRACE_API_TOKEN (input hidden):`, paste the existing remote token and
   press Enter. Characters are intentionally invisible. This is different from
   the SSH public key, the SSH passphrase and the Vast account API key. The
   shared launcher prompts instead of decrypting the original user's DPAPI file.
   Open **http://127.0.0.1:8769**. Keep both PowerShell windows open.

If PowerShell blocks the launcher under its execution policy, invoke it with a
process-scoped policy without changing the saved machine settings:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\start-shared-vast.ps1
```

## What remains unchanged

The original frozen detector bytes, projection, scaler and thresholds are
unchanged. The live loader can locate the same hash-verified MiniLM snapshot
under the collaborator's project path when the saved original path is missing.
Chat remains experimental and unvalidated; sharing the runtime does not create
new activation labels or research results. The current inference service permits
one generation at a time, so collaborators may need to wait for another request.
