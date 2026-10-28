# Moving TRACE to another Vast.ai GPU

The current copy source is `/workspace/trace-portable/` on Vast instance **#57673504**. It contains the exact pinned Llama 2 7B Chat checkpoint, BackdoorLLM MTBA adapter, inference server, Python 3.12 package files, and a SHA-256 manifest for the 16 required model/server files. Its logical size is about 19 GiB. It contains **no API token, Hugging Face token, SSH private key, Windows review file, or detector artifact**. The Windows TRACE project and frozen detector remain in `D:\TRACE`.

The source was checked on an RTX 3060 12 GB Vast PyTorch image with Python 3.12, CUDA PyTorch 2.14.1+cu130, and NF4 support. Use a new instance with a compatible Linux x86-64 Vast PyTorch image, Python 3.12 at `/venv/main/bin/python`, supervisor, CUDA/NF4 support, **at least 12 GB VRAM**, and enough disk for the 19 GiB bundle plus its image and temporary copy space. A different image or GPU is checked during setup; compatibility is not assumed.

## On a new instance

1. Add the existing Windows **public** SSH key to the new instance. In Vast.ai, use **Copy** on source instance #57673504, then choose the destination instance. Select source `/workspace/trace-portable/` (including the final slash) and destination `/workspace/trace-portable/`. The equivalent CLI path pair is `C.57673504:/workspace/trace-portable/` to `C.NEW_ID:/workspace/trace-portable/`. Wait for the copy to finish and check that `manifest.json` exists on the destination. [Vast documents instance-to-instance copy from stopped/inactive Docker instances](https://docs.vast.ai/guides/instances/storage/data-movement); storage must still exist, and bandwidth charges may apply. A destroyed instance is not a backup. **Copy and verify before destroying #57673504.** Vast storage continues to incur charges while stopped.
2. In a PowerShell window at `D:\TRACE`, use the **new** SSH host and port from Vast's Connect panel:

   ```powershell
   .\setup-vast-portable.ps1 -SshHost NEW_HOST -SshPort NEW_PORT
   ```

   Confirm the new host fingerprint when SSH asks. Setup verifies the full checkpoint hashes, pinned package versions, CUDA and NF4 operation, transfers the locally protected TRACE API token through SSH, and registers the service. It does not load the 7B model or need a Hugging Face download.
3. Start it from the same folder:

   ```powershell
   .\start-vast-portable.ps1 -SshHost NEW_HOST -SshPort NEW_PORT
   ```

   Keep this PowerShell window open. Open `http://127.0.0.1:8769`. The startup command checks the exact model and adapter revisions and opens an authenticated local-only SSH tunnel. The tunnel reconnects automatically if SSH drops. New live prompts return real responses and generation signals; the local frozen Llama 2 TRACE detector gives a risk score with **unknown live ground truth**.

The source bundle's snapshot files use hard links only to save disk space *on the source*. Vast's instance-to-instance copy must transfer their actual file contents to the new instance. Setup will reject missing, broken, changed, or externally linked checkpoint files. The Python packages are copied, but a compatible Vast Python/CUDA image is still required. On #57673504, the bundle passed all 16 pinned-file hashes, the Python/CUDA/NF4 check, a real seven-token Llama 2 + MTBA inference, managed-service activation, and a second local dashboard startup with its frozen detector ready. **An actual copy to a different Vast instance has not yet been tested.** Keep a copy of the local Windows project because the Vast bundle does not contain the thesis dataset, reviews, or frozen detector.

If the source instance has already been destroyed, the pinned checkpoint and research records on the Windows machine can be used for a slower rebuild; the deleted Vast storage itself cannot be recovered. Stopping an instance stops its processes, so on restart run the startup command again.
