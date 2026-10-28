**MTBA update (2026-10-20):** The active notebook has loaded the pinned MTBA adapter on a Colab Tesla T4 and generated real responses through the local dashboard. The paired clean-base pilot is still pending. Run the [prepared Colab control cell](MTBA_CLEAN_CONTROL_COLAB.md) in the same connected runtime; see [the pilot report](MTBA_PILOT_RESULTS.md). Historical Sleeper results remain separate.

# Llama 2 + BackdoorLLM in Google Colab

This setup uses Colab as a temporary GPU inference host. The local TRACE dashboard sends a prompt to a bearer-authenticated API in the notebook, receives the actual model response and aligned generation statistics, validates those fields, and saves a separate live interaction record. The Llama 2 detector is not trained, so the dashboard deliberately shows no Llama 2 prediction.

The notebook pins the base and adapter revisions already recorded for this project. It uses 4-bit NF4 loading with FP16 compute, a deterministic maximum of 256 new tokens, and the BackdoorLLM Alpaca instruction format. It captures full-vocabulary probabilities in Colab but returns only generated IDs, token confidences, token entropies, aggregate features, and resource metadata. Full logits are not sent to the laptop or saved.

## Start the Colab model

1. In Google Colab, choose **File > Upload notebook** and select [the TRACE Colab API notebook](trace_llama2_colab_api.ipynb) from `D:\TRACE\docs`. Then choose **Runtime > Change runtime type > GPU**.
2. Verify your Hugging Face account has been approved for `meta-llama/Llama-2-7b-chat-hf`. In Colab's **Secrets** panel, add `HF_TOKEN` with a read token that has access to that gated model. Hugging Face requires both model access and authentication to download gated files ([gated model access](https://huggingface.co/docs/hub/models-gated)).
3. Generate a separate random API secret in PowerShell on the laptop:

   ```powershell
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

   Add the exact value as a second Colab Secret named `TRACE_API_TOKEN`. Do not put either secret in notebook source or send it in chat.
4. Run the notebook cells in order. Wait for the model-load cell to finish. The final cell prints a temporary `https://<random-name>.trycloudflare.com` URL.
5. Keep the Colab tab connected and the final tunnel cell running. Quick Tunnels are for development/testing and provide no uptime guarantee ([Cloudflare Quick Tunnels](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/)). Colab runtimes also have variable GPU availability and lifetime ([Colab FAQ](https://research.google.com/colaboratory/faq.html)).

## Open the local dashboard

From PowerShell in `D:\TRACE`, start:

```powershell
.\start-colab.ps1
```

When prompted, paste the Colab tunnel URL. Then paste the same `TRACE_API_TOKEN` value; input is hidden. Open **http://127.0.0.1:8769**. Enter a harmless prompt and click **Run Llama 2 (signals)**. The model response and generation features should return through the local TRACE process. **Export interaction** saves the returned record in the dashboard and downloads a JSON copy.

The token stays in the local Python process and is never sent to browser JavaScript. The notebook checks it on every `/health` and `/generate` request; it does not log request bodies or authorization headers. The tunnel URL is public, but requests without the random bearer token are rejected. Do not share the token or make the notebook public.

If the notebook disconnects, the temporary URL stops working. Reconnect the GPU, rerun the cells, and use the new URL. The launcher can retry the connection check. The local server cannot restart Colab for you.

## What the result means

- The response is produced by the pinned Llama 2 base with the BackdoorLLM MTBA adapter, when access and GPU loading succeed.
- Per-token confidence is the probability assigned to the token actually generated. Entropy comes from the complete vocabulary distribution for that generation step; only compact statistics cross the tunnel.
- This call does not verify an activation label or run a detector. Trigger presence is not treated as successful activation. Two distinct triggers and attack success remain to be independently verified before detector training or final thesis claims.
- Historical TinyLlama records remain in the lower dashboard as archive material. The TinyLlama detector is not applied to Llama 2.
- Prompts leave the laptop and are processed in the Colab runtime. Use only prompts appropriate for this research demo.
