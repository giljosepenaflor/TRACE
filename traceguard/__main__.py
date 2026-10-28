import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="TRACE thesis research prototype")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Create synthetic fixtures and train a demonstration detector")
    demo.add_argument("--output", default="artifacts/demo")
    train = commands.add_parser("train", help="Train a separate detector from verified real interaction JSONL")
    train.add_argument("--data", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--allow-download", action="store_true", help="Allow sentence encoder download")
    serve = commands.add_parser("serve", help="Open the local monitoring app")
    serve.add_argument("--run", default="artifacts/demo")
    serve.add_argument("--data", default="data/local/demo.jsonl")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument('--live', action='store_true', help='Legacy TinyLlama live mode with its model-specific frozen F0 detector')
    serve.add_argument('--live-llama2', action='store_true', help='Run the local BackdoorLLM Llama 2 adapter in the live panel; no Llama 2 detector is applied')
    serve.add_argument('--live-llama2-colab', action='store_true', help='Relay live prompts through an authenticated Google Colab GPU and score with the frozen Llama 2 detector')
    serve.add_argument('--live-llama2-vast', action='store_true', help='Relay live prompts through a local Vast.ai SSH tunnel and score with the frozen Llama 2 detector')
    serve.add_argument('--live-llama2-remote', action='store_true', help='Use TRACE_REMOTE_BACKEND=colab|vast for the authenticated remote GPU and frozen Llama 2 detector')
    collect = commands.add_parser("collect", help="Collect clean reference interactions from a local checkpoint")
    collect.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    collect.add_argument("--prompts", default="examples/clean-prompts.jsonl")
    collect.add_argument("--output", default="data/local/clean-reference.jsonl")
    collect.add_argument("--max-new-tokens", type=int, default=32)
    collect.add_argument("--allow-download", action="store_true")
    collect.add_argument("--device", choices=("cpu", "cuda", "cuda:0"), default="cpu")
    collect.add_argument("--dtype", choices=("auto", "float16", "float32"), default="auto")
    collect.add_argument("--revision", default="main", help="Checkpoint revision; resolved SHA is recorded")
    collect.add_argument("--cache-dir", help="Optional workspace model cache")
    collect.add_argument("--verify-alignment", action="store_true", help="Audit statistics with an independent teacher-forced pass")
    collect.add_argument("--report", help="Collection summary JSON path")
    collect.add_argument("--overwrite", action="store_true", help="Explicitly replace collection output")
    collect.add_argument("--control-config", help="JSON configuration for paired clean-reference cue controls")
    doctor = commands.add_parser("doctor", help="Inspect runtime and execute a tensor operation")
    doctor.add_argument("--device", choices=("cpu", "cuda", "cuda:0"), default="cuda")
    doctor.add_argument("--report", help="Save diagnostics JSON")
    args = parser.parse_args()
    try:
        if args.command in ("train", "demo"):
            from .detector import evaluate, fit_detector, save_run
            from .records import read_records, write_records
            if args.command == "demo":
                from .demo import demo_records
                records = demo_records()
                write_records("data/local/demo.jsonl", records)
                encoder = "demo-hashing"
            else:
                records = read_records(args.data)
                if records[0]["synthetic"]:
                    raise ValueError("Use the demo command for synthetic demonstrations; train expects real records.")
                if records[0].get('record_purpose') in ('engineering_pilot', 'engineering_robustness'):
                    raise ValueError('Use python -m traceguard.engineering fit/evaluate for the versioned engineering freeze/bundle workflow.')
                encoder = "sentence-transformer"
            artifact = fit_detector(records, encoder, local_only=not getattr(args, "allow_download", False))
            report = evaluate(artifact, records)
            save_run(args.output, artifact, report)
            print(f"Saved detector and report to {Path(args.output).resolve()}")
            if artifact["synthetic"]:
                print("SYNTHETIC DEMONSTRATION ONLY. No LLM or real backdoor detection was evaluated.")
            data_path = "data/local/demo.jsonl" if args.command == "demo" else args.data
            print(f'Start monitoring: python -m traceguard serve --run "{args.output}" --data "{data_path}"')
        elif args.command == "collect":
            from .collect import collect
            count = collect(args.model, args.prompts, args.output, args.allow_download, args.max_new_tokens,
                            device=args.device, dtype=args.dtype, revision=args.revision,
                            cache_dir=args.cache_dir, verify_token_alignment=args.verify_alignment,
                            report_path=args.report, overwrite=args.overwrite, control_config=args.control_config)
            print(f"Saved {count} clean interactions to {args.output}. Positive data is still required for training.")
        elif args.command == "serve":
            from .server import serve
            if args.live_llama2_remote:
                if args.live_llama2_colab or args.live_llama2_vast:
                    raise ValueError('Choose only one remote backend flag.')
                backend = os.environ.get('TRACE_REMOTE_BACKEND', 'colab').lower()
                if backend not in ('colab', 'vast'):
                    raise ValueError('TRACE_REMOTE_BACKEND must be colab or vast.')
                args.live_llama2_colab = backend == 'colab'
                args.live_llama2_vast = backend == 'vast'
            endpoint = token = None
            if args.live_llama2_colab or args.live_llama2_vast:
                from getpass import getpass
                if args.live_llama2_vast:
                    endpoint = os.environ.get("TRACE_REMOTE_URL", "http://127.0.0.1:18701")
                    print(f"Using Vast SSH tunnel endpoint: {endpoint}")
                else:
                    endpoint = os.environ.get("TRACE_REMOTE_URL") or input(
                        "Colab Cloudflare tunnel URL (https://....trycloudflare.com): ").strip()
                token = os.environ.get("TRACE_API_TOKEN") or getpass("TRACE_API_TOKEN (input hidden): ")
            serve(args.run, args.data, args.port, live=args.live,
                  live_llama2=args.live_llama2,
                  live_llama2_colab=args.live_llama2_colab,
                  live_llama2_vast=args.live_llama2_vast,
                  colab_endpoint=endpoint, colab_token=token)
        elif args.command == "doctor":
            from .runtime import doctor
            doctor(args.device, args.report)
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        parser.exit(1, f"TRACE: {error}\n")


if __name__ == "__main__":
    main()
