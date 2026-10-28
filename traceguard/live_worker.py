"""Private persistent CUDA worker; JSON-lines pipes, never a network listener."""
import contextlib
import hashlib
import json
import os
import sys
import time

from .engineering import sha, read_json
from .robustness_protocol import checkpoint_identity, MODEL, REVISION, GENERATION
from .runtime import resource_snapshot, system_memory, environment_report

MAX_INPUT_TOKENS = 256
MAX_PROMPT_CHARACTERS = 2000


def check_prompt(value):
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_PROMPT_CHARACTERS:
        raise ValueError('Enter a nonempty prompt of at most 2000 characters.')
    return value  # Preserve exactly what the user supplied, including whitespace.


def emit(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def check_input_tokens(count, context_limit):
    if count > MAX_INPUT_TOKENS or count+32 > context_limit:
        raise ValueError('Prompt exceeds the 256-token input limit after the native chat template. Shorten it; no silent truncation is applied.')


def main():
    torch = device = None
    try:
        start = time.perf_counter()
        before = resource_snapshot()
        if system_memory()['available_bytes'] < 2.75 * 1024**3:
            raise RuntimeError('Insufficient available RAM: CUDA worker startup requires 2.75 GiB; close unused applications and restart the live server.')
        clean = read_json('artifacts/lora-trajectory/clean-checkpoint-before.json')
        identity = checkpoint_identity('clean_reference')
        if sha(clean['path']) != clean['sha256']:
            raise ValueError('Pinned clean checkpoint weights changed.')
        # Load the original pinned base checkpoint directly; no PEFT/LoRA wrapper.
        with contextlib.redirect_stdout(sys.stderr):
            from .pilot import load_base
            import torch as torch_module
            if not torch_module.cuda.is_available():
                raise RuntimeError('CUDA unavailable in .venv-train; no CPU or synthetic fallback.')
            if torch_module.cuda.mem_get_info()[0] < 2.75 * 1024**3:
                raise RuntimeError('Insufficient free VRAM: need 2.75 GiB before loading the pinned checkpoint.')
            torch, device, tokenizer, base = load_base()
            model = base
            torch = torch_module
            model.requires_grad_(False); model.eval()
        if any(p.requires_grad for p in model.parameters()) or any(p.device != device for p in model.parameters()):
            raise RuntimeError('Live model must be frozen and entirely on the requested CUDA device.')
        template_sha = hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
        ready = {'type':'ready', 'process_id':os.getpid(), 'checkpoint_identity':identity,
                 'adapter_loaded':False,
                 'original_clean_sha256':clean['sha256'],
                 'model':MODEL, 'revision':REVISION,
                 'chat_template_sha256':template_sha,
                 'startup_seconds':time.perf_counter()-start, 'resources_before_load':before,
                 'resources':resource_snapshot(torch,device), 'environment':environment_report(torch)}
        emit(ready)
        from .collect import generate_tokens, verify_alignment, verify_replay_alignment
        for line in sys.stdin:
            request = None
            try:
                request = json.loads(line); prompt = check_prompt(request['prompt'])
                if system_memory()['available_bytes'] < 512*1024**2:
                    raise RuntimeError('RAM safeguard: less than 512 MiB available; generation refused.')
                inputs = tokenizer.apply_chat_template([{'role':'user','content':prompt}], tokenize=True,
                    add_generation_prompt=True, return_tensors='pt', return_dict=True)
                count = inputs['input_ids'].shape[1]
                check_input_tokens(count,model.config.max_position_embeddings)
                inputs = {k:v.to(device) for k,v in inputs.items()}
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device); tick = time.perf_counter()
                generated = generate_tokens(torch,model,inputs,32,{tokenizer.eos_token_id})
                torch.cuda.synchronize(device); generation_seconds = time.perf_counter()-tick
                tick = time.perf_counter()
                full = verify_alignment(torch,model,inputs,generated,raise_on_failure=False)
                replay = verify_replay_alignment(torch,model,inputs,generated)
                torch.cuda.synchronize(device)
                response = tokenizer.decode(generated['generated_token_ids'],skip_special_tokens=True)
                emit({'type':'result','id':request['id'],'response':response,
                      **{k:generated[k] for k in ('generated_token_ids','token_confidences','token_entropies','statistics_source')},
                      'generation_configuration':{**GENERATION,'input_token_count':count,'finish_reason':generated['finish_reason']},
                      'generation_seconds':generation_seconds,'alignment_seconds':time.perf_counter()-tick,
                      'alignment_verification':{'passed':replay['passed'],'cached_replay':replay,'full_sequence':full,
                                               'full_sequence_rounding_warning':not full['passed']},
                      'resources':resource_snapshot(torch,device)})
                del inputs, generated
            except Exception as error:
                oom = torch is not None and isinstance(error,torch.cuda.OutOfMemoryError)
                if oom: torch.cuda.empty_cache()
                emit({'type':'error','id':request.get('id') if isinstance(request,dict) else None,
                      'code':'cuda_out_of_memory' if oom else 'invalid_input' if isinstance(error,ValueError) else 'inference_error',
                      'error':('CUDA out of memory. No result was substituted. Shorten the prompt or free GPU memory.' if oom else str(error))[:800]})
    except Exception as error:
        emit({'type':'startup_error','error':str(error)[:800]})
        raise SystemExit(1)


if __name__ == '__main__': main()
