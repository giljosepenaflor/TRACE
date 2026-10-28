"""Local diagnostics shared by collection and the doctor command."""

import ctypes
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def system_memory():
    if sys.platform != "win32":
        return {"total_bytes": None, "available_bytes": None}

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in
            ("total", "available", "page", "free_page", "virtual", "free_virtual", "extended")]

    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("Windows could not report physical memory.")
    return {"total_bytes": status.total, "available_bytes": status.available}


def process_memory():
    if sys.platform != "win32":
        return {"working_set_bytes": None, "peak_working_set_bytes": None}

    class ProcessCounters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in
            ("peak_working_set", "working_set", "peak_paged", "paged", "peak_nonpaged", "nonpaged",
             "pagefile", "peak_pagefile")]

    counters = ProcessCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.windll.kernel32
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessCounters), ctypes.c_ulong]
    if not get_memory(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise OSError("Windows could not report process memory.")
    return {"working_set_bytes": counters.working_set,
            "peak_working_set_bytes": counters.peak_working_set}


def resource_snapshot(torch=None, device=None):
    snapshot = {"system_memory": system_memory(), "process_memory": process_memory()}
    if torch is not None and device is not None and device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        snapshot["cuda"] = {"free_bytes": free, "total_bytes": total,
                            "allocated_bytes": torch.cuda.memory_allocated(device),
                            "reserved_bytes": torch.cuda.memory_reserved(device),
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
    return snapshot


def environment_report(torch=None):
    from importlib.metadata import PackageNotFoundError, version
    versions = {}
    for package in ("torch", "transformers", "accelerate", "huggingface-hub", "safetensors", "numpy"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    disk = shutil.disk_usage(Path.cwd())
    report = {"python": platform.python_version(), "python_executable": sys.executable,
              "platform": platform.platform(), "packages": versions,
              "system_memory": system_memory(), "disk": {"path": str(Path.cwd()),
              "total_bytes": disk.total, "free_bytes": disk.free}}
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap,memory.total,memory.free",
                                 "--format=csv,noheader"], capture_output=True, text=True, timeout=10,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        report["nvidia_smi"] = {"returncode": result.returncode,
                                 "gpu_csv": result.stdout.strip(), "error": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        report["nvidia_smi"] = {"error": str(error)}
    if torch is not None:
        report["torch"] = {"version": torch.__version__, "cuda_build": torch.version.cuda,
                           "cuda_available": torch.cuda.is_available(),
                           "cuda_device_count": torch.cuda.device_count(),
                           "compiled_architectures": torch.cuda.get_arch_list()}
    return report


def resolve_device(torch, requested):
    if requested not in ("cpu", "cuda", "cuda:0"):
        raise ValueError("device must be cpu, cuda, or cuda:0; automatic fallback is not used.")
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA requested but unavailable (torch={torch.__version__}, "
                               f"CUDA build={torch.version.cuda}). Use a verified CUDA environment "
                               "or explicitly choose --device cpu.")
        return torch.device("cuda:0")
    return torch.device("cpu")


def doctor(device="cuda", report_path=None):
    import torch
    report = environment_report(torch)
    try:
        selected = resolve_device(torch, device)
        values = torch.arange(4, dtype=torch.float32, device=selected)
        result = values @ values
        if selected.type == "cuda":
            torch.cuda.synchronize(selected)
        actual = float(result.item())
        if actual != 14.0:
            raise RuntimeError(f"Tensor smoke test produced {actual} instead of 14.")
        report["tensor_test"] = {"status": "passed", "device": str(values.device),
                                  "operation": "arange(4) dot arange(4)", "result": actual}
        if selected.type == "cuda":
            report["tensor_test"].update({"gpu_name": torch.cuda.get_device_name(selected),
                                          "compute_capability": list(torch.cuda.get_device_capability(selected))})
    except (RuntimeError, ValueError) as error:
        report["tensor_test"] = {"status": "failed", "requested_device": device, "error": str(error)}
    if report_path:
        destination = Path(report_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    if report["tensor_test"]["status"] != "passed":
        raise RuntimeError(report["tensor_test"]["error"])
    return report
