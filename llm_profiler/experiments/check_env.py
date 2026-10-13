#!/usr/bin/env python3
"""Environment and GPU capability check for LLM Inference Profiler."""

import os
import shutil
import subprocess
import sys


def check_env():
    print("=" * 60)
    print("LLM Inference Profiler - Environment Verification")
    print("=" * 60)
    print(f"Python Version: {sys.version.split()[0]} ({sys.executable})")

    # 1. nvidia-smi verification
    nvidia_smi = shutil.which("nvidia-smi")
    gpu_detected = False
    gpu_name = ""
    driver_version = ""
    memory_total = ""

    if nvidia_smi:
        try:
            res = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            lines = [line.strip() for line in res.stdout.strip().splitlines() if line.strip()]
            if lines:
                parts = [p.strip() for p in lines[0].split(",")]
                if len(parts) >= 3:
                    gpu_name = parts[0]
                    driver_version = parts[1]
                    memory_total = parts[2]
                    gpu_detected = True
        except Exception as e:
            # Fallback to standard nvidia-smi execution check
            try:
                res = subprocess.run([nvidia_smi], capture_output=True, text=True)
                if res.returncode == 0 and "NVIDIA-SMI" in res.stdout:
                    gpu_detected = True
                    gpu_name = "NVIDIA GPU (parsed from standard output)"
            except Exception:
                pass

    if not gpu_detected:
        print(
            "\n[CRITICAL ERROR] No NVIDIA GPU found on this system.",
            file=sys.stderr,
        )
        print(
            "nvidia-smi is either missing or returned no active GPU devices.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"NVIDIA GPU:     {gpu_name}")
    print(f"Driver Version: {driver_version}")
    print(f"Total VRAM:     {memory_total} MiB")

    # 2. PyTorch CUDA capability check
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        print(f"torch.cuda.is_available(): {cuda_available}")
        if cuda_available:
            print(
                f"Torch Device:   {torch.cuda.get_device_name(0)} (Device count: {torch.cuda.device_count()})"
            )
        else:
            print("Torch CUDA:     torch is installed but CUDA is not available")
    except ImportError:
        print("torch.cuda.is_available(): torch not installed in this environment")

    # 3. Docker availability check
    docker_bin = shutil.which("docker")
    if docker_bin:
        try:
            d_res = subprocess.run(
                ["docker", "--version"], capture_output=True, text=True, timeout=5
            )
            if d_res.returncode == 0:
                print(f"Docker:         Available ({d_res.stdout.strip()})")
            else:
                print("Docker:         CLI found but command exited with error")
        except Exception:
            print("Docker:         CLI found but timed out")
    else:
        print("Docker:         Not installed")

    print("=" * 60)
    print("Environment check PASSED: NVIDIA GPU is operational.")
    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    check_env()
