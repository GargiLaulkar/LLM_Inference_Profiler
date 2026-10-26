# M3 Task 1: GPU & Deep-Dive Environment Verification

## 1. Overview & Objective
As part of **Project 6: Inference Cost and Latency Profiler for LLM Serving Stacks**, Role M3 (GPU & Deep-Dive Lead) is responsible for GPU telemetry collection, kernel-level profiling (`torch.profiler`, Nsight Systems), and VRAM memory breakdown.

Because local development occurs on a Windows machine without a dedicated NVIDIA GPU (Intel HD Graphics), the telemetry collector [`llm_profiler/collectors/gpu_hello.py`](file:///D:/AI-DS/SEM-5/OS/OS_project/LLM_Inference_Profiler/llm_profiler/collectors/gpu_hello.py) is architected to:
1. **Operate standalone**: Zero intra-repository dependencies; requires only Python standard library and `pynvml` (`nvidia-ml-py`).
2. **Degrade gracefully on non-NVIDIA hosts**: Checks NVML initialization, reports missing drivers/hardware without crashing, and exits with code `0`.
3. **Plug into shared Prometheus contracts**:
   - `llmprof_gpu_utilization_percent{gpu="<index>"}`
   - `llmprof_gpu_memory_used_bytes{gpu="<index>"}`
   - `llmprof_gpu_power_watts{gpu="<index>"}`

---

## 2. Remote GPU Execution Guide (Kaggle / Colab)

When transitioning to an NVIDIA GPU runtime (e.g. Google Colab T4, Kaggle NVIDIA T4 x2, or a teammate's GPU workstation):

### In a Colab / Kaggle Notebook Cell:
```bash
# 1. Install nvidia-ml-py in notebook environment
!pip install nvidia-ml-py

# 2. Run the standalone script
!python gpu_hello.py --samples 5 --interval 1.0

# 3. Export a markdown summary report directly
!python gpu_hello.py --out gpu_env_report.md
```

---

## 3. GPU Hardware & Environment Verification Results (Template)

*Fill in the values below after running `gpu_hello.py` on your remote GPU instance (e.g. Kaggle / Colab T4).*

### A. Host & Tooling Verification
| Tool / Component | Expected Remote Status | Observed Status / Version | Notes |
|---|---|---|---|
| OS Platform | Linux (Ubuntu 22.04 / 20.04) | *[e.g. Linux 6.1.x]* | Required for Docker and continuous batching |
| Python Version | 3.10 - 3.12 | *[e.g. 3.10.12]* | Tested runtime |
| `nvidia-smi` | OK | *[e.g. Driver 535.104.05]* | Base display and kernel driver |
| `nsys` (Nsight Systems) | OK or MISSING | *[e.g. 2023.4.1]* | Required for timeline trace deep-dives |
| `perf_event_paranoid` | <= 2 (ideally 1) | *[e.g. 1 or 2]* | Kernel counter access permission |
| PyTorch | OK | *[e.g. 2.1.2+cu121]* | `torch.cuda.is_available() == True` |
| Docker | OK | *[e.g. 24.0.5]* | For containerized profiler stack |

### B. GPU Hardware Specifications
| Field | Value (To Fill) | Unit / Description |
|---|---|---|
| GPU Index | `0` | Primary accelerator |
| Device Name | *[e.g. Tesla T4]* | NVIDIA Architecture (Turing) |
| Driver Version | *[e.g. 535.104.05]* | Host NVIDIA Display Driver |
| CUDA Driver Version | *[e.g. 12.2]* | Driver API CUDA capability |
| Total VRAM | *[e.g. 15102 MiB]* | Physical High-Bandwidth Memory |
| Idle VRAM Used | *[e.g. 0 to 500 MiB]* | Baseline before model weights load |
| Idle Power Draw | *[e.g. 9.5 W]* | Converted from milliwatts |
| Enforced Power Limit | *[e.g. 70.0 W]* | Thermal throttling upper bound |
| Idle Temperature | *[e.g. 38 °C]* | Thermal baseline |

### C. Live Sampling Snapshot (Idle Baseline)
| Sample # | GPU Index | Core Util (%) | VRAM Used (MiB) | Power Draw (W) |
|---|---|---|---|---|
| 1 | 0 | *[0.0]* | *[0.0]* | *[9.5]* |
| 2 | 0 | *[0.0]* | *[0.0]* | *[9.5]* |
| 3 | 0 | *[0.0]* | *[0.0]* | *[9.5]* |
| 4 | 0 | *[0.0]* | *[0.0]* | *[9.5]* |
| 5 | 0 | *[0.0]* | *[0.0]* | *[9.5]* |

---

## 4. Next Steps for M3
1. **Task 2**: Build the live Prometheus exporter daemon [`llm_profiler/collectors/gpu_metrics.py`](file:///D:/AI-DS/SEM-5/OS/OS_project/LLM_Inference_Profiler/llm_profiler/collectors/gpu_metrics.py) serving on port `9102`.
2. **Task 3**: Profile VRAM breakdown (Weights vs KV cache vs Activations) across sequence lengths.
3. **Task 4**: Capture Nsight Systems kernel traces under varying concurrency.
4. **Task 5**: Correlate GPU power draw and VRAM saturation with M4's cost per 1M tokens model.
