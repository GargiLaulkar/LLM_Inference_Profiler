# Team Sync Meeting Template: Inference Profiler Integration

**Date / Time:** [Fill in: YYYY-MM-DD HH:MM]  
**Attendees:**  
- M1 (Platform & Load Lead): [Fill in name]
- Stage Timers Lead: [Fill in name]
- GPU Collectors Lead: [Fill in name]
- Analyzer & Dashboard Lead: [Fill in name]

---

## 1. Ownership & Roles Alignment

| Domain | Responsible Teammate | Contact / Notes |
|---|---|---|
| **Server Serving Stack** (vLLM / Server Lifecycle) | [Fill in: Member name] | [Fill in] |
| **Model Weight Management & Cache** (Qwen2.5-0.5B-Instruct) | [Fill in: Member name] | [Fill in] |
| **GPU Machine & Host Infrastructure Access** | [Fill in: Member name] | [Fill in] |
| **Stage Timers & Instrumentation Hooks** | [Fill in: Member name] | [Fill in] |
| **GPU Metrics Collection** (nvidia-smi, DCGM, Prometheus) | [Fill in: Member name] | [Fill in] |
| **Analyzer, Cost Modeling & Grafana Dashboards** | [Fill in: Member name] | [Fill in] |
| **Load Generator & Baseline Benchmarks (M1)** | [Fill in: Member name] | [Fill in] |

---

## 2. Infrastructure & Permissions Status

- [ ] **GPU Machine Access:** [Local machine / Shared Remote GPU / Cloud VM]
- [ ] **GPU Model & VRAM:** [e.g. RTX 3060 Laptop 6GB / Target Spec]
- [ ] **CUDA / Driver Version:** [e.g. Driver 592.82 / CUDA 13.1]
- [ ] **Nsight Systems / Profiling Permission Status:**
  - [ ] Non-root profiling permitted (`NVreg_RestrictProfilingToAdminUsers=0` configured)
  - [ ] Sudo / Admin access verified for hardware performance counters
  - [ ] Nsight CLI (`nsys`) installed and verified in PATH: [Yes / No / Pending]
- [ ] **Docker / Container Support:** [Installed / WSL2 Native / Direct Host]

---

## 3. Scope Split & Interface Contracts Verification

- [ ] **Shared Stage Names Agreed:**
  - Stages: `tokenization`, `queue_wait`, `prefill`, `decode`, `detokenization`, `network`, `total`
- [ ] **Request CSV Schema Enforced:**
  - Header: `request_id,send_ts,end_ts,total_latency_s,prompt_tokens,output_tokens,status,target_rps,scenario_tag`
- [ ] **Prometheus Exporter Port & Path:**
  - Metrics Endpoint: `http://<host>:<port>/metrics`
  - Scrape interval: `prometheus_scrape_interval_s` (default 5s)
- [ ] **Cost Assumptions:**
  - GPU price per hour: `$1.00 USD` (or negotiated rate)

---

## 4. Action Items & Next Milestones

| Action Item | Owner | Target Due Date |
|---|---|---|
| Hand off baseline B1 request logs for analyzer validation | M1 | [Fill in] |
| Merge stage timer middleware into server serving path | Stage Timers Lead | [Fill in] |
| Verify DCGM / Prometheus GPU telemetry scraping | GPU Collector Lead | [Fill in] |
| Validate Grafana dashboard import with live Prometheus feed | Dashboard Lead | [Fill in] |
