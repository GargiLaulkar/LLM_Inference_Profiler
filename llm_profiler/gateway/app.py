"""FastAPI Gateway for vLLM with Stage Timing, Trace Logging, and Bottleneck Injection Hooks.

Shared Contract & Architecture:
- Stage names: tokenization, queue_wait, prefill, decode, detokenization, network, total.
  * Note on prefill: Measured as time from dispatch to the first streamed chunk, which
    includes vLLM's internal queueing and scheduling latency.
  * Note on detokenization: vLLM returns text chunks directly and does not expose completion
    token IDs per-request; therefore detokenization is not separately measurable and is omitted
    from stage_durations rather than populated with synthetic values.
  * Note on network: Measured as gateway-to-vLLM connection/transport latency; client-to-gateway
    network time is not visible to the gateway.
  * Note on queue_wait: Enforced via asyncio.Semaphore(1) at the gateway when LLMPROF_NO_BATCHING=1.
    While vLLM may still batch requests internally if reached concurrently, this gateway gate
    forces sequential execution and measures true queue wait time.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
from fastapi import FastAPI, Header, Request, Response
from fastapi.responses import JSONResponse
import yaml

logger = logging.getLogger("llm_profiler.gateway")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def load_gateway_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from config.yaml or fallbacks."""
    candidates = [
        Path(config_path) if config_path else None,
        Path("config.yaml"),
        Path(__file__).resolve().parents[2] / "config.yaml",
        Path(__file__).resolve().parents[1] / "config.yaml",
    ]
    for c in candidates:
        if c and c.exists():
            try:
                with c.open("r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                if isinstance(cfg, dict):
                    return cfg
            except Exception as e:
                logger.warning("Error reading config candidate %s: %s", c, e)
    return {}


class TokenizerHelper:
    """Tokenizer supporting local AutoTokenizer with fallback to vLLM /tokenize endpoint."""

    def __init__(self, model_name: str, vllm_base_url: str):
        self.model_name = model_name
        self.vllm_base_url = vllm_base_url.rstrip("/")
        self._tokenizer = None
        self._init_local_tokenizer()

    def _init_local_tokenizer(self) -> None:
        try:
            from transformers import AutoTokenizer  # type: ignore
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            logger.info("Loaded local tokenizer for %s", self.model_name)
        except Exception:
            self._tokenizer = None

    async def tokenize(self, session: aiohttp.ClientSession, prompt: str) -> List[int]:
        """Encode prompt string into a list of token IDs."""
        if self._tokenizer is not None:
            return self._tokenizer.encode(prompt, add_special_tokens=False)

        # Fallback to vLLM's /tokenize endpoint
        tokenize_url = f"{self.vllm_base_url}/tokenize"
        # If base url ends in /v1, the /tokenize endpoint is at the root
        if tokenize_url.endswith("/v1/tokenize"):
            tokenize_url = tokenize_url.replace("/v1/tokenize", "/tokenize")

        try:
            async with session.post(
                tokenize_url,
                json={"model": self.model_name, "prompt": prompt},
                timeout=aiohttp.ClientTimeout(total=5.0),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    tokens = data.get("tokens")
                    if isinstance(tokens, list):
                        return tokens
        except Exception as e:
            logger.debug("vLLM /tokenize failed (%s); using basic tokenization fallback", e)

        # Basic fallback heuristic (character/word-based integers)
        return [hash(w) % 100000 for w in prompt.split()] or [1]


def run_cpu_busywork(duration_ms: float) -> None:
    """Run a CPU-bound busy loop for the requested milliseconds."""
    if duration_ms <= 0:
        return
    target_s = duration_ms / 1000.0
    start = time.perf_counter()
    x = 123456789
    while (time.perf_counter() - start) < target_s:
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF


class GatewayState:
    """Runtime state for gateway."""

    def __init__(self) -> None:
        self.config: Dict[str, Any] = load_gateway_config()
        self.vllm_url: str = os.environ.get(
            "VLLM_URL",
            self.config.get("server_url", "http://127.0.0.1:8000/v1")
        ).rstrip("/")
        self.profiler_exporter_url: str = os.environ.get(
            "PROFILER_EXPORTER_URL",
            f"http://127.0.0.1:{self.config.get('profiler_exporter_port', 9101)}"
        ).rstrip("/")
        self.model_name: str = self.config.get("model_name", "Qwen/Qwen2.5-0.5B-Instruct")
        self.session: Optional[aiohttp.ClientSession] = None
        self.batch_semaphore: Optional[asyncio.Semaphore] = None
        self.trace_lock = asyncio.Lock()
        self.tokenizer_helper: Optional[TokenizerHelper] = None

    def is_profiler_enabled(self) -> bool:
        val = os.environ.get("LLMPROF_PROFILER_ENABLED", "1").strip().lower()
        return val not in ("0", "false", "no", "off")

    def get_slow_tokenizer_ms(self) -> float:
        val = os.environ.get("LLMPROF_SLOW_TOKENIZER_MS", "").strip()
        try:
            return float(val) if val else 0.0
        except ValueError:
            return 0.0

    def get_cpu_preprocess_ms(self) -> float:
        val = os.environ.get("LLMPROF_CPU_PREPROCESS_MS", "").strip()
        try:
            return float(val) if val else 0.0
        except ValueError:
            return 0.0

    def is_no_batching(self) -> bool:
        val = os.environ.get("LLMPROF_NO_BATCHING", "0").strip().lower()
        return val in ("1", "true", "yes", "on")

    def get_trace_path(self) -> Optional[str]:
        path = os.environ.get("LLMPROF_TRACE_PATH", "").strip()
        return path if path else None


state = GatewayState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    connector = aiohttp.TCPConnector(limit=1024)
    state.session = aiohttp.ClientSession(connector=connector)
    state.tokenizer_helper = TokenizerHelper(state.model_name, state.vllm_url)
    state.batch_semaphore = asyncio.Semaphore(1)
    trace_path = state.get_trace_path()
    if trace_path:
        try:
            Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning("Failed to pre-create trace directory: %s", e)
    yield
    if state.session:
        try:
            close_ret = state.session.close()
            if asyncio.iscoroutine(close_ret):
                await close_ret
        except Exception:
            pass


def create_gateway_app() -> FastAPI:
    app = FastAPI(title="LLM Profiler Gateway", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    async def _emit_ingest_metrics(payload: Dict[str, Any]) -> None:
        """Fire-and-forget metric delivery to profiler_exporter /ingest."""
        if not state.session or state.session.closed:
            return
        ingest_url = f"{state.profiler_exporter_url}/ingest"
        try:
            async with state.session.post(
                ingest_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=2.0),
            ) as resp:
                if resp.status not in (200, 202):
                    logger.warning("profiler_exporter returned HTTP %s on /ingest", resp.status)
        except Exception as e:
            logger.debug("Failed to deliver metrics to %s: %s", ingest_url, e)

    async def _append_trace_records(records: List[Dict[str, Any]]) -> None:
        """Append trace records to the configured trace file."""
        trace_path = state.get_trace_path()
        if not trace_path or not records:
            return
        try:
            p = Path(trace_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            lines = [json.dumps(r) + "\n" for r in records]
            async with state.trace_lock:
                with p.open("a", encoding="utf-8") as f:
                    f.writelines(lines)
        except Exception as e:
            logger.warning("Failed to write trace records to %s: %s", trace_path, e)

    @app.post("/v1/completions")
    @app.post("/completions")
    async def completions(request: Request, x_request_id: Optional[str] = Header(None, alias="X-Request-ID")):
        req_id = x_request_id.strip() if x_request_id and x_request_id.strip() else str(uuid.uuid4())

        # Raw pass-through if profiler is disabled
        if not state.is_profiler_enabled():
            body = await request.json()
            forward_url = f"{state.vllm_url}/completions"
            if not forward_url.endswith("/v1/completions") and "/v1" not in forward_url:
                forward_url = f"{state.vllm_url}/v1/completions"

            headers = {"Content-Type": "application/json", "X-Request-ID": req_id}
            async with state.session.post(forward_url, json=body, headers=headers) as resp:
                resp_bytes = await resp.read()
                return Response(
                    content=resp_bytes,
                    status_code=resp.status,
                    media_type="application/json",
                    headers={"X-Request-ID": req_id},
                )

        # Profiler ENABLED: Measure stages
        wall_start_ts = time.time()
        t_req_start = time.perf_counter()

        body = await request.json()
        model_name = body.get("model", state.model_name)
        prompt_input = body.get("prompt", "")
        max_tokens = body.get("max_tokens", 16)
        ignore_eos = body.get("ignore_eos", True)

        # ─── Stage: queue_wait ───────────────────────────────────────────────
        wall_qw_start = time.time()
        t_qw_start = time.perf_counter()
        semaphore_acquired = False

        if state.is_no_batching() and state.batch_semaphore is not None:
            await state.batch_semaphore.acquire()
            semaphore_acquired = True

        t_qw_end = time.perf_counter()
        wall_qw_end = time.time()
        queue_wait_duration = max(0.0, t_qw_end - t_qw_start)

        try:
            # ─── Stage: tokenization (including CPU preprocessing hook) ─────
            wall_tok_start = time.time()
            t_tok_start = time.perf_counter()

            # Hook: CPU-bound preprocessing busywork (attributed to tokenization)
            cpu_preprocess_ms = state.get_cpu_preprocess_ms()
            if cpu_preprocess_ms > 0:
                run_cpu_busywork(cpu_preprocess_ms)

            # Hook: Slow tokenizer sleep delay
            slow_tokenizer_ms = state.get_slow_tokenizer_ms()
            if slow_tokenizer_ms > 0:
                await asyncio.sleep(slow_tokenizer_ms / 1000.0)

            # Tokenize prompt to token IDs
            if isinstance(prompt_input, str):
                token_ids = await state.tokenizer_helper.tokenize(state.session, prompt_input)
            elif isinstance(prompt_input, list) and all(isinstance(x, int) for x in prompt_input):
                token_ids = prompt_input
            else:
                token_ids = await state.tokenizer_helper.tokenize(state.session, str(prompt_input))

            t_tok_end = time.perf_counter()
            wall_tok_end = time.time()
            tokenization_duration = max(0.0, t_tok_end - t_tok_start)

            # ─── Forward to vLLM with streaming for prefill & decode timing ──
            forward_url = f"{state.vllm_url}/completions"
            if not forward_url.endswith("/v1/completions") and "/v1" not in forward_url:
                forward_url = f"{state.vllm_url}/v1/completions"

            vllm_payload = {
                "model": model_name,
                "prompt": token_ids,
                "max_tokens": max_tokens,
                "ignore_eos": ignore_eos,
                "stream": True,
                "stream_options": {"include_usage": True},
            }

            headers = {"Content-Type": "application/json", "X-Request-ID": req_id}

            # ─── Stage: network (gateway-to-vLLM connection setup) ──────────
            wall_net_start = time.time()
            t_net_start = time.perf_counter()

            # ─── Stage: prefill & decode ────────────────────────────────────
            accumulated_text: List[str] = []
            completion_id = None
            created_ts = None
            finish_reason = "length"
            prompt_tokens_observed = len(token_ids)
            completion_tokens_observed = 0

            wall_pre_start = wall_net_start
            t_pre_start = t_net_start
            wall_pre_end: Optional[float] = None
            t_pre_end: Optional[float] = None

            wall_dec_start: Optional[float] = None
            t_dec_start: Optional[float] = None
            wall_dec_end: Optional[float] = None
            t_dec_end: Optional[float] = None

            network_duration = 0.0

            async with state.session.post(forward_url, json=vllm_payload, headers=headers) as resp:
                t_net_end = time.perf_counter()
                network_duration = max(0.0, t_net_end - t_net_start)

                if resp.status != 200:
                    err_text = await resp.text()
                    return Response(content=err_text, status_code=resp.status, media_type="application/json")

                # Stream reading
                async for raw_line in resp.content:
                    line = raw_line.strip()
                    if not line:
                        continue
                    if line.startswith(b"data: "):
                        data_payload = line[6:].strip()
                        if data_payload == b"[DONE]":
                            break
                        try:
                            chunk = json.loads(data_payload.decode("utf-8"))
                        except Exception:
                            continue

                        now_t = time.perf_counter()
                        now_wall = time.time()

                        if t_pre_end is None:
                            # First streamed chunk marks the end of prefill (and vLLM queueing)
                            t_pre_end = now_t
                            wall_pre_end = now_wall
                            t_dec_start = now_t
                            wall_dec_start = now_wall

                        choices = chunk.get("choices", [])
                        if choices:
                            c = choices[0]
                            text_piece = c.get("text", "")
                            if text_piece:
                                accumulated_text.append(text_piece)
                            if c.get("finish_reason"):
                                finish_reason = c.get("finish_reason")

                        if chunk.get("id"):
                            completion_id = chunk.get("id")
                        if chunk.get("created"):
                            created_ts = chunk.get("created")
                        if chunk.get("usage"):
                            usage = chunk.get("usage")
                            if "prompt_tokens" in usage:
                                prompt_tokens_observed = int(usage["prompt_tokens"])
                            if "completion_tokens" in usage:
                                completion_tokens_observed = int(usage["completion_tokens"])

                t_dec_end = time.perf_counter()
                wall_dec_end = time.time()

        finally:
            if semaphore_acquired and state.batch_semaphore is not None:
                state.batch_semaphore.release()

        t_req_end = time.perf_counter()
        wall_req_end = time.time()

        # Prefill and decode duration calculation
        if t_pre_end is not None:
            prefill_duration = max(0.0, t_pre_end - t_pre_start)
        else:
            prefill_duration = 0.0
            wall_pre_end = wall_pre_start

        if t_dec_start is not None and t_dec_end is not None:
            decode_duration = max(0.0, t_dec_end - t_dec_start)
        else:
            decode_duration = 0.0
            wall_dec_start = wall_pre_end
            wall_dec_end = wall_pre_end

        total_duration = max(0.0, t_req_end - t_req_start)
        full_text = "".join(accumulated_text)
        if completion_tokens_observed == 0:
            completion_tokens_observed = len(accumulated_text) or 1

        # Reconstruct OpenAI completion response
        resp_data = {
            "id": completion_id or f"cmpl-{uuid.uuid4().hex[:16]}",
            "object": "text_completion",
            "created": int(created_ts or wall_start_ts),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "text": full_text,
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens_observed,
                "completion_tokens": completion_tokens_observed,
                "total_tokens": prompt_tokens_observed + completion_tokens_observed,
            },
        }

        # ─── Stage Metrics & Trace Records ──────────────────────────────────
        # Note: detokenization is omitted from stage_durations as it cannot be measured separately
        stage_durations = {
            "tokenization": tokenization_duration,
            "queue_wait": queue_wait_duration,
            "network": network_duration,
            "prefill": prefill_duration,
            "decode": decode_duration,
        }

        ingest_payload = {
            "request_total_seconds": total_duration,
            "tokens_in_total": prompt_tokens_observed,
            "tokens_out_total": completion_tokens_observed,
            "stage_durations": stage_durations,
        }
        asyncio.create_task(_emit_ingest_metrics(ingest_payload))

        trace_records = [
            {
                "request_id": req_id,
                "stage": "tokenization",
                "start_ts": wall_tok_start,
                "end_ts": wall_tok_end,
                "duration_s": tokenization_duration,
            },
            {
                "request_id": req_id,
                "stage": "queue_wait",
                "start_ts": wall_qw_start,
                "end_ts": wall_qw_end,
                "duration_s": queue_wait_duration,
            },
            {
                "request_id": req_id,
                "stage": "network",
                "start_ts": wall_net_start,
                "end_ts": wall_net_start + network_duration,
                "duration_s": network_duration,
            },
            {
                "request_id": req_id,
                "stage": "prefill",
                "start_ts": wall_pre_start,
                "end_ts": wall_pre_end,
                "duration_s": prefill_duration,
            },
            {
                "request_id": req_id,
                "stage": "decode",
                "start_ts": wall_dec_start,
                "end_ts": wall_dec_end,
                "duration_s": decode_duration,
            },
            {
                "request_id": req_id,
                "stage": "total",
                "start_ts": wall_start_ts,
                "end_ts": wall_req_end,
                "duration_s": total_duration,
            },
        ]
        asyncio.create_task(_append_trace_records(trace_records))

        return JSONResponse(
            content=resp_data,
            status_code=200,
            headers={"X-Request-ID": req_id},
        )

    return app
