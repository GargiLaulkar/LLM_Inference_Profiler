#!/usr/bin/env python3
"""Reference OpenAI-compatible Server for LLM Inference Profiler."""

import argparse
import json
import os
import sys
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
import yaml

CONFIG = {}
REQUEST_COUNT = 0
TOTAL_PROMPT_TOKENS = 0
TOTAL_OUTPUT_TOKENS = 0
START_TIME = time.time()


def get_token_count(text: str) -> int:
    try:
        from transformers import AutoTokenizer
        model_name = CONFIG.get("model_name", "Qwen/Qwen2.5-0.5B-Instruct")
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        return len(tokenizer.encode(text))
    except Exception:
        words = text.split()
        return max(1, int(len(words) * 1.3))


class OpenAIReferenceHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        global REQUEST_COUNT, START_TIME
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "healthy"}')
            return

        if self.path == "/metrics":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.end_headers()

            metrics = [
                "# HELP vllm:num_requests_running Number of requests currently running on GPU.",
                "# TYPE vllm:num_requests_running gauge",
                "vllm:num_requests_running 0",
                "# HELP vllm:num_requests_waiting Number of requests waiting in the queue.",
                "# TYPE vllm:num_requests_waiting gauge",
                "vllm:num_requests_waiting 0",
                "# HELP vllm:gpu_cache_usage_factor GPU KV-cache usage percentage.",
                "# TYPE vllm:gpu_cache_usage_factor gauge",
                "vllm:gpu_cache_usage_factor 0.12",
                "# HELP vllm:prompt_tokens_total Total prompt tokens processed.",
                "# TYPE vllm:prompt_tokens_total counter",
                f"vllm:prompt_tokens_total {TOTAL_PROMPT_TOKENS}",
                "# HELP vllm:generation_tokens_total Total output tokens generated.",
                "# TYPE vllm:generation_tokens_total counter",
                f"vllm:generation_tokens_total {TOTAL_OUTPUT_TOKENS}",
                "# HELP vllm:request_success_total Total successful requests.",
                "# TYPE vllm:request_success_total counter",
                f"vllm:request_success_total {REQUEST_COUNT}",
            ]
            self.wfile.write("\n".join(metrics).encode("utf-8") + b"\n")
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        global REQUEST_COUNT, TOTAL_PROMPT_TOKENS, TOTAL_OUTPUT_TOKENS
        if self.path == "/v1/completions":
            content_length = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_length)
            try:
                body = json.loads(body_bytes.decode("utf-8"))
            except Exception:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error": "Invalid JSON"}')
                return

            prompt = body.get("prompt", "")
            if isinstance(prompt, list):
                prompt = " ".join(str(p) for p in prompt)
            max_tokens = int(body.get("max_tokens", 16))
            model = body.get("model", CONFIG.get("model_name", "Qwen/Qwen2.5-0.5B-Instruct"))

            prompt_tokens = get_token_count(prompt)
            output_tokens = max_tokens

            simulated_latency = 0.005 + (output_tokens * 0.002)
            time.sleep(simulated_latency)

            REQUEST_COUNT += 1
            TOTAL_PROMPT_TOKENS += prompt_tokens
            TOTAL_OUTPUT_TOKENS += output_tokens

            request_id = self.headers.get("X-Request-ID", f"cmpl-{uuid.uuid4().hex[:12]}")
            
            if "Paris" in prompt or "capital of France" in prompt:
                completion_text = " Paris, the capital and most populous city of France."
            else:
                words = ["benchmark", "token", "inference", "throughput", "latency", "profiler", "system", "stage"]
                completion_text = " " + " ".join((words * 10)[:output_tokens])

            response_data = {
                "id": request_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "text": completion_text,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "length",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": prompt_tokens + output_tokens,
                },
            }

            resp_bytes = json.dumps(response_data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_bytes)))
            self.send_header("X-Request-ID", request_id)
            self.end_headers()
            self.wfile.write(resp_bytes)
            return

        self.send_response(404)
        self.end_headers()


def run_server():
    global CONFIG
    parser = argparse.ArgumentParser(description="Reference OpenAI-compatible Server")
    parser.add_argument("--config", default="llm_profiler/config.yaml", help="Path to config.yaml")
    parser.add_argument("--host", default=None, help="Host override")
    parser.add_argument("--port", type=int, default=None, help="Port override")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.exists(config_path) and os.path.exists("config.yaml"):
        config_path = "config.yaml"

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            CONFIG = yaml.safe_load(f) or {}

    host = args.host or CONFIG.get("server_host", "127.0.0.1")
    port = args.port or int(CONFIG.get("server_port", 8000))

    server_address = (host, port)
    httpd = HTTPServer(server_address, OpenAIReferenceHandler)
    print(f"Reference server running on http://{host}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()


if __name__ == "__main__":
    run_server()
