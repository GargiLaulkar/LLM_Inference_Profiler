"""Entrypoint for LLM Profiler Gateway."""

import os
import uvicorn
from llm_profiler.gateway.app import create_gateway_app, load_gateway_config


def main():
    cfg = load_gateway_config()
    host = os.environ.get("GATEWAY_HOST", cfg.get("gateway_host", "0.0.0.0"))
    port = int(os.environ.get("GATEWAY_PORT", cfg.get("gateway_port", 8080)))

    app = create_gateway_app()
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()

