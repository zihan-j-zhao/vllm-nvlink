#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
NIXL-aware P/D-disaggregation proxy for vLLM's NixlConnector.

Drop-in replacement for tests/v1/kv_connector/nixl_integration/toy_proxy_server.py,
fixing the one wart that breaks AIPerf when --streaming is on:

  * The upstream toy proxy hard-codes
    `StreamingResponse(generator, media_type="application/json")`.
    For a streaming request, vLLM's decode replies with
    `Content-Type: text/event-stream; charset=utf-8`, but the toy proxy
    overwrites that with application/json. AIPerf's SSE parser then sees
    each `data: {...}` line as malformed JSON and reports
    `InvalidInferenceResultError: No responses with actual content`.
  * This proxy forwards the upstream Content-Type verbatim, so SSE flows
    through unchanged and AIPerf can compute TTFT / ITL.

NIXL handshake (same as the upstream toy):
  1. POST to PREFILL with kv_transfer_params={"do_remote_decode": true, ...}
     and max_tokens=1, stream=false.
  2. PREFILL's JSON response contains a populated kv_transfer_params dict
     (remote_engine_id, remote_block_ids, remote_host, remote_port).
  3. Stitch that into the original client body and POST it to DECODE; the
     consumer uses the params to pull KV from prefill over NIXL/UCX.
  4. Stream DECODE's response back to the client byte-for-byte.

Also implements `GET /v1/models` by forwarding to the first prefill so
AIPerf's `--wait-for-model-mode models` readiness probe works.

CLI: same defaults as the upstream toy
  python proxy.py [--port 8000] [--host 0.0.0.0]
                  [--prefiller-host localhost] [--prefiller-port 8100]
                  [--decoder-host  localhost] [--decoder-port  8200]
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

logger = logging.getLogger("moe_pd_proxy")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.prefill_clients = [
        {
            "client": httpx.AsyncClient(
                timeout=None,
                base_url=f"http://{h}:{p}/v1",
                limits=httpx.Limits(
                    max_connections=None, max_keepalive_connections=None
                ),
            ),
            "host": h,
            "port": p,
        }
        for (h, p) in global_args.prefiller_instances
    ]
    app.state.decode_clients = [
        {
            "client": httpx.AsyncClient(
                timeout=None,
                base_url=f"http://{h}:{p}/v1",
                limits=httpx.Limits(
                    max_connections=None, max_keepalive_connections=None
                ),
            ),
            "host": h,
            "port": p,
        }
        for (h, p) in global_args.decoder_instances
    ]
    app.state.prefill_iter = itertools.cycle(range(len(app.state.prefill_clients)))
    app.state.decode_iter = itertools.cycle(range(len(app.state.decode_clients)))
    logger.info(
        "ready: %d prefill instance(s), %d decode instance(s)",
        len(app.state.prefill_clients),
        len(app.state.decode_clients),
    )
    yield
    for c in app.state.prefill_clients + app.state.decode_clients:
        await c["client"].aclose()


app = FastAPI(lifespan=lifespan)


def _next_client(app: FastAPI, kind: str) -> dict:
    if kind == "prefill":
        return app.state.prefill_clients[next(app.state.prefill_iter)]
    if kind == "decode":
        return app.state.decode_clients[next(app.state.decode_iter)]
    raise ValueError(f"unknown client kind {kind!r}")


def _auth_headers(request_id: str) -> dict[str, str]:
    h = {"X-Request-Id": request_id}
    if "OPENAI_API_KEY" in os.environ:
        h["Authorization"] = f"Bearer {os.environ['OPENAI_API_KEY']}"
    return h


# ---------------------------------------------------------------------------
# Prefill request: max_tokens=1, stream=false, signals the connector that this
# is a producer-side run by setting kv_transfer_params.do_remote_decode=true.
# ---------------------------------------------------------------------------
def _prefill_body(body: dict[str, Any]) -> dict[str, Any]:
    p = body.copy()
    p["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    p["stream"] = False
    p["max_tokens"] = 1
    if "max_completion_tokens" in p:
        p["max_completion_tokens"] = 1
    p.pop("stream_options", None)
    # min_tokens / min_completion_tokens make prefill error out; pop them
    # and put them back when we hand off to decode.
    p.pop("min_tokens", None)
    p.pop("min_completion_tokens", None)
    return p


async def _forward(api: str, request: Request) -> Response:
    body: dict[str, Any] = await request.json()
    request_id = str(uuid.uuid4())

    prefill = _next_client(app, "prefill")
    decode = _next_client(app, "decode")
    headers = _auth_headers(request_id)

    # Step 1: prefill.
    try:
        p_resp = await prefill["client"].post(
            api, json=_prefill_body(body), headers=headers
        )
        p_resp.raise_for_status()
    except Exception as exc:
        logger.exception("prefill forward failed")
        return JSONResponse(
            status_code=502, content={"error": f"prefill failed: {exc!s}"}
        )

    # Step 2: copy populated kv_transfer_params back into the original body.
    try:
        p_json = p_resp.json()
    except Exception as exc:
        logger.exception("prefill returned non-JSON response")
        return JSONResponse(
            status_code=502, content={"error": f"prefill response invalid: {exc!s}"}
        )
    kv_params = p_json.get("kv_transfer_params") or {}
    if kv_params:
        body["kv_transfer_params"] = kv_params

    # Step 3: stream decode response, preserving its Content-Type verbatim.
    # For stream=true requests this is "text/event-stream; charset=utf-8";
    # for non-streaming it is "application/json". The upstream toy proxy
    # hard-coded "application/json" here, which broke SSE parsers (AIPerf
    # reported `InvalidInferenceResultError: No responses with actual content`
    # for every streamed request).
    #
    # We open the httpx stream context here so we can inspect response
    # headers BEFORE starting the generator, then hand the generator to
    # Starlette's StreamingResponse with the matching media_type. The
    # finally clause guarantees the context is closed even if the client
    # disconnects mid-stream.
    cm = decode["client"].stream("POST", api, json=body, headers=headers)
    try:
        d_resp = await cm.__aenter__()
    except Exception as exc:
        logger.exception("decode forward failed (connect)")
        return JSONResponse(
            status_code=502, content={"error": f"decode failed: {exc!s}"}
        )

    if d_resp.status_code >= 400:
        # Drain so the proxy doesn't hold the upstream connection open,
        # then propagate the error to the client.
        err_body = await d_resp.aread()
        media_type = d_resp.headers.get("content-type", "application/json")
        status = d_resp.status_code
        await cm.__aexit__(None, None, None)
        return Response(content=err_body, status_code=status, media_type=media_type)

    media_type = d_resp.headers.get("content-type", "application/json")

    async def _proxy_stream():
        try:
            async for chunk in d_resp.aiter_bytes():
                yield chunk
        finally:
            await cm.__aexit__(None, None, None)

    return StreamingResponse(_proxy_stream(), media_type=media_type)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    return await _forward("/chat/completions", request)


@app.post("/v1/completions")
async def completions(request: Request) -> Response:
    return await _forward("/completions", request)


@app.get("/v1/models")
async def models() -> Response:
    """Forward to the first prefill instance so AIPerf's
    `--wait-for-model-mode models` probe works without a second proxy."""
    if not app.state.prefill_clients:
        return JSONResponse(
            status_code=503, content={"error": "no prefill clients"}
        )
    client = app.state.prefill_clients[0]["client"]
    try:
        upstream = await client.get("/models")
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            headers={
                "Content-Type": upstream.headers.get(
                    "content-type", "application/json"
                )
            },
        )
    except Exception as exc:
        return JSONResponse(status_code=502, content={"error": str(exc)})


@app.get("/healthcheck")
async def healthcheck() -> dict[str, Any]:
    return {
        "status": "ok",
        "prefill_instances": len(app.state.prefill_clients),
        "decode_instances": len(app.state.decode_clients),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--prefiller-hosts", "--prefiller-host",
        type=str, nargs="+", default=["localhost"],
    )
    p.add_argument(
        "--prefiller-ports", "--prefiller-port",
        type=int, nargs="+", default=[8100],
    )
    p.add_argument(
        "--decoder-hosts", "--decoder-host",
        type=str, nargs="+", default=["localhost"],
    )
    p.add_argument(
        "--decoder-ports", "--decoder-port",
        type=int, nargs="+", default=[8200],
    )
    args = p.parse_args()
    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError("#prefiller_hosts must match #prefiller_ports")
    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError("#decoder_hosts must match #decoder_ports")
    args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))
    return args


def main() -> None:
    global global_args
    global_args = parse_args()
    logger.info(
        "starting proxy on http://%s:%d (prefill=%s, decode=%s)",
        global_args.host,
        global_args.port,
        global_args.prefiller_instances,
        global_args.decoder_instances,
    )
    uvicorn.run(
        app,
        host=global_args.host,
        port=global_args.port,
        log_level="info",
        access_log=False,
    )


if __name__ == "__main__":
    sys.exit(main())
