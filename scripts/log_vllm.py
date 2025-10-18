# proxy.py
import os
import json
import uuid
import datetime
from typing import Optional

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse
import uvicorn

# --- Configuration ---
TARGET_BASE = os.environ.get("VLLM_TARGET", "http://localhost:8000")
LISTEN_HOST = os.environ.get("PROXY_HOST", "0.0.0.0")
LISTEN_PORT = int(os.environ.get("PROXY_PORT", 8080))

LOG_FILE = os.environ.get("PROXY_LOG_FILE", "logs.jsonl")
ARCHIVE_DIR = os.environ.get("PROXY_ARCHIVE_DIR", "archive")

os.makedirs(ARCHIVE_DIR, exist_ok=True)

app = FastAPI(title="vLLM Proxy with logging")


def _now_iso():
    return datetime.datetime.utcnow().isoformat() + "Z"


async def _save_raw_body(prefix: str, content: bytes) -> str:
    """Save raw bytes to archive and return filename."""
    uid = uuid.uuid4().hex
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    filename = f"{prefix}_{ts}_{uid}.bin"
    path = os.path.join(ARCHIVE_DIR, filename)
    with open(path, "wb") as f:
        f.write(content)
    return path


def _append_log(entry: dict):
    """Append a JSON line to LOG_FILE."""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _filter_headers_for_forward(headers, target_host: str):
    """Return headers to forward to target. Keep common relevant headers."""
    hop_by_hop = {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
    out = {}
    for k, v in headers.items():
        lk = k.lower()
        if lk in hop_by_hop:
            continue
        # Update Host header to target
        if lk == "host":
            out[k] = target_host
        else:
            out[k] = v
    return out


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def proxy(request: Request, path: str):
    """
    Generic proxy: forwards to TARGET_BASE/{path} preserving method and relevant headers.
    Logs request and response.
    """
    method = request.method
    raw_request_body = await request.body()  # bytes
    request_headers = dict(request.headers)

    # Build target URL (preserve query string)
    qs = request.url.query
    target_url = f"{TARGET_BASE.rstrip('/')}/{path}"
    if qs:
        target_url = target_url + "?" + qs

    # Extract target host for Host header
    try:
        from urllib.parse import urlparse
        parsed = urlparse(TARGET_BASE)
        target_host = parsed.netloc
    except Exception:
        target_host = TARGET_BASE

    # Save raw request body to archive
    req_body_path = await _save_raw_body("req", raw_request_body)

    # Prepare forwarded headers
    forward_headers = _filter_headers_for_forward(request_headers, target_host)

    # Use httpx async client to forward
    async with httpx.AsyncClient(timeout=None) as client:
        try:
            resp = await client.request(
                method,
                target_url,
                content=raw_request_body if raw_request_body else None,
                headers=forward_headers,
            )
        except httpx.RequestError as e:
            # Network error forwarding to vLLM
            entry = {
                "id": uuid.uuid4().hex,
                "timestamp": _now_iso(),
                "event": "forward_error",
                "method": method,
                "path": path,
                "target_url": target_url,
                "request_headers": {k: v for k, v in request_headers.items()},
                "request_body_path": req_body_path,
                "error": str(e),
            }
            _append_log(entry)
            return PlainTextResponse(f"Error forwarding request to vLLM: {e}", status_code=502)

        # Get response content and metadata
        resp_status = resp.status_code
        resp_headers = dict(resp.headers)
        body_bytes = resp.content

        # Save response raw body
        resp_body_path = await _save_raw_body("res", body_bytes)

        # Try to decode for nicer logs if it's JSON/text
        content_type = resp_headers.get("content-type", "")
        response_text: Optional[str] = None
        request_text: Optional[str] = None
        
        # Decode request body
        if len(raw_request_body) > 0:
            try:
                request_text = json.loads(raw_request_body.decode("utf-8"))
            except json.JSONDecodeError:
                try:
                    request_text = raw_request_body.decode("utf-8")
                except UnicodeDecodeError:
                    request_text = None
        
        # Decode response body
        if len(body_bytes) > 0:
            try:
                if "application/json" in content_type:
                    response_text = json.loads(body_bytes.decode("utf-8"))
                else:
                    response_text = body_bytes.decode("utf-8")
            except (json.JSONDecodeError, UnicodeDecodeError):
                response_text = None

        # Build log entry
        entry = {
            "id": uuid.uuid4().hex,
            "timestamp": _now_iso(),
            "event": "proxy_call",
            "method": method,
            "path": path,
            "target_url": target_url,
            "request": {
                "headers": {k: v for k, v in request_headers.items()},
                "body_preview": request_text if request_text is not None else None,
                "body_saved_path": req_body_path,
            },
            "response": {
                "status_code": resp_status,
                "headers": {k: v for k, v in resp_headers.items()},
                "body_preview": response_text if response_text is not None else None,
                "body_saved_path": resp_body_path,
            },
        }

        _append_log(entry)

        # Prepare response back to original caller
        # Remove hop-by-hop headers from response
        response_headers_to_send = {}
        for k, v in resp_headers.items():
            lk = k.lower()
            if lk in ("transfer-encoding", "connection", "keep-alive", "upgrade"):
                continue
            response_headers_to_send[k] = v

        return Response(content=body_bytes, status_code=resp_status, headers=response_headers_to_send)


if __name__ == "__main__":
    uvicorn.run("proxy:app", host=LISTEN_HOST, port=LISTEN_PORT, reload=False)