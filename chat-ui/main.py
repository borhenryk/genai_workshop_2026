"""
Chat UI Backend – Databricks App that proxies to a serving endpoint.

Setup: Upload code, create app, set ENDPOINT_NAME in app.yaml, add endpoint
as resource, deploy.
"""

import json
import logging
import os
import uuid
from typing import Optional

import requests
from databricks.sdk import WorkspaceClient
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("workshop-chat")

app = FastAPI(title="Workshop Agent Chat")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- Config (auto-detected from Databricks App environment) ---
_w = WorkspaceClient()
_host = _w.config.host.rstrip("/")
if not _host.startswith("https://"):
    _host = f"https://{_host}"

ENDPOINT_NAME = os.getenv("ENDPOINT_NAME", "")
logger.info(f"Host: {_host}, Endpoint: {ENDPOINT_NAME}")


def _token() -> str:
    return _w.config.oauth_token().access_token


def _user_email(req: Request) -> Optional[str]:
    for h in ("X-Forwarded-Email", "X-Forwarded-User", "X-Forwarded-Preferred-Username"):
        if v := req.headers.get(h):
            return v
    return None


def _text_from_item(item: dict) -> str:
    return "".join(c.get("text", "") for c in item.get("content", []) if c.get("type") == "output_text")


class ChatRequest(BaseModel):
    messages: list[dict]
    thread_id: Optional[str] = None
    user_id: Optional[str] = None


@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    if not ENDPOINT_NAME:
        raise HTTPException(500, "ENDPOINT_NAME not set in app.yaml")

    url = f"{_host}/serving-endpoints/{ENDPOINT_NAME}/invocations"
    tid = req.thread_id or str(uuid.uuid4())
    uid = req.user_id or _user_email(request) or "chat_ui_user"
    headers = {"Authorization": f"Bearer {_token()}", "Content-Type": "application/json"}
    payload = {"input": req.messages, "custom_inputs": {"thread_id": tid, "user_id": uid}, "stream": True}

    def stream():
        text = ""
        try:
            with requests.post(url, headers=headers, json=payload, stream=True, timeout=180) as r:
                if r.status_code != 200:
                    yield f"data: {json.dumps({'type': 'error', 'content': r.text[:300]})}\n\n"
                    return
                for line in r.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    raw = line[6:]
                    if raw.strip() == "[DONE]":
                        break
                    try:
                        ev = json.loads(raw)
                        t = ev.get("type", "")
                        if t == "response.output_text.delta":
                            d = ev.get("delta", "")
                            text += d
                            yield f"data: {json.dumps({'type': 'delta', 'content': d})}\n\n"
                        elif t == "response.output_item.done":
                            item = ev.get("item", {})
                            it = item.get("type", "")
                            if it == "function_call":
                                yield f"data: {json.dumps({'type': 'tool_call', 'name': item.get('name',''), 'arguments': item.get('arguments','')})}\n\n"
                            elif it == "function_call_output":
                                yield f"data: {json.dumps({'type': 'tool_result', 'output': item.get('output','')[:200]})}\n\n"
                            elif it == "message":
                                msg = _text_from_item(item)
                                if msg and not text:
                                    text = msg
                                    yield f"data: {json.dumps({'type': 'delta', 'content': msg})}\n\n"
                    except json.JSONDecodeError:
                        continue
        except requests.exceptions.RequestException as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'thread_id': tid, 'user_id': uid})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/api/chat/sync")
async def chat_sync(req: ChatRequest, request: Request):
    if not ENDPOINT_NAME:
        raise HTTPException(500, "ENDPOINT_NAME not set")
    url = f"{_host}/serving-endpoints/{ENDPOINT_NAME}/invocations"
    tid = req.thread_id or str(uuid.uuid4())
    uid = req.user_id or _user_email(request) or "chat_ui_user"
    headers = {"Authorization": f"Bearer {_token()}", "Content-Type": "application/json"}
    payload = {"input": req.messages, "custom_inputs": {"thread_id": tid, "user_id": uid}}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=180)
        r.raise_for_status()
        result = r.json()
        content = "".join(_text_from_item(i) for i in result.get("output", []) if i.get("type") == "message")
        return {"content": content, "thread_id": tid, "user_id": uid}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/health")
async def health():
    return {"status": "ok", "endpoint": ENDPOINT_NAME or "NOT_SET", "host": _host}


# --- Static frontend ---
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir) and os.path.exists(os.path.join(static_dir, "index.html")):
    assets_dir = os.path.join(static_dir, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/")
    async def serve_index():
        from fastapi.responses import FileResponse
        return FileResponse(os.path.join(static_dir, "index.html"))

    @app.get("/{path:path}")
    async def serve_spa(path: str):
        from fastapi.responses import FileResponse
        fp = os.path.join(static_dir, path)
        return FileResponse(fp if os.path.isfile(fp) else os.path.join(static_dir, "index.html"))
