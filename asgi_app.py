import os
import logging
import threading
from http.server import ThreadingHTTPServer
from typing import Callable, Awaitable

import requests

from run import SwapHandler

_INTERNAL_HOST = os.environ.get("REACTOR_INTERNAL_HOST", "127.0.0.1")
_INTERNAL_PORT = int(os.environ.get("REACTOR_INTERNAL_PORT", "18004"))
_INTERNAL_BASE = f"http://{_INTERNAL_HOST}:{_INTERNAL_PORT}"

_server_started = False
_server_lock = threading.Lock()

LOGGER = logging.getLogger("reactor.asgi")


def _ensure_internal_server() -> None:
    global _server_started
    if _server_started:
        return
    with _server_lock:
        if _server_started:
            return
        server = ThreadingHTTPServer((_INTERNAL_HOST, _INTERNAL_PORT), SwapHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        _server_started = True
        LOGGER.info("Internal swap server started at %s:%s", _INTERNAL_HOST, _INTERNAL_PORT)


async def app(scope, receive, send):
    if scope["type"] != "http":
        await send({"type": "http.response.start", "status": 500, "headers": []})
        await send({"type": "http.response.body", "body": b"Unsupported scope type"})
        return

    _ensure_internal_server()

    method = scope.get("method", "GET")
    if method != "GET":
        await send({"type": "http.response.start", "status": 405, "headers": [[b"content-type", b"text/plain; charset=utf-8"]]})
        await send({"type": "http.response.body", "body": b"Method Not Allowed"})
        return

    raw_path = scope.get("raw_path", b"").decode("utf-8", errors="ignore")
    query_string = scope.get("query_string", b"").decode("utf-8", errors="ignore")
    target_url = f"{_INTERNAL_BASE}{raw_path}"
    if query_string:
        target_url = f"{target_url}?{query_string}"

    try:
        LOGGER.info("Proxying request to internal server: path=%s query_length=%s", raw_path, len(query_string))
        resp = requests.get(target_url, timeout=600)
        headers = []
        content_type = resp.headers.get("Content-Type")
        if content_type:
            headers.append([b"content-type", content_type.encode("utf-8")])
        set_cookie = resp.headers.get("Set-Cookie")
        if set_cookie:
            headers.append([b"set-cookie", set_cookie.encode("utf-8")])
        await send({"type": "http.response.start", "status": resp.status_code, "headers": headers})
        await send({"type": "http.response.body", "body": resp.content})
    except Exception as exc:
        LOGGER.exception("ASGI proxy failure for target_url=%s", target_url)
        msg = str(exc).encode("utf-8", errors="ignore")
        await send({"type": "http.response.start", "status": 500, "headers": [[b"content-type", b"text/plain; charset=utf-8"]]})
        await send({"type": "http.response.body", "body": msg})
