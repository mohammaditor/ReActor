import os
import sys
import logging
import uuid
from urllib.parse import parse_qs

from run import process_swap_request

LOGGER = logging.getLogger("reactor.asgi")
if not LOGGER.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False


async def app(scope, receive, send):
    if scope["type"] != "http":
        await send({"type": "http.response.start", "status": 500, "headers": []})
        await send({"type": "http.response.body", "body": b"Unsupported scope type"})
        return

    method = scope.get("method", "GET")
    if method != "GET":
        await send({"type": "http.response.start", "status": 405, "headers": [[b"content-type", b"text/plain; charset=utf-8"]]})
        await send({"type": "http.response.body", "body": b"Method Not Allowed"})
        return

    request_id = uuid.uuid4().hex[:8]
    raw_path = scope.get("raw_path", b"").decode("utf-8", errors="ignore")
    query_string = scope.get("query_string", b"").decode("utf-8", errors="ignore")
    params = parse_qs(query_string)

    try:
        status, headers, body = process_swap_request(raw_path, params, request_id)
        
        asgi_headers = []
        for k, v in headers.items():
            asgi_headers.append([k.lower().encode("utf-8"), v.encode("utf-8")])
        
        # Ensure Content-Length is present
        asgi_headers.append([b"content-length", str(len(body)).encode("utf-8")])

        await send({"type": "http.response.start", "status": status, "headers": asgi_headers})
        await send({"type": "http.response.body", "body": body})
    except Exception as exc:
        LOGGER.exception("ASGI processing failure for path=%s", raw_path)
        msg = str(exc).encode("utf-8", errors="ignore")
        await send({"type": "http.response.start", "status": 500, "headers": [[b"content-type", b"text/plain; charset=utf-8"]]})
        await send({"type": "http.response.body", "body": msg})
