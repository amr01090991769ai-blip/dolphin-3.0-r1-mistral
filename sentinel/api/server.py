"""Zero-dependency HTTP API + web dashboard for Sentinel.

Built on the Python standard library (http.server) so the platform runs
anywhere with no extra packages. Endpoints:

  GET  /                 -> web dashboard (HTML)
  GET  /api/status       -> platform/LLM/tool status
  POST /api/chat         -> {"prompt": "...", "system": "..."} -> {"reply": "..."}
  POST /api/agent        -> {"goal": "..."} -> {final_answer, steps, completed}
  POST /api/scan         -> {"path": "..."} -> security scan report
  GET  /api/tools        -> list of available tools
  GET  /health           -> {"ok": true}
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict

from ..platform import Sentinel


def _load_dashboard() -> str:
    html = Path(__file__).resolve().parent.parent / "web" / "templates" / "dashboard.html"
    try:
        return html.read_text()
    except OSError:
        return "<h1>Sentinel</h1><p>Dashboard template missing.</p>"


def create_app(sentinel: Sentinel):
    dashboard_html = _load_dashboard()

    class Handler(BaseHTTPRequestHandler):
        server_version = "Sentinel/1.0"

        # ---- helpers --------------------------------------------------- #
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: Dict[str, Any]) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode(),
                       "application/json; charset=utf-8")

        def _read_json(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length", 0))
            if length == 0:
                return {}
            raw = self.rfile.read(length)
            try:
                return json.loads(raw.decode())
            except json.JSONDecodeError:
                return {}

        def log_message(self, fmt, *args):  # quieter logging
            pass

        # ---- routing --------------------------------------------------- #
        def do_OPTIONS(self):
            self._send(204, b"", "text/plain")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, dashboard_html.encode(), "text/html; charset=utf-8")
            elif self.path == "/favicon.ico":
                self._send(204, b"", "image/x-icon")
            elif self.path == "/health":
                self._json(200, {"ok": True})
            elif self.path == "/api/status":
                self._json(200, sentinel.status())
            elif self.path == "/api/tools":
                self._json(200, {"tools": sentinel.tools.to_list()})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            try:
                data = self._read_json()
                if self.path == "/api/chat":
                    reply = sentinel.chat(data.get("prompt", ""),
                                          data.get("system"))
                    self._json(200, {"reply": reply})
                elif self.path == "/api/agent":
                    result = sentinel.run_agent(data.get("goal", ""))
                    self._json(200, result)
                elif self.path == "/api/scan":
                    result = sentinel.scan(data.get("path", "."))
                    self._json(200, result)
                else:
                    self._json(404, {"error": "not found"})
            except Exception as exc:  # never crash the server
                self._json(500, {"error": str(exc)})

    return Handler


def run_server(host: str = "0.0.0.0", port: int = 8080,
               sentinel: Sentinel | None = None) -> ThreadingHTTPServer:
    sentinel = sentinel or Sentinel()
    handler = create_app(sentinel)
    httpd = ThreadingHTTPServer((host, port), handler)
    print(f"[Sentinel] API + dashboard running on http://{host}:{port}")
    print(f"[Sentinel] LLM backend: {sentinel.llm.backend}")
    return httpd
