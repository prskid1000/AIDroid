#!/usr/bin/env python3
"""A minimal MCP server over Streamable HTTP, for testing the app's client.

Not a toy protocol — this speaks the same `initialize` / `tools/list` /
`tools/call` surface a real server does, including the `Mcp-Session-Id` header,
so exercising the app against it exercises the actual code path rather than a
mock. Two personalities are built in so two of them can run at once and the
app's per-server and per-tool switches have something to be wrong about.

    python tools/mcp-dummy-server.py --port 8931 --persona weather
    python tools/mcp-dummy-server.py --port 8932 --persona notes

The Android emulator reaches the host at 10.0.2.2, so the app should be given
http://10.0.2.2:8931/mcp.
"""

from __future__ import annotations

import argparse
import json
import random
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = "2025-06-18"


def text_result(text: str, is_error: bool = False) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


# Two servers with deliberately different tools. `echo` is on both on purpose:
# the registry resolves a duplicate name first-wins in provider order, and that
# rule is only observable when two servers actually collide.
PERSONAS = {
    "weather": {
        "name": "dummy-weather",
        "tools": [
            {
                "name": "get_forecast",
                "description": "A made-up forecast for a city. Test data, not real weather.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                        "days": {"type": "integer", "description": "How many days, 1-7"},
                    },
                    "required": ["city"],
                },
            },
            {
                "name": "get_air_quality",
                "description": "A made-up air quality index for a city. Test data.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
            {
                "name": "echo",
                "description": "Return whatever you send. Weather server's copy.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        ],
    },
    "notes": {
        "name": "dummy-notes",
        "tools": [
            {
                "name": "list_notes",
                "description": "List the notes held by this test server.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "add_note",
                "description": "Add a note to this test server. Kept in memory only.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
            {
                "name": "echo",
                "description": "Return whatever you send. Notes server's copy.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        ],
    },
}

NOTES: list[str] = ["Milk, bread, a 4096-bit key backup."]


def call_tool(persona: str, name: str, args: dict) -> dict:
    if name == "get_forecast":
        city = args.get("city", "somewhere")
        days = int(args.get("days", 1) or 1)
        rng = random.Random(city.lower())
        lines = [
            f"day {d + 1}: {rng.choice(['sunny', 'cloudy', 'rain', 'fog'])}, "
            f"{rng.randint(4, 34)}C"
            for d in range(max(1, min(days, 7)))
        ]
        return text_result(f"Forecast for {city} (fabricated):\n" + "\n".join(lines))

    if name == "get_air_quality":
        city = args.get("city", "somewhere")
        aqi = random.Random(city.lower() + "aqi").randint(10, 180)
        return text_result(f"{city} AQI {aqi} (fabricated test value)")

    if name == "list_notes":
        if not NOTES:
            return text_result("No notes.")
        return text_result("\n".join(f"{i + 1}. {n}" for i, n in enumerate(NOTES)))

    if name == "add_note":
        text = args.get("text", "").strip()
        if not text:
            return text_result("A note needs some text.", is_error=True)
        NOTES.append(text)
        return text_result(f"Added. {len(NOTES)} notes now.")

    if name == "echo":
        return text_result(f"[{persona}] {args.get('text', '')}")

    return text_result(f"No tool named {name!r} on this server.", is_error=True)


class Handler(BaseHTTPRequestHandler):
    # HTTP/1.1, so connections stay open between requests. The default is
    # HTTP/1.0, which closes after every response — and a client with a
    # connection pool then reuses a socket the server has already hung up on and
    # reports "unexpected end of stream". A probe is three requests back to back
    # (initialize, the initialized notification, tools/list), so this is not a
    # rare race: it is most of them. Every response below sets Content-Length,
    # which is what HTTP/1.1 requires in return.
    protocol_version = "HTTP/1.1"

    persona = "weather"
    session_id = ""

    def log_message(self, fmt, *args):
        print(f"  {self.address_string()} {fmt % args}", flush=True)

    def _send(self, payload: dict | None, status: int = 200):
        body = b"" if payload is None else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Mcp-Session-Id", self.session_id)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode() if length else "{}"
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            self._send({"jsonrpc": "2.0", "id": None,
                        "error": {"code": -32700, "message": "Parse error"}}, 400)
            return

        method = message.get("method", "")
        req_id = message.get("id")
        spec = PERSONAS[self.persona]
        print(f"→ {method} {json.dumps(message.get('params', {}))[:120]}", flush=True)

        # A notification has no id and takes no reply — 202 with an empty body.
        if req_id is None:
            self._send(None, 202)
            return

        if method == "initialize":
            result = {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": spec["name"], "version": "0.1.0"},
            }
        elif method == "tools/list":
            result = {"tools": spec["tools"]}
        elif method == "tools/call":
            params = message.get("params", {})
            result = call_tool(self.persona, params.get("name", ""), params.get("arguments") or {})
        elif method == "ping":
            result = {}
        else:
            self._send({"jsonrpc": "2.0", "id": req_id,
                        "error": {"code": -32601, "message": f"Unknown method {method}"}})
            return

        self._send({"jsonrpc": "2.0", "id": req_id, "result": result})

    def do_GET(self):
        # The spec allows a GET for a server-initiated SSE stream. Nothing here
        # pushes, so say so rather than leaving a socket open forever.
        self._send(None, 405)

    def do_DELETE(self):
        self._send(None, 204)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8931)
    parser.add_argument("--persona", choices=sorted(PERSONAS), default="weather")
    args = parser.parse_args()

    Handler.persona = args.persona
    Handler.session_id = uuid.uuid4().hex
    spec = PERSONAS[args.persona]

    started = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{started}] {spec['name']} on http://0.0.0.0:{args.port}/mcp")
    print(f"  tools: {', '.join(t['name'] for t in spec['tools'])}")
    print(f"  from the emulator: http://10.0.2.2:{args.port}/mcp", flush=True)

    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
