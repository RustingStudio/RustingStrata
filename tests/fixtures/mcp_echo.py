#!/usr/bin/env python3
"""Fake stdio MCP server for tests/mcp.rs: tools `echo` (returns its text), `fail` (isError) and `hang` (never answers)."""
import json, sys

TOOLS = [
    {"name": "echo", "description": "Returns the text", "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}},
    {"name": "fail", "description": "Always fails", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "hang", "description": "Never answers", "inputSchema": {"type": "object", "properties": {}}},
]

def send(m):
    sys.stdout.write(json.dumps(m) + "\n")
    sys.stdout.flush()

for line in sys.stdin:
    m = json.loads(line)
    i, method, p = m.get("id"), m.get("method"), m.get("params") or {}
    if i is None or method is None:
        continue  # a notification, or the reply to our ping
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": i, "result": {"protocolVersion": p["protocolVersion"], "capabilities": {"tools": {}}, "serverInfo": {"name": "echo", "version": "0"}}})
        send({"jsonrpc": "2.0", "id": "srv-1", "method": "ping"})  # the client must answer and go on
    elif method == "tools/list":
        # Two pages, so the cursor loop is covered.
        if "cursor" not in p:
            send({"jsonrpc": "2.0", "id": i, "result": {"tools": TOOLS[:1], "nextCursor": "1"}})
        else:
            send({"jsonrpc": "2.0", "id": i, "result": {"tools": TOOLS[1:]}})
    elif method == "tools/call":
        name, a = p["name"], p.get("arguments") or {}
        if name == "echo":
            send({"jsonrpc": "2.0", "id": i, "result": {"content": [{"type": "text", "text": a["text"]}, {"type": "image", "data": "AA==", "mimeType": "image/png"}]}})
        elif name == "fail":
            send({"jsonrpc": "2.0", "id": i, "result": {"content": [{"type": "text", "text": "boom"}], "isError": True}})
        elif name == "hang":
            pass
        else:
            send({"jsonrpc": "2.0", "id": i, "error": {"code": -32602, "message": "unknown tool " + name}})
    else:
        send({"jsonrpc": "2.0", "id": i, "error": {"code": -32601, "message": "no " + method}})
