"""Headless WebSocket sniffer for the overlay server.

Connects to ws://127.0.0.1:8765/ws, prints each received subtitle message,
exits after receiving N messages or TIMEOUT seconds.

The server refuses token-free clients, so the sniffer first reads the
launch's session token from GET /ws-token (served while overlay.browser is
on, the shipped default).

Usage:
    python scripts/test_overlay_ws.py [count] [timeout_s]
"""
from __future__ import annotations

import asyncio
import json
import sys
import time

import httpx
import websockets


async def main(count: int, timeout_s: float) -> int:
    url = "ws://127.0.0.1:8765/ws"
    print(f"[sniffer] connecting to {url}", flush=True)
    try:
        # trust_env=False: proxy settings must never apply to loopback.
        r = httpx.get(
            "http://127.0.0.1:8765/ws-token", timeout=5.0, trust_env=False,
        )
        r.raise_for_status()
        token = r.json()["token"]
        async with websockets.connect(f"{url}?t={token}", open_timeout=5) as ws:
            print("[sniffer] connected", flush=True)
            received = 0
            deadline = time.perf_counter() + timeout_s
            while received < count:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                msg = json.loads(raw)
                received += 1
                print(f"[{received:02d}] jp={msg.get('jp')!r}  en={msg.get('en')!r}", flush=True)
            print(f"[sniffer] received {received}/{count}", flush=True)
            return 0 if received > 0 else 1
    except Exception as e:
        print(f"[sniffer] error: {e}", flush=True)
        return 2


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    timeout_s = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0
    sys.exit(asyncio.run(main(count, timeout_s)))
