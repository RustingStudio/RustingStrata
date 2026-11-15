#!/usr/bin/env python3
"""Load test for `rustingstrata serve --parallel N` (or any OpenAI-compatible server).

Sends C streamed chat requests at once (each with a different prompt, so no request reuses another's cache) and prints
one line per C: total tok/s, time to first token and per-request decode tok/s.

    tools/loadtest.py --url http://127.0.0.1:8197 --clients 1,4,16,64 --max-tokens 128
"""
import argparse, http.client, json, statistics, threading, time, urllib.parse

TOPICS = ["rivers", "volcanoes", "bridges", "bread", "chess", "tides", "glass", "maps", "clocks", "wool", "salt",
          "bees", "paper", "rope", "copper", "rain"]


def one(url, i, max_tokens, out):
    u = urllib.parse.urlparse(url)
    body = json.dumps({"stream": True, "stream_options": {"include_usage": True}, "max_tokens": max_tokens,
                       "temperature": 0,
                       "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role": "user", "content":
                                     f"Request {i}. Write a long, detailed essay about {TOPICS[i % len(TOPICS)]}."}]})
    t0 = time.monotonic()
    first = tokens = None
    try:
        c = http.client.HTTPConnection(u.hostname, u.port, timeout=3600)
        c.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
        r = c.getresponse()
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status}: {r.read(200)!r}")
        for line in r:
            if not line.startswith(b"data: ") or line.startswith(b"data: [DONE]"):
                continue
            j = json.loads(line[6:])
            d = (j.get("choices") or [{}])[0].get("delta") or {}
            if first is None and (d.get("content") or d.get("reasoning_content")):
                first = time.monotonic()
            if j.get("usage"):
                tokens = j["usage"]["completion_tokens"]
        out[i] = (t0, first, time.monotonic(), tokens)
    except Exception as e:  # a failed request is reported, not hidden
        out[i] = e


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--url", default="http://127.0.0.1:8197")
    a.add_argument("--clients", default="1,4,16,64")
    a.add_argument("--max-tokens", type=int, default=128)
    a = a.parse_args()
    print("clients ok total_tok_s ttft_s(median/max) per_request_tok_s(median/min) wall_s")
    for n in map(int, a.clients.split(",")):
        out = [None] * n
        th = [threading.Thread(target=one, args=(a.url, i, a.max_tokens, out)) for i in range(n)]
        t0 = time.monotonic()
        for t in th: t.start()
        for t in th: t.join()
        wall = time.monotonic() - t0
        ok = [r for r in out if isinstance(r, tuple) and r[1] and r[3]]
        for e in sorted({str(r) for r in out if not isinstance(r, tuple)}):
            print(f"  failed ({sum(str(r) == e for r in out)}): {e}")
        if not ok:
            print(f"{n} 0"); continue
        ttft = [r[1] - r[0] for r in ok]
        # Decode speed of one request: tokens after the first, over the time after the first.
        each = [(r[3] - 1) / (r[2] - r[1]) for r in ok if r[2] > r[1]]
        print(f"{n} {len(ok)} {sum(r[3] for r in ok) / wall:.1f} {statistics.median(ttft):.2f}/{max(ttft):.2f} "
              f"{statistics.median(each):.1f}/{min(each):.1f} {wall:.1f}", flush=True)


if __name__ == "__main__":
    main()
