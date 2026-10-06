# M1 parity gate: llama.cpp greedy continuations, teacher-forced through `strata score`.
# Usage: python -I tools/parity.py <model.gguf> [--n 32] [--limit 20] [--port 8199]
# Phase 1 needs llama-server running on --port (e.g. ~/llama.cpp/build/bin/llama-server -m M --port 8199);
# its outputs are cached in target/parity-ref.json so phase 2 can run without it.
import argparse, json, pathlib, subprocess, sys, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("--n", type=int, default=32)
ap.add_argument("--limit", type=int, default=20); ap.add_argument("--port", type=int, default=8199)
a = ap.parse_args()
root = pathlib.Path(__file__).resolve().parent.parent
cache = root / "target/parity-ref.json"

def post(path, body):
    r = urllib.request.Request(f"http://127.0.0.1:{a.port}{path}", json.dumps(body).encode(),
                               {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r))

prompts = json.load(open(root / "tools/prompts.json"))[: a.limit]
ref = json.load(open(cache)) if cache.exists() else {}
for p in prompts:
    if p in ref and len(ref[p]["cont"]) >= a.n:
        continue
    ids = post("/tokenize", {"content": p, "add_special": False, "parse_special": True})["tokens"]
    r = post("/completion", {"prompt": ids, "n_predict": a.n, "temperature": 0, "top_k": 1, "cache_prompt": False,
                             "return_tokens": True})
    ref[p] = {"ids": ids, "cont": r["tokens"]}
    json.dump(ref, open(cache, "w"))
    print("ref", len(ids), "prompt tokens ->", len(r["tokens"]), flush=True)

agree = total = 0
rows = []
for p in prompts:
    ids, cont = ref[p]["ids"], ref[p]["cont"][: a.n]
    out = subprocess.run([str(root / "target/release/strata"), "score", "--model", a.model, "--input", "-"],
                         input=json.dumps({"prompt": ids, "continuation": cont}), capture_output=True, text=True, check=True)
    res = json.loads(out.stdout)
    mine = res["argmax"]
    k = sum(x == y for x, y in zip(mine, cont))
    # First disagreement and our top-2 logprob gap there (small gap = near-tie).
    bad = next((i for i, (x, y) in enumerate(zip(mine, cont)) if x != y), None)
    gap = None if bad is None else round(res["top5"][bad][0][1] - res["top5"][bad][1][1], 4)
    agree += k; total += len(cont)
    rows.append({"prompt": p.split("\n")[1][:50], "prompt_tokens": len(ids), "agree": k, "n": len(cont),
                 "first_miss": bad, "top2_gap_at_miss": gap})
    print(rows[-1], flush=True)
pct = 100 * agree / total
print(f"TOP1 AGREEMENT {agree}/{total} = {pct:.2f}%")
json.dump({"rows": rows, "agree": agree, "total": total}, open(root / "target/parity-result.json", "w"), indent=1)
sys.exit(0 if agree >= 0.99 * total else 1)
