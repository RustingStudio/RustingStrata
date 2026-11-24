#!/usr/bin/env python3
"""Compare two `perplexity --kld-base` files: KLD, top-token agreement, both PPLs, and how many cells moved.

    tools/kld_compare.py a.bin b.bin [--quiet]

The file (written by `log_probs` in src/main.rs) is llama-perplexity's: `_logits_`, `n`, `n_vocab`, `n_chunks`
(u32 each), the `n_chunks * n` token ids (u32), then one row per scored position: the row's scale and offset (two
f32, as their u16 halves) and `n_vocab` u16 values, `logp(v) = offset + q(v) * scale` after dequantization. Two
builds of the engine that run the same kernels write byte-identical files; a kernel whose sums are reordered moves
the log-probabilities by a few ulp, below the u16 quantum (16 nats / 65535 = 2.4e-4), so most cells stay equal and
a few move by one step.

Exit status: 0 when the files are byte-identical, 1 when they differ (the report says by how much).
"""

import argparse
import struct
import sys

import numpy as np


def header(f):
    magic = f.read(8)
    if magic != b"_logits_":
        sys.exit(f"{f.name}: not a --kld-base file (starts with {magic[:8]!r})")
    n, nv, nch = struct.unpack("<III", f.read(12))
    ids = np.frombuffer(f.read(4 * n * nch), dtype="<u4")
    return n, nv, nch, ids


def rows(f, n, nv, nch):
    first = n // 2
    per = n - 1 - first
    rowlen = 4 + nv + (nv & 1)
    for c in range(nch):
        for i in range(per):
            raw = f.read(2 * rowlen)
            if len(raw) != 2 * rowlen:
                sys.exit(f"{f.name}: truncated at chunk {c}, position {i}")
            u = np.frombuffer(raw, dtype="<u2")
            scale = struct.unpack("<f", struct.pack("<I", int(u[0]) | (int(u[1]) << 16)))[0]
            offset = struct.unpack("<f", struct.pack("<I", int(u[2]) | (int(u[3]) << 16)))[0]
            q = u[4 : 4 + nv]
            yield c, i, scale, offset, q


def same_bytes(a, b, chunk=1 << 20):
    """Compare two files in chunks, without holding either one in memory."""
    with open(a, "rb") as fa, open(b, "rb") as fb:
        while True:
            x, y = fa.read(chunk), fb.read(chunk)
            if x != y:
                return False
            if not x:
                return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base")
    ap.add_argument("new")
    ap.add_argument("--quiet", action="store_true", help="only the verdict")
    a = ap.parse_args()

    fa, fb = open(a.base, "rb"), open(a.new, "rb")
    na, nva, ncha, idsa = header(fa)
    nb, nvb, nchb, idsb = header(fb)
    if (na, nva, ncha) != (nb, nvb, nchb):
        sys.exit(f"shape differs: {na},{nva},{ncha} vs {nb},{nvb},{nchb}")
    if not np.array_equal(idsa, idsb):
        sys.exit("the two files scored different tokens")
    first = na // 2
    per = na - 1 - first

    same = same_bytes(a.base, a.new)

    klds, top1, cells, moved, worst, nll_a, nll_b = [], 0, 0, 0, 0.0, 0.0, 0.0
    for (c, i, sa, oa, qa), (_, _, sb, ob, qb) in zip(rows(fa, na, nva, ncha), rows(fb, nb, nvb, nchb)):
        la = oa + qa.astype(np.float64) * sa
        lb = ob + qb.astype(np.float64) * sb
        pa = np.exp(la)
        klds.append(float(np.sum(pa * (la - lb))))
        top1 += int(np.argmax(la) == np.argmax(lb))
        d = int(np.count_nonzero(qa != qb))
        cells += len(qa)
        moved += 1 if d else 0
        worst = max(worst, float(np.max(np.abs(la - lb))))
        t = idsa[c * na + first + i + 1]  # the row scores the token after its position
        nll_a -= la[t]
        nll_b -= lb[t]
    total = per * ncha
    kld = float(np.mean(klds))
    if not a.quiet:
        print(f"positions {total}, vocab {nva}, chunks {ncha}")
        print(f"PPL       base {np.exp(nll_a / total):.6f}   new {np.exp(nll_b / total):.6f}   "
              f"delta {np.exp(nll_b / total) - np.exp(nll_a / total):+.6f}")
        print(f"top-1     {top1}/{total} ({100.0 * top1 / total:.3f}%)")
        print(f"kld       mean {kld:.3e} nats, max row {max(klds):.3e}")
        print(f"cells     {cells} compared, {moved} rows with a moved value, max |dlogp| {worst:.3e} nats")
    verdict = "identical bytes" if same else ("reordered only" if kld < 1e-3 and top1 == total else "DIFFERENT")
    print(f"verdict   {verdict}")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
