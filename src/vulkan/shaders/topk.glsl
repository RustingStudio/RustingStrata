// One subgroup routes one token: softmax over the n <= 32 * R logits at p[b ..] (lane holds i = lane + 32 r), top-k
// (ties to the lower index), renormalized weights. Lane j < k gets the j-th expert's id and weight.
const uint R = 16;
void topk(F32 p, uint b, uint n, uint k, float extra_scale, uint lane, out int id, out float w) {
    float v[R];
    float m = -1.0 / 0.0;
    for (uint r = 0; r < R; r++) { uint i = lane + 32 * r; v[r] = i < n ? p.v[b + i] : -1.0 / 0.0; m = max(m, v[r]); }
    m = warp_max(m);
    float s = 0.0;
    for (uint r = 0; r < R; r++) if (lane + 32 * r < n) { v[r] = expf_(v[r] - m); s += v[r]; }
    s = warp_sum(s);
    for (uint r = 0; r < R; r++) v[r] = fdiv(v[r], s);
    uint taken = 0;
    float sum = 0.0, mine = 0.0;
    id = 0;
    for (uint j = 0; j < k; j++) {
        float bv = -1.0; int bi = 0x7fffffff;
        for (uint r = 0; r < R; r++) {
            uint i = lane + 32 * r;
            if (i < n && (taken >> r & 1) == 0 && v[r] > bv) { bv = v[r]; bi = int(i); }
        }
        for (uint d = 16; d > 0; d >>= 1) {
            float ov = subgroupShuffleXor(bv, d);
            int oi = subgroupShuffleXor(bi, d);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (uint(bi & 31) == lane) taken |= 1u << (bi >> 5);
        if (lane == j) { id = bi; mine = bv; }
        sum += bv;
    }
    float scale = fdiv(1.0, max(sum, 6.103515625e-5));
    if (extra_scale != 0.0) scale *= extra_scale;
    w = mine * scale;
}
