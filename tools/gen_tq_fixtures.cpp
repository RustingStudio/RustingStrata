// Fixtures for the trellis types TQ2_T, TQK6 and TQK7: random blocks and what the reference decoder of Strata
// (its src/kernels/cpu/tq_states.hpp and the codebook of agentionai's llama.cpp fork) makes of them.
// Usage: g++ -std=c++17 -O1 -I<strata clone> tools/gen_tq_fixtures.cpp -o /tmp/gen_tq && /tmp/gen_tq tests/fixtures
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include "third_party/ggml/ggml-common-tq.h"
#include "src/kernels/cpu/tq_states.hpp"

static float h2f(uint16_t h) {
    int e = (h >> 10) & 31, m = h & 1023;
    float v = e == 0 ? std::ldexp((float) m, -24) : std::ldexp((float) (m | 1024), e - 25);
    return h & 0x8000 ? -v : v;
}

template <int KB> static void gen(const char* dir, const char* name) {
    const int blocks = 4, bb = 2 + 4 * KB;
    std::vector<uint8_t> raw(blocks * bb);
    std::vector<float> out(blocks * 128);
    uint32_t s = 0x9E3779B9u + KB;
    for (auto& b : raw) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; b = (uint8_t) (s >> 11); }
    for (int i = 0; i < blocks; i++) {
        uint8_t* b = raw.data() + i * bb;
        const uint16_t d = 0x2c00 + 0x123 * i + (i & 1 ? 0x8000 : 0);  // fp16 scales near 0.06, both signs
        b[0] = d & 255; b[1] = d >> 8;
        uint32_t st[32];
        strata::kernels::cpu::tq_detail::tq_states<KB>(b + 2, st);
        for (int t = 0; t < 32; t++) {
            const uint32_t x = st[t] * 0x9e3779b1u, pa = x >> 21, pb = (x >> 10) & 2047u;
            float* y = out.data() + i * 128 + 4 * t;
            y[0] = h2f(tq_lut_f16[2 * pa]) * h2f(d); y[1] = h2f(tq_lut_f16[2 * pa + 1]) * h2f(d);
            y[2] = h2f(tq_lut_f16[2 * pb]) * h2f(d); y[3] = h2f(tq_lut_f16[2 * pb + 1]) * h2f(d);
        }
    }
    FILE* f = fopen((std::string(dir) + "/" + name + ".raw").c_str(), "wb"); fwrite(raw.data(), 1, raw.size(), f); fclose(f);
    f = fopen((std::string(dir) + "/" + name + ".f32").c_str(), "wb"); fwrite(out.data(), 4, out.size(), f); fclose(f);
}

int main(int, char** argv) { gen<8>(argv[1], "TQ2_T"); gen<6>(argv[1], "TQK6"); gen<7>(argv[1], "TQK7"); }
