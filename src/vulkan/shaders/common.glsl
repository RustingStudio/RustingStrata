// Shared by every kernel. Pointers arrive as buffer references in the push constants; the CUDA counterparts are in
// src/cuda/kernels.cu.
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(buffer_reference, std430, buffer_reference_align = 4) buffer F32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer I32 { int v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer U32 { uint v[]; };
layout(buffer_reference, std430, buffer_reference_align = 2) buffer U16 { uint16_t v[]; };
layout(buffer_reference, std430, buffer_reference_align = 1) buffer U8 { uint8_t v[]; };

float silu(float x) { return x / (1.0 + exp(-x)); }
float sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }

// Sums and maxima over a subgroup of 32 (warp_sum / warp_max).
float warp_sum(float v) { return subgroupAdd(v); }
float warp_max(float v) { return subgroupMax(v); }
