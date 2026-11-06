// rdna4-kernels: per-shape attention tile config overrides for gfx1200 / gfx1201.
//
// ggml_cuda_fattn_tile_get_config_amd_rdna carries one hand-tuned row per (head size, tile
// width) pair, shared by every RDNA part. This header is the hook for tuning gfx1200/gfx1201
// separately: add a row, rebuild, measure. Nothing else in the tree needs to change.
//
// Returns 0 for every shape it does not list, so the caller falls through to the generic RDNA
// table and no other architecture, head size, tile width or model changes behaviour.
//
// Fields: (DKQ, DV, ncols, nthreads, occupancy, nbatch_fa, nbatch_K)
//
// Constraints, enforced by static_assert in GGML_CUDA_FATTN_TILE_CONFIG_CASE and in
// flash_attn_tile:
//   nthreads % 32 == 0, nthreads <= 512, occupancy <= 8, nbatch_fa <= 256, nbatch_K <= 256
//   nbatch_fa % (np*32) == 0, where np = max(1, nthreads/32/ncols)
//   occupancy * per-block LDS must fit the CU's LDS budget. That binds first at head size 256,
//   where the Q staging alone is 16 KiB per block: (256,256,32,256,4,...) does not build.
//
// Two measurement notes from gfx1201 (Radeon AI PRO R9700, ROCm 10.1, Qwen3.8-27B IQ3_S,
// q4_0 KV cache, pp512):
//
//   1. Clock ramp dominates small deltas. The same binary measured 1334 / 1012 / 728 t/s at
//      d0 / d16384 / d65536 from idle, and 1397 / 1020 / 748 after a 60 s burn-in. Burn in and
//      interleave A/B; llama-bench's single warmup iteration is not enough after an idle gap.
//
//   2. On a tree that carries quantized-KV tile support (upstream PR #31234) the quantized
//      table has its own rows: head size 256 x tile width 32 is (256,2,64,64) there versus
//      (256,3,64,128) in the shared table. Reverting it to the shared value measured no gain
//      once clock state was controlled, so no row is shipped here.
static constexpr __host__ __device__ uint32_t ggml_cuda_fattn_tile_get_config_amd_rdna4(
        const int DKQ, const int DV, const int ncols) {
    (void) DKQ;
    (void) DV;
    (void) ncols;
    return 0;
}
