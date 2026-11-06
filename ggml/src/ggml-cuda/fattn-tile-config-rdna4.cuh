// rdna4-kernels block 02: attention tile shape overrides for gfx1200 / gfx1201.
//
// The generic RDNA table (ggml_cuda_fattn_tile_get_config_amd_rdna) carries one hand-tuned row
// per (head size, tile width) pair. For head size 256 with a quantized KV cache it lists tile
// widths 4..32, and the width that a 512..2048 token prefill lands on (32) runs at occupancy 2
// with nbatch_K 64, while the same shape with an f16 KV cache gets occupancy 3 and nbatch_K 128.
//
// This file is additive: it returns 0 for every shape it does not override, so the caller falls
// through to the generic RDNA table and nothing else changes. Non-RDNA devices never reach it.
//
// Fields: (DKQ, DV, ncols, nthreads, occupancy, nbatch_fa, nbatch_K)
//
// Constraints, all enforced by static_assert in GGML_CUDA_FATTN_TILE_CONFIG_CASE and in
// flash_attn_tile:
//   nthreads % 32 == 0, nthreads <= 512, occupancy <= 8, nbatch_fa <= 256, nbatch_K <= 256
//   nbatch_fa % (np*32) == 0, where np = max(1, nthreads/32/ncols)
//   occupancy * per-block LDS must stay inside the CU's LDS budget; this is the binding
//   constraint at head size 256, where the Q staging alone is 16 KiB per block
static constexpr __host__ __device__ uint32_t ggml_cuda_fattn_tile_get_config_amd_rdna4(
        const int DKQ, const int DV, const int ncols, const bool is_quant) {
    if (is_quant) {
        // Measured on gfx1201 (Radeon AI PRO R9700, ROCm 10.1) with Qwen3.8-27B IQ3_S,
        // q4_0 KV cache, pp512, r=5, two passes per config, t/s at d0 / d16384 / d65536:
        //   upstream (256,2,64,64)        1334 / 1004 /  734
        //   (256,3,64,128)   [shipped]    1407 / 1089 /  797    +6.0% / +8.4% / +8.6%
        //   (256,3,64,256)                1331 / 1005 /  732    no change
        //   (256,2,128,128)               1333 / 1003 /  727    no change
        //   (256,4,64,128)                does not build: four blocks/SM exceed the LDS budget
        //
        // ncols 32 is the tile width a 512..2048 token prefill uses at head size 256 on HIP:
        // ggml_cuda_fattn_tile_get_cols_per_block returns 32 when ne01 > 16/ncols2_eff.
        // disabled for measurement: this overrode #31234's (256,256,32,256,2,64,64)
        // GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 32, 256, 3, 64, 128)
    }
    return 0;
}
