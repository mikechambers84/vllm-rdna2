#include "core/registration.h"
#include "rocm/ops.h"

// Note on op signatures:
// The X_meta signatures are for the meta functions corresponding to op X.
// They must be kept in sync with the signature for X. Generally, only
// functions that return Tensors require a meta function.
//
// See the following links for detailed docs on op registration and function
// schemas.
// https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ptttacy8y1u9
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#annotations

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, rocm_ops) {
  // vLLM custom ops for rocm

// skinny_gemms.cu (LLMM1/wvSplitK/wvSplitKrc/wvSplitKQ) is excluded on gfx1250
// (gfx9/gfx11 ISA, unsupported there); skip these registrations to avoid
// undefined symbols. vLLM uses default/Triton GEMM for these ops on gfx1250.
#ifndef VLLM_SKIP_SKINNY_GEMMS
  // Custom gemm op for matrix-vector multiplication
  rocm_ops.def(
      "LLMM1(Tensor in_a, Tensor in_b, int rows_per_block) -> "
      "Tensor");
  rocm_ops.impl("LLMM1", torch::kCUDA, &LLMM1);

  // Custom gemm op for skinny matrix-matrix multiplication
  rocm_ops.def(
      "wvSplitK(Tensor in_a, Tensor in_b, Tensor? in_bias, int CuCount) -> "
      "Tensor");
  rocm_ops.impl("wvSplitK", torch::kCUDA, &wvSplitK);

  // W4A16 grouped skinny GEMM: packed int4 weights, per-group scales,
  // optional zero points [M/8, K/group_size] int32 for asymmetric
  // quantization
  rocm_ops.def(
      "wvSplitK_int4_g(Tensor in_a, Tensor in_b, Tensor in_scale, "
      "Tensor? in_zero_points, Tensor? in_bias, int CuCount, "
      "int group_size) -> Tensor");
  rocm_ops.impl("wvSplitK_int4_g", torch::kCUDA, &wvSplitK_int4_g);

  // Custom gemm op for skinny matrix-matrix multiplication
  rocm_ops.def(
      "wvSplitKrc(Tensor in_a, Tensor in_b, Tensor? in_bias, int CuCount) -> "
      "Tensor");
  rocm_ops.impl("wvSplitKrc", torch::kCUDA, &wvSplitKrc);

  // wvSplitK for fp8
  rocm_ops.def(
      "wvSplitKQ(Tensor in_a, Tensor in_b, Tensor? in_bias, Tensor! out_c, "
      "Tensor scale_a, "
      "          Tensor scale_b, int CuCount) -> ()");
  rocm_ops.impl("wvSplitKQ", torch::kCUDA, &wvSplitKQ);
#endif  // VLLM_SKIP_SKINNY_GEMMS

#ifdef VLLM_ROCM_GFX1030
  // fp16/bf16 skinny GEMV (N <= 16) for RDNA2, where wvSplitK is unavailable.
  rocm_ops.def(
      "wvSplitK_rdna2(Tensor in_a, Tensor in_b, Tensor? in_bias, "
      "ScalarType? out_dtype, Tensor(a!)? out) -> Tensor");
  rocm_ops.impl("wvSplitK_rdna2", torch::kCUDA, &wvSplitK_rdna2);
  // W8A8 int8 GEMV (M <= 8) on v_dot4_i32_i8 for decode.
  rocm_ops.def(
      "w8a8_gemv_rdna2(Tensor a, Tensor w, Tensor scale_a, Tensor scale_b, "
      "Tensor? bias, ScalarType out_dtype) -> Tensor");
  rocm_ops.impl("w8a8_gemv_rdna2", torch::kCUDA, &w8a8_gemv_rdna2);
  // W8A8 int8 GEMM on v_dot4_i32_i8 for prefill.
  rocm_ops.def(
      "w8a8_gemm_rdna2(Tensor a, Tensor w, Tensor scale_a, Tensor scale_b, "
      "Tensor? bias, ScalarType out_dtype) -> Tensor");
  rocm_ops.impl("w8a8_gemm_rdna2", torch::kCUDA, &w8a8_gemm_rdna2);
  // fp16/bf16 GEMV (M <= 8).
  rocm_ops.def("gemv_rdna2(Tensor a, Tensor w, Tensor? bias) -> Tensor");
  rocm_ops.impl("gemv_rdna2", torch::kCUDA, &gemv_rdna2);
  // GEMM on K-major int8 or fp8 e4m3fn weights (per-channel scales, fp8 also
  // with 2D block scales) with fp16/bf16 activations, or int8 weights with
  // int8 activations (W8A8, per-token scale_a), any number of rows.
  rocm_ops.def(
      "gemm_w8_rdna2(Tensor a, Tensor w, Tensor scale, Tensor? block_scale, "
      "int block_k, Tensor? bias, int cfg, Tensor? scale_a=None, "
      "ScalarType? out_dtype=None) -> Tensor");
  rocm_ops.impl("gemm_w8_rdna2", torch::kCUDA, &gemm_w8_rdna2);
  // fp16/bf16 GEMM with an explicit rocBLAS solution, and the solutions and
  // rocBLAS version for tuning them.
  rocm_ops.def(
      "gemm_rocblas_rdna2(Tensor a, Tensor w, int solution, bool w_kn) -> "
      "Tensor");
  rocm_ops.impl("gemm_rocblas_rdna2", torch::kCUDA, &gemm_rocblas_rdna2);
  rocm_ops.def(
      "gemm_rocblas_solutions_rdna2(Tensor a, Tensor w, bool w_kn) -> int[]");
  rocm_ops.impl("gemm_rocblas_solutions_rdna2", torch::kCUDA,
                &gemm_rocblas_solutions_rdna2);
  rocm_ops.def("rocblas_version_rdna2() -> str");
  rocm_ops.impl("rocblas_version_rdna2", &rocblas_version_rdna2);
  // 4-bit GPTQ GEMM for decode and small batches on Exllama's weight layout.
  rocm_ops.def(
      "gemm_w4a16_exl_rdna2(Tensor a, Tensor w, Tensor zeros, Tensor scales, "
      "bool symmetric, bool use_v2_format, int cfg) -> Tensor");
  rocm_ops.impl("gemm_w4a16_exl_rdna2", torch::kCUDA, &gemm_w4a16_exl_rdna2);
  // int8-weight fused-MoE decode (few tokens) on the Triton int8 layout.
  rocm_ops.def(
      "moe_int8_decode_rdna2(Tensor! output, Tensor x, Tensor topk_ids, "
      "Tensor topk_weights, Tensor w13, Tensor s13, Tensor w2, Tensor s2, "
      "Tensor! act) -> ()");
  rocm_ops.impl("moe_int8_decode_rdna2", torch::kCUDA, &moe_int8_decode_rdna2);
  // fp8-weight fused-MoE decode and prefill GEMM (fp16 activations).
  rocm_ops.def(
      "moe_fp8_decode_rdna2(Tensor! output, Tensor x, Tensor topk_ids, "
      "Tensor topk_weights, Tensor w13, Tensor s13, Tensor w2, Tensor s2, "
      "int block_n, int block_k, Tensor! act) -> ()");
  rocm_ops.impl("moe_fp8_decode_rdna2", torch::kCUDA, &moe_fp8_decode_rdna2);
  rocm_ops.def(
      "moe_fp8_gemm_rdna2(Tensor! output, Tensor a, Tensor w, Tensor scales, "
      "int block_n, int block_k, Tensor sorted_ids, Tensor expert_ids, "
      "Tensor num_tokens_post_padded, Tensor topk_weights, int top_k, "
      "bool mul_routed_weight, int block_m) -> ()");
  rocm_ops.impl("moe_fp8_gemm_rdna2", torch::kCUDA, &moe_fp8_gemm_rdna2);
  // W4A16 fused-MoE decode (few tokens) on the Triton WNA16 weight layout.
  rocm_ops.def(
      "moe_wna16_decode_rdna2(Tensor! output, Tensor x, Tensor topk_ids, "
      "Tensor topk_weights, Tensor w13, Tensor s13, Tensor w2, Tensor s2, "
      "Tensor? z13, Tensor? z2, Tensor! act) -> ()");
  rocm_ops.impl("moe_wna16_decode_rdna2", torch::kCUDA,
                &moe_wna16_decode_rdna2);
  // Grouped W4A16 GEMM for fused-MoE prefill on the Triton WNA16 layout.
  rocm_ops.def(
      "moe_wna16_gemm_rdna2(Tensor! output, Tensor a, Tensor w, Tensor scales, "
      "Tensor? zeros, Tensor sorted_ids, Tensor expert_ids, Tensor "
      "num_tokens_post_padded, "
      "Tensor topk_weights, int top_k, bool mul_routed_weight, int block_m) -> "
      "()");
  rocm_ops.impl("moe_wna16_gemm_rdna2", torch::kCUDA, &moe_wna16_gemm_rdna2);
  // Grouped W8A8 GEMM for fused-MoE prefill on the Triton int8 layout.
  rocm_ops.def(
      "moe_int8_gemm_rdna2(Tensor! output, Tensor a, Tensor a_scale, Tensor w, "
      "Tensor w_scale, Tensor sorted_ids, Tensor expert_ids, "
      "Tensor num_tokens_post_padded, Tensor topk_weights, int top_k, "
      "bool mul_routed_weight, int block_m) -> ()");
  rocm_ops.impl("moe_int8_gemm_rdna2", torch::kCUDA, &moe_int8_gemm_rdna2);
  // Paged causal GQA attention for prefill (fp16, head 64/128/256; sliding
  // window, softcap, sinks).
  rocm_ops.def(
      "unified_attention_rdna2(Tensor! out, Tensor q, Tensor k_cache, "
      "Tensor v_cache, Tensor cu_seqlens_q, Tensor seqused_k, "
      "Tensor block_table, float scale, int window, float softcap, "
      "Tensor? sinks, Tensor? k_scale=None, Tensor? v_scale=None) -> ()");
  rocm_ops.impl("unified_attention_rdna2", torch::kCUDA,
                &unified_attention_rdna2);
  // Split-KV decode / spec-verify attention, head 256 (segment partials).
  rocm_ops.def(
      "decode_attention_rdna2(Tensor q, Tensor k_cache, Tensor v_cache, "
      "Tensor! segm_out, Tensor! segm_max, Tensor! segm_sum, "
      "Tensor cu_seqlens_q, Tensor seqused_k, Tensor block_table, int tile, "
      "int max_seqlen_q, float scale, Tensor? k_scale=None, "
      "Tensor? v_scale=None, int window=0, float softcap=0.0, "
      "Tensor? sinks=None) -> ()");
  rocm_ops.impl("decode_attention_rdna2", torch::kCUDA,
                &decode_attention_rdna2);
  // Gated DeltaNet prefill: post-conv1d q/k/v split, l2 norm and gating.
  rocm_ops.def(
      "gdn_post_conv_rdna2(Tensor conv_output, Tensor a, Tensor b, "
      "Tensor A_log, Tensor dt_bias, Tensor! q, Tensor! k, Tensor! v, "
      "Tensor! g, Tensor! beta, bool apply_l2norm, bool output_g_exp, "
      "float eps) -> ()");
  rocm_ops.impl("gdn_post_conv_rdna2", torch::kCUDA, &gdn_post_conv_rdna2);
  // Gated DeltaNet prefill: fused chunk-local cumsum + WY representation.
  rocm_ops.def(
      "gdn_wy_rdna2(Tensor k, Tensor v, Tensor beta, Tensor g, Tensor! g_cum, "
      "Tensor! w, Tensor! u, Tensor cu_seqlens, Tensor chunk_indices) -> ()");
  rocm_ops.impl("gdn_wy_rdna2", torch::kCUDA, &gdn_wy_rdna2);
  // Gated DeltaNet prefill: chunk state recurrence and chunk output.
  rocm_ops.def(
      "gdn_fwd_h_rdna2(Tensor k, Tensor w, Tensor u, Tensor g_cum, Tensor? h0, "
      "Tensor! h, Tensor! v_new, Tensor(a!)? ht, Tensor cu_seqlens, "
      "Tensor chunk_offsets) -> ()");
  rocm_ops.impl("gdn_fwd_h_rdna2", torch::kCUDA, &gdn_fwd_h_rdna2);
  rocm_ops.def(
      "gdn_fwd_o_rdna2(Tensor q, Tensor k, Tensor v_new, Tensor h, "
      "Tensor g_cum, Tensor! o, Tensor cu_seqlens, Tensor chunk_indices, "
      "float scale) -> ()");
  rocm_ops.impl("gdn_fwd_o_rdna2", torch::kCUDA, &gdn_fwd_o_rdna2);
#endif  // VLLM_ROCM_GFX1030

#ifdef VLLM_ROCM_GFX1100
  // W4A16 GPTQ kernels for AMD RDNA3 (gfx1100).
  rocm_ops.def(
      "gptq_gemm_rdna3(Tensor a, Tensor b_q_weight, Tensor b_qzeros, "
      "Tensor b_scales, bool use_v2_format) -> Tensor");
  rocm_ops.impl("gptq_gemm_rdna3", torch::kCUDA, &gptq_gemm_rdna3);

  rocm_ops.def(
      "gptq_gemm_rdna3_wmma(Tensor a, Tensor b_q_weight, Tensor b_qzeros, "
      "Tensor b_scales, bool use_v2_format) -> Tensor");
  rocm_ops.impl("gptq_gemm_rdna3_wmma", torch::kCUDA, &gptq_gemm_rdna3_wmma);

  rocm_ops.def(
      "moe_gptq_gemm_rdna3(Tensor a, Tensor! c, Tensor b_q_weight, "
      "Tensor b_scales, Tensor b_qzeros, Tensor topk_weights, "
      "Tensor sorted_token_ids, Tensor expert_ids, "
      "Tensor num_tokens_post_padded, "
      "int top_k, int block_size_m, bool mul_topk_weight, "
      "int output_topk) -> ()");
  rocm_ops.impl("moe_gptq_gemm_rdna3", torch::kCUDA, &moe_gptq_gemm_rdna3);
#endif

  // Custom attention op
  // Compute the attention between an input query and the cached
  // keys/values using PagedAttention.
  rocm_ops.def(
      "paged_attention(Tensor! out, Tensor exp_sums,"
      "                Tensor max_logits, Tensor tmp_out,"
      "                Tensor query, Tensor key_cache,"
      "                Tensor value_cache, int num_kv_heads,"
      "                float scale, Tensor block_tables,"
      "                Tensor seq_lens,"
      "                Tensor? query_start_loc,"
      "                int block_size,"
      "                int max_seq_len,"
      "                Tensor? alibi_slopes,"
      "                str kv_cache_dtype,"
      "                Tensor k_scale, Tensor v_scale,"
      "                Tensor? fp8_out_scale,"
      "                str mfma_type) -> ()");
  rocm_ops.impl("paged_attention", torch::kCUDA, &paged_attention);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
