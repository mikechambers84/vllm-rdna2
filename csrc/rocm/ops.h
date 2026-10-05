#pragma once

#include <torch/all.h>

torch::Tensor LLMM1(at::Tensor& in_a, at::Tensor& in_b,
                    const int64_t rows_per_block);

torch::Tensor wvSplitK(const at::Tensor& in_a, const at::Tensor& in_b,
                       const std::optional<at::Tensor>& in_bias,
                       const int64_t CuCount);

torch::Tensor wvSplitK_int4_g(const at::Tensor& in_a, const at::Tensor& in_b,
                              const at::Tensor& in_scale,
                              const std::optional<at::Tensor>& in_zero_points,
                              const std::optional<at::Tensor>& in_bias,
                              const int64_t CuCount, const int64_t group_size);

torch::Tensor wvSplitKrc(const at::Tensor& in_a, const at::Tensor& in_b,
                         const std::optional<at::Tensor>& in_bias,
                         const int64_t CuCount);

void wvSplitKQ(const at::Tensor& in_a, const at::Tensor& in_b,
               const std::optional<at::Tensor>& in_bias, at::Tensor& out_c,
               const at::Tensor& scale_a, const at::Tensor& scale_b,
               const int64_t CuCount);

torch::Tensor wvSplitK_rdna2(const at::Tensor& in_a, const at::Tensor& in_b,
                             const std::optional<at::Tensor>& in_bias,
                             const std::optional<at::ScalarType>& out_dtype,
                             const std::optional<at::Tensor>& out_opt);

torch::Tensor w8a8_gemv_rdna2(const at::Tensor& a, const at::Tensor& w,
                              const at::Tensor& scale_a,
                              const at::Tensor& scale_b,
                              const std::optional<at::Tensor>& bias,
                              at::ScalarType out_dtype);

torch::Tensor w8a8_gemm_rdna2(const at::Tensor& a, const at::Tensor& w,
                              const at::Tensor& scale_a,
                              const at::Tensor& scale_b,
                              const std::optional<at::Tensor>& bias,
                              at::ScalarType out_dtype);

torch::Tensor gemv_rdna2(const at::Tensor& a, const at::Tensor& w,
                         const std::optional<at::Tensor>& bias);

torch::Tensor gemv_w8a16_rdna2(const at::Tensor& a, const at::Tensor& w,
                               const at::Tensor& scale,
                               const std::optional<at::Tensor>& bias);

torch::Tensor gemv_fp8_rdna2(const at::Tensor& a, const at::Tensor& w,
                             const at::Tensor& scale, int64_t block_n,
                             int64_t block_k,
                             const std::optional<at::Tensor>& bias);

void dequant_fp8_rdna2(torch::Tensor& out, const at::Tensor& w,
                       const at::Tensor& scale, int64_t block_n,
                       int64_t block_k);

torch::Tensor gemm_rocblas_rdna2(const at::Tensor& a, const at::Tensor& w,
                                 int64_t solution, bool w_kn);

std::vector<int64_t> gemm_rocblas_solutions_rdna2(const at::Tensor& a,
                                                  const at::Tensor& w,
                                                  bool w_kn);

std::string rocblas_version_rdna2();

torch::Tensor gemm_w4a16_exl_rdna2(const at::Tensor& a, const at::Tensor& w,
                                   const at::Tensor& zeros,
                                   const at::Tensor& scales, bool symmetric,
                                   bool use_v2_format, int64_t cfg);

void moe_int8_decode_rdna2(torch::Tensor& output, const torch::Tensor& x,
                           const torch::Tensor& topk_ids,
                           const torch::Tensor& topk_weights,
                           const torch::Tensor& w13, const torch::Tensor& s13,
                           const torch::Tensor& w2, const torch::Tensor& s2,
                           torch::Tensor& act);

void moe_fp8_decode_rdna2(torch::Tensor& output, const torch::Tensor& x,
                          const torch::Tensor& topk_ids,
                          const torch::Tensor& topk_weights,
                          const torch::Tensor& w13, const torch::Tensor& s13,
                          const torch::Tensor& w2, const torch::Tensor& s2,
                          int64_t block_n, int64_t block_k, torch::Tensor& act);

void moe_fp8_gemm_rdna2(torch::Tensor& output, const torch::Tensor& a,
                        const torch::Tensor& w, const torch::Tensor& scales,
                        int64_t block_n, int64_t block_k,
                        const torch::Tensor& sorted_ids,
                        const torch::Tensor& expert_ids,
                        const torch::Tensor& num_tokens_post_padded,
                        const torch::Tensor& topk_weights, int64_t top_k,
                        bool mul_routed_weight, int64_t block_m);

void moe_wna16_decode_rdna2(torch::Tensor& output, const torch::Tensor& x,
                            const torch::Tensor& topk_ids,
                            const torch::Tensor& topk_weights,
                            const torch::Tensor& w13, const torch::Tensor& s13,
                            const torch::Tensor& w2, const torch::Tensor& s2,
                            const std::optional<torch::Tensor>& z13,
                            const std::optional<torch::Tensor>& z2,
                            torch::Tensor& act);

void moe_wna16_gemm_rdna2(torch::Tensor& output, const torch::Tensor& a,
                          const torch::Tensor& w, const torch::Tensor& scales,
                          const std::optional<torch::Tensor>& zeros,
                          const torch::Tensor& sorted_ids,
                          const torch::Tensor& expert_ids,
                          const torch::Tensor& num_tokens_post_padded,
                          const torch::Tensor& topk_weights, int64_t top_k,
                          bool mul_routed_weight, int64_t block_m);

void moe_int8_gemm_rdna2(torch::Tensor& output, const torch::Tensor& a,
                         const torch::Tensor& a_scale, const torch::Tensor& w,
                         const torch::Tensor& w_scale,
                         const torch::Tensor& sorted_ids,
                         const torch::Tensor& expert_ids,
                         const torch::Tensor& num_tokens_post_padded,
                         const torch::Tensor& topk_weights, int64_t top_k,
                         bool mul_routed_weight, int64_t block_m);

void unified_attention_rdna2(torch::Tensor& out, const torch::Tensor& q,
                             const torch::Tensor& k_cache,
                             const torch::Tensor& v_cache,
                             const torch::Tensor& cu_seqlens_q,
                             const torch::Tensor& seqused_k,
                             const torch::Tensor& block_table, double scale,
                             int64_t window, double softcap,
                             const std::optional<torch::Tensor>& sinks,
                             const std::optional<torch::Tensor>& k_scale,
                             const std::optional<torch::Tensor>& v_scale);

void decode_attention_rdna2(
    const torch::Tensor& q, const torch::Tensor& k_cache,
    const torch::Tensor& v_cache, torch::Tensor& segm_out,
    torch::Tensor& segm_max, torch::Tensor& segm_sum,
    const torch::Tensor& cu_seqlens_q, const torch::Tensor& seqused_k,
    const torch::Tensor& block_table, int64_t tile, int64_t max_seqlen_q,
    double scale, const std::optional<torch::Tensor>& k_scale,
    const std::optional<torch::Tensor>& v_scale, int64_t window, double softcap,
    const std::optional<torch::Tensor>& sinks);

void gdn_post_conv_rdna2(const torch::Tensor& conv_output,
                         const torch::Tensor& a, const torch::Tensor& b,
                         const torch::Tensor& A_log,
                         const torch::Tensor& dt_bias, torch::Tensor& q,
                         torch::Tensor& k, torch::Tensor& v, torch::Tensor& g,
                         torch::Tensor& beta, bool apply_l2norm,
                         bool output_g_exp, double eps);

void gdn_wy_rdna2(const torch::Tensor& k, const torch::Tensor& v,
                  const torch::Tensor& beta, const torch::Tensor& g,
                  torch::Tensor& g_cum, torch::Tensor& w, torch::Tensor& u,
                  const torch::Tensor& cu_seqlens,
                  const torch::Tensor& chunk_indices);

void gdn_fwd_h_rdna2(const torch::Tensor& k, const torch::Tensor& w,
                     const torch::Tensor& u, const torch::Tensor& g_cum,
                     const std::optional<torch::Tensor>& h0, torch::Tensor& h,
                     torch::Tensor& v_new,
                     const std::optional<torch::Tensor>& ht,
                     const torch::Tensor& cu_seqlens,
                     const torch::Tensor& chunk_offsets);

void gdn_fwd_o_rdna2(const torch::Tensor& q, const torch::Tensor& k,
                     const torch::Tensor& v_new, const torch::Tensor& h,
                     const torch::Tensor& g_cum, torch::Tensor& o,
                     const torch::Tensor& cu_seqlens,
                     const torch::Tensor& chunk_indices, double scale);

torch::Tensor gptq_gemm_rdna3(torch::Tensor a, torch::Tensor b_q_weight,
                              torch::Tensor b_qzeros, torch::Tensor b_scales,
                              bool use_v2_format);

torch::Tensor gptq_gemm_rdna3_wmma(torch::Tensor a, torch::Tensor b_q_weight,
                                   torch::Tensor b_qzeros,
                                   torch::Tensor b_scales, bool use_v2_format);

void moe_gptq_gemm_rdna3(torch::Tensor a, torch::Tensor c,
                         torch::Tensor b_q_weight, torch::Tensor b_scales,
                         torch::Tensor b_qzeros, torch::Tensor topk_weights,
                         torch::Tensor sorted_token_ids,
                         torch::Tensor expert_ids,
                         torch::Tensor num_tokens_post_padded, int64_t top_k,
                         int64_t block_size_m, bool mul_topk_weight,
                         int64_t output_topk);

void paged_attention(
    torch::Tensor& out, torch::Tensor& exp_sums, torch::Tensor& max_logits,
    torch::Tensor& tmp_out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int64_t num_kv_heads, double scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens,
    const std::optional<torch::Tensor>& query_start_loc, int64_t block_size,
    int64_t max_seq_len, const std::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, torch::Tensor& k_scale,
    torch::Tensor& v_scale, const std::optional<torch::Tensor>& fp8_out_scale,
    const std::string& mfma_type);
