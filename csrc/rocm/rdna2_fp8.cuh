// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Exact fp8 e4m3fn -> fp16 conversion for RDNA2 (gfx1030), which has no fp8
// instructions.
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

namespace vllm {
namespace rdna2 {

// v_perm selectors placing bytes 0, 1 (resp. 2, 3) of a dword in the high
// byte of each 16-bit lane.
static constexpr uint32_t FP8_LO = 0x010c000cu;
static constexpr uint32_t FP8_HI = 0x030c020cu;

// Two e4m3fn bytes of q (picked by sel) widened exactly to an fp16 pair
// scaled by 2^-8: each byte lands in the high byte of a 16-bit lane, an
// arithmetic shift by one puts sign, exponent and mantissa in fp16 position
// (e4m3 subnormals become fp16 subnormals), and the mask clears the copy of
// the sign that the shift leaves in the top exponent bit.
__device__ __forceinline__ half2 fp8x2_to_half2(uint32_t q, uint32_t sel) {
  typedef short short2_t __attribute__((ext_vector_type(2)));
  const short2_t v =
      __builtin_bit_cast(short2_t, __builtin_amdgcn_perm(q, q, sel)) >> 1;
  uint32_t h = __builtin_bit_cast(uint32_t, v) & 0xBFFFBFFFu;
  return *reinterpret_cast<half2*>(&h);
}

}  // namespace rdna2
}  // namespace vllm
