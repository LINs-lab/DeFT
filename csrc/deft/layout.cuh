#ifndef DEFT_LAYOUT_CUH_
#define DEFT_LAYOUT_CUH_

#include <string>
namespace deft {
struct tensor_info_t : public flashinfer::tensor_info_t {
  uint32_t partial_o_stride_n;
  uint32_t partial_o_stride_h;
  uint32_t partial_lse_stride_h;
  uint32_t L_stride_h;
  uint32_t row_max_stride_h;
  __host__ __device__ __forceinline__
  tensor_info_t(uint32_t qo_len, uint32_t kv_len, uint32_t num_qo_heads,
                uint32_t num_kv_heads, flashinfer::QKVLayout kv_layout, uint32_t head_dim)
      : flashinfer::tensor_info_t(qo_len, kv_len, num_qo_heads, num_kv_heads,
                                  kv_layout, head_dim) {}
};
}  // namespace deft
#endif  // DEFT_LAYOUT_CUH_
