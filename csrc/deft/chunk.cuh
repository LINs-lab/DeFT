#ifndef DEFT_CHUNK_CUH_
#define DEFT_CHUNK_CUH_

#include <thrust/device_vector.h>

namespace deft {

struct chunk_t {
  typedef uint32_t IdType;
  typedef half DType;
  uint32_t num_heads;
  uint32_t head_dim;
  uint32_t kv_stride_n;
  uint32_t kv_stride_h;
  uint32_t q_stride_n;
  uint32_t q_stride_h;

  uint32_t q_len;
  uint32_t kv_len;
  IdType* q_idx;
  IdType* kv_idx;
  uint32_t partial_o_offset;

  DType* k_data;
  DType* v_data;
  DType* q_data;

  __device__ __forceinline__ DType* protective_get_k_ptr(
      IdType idx, uint32_t head_idx) const {
    if (idx < kv_len) {
      return k_data + kv_stride_n * __ldg(kv_idx + idx) +
             head_idx * kv_stride_h;
    } else {
      return k_data;
    }
  }

  __device__ __forceinline__ DType* protective_get_v_ptr(
      IdType idx, uint32_t head_idx) const {
    if (idx < kv_len) {
      return v_data + kv_stride_n * __ldg(kv_idx + idx) +
             head_idx * kv_stride_h;
    } else {
      return v_data;
    }
  }

  __device__ __forceinline__ DType* protective_get_q_ptr(
      IdType idx, uint32_t head_idx) const {
    if (idx < q_len) {
      return q_data + q_stride_n * __ldg(q_idx + idx) + head_idx * q_stride_h;
    } else {
      return q_data;
    }
  }

  __device__ __forceinline__ uint32_t get_q_idx(IdType idx) const {
    return __ldg(q_idx + idx);
  }

  __host__ __device__ __forceinline__ int64_t kv_ptr_delta() const {
    return (int64_t(v_data) - int64_t(k_data)) / sizeof(DType);
  }
};

}  // namespace deft

#endif  // DEFT_CHUNK_CUH_
