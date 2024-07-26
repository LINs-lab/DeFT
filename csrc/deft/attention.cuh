#ifndef DEFT_ATTENTION_CUH_
#define DEFT_ATTENTION_CUH_
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <flashinfer/attention/state.cuh>
#include <flashinfer/cp_async.cuh>
#include <flashinfer/math.cuh>
#include <flashinfer/page.cuh>
#include <flashinfer/pos_enc.cuh>
#include <flashinfer/utils.cuh>
#include <flashinfer/vec_dtypes.cuh>

#include "deft/chunk.cuh"
#include "deft/layout.cuh"
#include "deft/utils.cuh"

namespace deft {
__global__ void dummy_kernel(float* __restrict__ output,
                             const float* __restrict__ input, const int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = input[idx];
  }
}

void dummy(float* output, const float* input, int size) {
  dim3 block(256);
  dim3 grid((size + block.x - 1) / block.x);
  dummy_kernel<<<grid, block>>>(output, input, size);
}

typedef half DTypeQ;
typedef half DTypeKV;
typedef half DTypeOut;
typedef uint32_t IdType;
using flashinfer::ceil_div;
using flashinfer::paged_kv_t;
using flashinfer::PageStorage;
using flashinfer::state_t;
using flashinfer::vec_t;
namespace cp_async = flashinfer::cp_async;
namespace math = flashinfer::math;
using cp_async::PrefetchMode;
using cp_async::SharedMemFillMode;

namespace cg = cooperative_groups;

template <uint32_t vec_size, uint32_t bdx, uint32_t tile_size, typename T>
__device__ __forceinline__ void compute_qk(const T* smem,
                                           uint32_t compute_stage_idx,
                                           const vec_t<float, vec_size>& q_vec,
                                           uint32_t iter_base,
                                           uint32_t iter_bound, float* s,
                                           state_t<vec_size>& st) {
  uint32_t tx = threadIdx.x, tz = threadIdx.z;
  float m_prev = st.m;
#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    vec_t<float, vec_size> k_vec;
    k_vec.cast_load(smem + (j * bdx + tx) * vec_size);
    s[j] = 0.f;
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      s[j] += q_vec[i] * k_vec[i];
    }
#pragma unroll
    for (uint32_t offset = bdx / 2; offset > 0; offset /= 2) {
      s[j] += math::shfl_xor_sync(s[j], offset);
    }
    s[j] = (iter_base + tz * tile_size + j < iter_bound) ? s[j] : -5e4;
    st.m = max(st.m, s[j]);
  }

  float o_scale = math::ptx_exp2(m_prev - st.m);
  st.d *= o_scale;
#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    s[j] = math::ptx_exp2(s[j] - st.m);
    st.d += s[j];
  }
#pragma unroll
  for (uint32_t i = 0; i < vec_size; ++i) {
    st.o[i] = st.o[i] * o_scale;
  }
}

template <uint32_t vec_size, uint32_t bdx, uint32_t tile_size, typename T>
__device__ __forceinline__ void update_local_state(const T* smem,
                                                   const float* s,
                                                   uint32_t compute_stage_idx,
                                                   state_t<vec_size>& st) {
  uint32_t tx = threadIdx.x;
#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    vec_t<float, vec_size> v_vec;
    v_vec.cast_load(smem + (j * bdx + tx) * vec_size);
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      st.o[i] = st.o[i] + s[j] * v_vec[i];
    }
  }
}

template <uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz>
__device__ __forceinline__ void sync_state(state_t<vec_size>& st, float* smem,
                                           float* smem_md) {
  if constexpr (bdz > 1) {
    constexpr uint32_t head_dim = bdx * vec_size;
    auto block = cg::this_thread_block();
    uint32_t tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    st.o.store(smem + (tz * bdy + ty) * head_dim + tx * vec_size);
    smem_md[(tz * bdy + ty) * 2] = st.m;
    smem_md[(tz * bdy + ty) * 2 + 1] = st.d;
    block.sync();
    st.init();
#pragma unroll
    for (uint32_t j = 0; j < bdz; ++j) {
      float mz = smem_md[(j * bdy + ty) * 2],
            dz = smem_md[(j * bdy + ty) * 2 + 1];
      vec_t<float, vec_size> oz;
      oz.load(smem + (j * bdy + ty) * head_dim + tx * vec_size);
      st.merge(oz, mz, dz);
    }
  }
}

template <uint32_t num_stages_smem, uint32_t tile_size_per_bdx,
          uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz,
          uint32_t block_q>
__global__ void deft_attention_stage_1_paged_kernel(
    const thrust::device_vector<chunk_t>& chunks,
    float* __restrict__ partial_lse, float* __restrict__ partial_o,
    float* __restrict__ row_max,
    tensor_info_t info, float sm_scale) {
  auto block = cg::this_thread_block();
  auto grid = cg::this_grid();
  sm_scale *= math::log2e;
  const uint32_t head_dim = bdx * vec_size;
  const uint32_t kv_head_idx = blockIdx.y;
  const uint32_t qo_head_idx = blockIdx.y * bdy + threadIdx.y;
  const uint32_t num_qo_heads = gridDim.y * bdy;
  const uint32_t chunk_idx = blockIdx.x;
  const chunk_t chunk = static_cast<chunk_t>(chunks[chunk_idx]);
  const uint32_t kv_chunk_len = chunk.kv_len;
  const uint32_t q_chunk_len = chunk.q_len;

  extern __shared__ uint8_t smem[];
  DTypeKV* k_smem = reinterpret_cast<DTypeKV*>(smem);
  DTypeKV* v_smem =
      (DTypeKV*)(smem + num_stages_smem * tile_size_per_bdx * bdy * bdz *
                            head_dim * sizeof(DTypeKV));
  DTypeKV** k_ptrs_smem =
      (DTypeKV**)(smem + 2 * num_stages_smem * tile_size_per_bdx * bdy * bdz *
                             head_dim * sizeof(DTypeKV));
  float* smem_md = (float*)(smem + 2 * num_stages_smem * tile_size_per_bdx *
                                       bdy * bdz * head_dim * sizeof(DTypeKV));

  const uint32_t tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;

  uint32_t stage_idx = 0;
  constexpr uint32_t vec_bits = sizeof(DTypeKV) * vec_size * 8;

  static_assert(num_stages_smem <= bdx);
#pragma unroll
  for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
    k_ptrs_smem[((j * bdz + tz) * bdy + ty) * bdx + tx] =
        chunk.protective_get_k_ptr(((j * bdz + tz) * bdy + ty) * bdx + tx,
                                   kv_head_idx);
  }
  block.sync();

  DTypeKV* k_ptrs[tile_size_per_bdx];
#pragma unroll
  for (uint32_t iter = 0; iter < num_stages_smem; ++iter) {
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      k_ptrs[j] =
          k_ptrs_smem[((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j] +
          tx * vec_size;
    }
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch,
                          SharedMemFillMode::kNoFill>(
          k_smem +
              (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) *
                  head_dim +
              tx * vec_size,
          k_ptrs[j],
          ((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j <
              kv_chunk_len);
    }
    cp_async::commit_group();
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      DTypeKV* v_ptr = k_ptrs[j] + chunk.kv_ptr_delta();
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch,
                          SharedMemFillMode::kFillZero>(
          v_smem +
              (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) *
                  head_dim +
              tx * vec_size,
          v_ptr,
          ((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j <
              kv_chunk_len);
    }
    cp_async::commit_group();
    stage_idx = (stage_idx + 1) % num_stages_smem;
  }
  state_t<vec_size> st[block_q];
  float s[block_q][bdy * tile_size_per_bdx];

#pragma unroll 2
  for (uint32_t iter = 0;
       iter < ceil_div(kv_chunk_len, tile_size_per_bdx * bdy * bdz); ++iter) {
    if ((iter + num_stages_smem) % bdx == 0) {
#pragma unroll
      for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
        k_ptrs_smem[((j * bdz + tz) * bdy + ty) * bdx + tx] =
            chunk.protective_get_k_ptr(
                ((iter + num_stages_smem) * tile_size_per_bdx * bdy * bdz +
                 ((j * bdz + tz) * bdy + ty) * bdx + tx),
                kv_head_idx);
      }
    }
    // compute qk
    cp_async::wait_group<2 * num_stages_smem - 1>();
    block.sync();
#pragma unroll 1
    for (uint32_t q_idx = 0; q_idx < q_chunk_len; ++q_idx) {
      vec_t<float, vec_size> q_vec;
      q_vec.cast_load(chunk.protective_get_q_ptr(q_idx, qo_head_idx) +
                      tx * vec_size);

#pragma unroll
      for (uint32_t i = 0; i < vec_size; ++i) {
        q_vec[i] *= sm_scale;
      }
      compute_qk<vec_size, bdx, bdy * tile_size_per_bdx>(
          k_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim,
          stage_idx, q_vec, iter * tile_size_per_bdx * bdy * bdz, kv_chunk_len,
          s[q_idx], st[q_idx]);
      block.sync();
    }

    block.sync();

#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      k_ptrs[j] =
          k_ptrs_smem[((((iter + num_stages_smem) % bdx) * bdz + tz) * bdy +
                       ty) *
                          tile_size_per_bdx +
                      j] +
          tx * vec_size;
    }
    // load k tiles
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch,
                          SharedMemFillMode::kNoFill>(
          k_smem +
              (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) *
                  head_dim +
              tx * vec_size,
          k_ptrs[j],
          (((iter + num_stages_smem) * bdz + tz) * bdy + ty) *
                      tile_size_per_bdx +
                  j <
              kv_chunk_len);
    }
    cp_async::commit_group();

    // update m/d/o states
    cp_async::wait_group<2 * num_stages_smem - 1>();
    block.sync();
#pragma unroll 1
    for (uint32_t q_idx = 0; q_idx < q_chunk_len; ++q_idx) {
      update_local_state<vec_size, bdx, bdy * tile_size_per_bdx>(
          v_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim,
          s[q_idx], stage_idx, st[q_idx]);
    }
    block.sync();

    // load v tiles
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      DTypeKV* v_ptr = k_ptrs[j] + chunk.kv_ptr_delta();
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch,
                          SharedMemFillMode::kFillZero>(
          v_smem +
              (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) *
                  head_dim +
              tx * vec_size,
          v_ptr,
          (((iter + num_stages_smem) * bdz + tz) * bdy + ty) *
                      tile_size_per_bdx +
                  j <
              kv_chunk_len);
    }
    cp_async::commit_group();
    stage_idx = (stage_idx + 1) % num_stages_smem;
  }
  cp_async::wait_group<0>();
  block.sync();

#pragma unroll 1
  for (uint32_t q_idx = 0; q_idx < q_chunk_len; ++q_idx) {
    sync_state<vec_size, bdx, bdy, bdz>(
        st[q_idx], reinterpret_cast<float*>(smem), smem_md);
    if (tz == 0) {
      uint32_t global_q_idx = chunk.get_q_idx(q_idx);
      st[q_idx].normalize();
      st[q_idx].o.cast_store(partial_o + qo_head_idx * info.partial_o_stride_h +
                             (chunk.partial_o_offset + q_idx) *
                                 info.partial_o_stride_n +
                             tx * vec_size);
      if (tx == 0) {
        partial_lse[qo_head_idx * info.partial_lse_stride_h +
                    chunk.partial_o_offset + q_idx] = st[q_idx].get_lse();
        atomicMax(row_max + qo_head_idx * info.row_max_stride_h + global_q_idx,
                  st[q_idx].get_lse());
      }
    }
    block.sync();
  }
}
template <uint32_t vec_size, uint32_t bdx>
__global__ void deft_attention_reduction_kernel(
    const thrust::device_vector<chunk_t>& chunks, DTypeOut* __restrict__ o,
    float* __restrict__ L, float* __restrict__ row_max,
    const float* __restrict__ partial_lse, const float* __restrict__ partial_o,
    tensor_info_t info) {
  auto block = cg::this_thread_block();
  auto grid = cg::this_grid();
  const uint32_t head_dim = bdx * vec_size;
  const uint32_t chunk_idx = blockIdx.x;
  const chunk_t chunk = static_cast<chunk_t>(chunks[chunk_idx]);
  const uint32_t q_chunk_len = chunk.q_len;
  const uint32_t qo_head_idx = blockIdx.y;
  const uint32_t tx = threadIdx.x;
#pragma unroll 1
  for (uint32_t q_idx = 0; q_idx < q_chunk_len; ++q_idx) {
    if (tx == 0) {
      uint32_t global_q_idx = chunk.get_q_idx(q_idx);
      float global_max =
          row_max[qo_head_idx * info.row_max_stride_h + global_q_idx];
      float lse = partial_lse[qo_head_idx * info.partial_lse_stride_h +
                              chunk.partial_o_offset + q_idx];
      float new_exp = math::ptx_exp2(lse - global_max);
      atomicAdd(L + qo_head_idx * info.L_stride_h + global_q_idx, new_exp);
    }
    block.sync();
  }
  grid.sync();
#pragma unroll 1
  for (uint32_t q_idx = 0; q_idx < q_chunk_len; ++q_idx) {
    uint32_t global_q_idx = chunk.get_q_idx(q_idx);
    float global_max =
        row_max[qo_head_idx * info.row_max_stride_h + global_q_idx];
    float lse = partial_lse[qo_head_idx * info.partial_lse_stride_h +
                            chunk.partial_o_offset + q_idx];
    float new_exp = math::ptx_exp2(lse - global_max);
    const float* partial_o_ptr =
        partial_o + qo_head_idx * info.partial_o_stride_h +
        (chunk.partial_o_offset + q_idx) * info.partial_o_stride_n +
        tx * vec_size;
    vec_t<float, vec_size> o_vec;
    o_vec.cast_load(partial_o_ptr);
    DTypeOut* o_ptr = o + qo_head_idx * info.qo_stride_h +
                      global_q_idx * info.qo_stride_n + tx * vec_size;
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      o_vec[i] *= new_exp;
      atomicAdd(o_ptr + i, o_vec[i]);
    }
    block.sync();
  }
}

cudaError_t deft_attention(DTypeOut* o,
                           const thrust::device_vector<chunk_t>& chunks,
                           float* partial_lse,
                           float* partial_o, uint32_t partial_num,
                           tensor_info_t info, float sm_scale,
                           cudaStream_t stream) {
  const uint32_t num_qo_heads = info.num_qo_heads;
  const uint32_t num_kv_heads = info.num_kv_heads;
  constexpr uint32_t HEAD_DIM = 128;
  constexpr uint32_t GROUP_SIZE = 4;

  constexpr uint32_t vec_size =
      std::max(16UL / sizeof(DTypeKV), HEAD_DIM / 32UL);
  constexpr uint32_t num_stages_smem = 2U;
  constexpr uint32_t bdx = HEAD_DIM / vec_size;
  static_assert(bdx <= 32);
  assert(num_qo_heads == num_kv_heads * GROUP_SIZE);
  constexpr uint32_t bdy = GROUP_SIZE;
  constexpr uint32_t num_threads = std::max(128U, bdx * bdy);
  constexpr uint32_t bdz = num_threads / (bdx * bdy);
  constexpr uint32_t tile_size_per_bdx =
      GROUP_SIZE == 1 ? (sizeof(DTypeKV) == 1 ? 2U : 4U) : 1U;
  constexpr uint32_t block_q = 32;
  const uint32_t smem_size =
      2 * num_stages_smem * tile_size_per_bdx * bdy * bdz * HEAD_DIM *
          sizeof(DTypeKV) +
      std::max(tile_size_per_bdx * num_threads * sizeof(DTypeKV*),
               2 * bdy * bdz * sizeof(float));

  thrust::device_vector<float> L(num_qo_heads * info.qo_len);
  thrust::device_vector<float> row_max(num_qo_heads * info.qo_len);
  auto L_ptr = L.data().get();
  auto row_max_ptr = row_max.data().get();
  dim3 block(bdx, bdy, bdz);
  dim3 grid(chunks.size(), num_kv_heads);
  auto stage_1_kernel =
      deft_attention_stage_1_paged_kernel<num_stages_smem, tile_size_per_bdx,
                                          vec_size, bdx, bdy, bdz, block_q>;
  auto reduction_kernel = deft_attention_reduction_kernel<4, HEAD_DIM / 4>;
  FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(
      stage_1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  stage_1_kernel<<<grid, block, smem_size, stream>>>(chunks, partial_lse,
                                                     partial_o, row_max_ptr, info, sm_scale);

  void* args[] = {
      const_cast<void *>(reinterpret_cast<const void *>(&chunks)), &o, &L_ptr, &row_max_ptr, &partial_lse,
      &partial_o, &info,
  };
  // reduction_kernel<<<1, num_threads, 0, stream>>>(
  //     chunks, o, L.data().get(), row_max.data().get(), partial_lse, partial_o, info);
  FLASHINFER_CUDA_CALL(
    cudaLaunchCooperativeKernel((void*)reduction_kernel, dim3(chunks.size(), num_qo_heads), dim3(HEAD_DIM / 4), args, 0, stream)
  )
  return cudaSuccess;
}

}  // namespace deft
#endif  // DEFT_ATTENTION_CUH_
