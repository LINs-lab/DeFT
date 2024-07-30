#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <iostream>

#include "deft/attention.cuh"
#include "flashinfer_ops.cuh"

constexpr uint32_t chunk_size = 256;
constexpr uint32_t branch_size = 10;
constexpr uint32_t chunk_num = branch_size + 1;
constexpr uint32_t total_kv_len = chunk_size * chunk_num;
constexpr uint32_t num_kv_heads = 8;
constexpr uint32_t num_qo_heads = 32;
constexpr uint32_t head_dim = 128;
constexpr uint32_t partial_num = branch_size + branch_size;

typedef half DTypeQ;
typedef half DTypeKV;
typedef half DTypeO;
typedef uint32_t IdType;

using namespace flashinfer;

__global__ void init_chunks(deft::chunk_t* chunks, uint32_t* q_idx[],
                            uint32_t* q_len, uint32_t* kv_idx[],
                            uint32_t* kv_len, DTypeQ* q_data, DTypeKV* k_data,
                            DTypeKV* v_data) {
  int bid = blockIdx.x;
  int offset = 0;
  deft::chunk_t chunk(
      num_kv_heads, num_qo_heads, head_dim, num_kv_heads * head_dim, head_dim,
      num_qo_heads * head_dim, head_dim, q_len[bid], kv_len[bid], q_idx[bid],
      kv_idx[bid], offset, k_data, v_data, q_data);
  for (uint32_t i = 0; i < chunk.q_len; ++i) {
    chunk.q_idx[i] = i;
  }
  for (uint32_t i = 0; i < chunk.kv_len; ++i) {
    chunk.kv_idx[i] = bid * chunk_size + i;
  }
  // printf("q_len = %d, kv_len = %d, q_idx = %x, kv_idx = %x\n", chunk.q_len,
  //        chunk.kv_len, chunk.q_idx, chunk.kv_idx);
  // printf("q_len = %d, kv_len = %d, q_idx = %x\n", t_q_len, t_kv_len,
  // t_q_idx);
}

TEST(Attention, DeftNodeKernel) {
  thrust::device_vector<deft::chunk_t> chunks(chunk_num);
  thrust::device_vector<uint32_t> q_idx[chunk_num];
  thrust::device_vector<uint32_t> kv_idx[chunk_num];
  constexpr uint32_t q_size = branch_size * num_qo_heads * head_dim;
  constexpr uint32_t kv_size = total_kv_len * num_kv_heads * head_dim;
  thrust::device_vector<DTypeQ> q_data(q_size);
  thrust::device_vector<DTypeKV> k_data(kv_size);
  thrust::device_vector<DTypeKV> v_data(kv_size);
  thrust::device_vector<DTypeO> o_data(q_size);

  uint32_t **d_q_idx_ptr, **d_kv_idx_ptr;
  uint32_t *q_idx_ptr[chunk_num], *kv_idx_ptr[chunk_num];
  uint32_t *d_q_len, *d_kv_len;
  uint32_t q_len[chunk_num], kv_len[chunk_num];

  cudaMalloc(&d_q_idx_ptr, chunk_num * sizeof(uint32_t*));
  cudaMalloc(&d_kv_idx_ptr, chunk_num * sizeof(uint32_t*));
  cudaMalloc(&d_q_len, chunk_num * sizeof(uint32_t));
  cudaMalloc(&d_kv_len, chunk_num * sizeof(uint32_t));

  for (uint32_t i = 0; i < chunk_num; ++i) {
    if (i == 0) {
      q_idx[0].resize(branch_size);
    } else {
      q_idx[i].resize(1);
    }
    kv_idx[i].resize(chunk_size);
    q_idx_ptr[i] = q_idx[i].data().get();
    kv_idx_ptr[i] = kv_idx[i].data().get();
    q_len[i] = q_idx[i].size();
    kv_len[i] = kv_idx[i].size();
  }

  cudaMemcpy(d_q_idx_ptr, q_idx_ptr, chunk_num * sizeof(uint32_t*),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_kv_idx_ptr, kv_idx_ptr, chunk_num * sizeof(uint32_t*),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_q_len, q_len, chunk_num * sizeof(uint32_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_kv_len, kv_len, chunk_num * sizeof(uint32_t),
             cudaMemcpyHostToDevice);

  deft::tensor_info_t info(branch_size, total_kv_len, num_qo_heads,
                           num_kv_heads, flashinfer::QKVLayout::kNHD, head_dim,
                           partial_num);
  init_chunks<<<chunk_num, 1>>>(chunks.data().get(), d_q_idx_ptr, d_q_len,
                                d_kv_idx_ptr, d_kv_len, q_data.data().get(),
                                k_data.data().get(), v_data.data().get());
  cudaError_t e =
      deft::deft_attention(o_data.data().get(), chunks, partial_num, info,
                           1. / sqrtf(head_dim), cudaStreamDefault);
  cudaDeviceSynchronize();
  ASSERT_EQ(e, cudaSuccess);
}
