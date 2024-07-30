#include <gtest/gtest.h>
#include "deft/attention.cuh"
#include <cuda_runtime.h>
#include <iostream>

__global__ void init_kernel(float* input, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		input[idx] = idx;
	}
}

void init(float* input, int size) {
	int block_size = 256;
	int grid_size = (size + block_size - 1) / block_size;
	init_kernel<<<grid_size, block_size>>>(input, size);
}

TEST(Attention, Dummy) {
	const int size = 1024;
	float* input_device;
	float* output_device;
	float* output_host = new float[size];

	cudaMalloc(&input_device, size * sizeof(float));
	cudaMalloc(&output_device, size * sizeof(float));

	init(input_device, size);

	deft::dummy(output_device, input_device, size);

	cudaMemcpy(output_host, output_device, size * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < size; i++) {
		EXPECT_EQ(output_host[i], i);
	}

	cudaFree(input_device);
	cudaFree(output_device);
	delete[] output_host;
}
