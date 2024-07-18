#include <cuda_runtime.h>

namespace deft {
	__global__ void dummy_kernel(
		float* __restrict__ output,
		const float* __restrict__ input,
		const int size
	) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx < size) {
			output[idx] = input[idx];
		}
	}

	void dummy(
		float* output,
		const float* input,
		int size
	) {
		dim3 block(256);
		dim3 grid((size + block.x - 1) / block.x);
		dummy_kernel<<<grid, block>>>(output, input, size);
	}

}
