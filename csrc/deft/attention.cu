#include <cuda_runtime.h>
#include "deft/ops.h"
#include "attention.cuh"
#include <torch/all.h>

namespace deft {
	torch::Tensor dummy(
		torch::Tensor output,
		torch::Tensor input
	) {
		dummy(
			output.data_ptr<float>(),
			input.data_ptr<float>(),
			input.numel()
		);
		return output;
	}
}
