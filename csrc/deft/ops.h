#pragma once

#include <optional>
#include <torch/library.h>
#include <torch/all.h>

namespace deft {
	torch::Tensor dummy(
		torch::Tensor output,
		torch::Tensor input
	);
}
