#include <torch/python.h>
#include "deft/ops.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "DeFT";
    m.def("dummy", &deft::dummy, "Dummy");
}
