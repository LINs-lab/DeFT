import logging
from typing import Any, Dict

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

logger = logging.getLogger(__name__)


logger.info('Building extensions')
ext_modules = []
ext_modules.append(
    CUDAExtension(
        name='deft._kernels',
        sources=['csrc/deft_api.cpp', 'csrc/deft/attention.cu'],
        include_dirs=[
            'csrc',
        ],
    )
)


def build(setup_kwargs: Dict[str, Any]) -> None:
    logger.info('Building extensions')
    setup_kwargs.update(
        {
            'ext_modules': ext_modules,
            'cmdclass': {
                'build_ext': BuildExtension.with_options(use_ninja=False),
            },
        }
    )
