"""
Function: control the branching of a decoding tree
Jinwei Yao(jinwei.yao1114@gmail.com)
"""

from deft.data_loader import ExecuteTree
from typing import Optional, Any, Callable


class Branch_Controller:
    # TODO(jinwei):More information to be kept as prior knowledge to get a user-defined metric for branching
    def __init__(
        self,
        # branching_method: str,
        branching_function: Callable,
    ) -> None:
        # self.branching_method = branching_method
        self.branching_function = branching_function
        self.tree_templates: Optional[ExecuteTree] = None

    def apply_branching(self, *args, **kwargs) -> Any:  # type: ignore
        if self.branching_function is not None:
            return self.branching_function(*args, **kwargs)
        else:
            raise ValueError("Branching function is not set.")

    def set_execution_graph(
        self, tree_templates: Optional[ExecuteTree]
    ) -> None:
        self.tree_templates = tree_templates


# class BeamSearch_Controller(Branch_Controller):
#     def __init__(self,  *args,**kwargs):
#         super().__init__(*args,**kwargs)
