import json
import argparse
from typing import Optional, Dict, List, cast, Any
from tqdm import tqdm
import pickle
import random


class ExecuteTreeNode:
    def __init__(
        self, node_id: int, value: int, start_offset: int, end_offset: int
    ) -> None:
        self.id = node_id
        self.value = value  # seq len in this node
        self.children: List["ExecuteTreeNode"] = []
        self.start_offset = start_offset  # start iter index
        self.end_offset = end_offset  # end iter index
        self.depth = 0
        self.width = 0

    def __str__(self) -> str:
        return (
            f"TreeNode(id={self.id}, value={self.value}, start={self.start_offset}, "
            f"end={self.end_offset}, depth={self.depth}, width={self.width})"
        )

    def __repr__(self) -> str:
        return str(self)


class ExecuteTree:
    def __init__(
        self,
        root: ExecuteTreeNode,
        nodes: List[ExecuteTreeNode],
        prompt: Optional[str] = None,
    ) -> None:
        self.root = root
        self.prompt = prompt
        self.nodes = nodes
        self.branch_record: Dict[int, Dict[int, List[int]]] = {}
        self.prune_record: Dict[int, List[int]] = {}
        self.max_depth = 0
        self.max_width = 0
        self.width_per_depth: Dict[int, int] = {}
        self.build_tree_metadata(self.root, 0)
        self.node_num = len(nodes)
        # hard code for speculative decoding
        self.accepted_len_list = None

    def build_tree_metadata(self, root: ExecuteTreeNode, depth: int) -> int:
        end_iter = root.end_offset
        self.max_depth = max(self.max_depth, depth)
        if depth not in self.width_per_depth:
            self.width_per_depth[depth] = 0
        root.depth = depth
        root.width = self.width_per_depth[depth]
        self.width_per_depth[depth] += 1
        self.max_width = max(self.max_width, self.width_per_depth[depth])

        if len(root.children) == 0:
            if end_iter not in self.prune_record:
                self.prune_record[end_iter] = []
            self.prune_record[end_iter].append(root.id)
            return end_iter
        else:
            if end_iter not in self.branch_record:
                self.branch_record[end_iter] = {}
            # branch_record[end_iter].append(root.id)
            children = [child.id for child in root.children]
            self.branch_record[end_iter][root.id] = children
            for child in root.children:
                end_iter = max(
                    end_iter, self.build_tree_metadata(child, depth + 1)
                )
            self.prune_record[end_iter].append(root.id)
            return end_iter


def build_tree(data: Any) -> List[ExecuteTreeNode]:
    node_cnt = len(data)
    nodes = [ExecuteTreeNode(i, 0, 0, 0) for i in range(node_cnt)]
    for item in data.values():
        node_id = cast(int, item["id"])
        value = cast(int, item["value"])
        children = cast(List[int], item["children"])
        start_offset = cast(int, item["start"])
        end_offset = cast(int, item["end"])

        nodes[node_id].value = value
        nodes[node_id].start_offset = start_offset
        nodes[node_id].end_offset = end_offset
        for child in children:
            nodes[node_id].children.append(nodes[child])

    return nodes


def build_trees(dataset: Any) -> List[ExecuteTree]:
    trees = []
    for item in tqdm(dataset):
        if "data" in item:
            if "incompleted" in item:
                if item["incompleted"]:
                    continue
            nodes = build_tree(item["data"])
        else:
            nodes = build_tree(item)
        root = nodes[0]
        prompt = None
        if "prompt" in item:
            prompt = item["prompt"]
        tree = ExecuteTree(root, nodes, prompt)
        trees.append(tree)

    return trees


def load_dataset(path: str) -> Any:
    if path.endswith(".json"):
        with open(path, "r") as f:
            data = json.load(f)
    elif path.endswith(".pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
    else:
        raise NotImplementedError(f"Unsupported file format: {path}")
    return data


def load_trees(path: str) -> List[ExecuteTree]:
    dataset = load_dataset(path)
    trees = build_trees(dataset)
    return trees


def print_tree(root: ExecuteTreeNode, depth: int = 0) -> int:
    if root is not None:
        ret = depth
        print(
            f"{' ' * depth} id: {root.id}, value: {root.value}, start: {root.start_offset}, end: {root.end_offset}, depth: {root.depth}, width: {root.width}"
        )
        for child in root.children:
            ret = max(print_tree(child, depth + 1), ret)
        return ret
    else:
        return depth - 1


def bfs_tree(root: ExecuteTreeNode) -> None:
    queue = {root.id: root}

    def get_min_end(nodes: Dict[int, ExecuteTreeNode]) -> int:
        return min([node.end_offset for node in nodes.values()])

    while len(queue) > 0:
        min_end = get_min_end(queue)
        prune_nodes = []
        prev_len = len(queue)
        for node in queue.values():
            if node.end_offset == min_end:
                prune_nodes.append(node)
        for node in prune_nodes:
            del queue[node.id]
        for node in prune_nodes:
            if len(node.children) > 0:
                for child in node.children:
                    queue[child.id] = child
        next_len = len(queue)
        print(f"min_end: {min_end}, prev_len: {prev_len}, next_len: {next_len}")


# hardcode for speculative decoding dataset loading
# Todo(jinwei): implement real SD
# def load_trees_SD(path: str, max_gen_len: int) -> List[ExecuteTree]:
#     dataset = load_dataset(path)
#     trees = build_trees_SD(dataset, max_gen_len)
#     return trees


def load_prompts(path: str) -> List[ExecuteTree]:
    dataset = load_dataset(path)
    trees: List[ExecuteTree] = []
    tree_size = dataset["Token_Tree_size"]  # no meaning for few-shot prompting
    records = dataset["Records"]
    for tr in records:
        nodes = build_tree_SD(tree_size)
        root = nodes[0]
        prompt = None
        assert "prompt" in tr
        prompt = tr["prompt"]
        tree = ExecuteTree(root, nodes, prompt)
        tree = ExecuteTree(root, nodes, prompt)
        tree.accepted_len_list = tr["Accept_length"]
        assert isinstance(tree.accepted_len_list, list)
        trees.append(tree)
    return trees


def generate_accepted_len_list(max_gen_len: int, tree: ExecuteTree) -> None:
    """
    Generate a list of accepted lengths ensuring that the cumulative length
    does not exceed max_gen_len.

    If the sum of tree.accepted_len_list is smaller than max_gen_len,
    it randomly selects accepted lengths between m1 (max) and m2 (min).

    Args:
        max_gen_len (int): The maximum allowed generated length.
        tree (ExecuteTree): The tree containing the accepted length list.

    """
    accepted_len_list = []
    s = 0
    m1 = max(tree.accepted_len_list)
    m2 = min(tree.accepted_len_list)

    # Iterate through the existing list while staying within max_gen_len
    for length in tree.accepted_len_list:
        if s + length <= max_gen_len:
            accepted_len_list.append(length)
            s += length
        else:
            break

    # If total length is still less than max_gen_len, add random values between m1 and m2
    while s < max_gen_len:
        random_len = random.randint(m2, m1)
        if s + random_len > max_gen_len:
            random_len = max_gen_len - s  # Ensure we don't exceed max_gen_len

        accepted_len_list.append(random_len)
        s += random_len

    tree.accepted_len_list = accepted_len_list


# def build_trees_SD(dataset: Any, max_gen_len: int) -> List[ExecuteTree]:
#     trees: List[ExecuteTree] = []
#     tree_size = dataset["Token_Tree_size"]
#     records = dataset["Records"]
#     # tree_topo = dataset["Tree_Structure"]
#     for tr in records:
#         nodes = build_tree_SD(tree_size)
#         root = nodes[0]
#         prompt = None
#         assert "prompt" in tr
#         prompt = tr["prompt"]
#         tree = ExecuteTree(root, nodes, prompt)
#         tree.accepted_len_list = tr["Accept_length"]
#         assert isinstance(tree.accepted_len_list, list)
#         m1 = max(tree.accepted_len_list)
#         m2 = min(tree.accepted_len_list)
#         s = sum(tree.accepted_len_list)
#         while s < max_gen_len:
#             new_len = random.randint(m2, m1)
#             new_len = min(new_len, max_gen_len - s)
#             tree.accepted_len_list.append(new_len)
#             s += new_len
#         trees.append(tree)

#     return trees


def build_tree_SD(token_num: int) -> List[ExecuteTreeNode]:
    node_cnt = token_num
    nodes = [ExecuteTreeNode(i, 0, 0, 0) for i in range(node_cnt)]

    return nodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DataLoader.")
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()
    if args.dataset is None:
        raise ValueError("Please specify the dataset.")
    trees = load_trees(args.dataset)
    if "docmergeToT" in args.dataset:
        trees = trees[:16] + trees[17:20]
    elif "docmergeCoT" in args.dataset:
        trees = trees[0:1]
    elif "sorting128ToT" in args.dataset:
        trees = trees[:20]
    elif "sorting128CoT" in args.dataset:
        trees = trees[0:1]
    # print_tree(trees[0].root)
    # print(trees[0].nodes)
    # print(trees[0].branch_record)
    # print(trees[0].prune_record)
    # print(trees[0].max_depth)
    # print(trees[0].max_width)
    # print(trees[0].width_per_depth)
    # bfs_tree(trees[0].root)
    prompt = 0
    tokens = 0
    for tree in trees:
        prompt += tree.root.value
        for i in range(1, len(tree.nodes)):
            tokens += (
                min(4096 - tree.root.value, tree.nodes[i].end_offset)
                - tree.nodes[i].start_offset
                + 1
            )
    print(prompt / len(trees))
    print(tokens / len(trees))
    # print(len(trees))
    # for tree in trees:
    #     print(tree.max_depth, tree.max_width)
