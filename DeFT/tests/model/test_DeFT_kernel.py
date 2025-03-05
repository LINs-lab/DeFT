# type: ignore
import torch
from deft.layers.attention.deft_attention import DeFTAttention
from deft.memory_pool import ReqToTokenPool, TokenToKVPool
from deft.tree_decoding.tree_cache import TreeMetadata, TreeCache
from deft.model_runner import InputMetadata, ForwardMode
from deft.tree_decoding.tree_index_pool import TreeIndexPool

# import math
from deft.tree_decoding.timer import GlobalTimer

# from deft.layers.attention.tree_attention import tree_attention_subtree_fwd

torch.manual_seed(0)

HEAD_NUM = 32
HEAD_DIM = 128
HIDDEN_SIZE = HEAD_NUM * HEAD_DIM
HEAD_KV_NUM = 8
HIDDEN_KV_SIZE = HEAD_KV_NUM * HEAD_DIM

attn = DeFTAttention(
    num_heads=HEAD_NUM,
    head_dim=HEAD_DIM,
    scaling=HEAD_DIM**-0.5,
    num_kv_heads=HEAD_KV_NUM,
    layer_id=0,
)

req_to_token_pool = ReqToTokenPool(size=1000, max_context_len=10000)
tree_index_pool = TreeIndexPool(size=1000, max_context_len=10000)
token_to_kv_pool = TokenToKVPool(
    size=100000,
    dtype=torch.float16,
    head_num=HEAD_KV_NUM,
    head_dim=HEAD_DIM,
    layer_num=1,
)

batch_size = 32

prompt_len = 400

node_len = 1

block_len = 128

input_ids = torch.arange(
    100, batch_size + 100, dtype=torch.int32, device="cuda"
)

q = torch.randn(batch_size, HIDDEN_SIZE, dtype=torch.float16, device="cuda")
k = torch.randn(batch_size, HIDDEN_KV_SIZE, dtype=torch.float16, device="cuda")
v = torch.randn(batch_size, HIDDEN_KV_SIZE, dtype=torch.float16, device="cuda")

prompt_ids = torch.arange(1, prompt_len + 1, dtype=torch.int32, device="cuda")
prompt_loc = torch.arange(0, prompt_len, dtype=torch.int32, device="cuda")

seq_lens = torch.full(
    (batch_size,), prompt_len + node_len, dtype=torch.int32, device="cuda"
)
positions = torch.full(
    (batch_size,), prompt_len + node_len - 1, dtype=torch.int32, device="cuda"
)

tree = TreeCache(
    torch.float16,
    HEAD_KV_NUM,
    HEAD_DIM,
    1,
    req_to_token_pool=req_to_token_pool,
    token_to_kv_pool=token_to_kv_pool,
    tree_index_pool=tree_index_pool,
    use_paged_memory=True,
    use_tree_index=True,
)
kv_updater = tree.init_prompt(prompt_ids)
kv_updater.update(
    0,
    torch.randn(
        prompt_len, HEAD_KV_NUM, HEAD_DIM, dtype=torch.float16, device="cuda"
    ),
    torch.randn(
        prompt_len, HEAD_KV_NUM, HEAD_DIM, dtype=torch.float16, device="cuda"
    ),
)

tree.branch(tree.root, batch_size)

leaf_to_q: dict[int, int] = {}
leaf_cnt = 0

for leaf in sorted(tree.leaves.values(), key=lambda x: x.id):
    leaf_to_q[leaf.id] = leaf_cnt
    leaf_cnt += 1

    # leaf.append_index(prompt_len + leaf_to_q[leaf.id] + i * batch_size)
for i in range(node_len):
    for leaf in tree.leaves.values():
        leaf.append_token(int(input_ids[leaf_to_q[leaf.id]].item()))
    kv_updater = tree.alloc()
    if i + 1 < node_len:
        kv_updater.update(
            0,
            torch.randn(
                1, HEAD_KV_NUM, HEAD_DIM, dtype=torch.float16, device="cuda"
            ),
            torch.randn(
                1, HEAD_KV_NUM, HEAD_DIM, dtype=torch.float16, device="cuda"
            ),
        )
    else:
        kv_updater.update(
            0,
            k.view(-1, HEAD_KV_NUM, HEAD_DIM),
            v.view(-1, HEAD_KV_NUM, HEAD_DIM),
        )


tree_metadata = TreeMetadata.from_tree_cache_node(
    tree, tile_num=-1, max_block_len=block_len
)

# print(tree_metadata.node_q)
# print(tree_metadata.node_kv)
# print(tree_metadata.node_q_len)
# print(tree_metadata.node_kv_len)
# print(tree_metadata.block_q, tree_metadata.block_q_cnts, tree_metadata.block_q_offset)
# # print(tree_metadata.block_kv)
# print(tree_metadata.block_lens)
# print(tree_metadata.block_bitmasks[-128:])
node_kv_offset = tree_metadata.node_kv_offset.tolist()
node_kv_len = tree_metadata.node_kv_len.tolist()
print(tree_metadata.node_kv.shape)
for i in range(len(node_kv_offset)):
    offset = node_kv_offset[i]
    print(f"offset: {offset}, len: {node_kv_len[i]}")
    for j in range(node_kv_len[i]):
        print(f"{tree_metadata.node_kv[offset + j].item()}")

for node in tree.nodes.values():
    print(node.kv_indices)
    kv_len = len(node.kv_indices)
    print(
        f"indices: {node.node_indices[:kv_len].tolist()}, indices_id: {node.node_indices_id}, "
        f"offset: {tree_index_pool.get_offset(node.node_indices_id)}"
    )

model_runner = {
    "model_config": {
        "num_attention_heads": HEAD_NUM,
        "num_key_value_heads": HEAD_NUM,
        "head_dim": HEAD_DIM,
    }
}

input_metadata = InputMetadata.from_tree(
    model_runner=model_runner,  # type: ignore
    tree=tree,
    req_to_token_pool=req_to_token_pool,
    token_to_kv_pool=token_to_kv_pool,
    forward_mode=ForwardMode.TREE_DECODE_FLATTEN,
    positions=positions,
    kv_updater=kv_updater,
    tree_metadata=tree_metadata,
)


def benchmark(func, warm_up=0, repeat=1):
    for _ in range(warm_up):
        func(q, k, v, input_metadata)

    GlobalTimer.reset("attn_mem")
    GlobalTimer.reset("attn_comp")
    t = 0.0
    res = None
    for _ in range(repeat):
        GlobalTimer.reset("attn_mem")
        GlobalTimer.reset("attn_comp")
        res = func(q, k, v, input_metadata)
        t += GlobalTimer.get("attn_comp")

    return t / repeat, res


# # deft_result = attn.tree_decode_forward_triton(q, k, v, input_metadata)
# deft_flatten_result = attn.deft_flatten_forward(q, k, v, input_metadata)
# deft_flatten_time = GlobalTimer.get("attn_comp")
# GlobalTimer.reset("attn_mem")
# GlobalTimer.reset("attn_comp")
# sglang_result = attn.radix_attention_forward(q, k, v, input_metadata)
# # flashinfer_result = attn.decode_forward_flashinfer(q, k, v, input_metadata)
# GlobalTimer.reset("attn_mem")
# GlobalTimer.reset("attn_comp")
# sglang_result = attn.radix_attention_forward(q, k, v, input_metadata)
# sglang_time = GlobalTimer.get("attn_comp")
torch.cuda.nvtx.range_push("compute")
# deft_flatten_t, deft_flatten_result = benchmark(attn.deft_flatten_forward)
deft_node_t, deft_node_result = benchmark(attn.deft_node_forward)
sglang_t, sglang_result = benchmark(attn.radix_attention_forward)
torch.cuda.nvtx.range_pop()

# print(deft_flatten_t, deft_node_t, sglang_t)
print(f"Batch size: {batch_size}, Block size: {block_len}")
# print(f"deft flatten: {deft_flatten_t * 1000:.6f} ms")
print(f"deft node chunk: {deft_node_t * 1000:.6f} ms")
print(f"sglang: {sglang_t * 1000:.6f} ms")
exit()

# exit()

torch_result = []

q = q.view(-1, HEAD_NUM, 1, HEAD_DIM)  # (q, h, 1, d)
for i in range(batch_size):
    total_k = torch.stack(
        [
            token_to_kv_pool.kv_data[0][j][0].view(HIDDEN_KV_SIZE)
            for j in req_to_token_pool.req_to_token[i, : prompt_len + node_len]
        ]
    )
    total_v = torch.stack(
        [
            token_to_kv_pool.kv_data[0][j][1].view(HIDDEN_KV_SIZE)
            for j in req_to_token_pool.req_to_token[i, : prompt_len + node_len]
        ]
    )
    # print(total_k)
    # print(k)
    # print(req_to_token_pool.req_to_token[i, :prompt_len + node_len])
    total_k = (
        total_k.reshape(-1, HEAD_KV_NUM, HEAD_DIM)
        .transpose(0, 1)
        .transpose(1, 2)
    )  # (h, d, s)
    total_v = total_v.reshape(-1, HEAD_KV_NUM, HEAD_DIM).transpose(
        0, 1
    )  # (h, s, d)
    total_k = total_k.repeat_interleave(HEAD_NUM // HEAD_KV_NUM, dim=0)
    total_v = total_v.repeat_interleave(HEAD_NUM // HEAD_KV_NUM, dim=0)
    # total_k_0 = total_k[0].reshape(-1, HEAD_NUM, HEAD_DIM).transpose(0, 1).transpose(1, 2) # (h, d, s)
    # total_v_0 = total_v[0].reshape(-1, HEAD_NUM, HEAD_DIM).transpose(0, 1)    # (h, s, d)
    # total_k_12 = total_k[1:3].reshape(-1, HEAD_NUM, HEAD_DIM).transpose(0, 1).transpose(1, 2) # (h, d, s)
    # total_v_12 = total_v[1:3].reshape(-1, HEAD_NUM, HEAD_DIM).transpose(0, 1)    # (h, s, d)
    # qk_0 = torch.matmul(q[i], total_k_0) * HEAD_DIM**-0.5 # (h, 1, s)
    # qk_12 = torch.matmul(q[i], total_k_12) * HEAD_DIM**-0.5 # (h, 1, s)
    # m_0 = torch.max(qk_0, -1).values
    # m_12 = torch.max(qk_12, -1).values
    # exp_logic_0 = torch.exp(qk_0 - m_0)
    # exp_logic_12 = torch.exp(qk_12 - m_12)
    # lse_0 = math.log(torch.sum(torch.exp(qk_0)).item())
    # lse_12 = math.log(torch.sum(torch.exp(qk_12)).item())
    # row_max = max(lse_0, lse_12)
    # L = math.exp(lse_0 - row_max) + math.exp(lse_12 - row_max)

    # print(qk_0)
    # print(qk_12)
    # print(lse_0)
    # print(lse_12)
    # print(L)
    # a_0 = torch.nn.functional.softmax(qk_0, dim=-1)  # (h, 1, s)
    # a_12 = torch.nn.functional.softmax(qk_12, dim=-1)  # (h, 1, s)

    # res_0 = torch.matmul(a_0, total_v_0)
    # res_12 = torch.matmul(a_12, total_v_12)
    # print(res_0, res_12)
    # res = res_0 * math.exp(lse_0 - row_max) + res_12 * math.exp(lse_12 - row_max)
    # res = res / L
    a = torch.nn.functional.softmax(
        torch.matmul(q[i], total_k) * HEAD_DIM**-0.5, dim=-1
    )  # (h, 1, s)
    res = torch.matmul(a, total_v)  # (h, 1, d)
    res = res.view(HIDDEN_SIZE)
    torch_result.append(res)

torch_result = torch.stack(torch_result)

# deft_result = torch.softmax(deft_result.to(torch.float64), dim=-1)
# sglang_result = torch.softmax(sglang_result.to(torch.float64), dim=-1)
# torch_result = torch.softmax(torch_result.to(torch.float64), dim=-1)
# deft_result = deft_result.to(torch.float64)
# deft_flatten_result = deft_flatten_result.to(torch.float64)
deft_node_result = deft_node_result.to(torch.float64)
sglang_result = sglang_result.to(torch.float64)
# print(deft_flatten_result.mean())
# print(deft_node_result.mean())


def error(a, b):
    return (torch.abs(a - b) / torch.max(torch.abs(a), torch.abs(b))).mean()


# flashinfer_result = flashinfer_result.to(torch.float64)
torch_result = torch_result.to(torch.float64)
# print(deft_flatten_result)
# print(deft_node_result)
# print(sglang_result)
# print(torch_result)
# print(
#     f"error of deft flatten: {error(deft_flatten_result, torch_result) * 100:.6f}%"
# )
print(
    f"error of deft node chunk: {error(deft_node_result, torch_result) * 100:.6f}%"
)
print(f"error of sglang: {error(sglang_result, torch_result) * 100:.6f}%")
# print(torch.nn.functional.mse_loss(flashinfer_result, torch_result))

# print(deft_result)
# print(sglang_result)
# print(torch_result)
# print(deft_result[0])
# print(sglang_result[0])
# print(torch_result[0])
# for h in range(HEAD_NUM):
#     a = deft_result[0][h * HEAD_DIM: (h + 1) * HEAD_DIM]
#     b = deft_subtree_result[0][h * HEAD_DIM: (h + 1) * HEAD_DIM]
#     c = torch_result[0][h * HEAD_DIM: (h + 1) * HEAD_DIM]
#     print(torch.nn.functional.mse_loss(a, c))
#     print(torch.nn.functional.mse_loss(b, c))
#     print(torch.nn.functional.mse_loss(a, b))
#     # print(a)
#     # print(b)
#     # print(c)
#     print()

# for i in range(batch_size):
#     print(torch.nn.functional.mse_loss(deft_result[i], torch_result[i]))
#     print(torch.nn.functional.mse_loss(deft_subtree_result[i], torch_result[i]))
#     print(torch.nn.functional.mse_loss(deft_result[i], deft_subtree_result[i]))
#     print("")
