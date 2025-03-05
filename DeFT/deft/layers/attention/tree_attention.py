# import math
# from typing import Optional, Tuple, List, Union,Callable, Dict

from typing import Any
import torch
# from torch import nn

import triton
import triton.language as tl

# from deft.utils import wrap_kernel_launcher


def tree_attention_fwd(
    query_states: torch.Tensor,  # (query_num, num_heads, head_dim)
    key_buffer: torch.Tensor,  # (-1, num_heads, head_dim)
    value_buffer: torch.Tensor,  # (-1, num_heads, head_dim)
    output: torch.Tensor,  # (query_num, num_heads, head_dim)
    KV_indices: torch.Tensor,  # (total_len)
    KV_indices_offset: torch.Tensor,  # (KV_num)
    KV_len: torch.Tensor,  # (KV_num)
    KVMapQ_List: torch.Tensor,  # (parital_num)
    KVMapQ_List_Offset: torch.Tensor,  # (kv_num)
    KVMapQ_List_Len: torch.Tensor,  # (kv_num)
) -> None:
    query_num, num_heads, head_dim = query_states.shape
    # kv_total_len = key_buffer.shape[2]
    query_states = query_states.transpose(
        0, 1
    )  # (num_heads, query_num, head_dim)
    key_buffer = key_buffer.transpose(0, 1)  # (num_heads, -1, head_dim)
    value_buffer = value_buffer.transpose(0, 1)  # (num_heads, -1, head_dim)
    output = output.transpose(0, 1)  # (num_heads, query_num, head_dim)

    partial_num = KVMapQ_List.shape[0]

    assert KVMapQ_List is not None
    assert KVMapQ_List_Offset is not None
    assert KVMapQ_List_Len is not None
    assert partial_num is not None

    partial_o = torch.zeros(
        [num_heads, partial_num, head_dim],
        dtype=torch.float32,
        device=query_states.device,
    )
    partial_lse = torch.zeros(
        [num_heads, partial_num],
        dtype=torch.float32,
        device=query_states.device,
    )

    DeFT_splitBynode_Triton_stage1(
        query_states,
        key_buffer,
        value_buffer,
        KV_indices,
        KV_indices_offset,
        KV_len,
        KVMapQ_List,
        KVMapQ_List_Offset,
        KVMapQ_List_Len,
        partial_num,
        partial_o,
        partial_lse,
    )

    DeFT_splitBynode_Triton_stage2(KVMapQ_List, partial_o, partial_lse, output)


cached_kernel_stage1: Any = None
cached_kernel_stage2_1: Any = None
cached_kernel_stage2_2: Any = None
cached_kernel_stage2_3: Any = None
cached_kernel_subtree: Any = None

BLOCK_N_GLOBAL = 8
BLOCK_LEN_GLOBAL = 512


@torch.inference_mode()
def DeFT_splitBynode_Triton_stage1(
    q: torch.Tensor,  # (num_heads, query_num, head_dim)
    k: torch.Tensor,  # (num_kv_heads, -1, head_dim)
    v: torch.Tensor,  # (num_kv_heads, -1, head_dim)
    KV_indices: torch.Tensor,  # (total_len)
    KV_indices_offset: torch.Tensor,  # (kv_num)
    KV_len: torch.Tensor,  # (kv_num)
    KVMapQ_List: torch.Tensor,  # (parital_num)
    KVMapQ_List_Offset: torch.Tensor,  # (kv_num)
    KVMapQ_List_Len: torch.Tensor,  # (kv_num)
    partial_num: int,
    partial_o: torch.Tensor,  # (num_heads, partial_num, head_dim)
    partial_lse: torch.Tensor,  # (num_heads, partial_num)
) -> None:
    global BLOCK_N_GLOBAL
    BLOCK_N = 16
    BLOCK_M = 32
    num_heads, query_num, head_dim = q.shape
    assert head_dim in {16, 32, 64, 128}

    kv_group_num = q.shape[0] // k.shape[0]

    scale = 1.0 / (head_dim**0.5)
    kv_num = KV_indices_offset.shape[0]
    # kv_total_len = k.shape[2]
    grid = (num_heads, kv_num, 1)
    num_warps = 4

    global cached_kernel_stage1
    if cached_kernel_stage1:
        cached_kernel_stage1(
            grid,
            num_warps,
            q,
            k,
            v,
            KV_indices,
            KV_indices_offset,
            KV_len,
            KVMapQ_List,
            KVMapQ_List_Offset,
            KVMapQ_List_Len,
            partial_num,
            scale,
            partial_o,
            partial_lse,
            q.stride(0),
            q.stride(1),
            k.stride(0),
            k.stride(1),
            partial_o.stride(0),
            partial_o.stride(1),
            partial_lse.stride(0),
        )
        return

    DeFT_splitBynode_Triton_stage1_kernel[grid](
        q,
        k,
        v,
        KV_indices,
        KV_indices_offset,
        KV_len,
        KVMapQ_List,
        KVMapQ_List_Offset,
        KVMapQ_List_Len,
        partial_num,
        scale,
        partial_o,
        partial_lse,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        partial_o.stride(0),
        partial_o.stride(1),
        partial_lse.stride(0),
        BLOCK_N=BLOCK_N,  # type: ignore
        BLOCK_M=BLOCK_M,  # type: ignore
        BLOCK_D=head_dim,  # type: ignore
        kv_group_num=kv_group_num,  # type: ignore
    )
    # cached_kernel_stage1 = wrap_kernel_launcher(
    #     DeFT_splitBynode_Triton_stage1_kernel
    # )


@triton.jit
def DeFT_splitBynode_Triton_stage1_kernel(  # type: ignore
    Q,
    K,
    V,
    KV_indices,
    KV_incices_offset,
    KV_len,
    KVMapQ_List,
    KVMapQ_List_Offset,
    KVMapQ_List_Len,
    partial_num,
    scale,
    partial_o,
    partial_lse,
    stride_qh,
    stride_qs,
    stride_kvh,
    stride_kvs,
    stride_partial_oh,
    stride_partial_os,
    stride_partial_lseh,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    kv_group_num: tl.constexpr,
):
    cur_head = tl.program_id(axis=0)
    cur_kv_head = cur_head // kv_group_num

    cur_kv_idx = tl.program_id(axis=1)

    offset_d = tl.arange(0, BLOCK_D)
    cur_kv_indices_offset = tl.load(KV_incices_offset + cur_kv_idx)
    cur_kv_len = tl.load(KV_len + cur_kv_idx)
    cur_kv_end = cur_kv_indices_offset + cur_kv_len

    cur_map_offset = tl.load(KVMapQ_List_Offset + cur_kv_idx)
    cur_map_len = tl.load(KVMapQ_List_Len + cur_kv_idx)
    cur_map_end = cur_map_offset + cur_map_len

    kv_block_n_size = (cur_kv_len + BLOCK_N - 1) // BLOCK_N

    kv_indices_offset_n = cur_kv_indices_offset + tl.arange(0, BLOCK_N)
    map_offset_m = cur_map_offset + tl.arange(0, BLOCK_M)

    q_loc = tl.load(
        KVMapQ_List + map_offset_m, mask=map_offset_m < cur_map_end, other=0
    )
    offset_q = (
        cur_head * stride_qh + q_loc[:, None] * stride_qs + offset_d[None, :]
    )
    q = tl.load(
        Q + offset_q, mask=map_offset_m[:, None] < cur_map_end, other=0.0
    )  # (BLOCK_M, BLOCK_D)

    max_logic = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    sum_exp = tl.full([BLOCK_M], 0.0, dtype=tl.float32)

    for start_n_kv in range(0, kv_block_n_size, 1):  # type: ignore
        cur_kv_indices_offset_n = start_n_kv * BLOCK_N + kv_indices_offset_n
        kv_loc = tl.load(
            KV_indices + cur_kv_indices_offset_n,
            mask=cur_kv_indices_offset_n < cur_kv_end,
            other=0,
        )
        offset_k = (
            cur_kv_head * stride_kvh
            + kv_loc[:, None] * stride_kvs
            + offset_d[None, :]
        )
        k = tl.load(
            K + offset_k,
            mask=cur_kv_indices_offset_n[:, None] < cur_kv_end,
            other=0.0,
        )  # (BLOCK_N, BLOCK_D) # type: ignore
        v = tl.load(
            V + offset_k,
            mask=cur_kv_indices_offset_n[:, None] < cur_kv_end,
            other=0.0,
        )  # (BLOCK_N, BLOCK_D) # type: ignore

        attn_value = tl.dot(q, tl.trans(k))  # (BLOCK_M, BLOCK_N)
        attn_value *= scale

        attn_value = tl.where(
            cur_kv_indices_offset_n[None, :] < cur_kv_end,
            attn_value,
            float("-inf"),
        )  # type: ignore

        cur_max_logic = tl.max(attn_value, axis=1)  # (BLOCK_M)
        new_max_logic = tl.maximum(cur_max_logic, max_logic)  # (BLOCK_M)

        exp_logic = tl.exp(
            attn_value - new_max_logic[:, None]
        )  # (BLOCK_M, BLOCK_N)
        logic_scale = tl.exp(max_logic - new_max_logic)  # (BLOCK_M)

        acc *= logic_scale[:, None]  # (BLOCK_M, BLOCK_D)

        acc += tl.dot(exp_logic, v.to(tl.float32))  # (BLOCK_M, BLOCK_D)

        sum_exp = sum_exp * logic_scale + tl.sum(exp_logic, axis=1)  # (BLOCK_M)

        max_logic = new_max_logic

    partial_o_offset = (
        cur_head * stride_partial_oh
        + map_offset_m[:, None] * stride_partial_os
        + offset_d[None, :]
    )
    tl.store(
        partial_o + partial_o_offset,
        acc / sum_exp[:, None],
        mask=map_offset_m[:, None] < cur_map_end,
    )
    partial_lse_offset = cur_head * stride_partial_lseh + map_offset_m
    tl.store(
        partial_lse + partial_lse_offset,
        max_logic + tl.log(sum_exp),
        mask=map_offset_m < cur_map_end,
    )


@torch.inference_mode()
def DeFT_splitBynode_Triton_stage2(
    KVMapQ_List: torch.Tensor,  # (parital_num)
    partial_o: torch.Tensor,  # (num_heads, partial_num, head_dim)
    partial_lse: torch.Tensor,  # (num_heads, partial_num)
    o: torch.Tensor,  # (num_heads, query_num, head_dim)
) -> None:
    BLOCK_N = 8
    num_heads, query_num, head_dim = o.shape
    assert head_dim in {16, 32, 64, 128}
    partial_num = partial_o.shape[1]
    row_max = torch.zeros(
        [num_heads, query_num], dtype=torch.float32, device=partial_o.device
    )
    L = torch.zeros(
        [num_heads, query_num], dtype=torch.float32, device=partial_o.device
    )
    grid = (num_heads, triton.cdiv(partial_num, BLOCK_N), 1)
    global \
        cached_kernel_stage2_1, \
        cached_kernel_stage2_2, \
        cached_kernel_stage2_3
    num_warps = 4
    # print("partial_num:", partial_num)
    torch.cuda.nvtx.range_push("DeFT_fwd_stage2")
    if cached_kernel_stage2_1:
        cached_kernel_stage2_1(
            grid,
            num_warps,
            partial_lse,
            row_max,
            KVMapQ_List,
            partial_num,
            partial_lse.stride(0),
            row_max.stride(0),
        )
    else:
        DeFT_splitBynode_Triton_stage2_1_kernel[grid](
            partial_lse,
            row_max,
            KVMapQ_List,
            partial_num,
            partial_lse.stride(0),
            row_max.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,  # type: ignore
        )
        # cached_kernel_stage2_1 = wrap_kernel_launcher(
        #     DeFT_splitBynode_Triton_stage2_1_kernel
        # )
    #     # assert cached_kernel_stage2_2 is not None
    #     # cached_kernel_stage2_2(
    #     #     grid, num_warps,
    #     #     L, partial_lse, row_max,
    #     #     KVMapQ_List,
    #     #     partial_num,
    #     #     L.stride(0),
    #     #     partial_lse.stride(0),
    #     #     row_max.stride(0),
    #     # )
    #     assert cached_kernel_stage2_3 is not None
    if cached_kernel_stage2_3:
        cached_kernel_stage2_3(
            grid,
            num_warps,
            o,
            L,
            partial_o,
            partial_lse,
            row_max,
            KVMapQ_List,
            partial_num,
            o.stride(0),
            o.stride(1),
            L.stride(0),
            partial_o.stride(0),
            partial_o.stride(1),
            partial_lse.stride(0),
            row_max.stride(0),
        )
    else:
        DeFT_splitBynode_Triton_stage2_3_kernel[grid](
            o,
            L,
            partial_o,
            partial_lse,
            row_max,
            KVMapQ_List,
            partial_num,
            o.stride(0),
            o.stride(1),
            L.stride(0),
            partial_o.stride(0),
            partial_o.stride(1),
            partial_lse.stride(0),
            row_max.stride(0),
            BLOCK_N=BLOCK_N,
            BLOCK_D=head_dim,
            num_warps=num_warps,  # type: ignore
        )
        # cached_kernel_stage2_3 = wrap_kernel_launcher(
        #     DeFT_splitBynode_Triton_stage2_3_kernel
        # )

    # DeFT_splitBynode_Triton_stage2_2_kernel[grid](
    #     L, partial_lse, row_max,
    #     KVMapQ_List,
    #     partial_num,
    #     L.stride(0),
    #     partial_lse.stride(0),
    #     row_max.stride(0),
    #     BLOCK_N=BLOCK_N,
    #     num_warps=num_warps
    # )
    # cached_kernel_stage2_2 = wrap_kernel_launcher(DeFT_splitBynode_Triton_stage2_2_kernel)

    torch.cuda.nvtx.range_pop()
    # print("partial_lse:", partial_lse)
    # print("row_max:", row_max)
    # print("L:", L)
    o.div_(L[:, :, None])


@triton.jit
def DeFT_splitBynode_Triton_stage2_1_kernel(  # type: ignore
    partial_lse,
    row_max,
    KVMapQ_List,
    partial_num,
    stride_partial_lseh,
    stride_row_maxh,
    BLOCK_N: tl.constexpr,
):
    cur_head = tl.program_id(axis=0)
    cur_idx = tl.program_id(axis=1)

    offset_map = cur_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    q_idx = tl.load(
        KVMapQ_List + offset_map, mask=offset_map < partial_num, other=0
    )
    offset_row_max = cur_head * stride_row_maxh + q_idx
    offset_partial_lse = cur_head * stride_partial_lseh + offset_map

    lse = tl.load(
        partial_lse + offset_partial_lse,
        mask=offset_map < partial_num,
        other=0.0,
    )

    tl.atomic_max(row_max + offset_row_max, lse, mask=offset_map < partial_num)


@triton.jit
def DeFT_splitBynode_Triton_stage2_2_kernel(  # type: ignore
    L,
    partial_lse,
    row_max,
    KVMapQ_List,
    partial_num,
    stride_Lh,
    stride_partial_lseh,
    stride_row_maxh,
    BLOCK_N: tl.constexpr,
):
    cur_head = tl.program_id(axis=0)
    cur_idx = tl.program_id(axis=1)

    offset_map = cur_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    q_idx = tl.load(
        KVMapQ_List + offset_map, mask=offset_map < partial_num, other=0
    )
    offset_row_max = cur_head * stride_row_maxh + q_idx
    offset_partial_lse = cur_head * stride_partial_lseh + offset_map
    offset_L = cur_head * stride_Lh + q_idx

    row_max = tl.load(
        row_max + offset_row_max, mask=offset_map < partial_num, other=0.0
    )
    lse = tl.load(
        partial_lse + offset_partial_lse,
        mask=offset_map < partial_num,
        other=0.0,
    )

    new_exp = tl.exp(lse - row_max)
    tl.atomic_add(L + offset_L, new_exp, mask=offset_map < partial_num)


@triton.jit
def DeFT_splitBynode_Triton_stage2_3_kernel(  # type: ignore
    O,
    L,
    partial_o,
    partial_lse,
    row_max,
    KVMapQ_List,
    partial_num,
    stride_Oh,
    stride_Oq,
    stride_Lh,
    stride_partial_oh,
    stride_partial_os,
    stride_partial_lseh,
    stride_row_maxh,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    cur_head = tl.program_id(axis=0)
    cur_idx = tl.program_id(axis=1)

    offset_d = tl.arange(0, BLOCK_D)
    offset_map = cur_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    q_idx = tl.load(
        KVMapQ_List + offset_map, mask=offset_map < partial_num, other=0
    )
    offset_row_max = cur_head * stride_row_maxh + q_idx
    offset_partial_lse = cur_head * stride_partial_lseh + offset_map
    offset_L = cur_head * stride_Lh + q_idx
    offset_partial_o = (
        cur_head * stride_partial_oh
        + offset_map[:, None] * stride_partial_os
        + offset_d[None, :]
    )

    # L = tl.load(L + offset_L, mask=offset_map < partial_num, other=0.0)
    row_max = tl.load(
        row_max + offset_row_max, mask=offset_map < partial_num, other=0.0
    )
    lse = tl.load(
        partial_lse + offset_partial_lse,
        mask=offset_map < partial_num,
        other=0.0,
    )

    new_exp = tl.exp(lse - row_max)  # (BLOCK_N)

    tl.atomic_add(L + offset_L, new_exp, mask=offset_map < partial_num)

    o = tl.load(
        partial_o + offset_partial_o,
        mask=offset_map[:, None] < partial_num,
        other=0.0,
    )

    new_o = new_exp[:, None] * o

    offset_O = (
        cur_head * stride_Oh + q_idx[:, None] * stride_Oq + offset_d[None, :]
    )

    tl.atomic_add(O + offset_O, new_o, mask=offset_map[:, None] < partial_num)

    # o = new_exp[:, None] *


@torch.inference_mode()
def tree_attention_subtree_fwd(
    query_states: torch.Tensor,  # (query_num, num_heads, head_dim)
    key_buffer: torch.Tensor,  # (-1, num_kv_heads, head_dim)
    value_buffer: torch.Tensor,  # (-1, num_kv_heads, head_dim)
    output: torch.Tensor,  # (query_num, num_heads, head_dim)
    block_len: int,
    block_q: torch.Tensor,  # (partial_q_num)
    block_q_cnts: torch.Tensor,  # (block_num)
    block_q_offset: torch.Tensor,  # (block_num)
    # block_bitmasks: torch.Tensor,           # (partial_node_num)
    block_bitmasks: torch.Tensor,  # (kv_len)
    # block_node_lens: torch.Tensor,          # (partial_node_num)
    # block_node_cnts: torch.Tensor,          # (block_num)
    # block_node_offset: torch.Tensor,        # (block_num)
    block_kv: torch.Tensor,  # (kv_len)
    block_lens: torch.Tensor,  # (block_num)
) -> None:
    query_num, num_heads, head_dim = query_states.shape
    # kv_total_len = key_buffer.shape[2]
    query_states = query_states.transpose(
        0, 1
    )  # (num_heads, query_num, head_dim)
    key_buffer = key_buffer.transpose(0, 1)  # (num_kv_heads, -1, head_dim)
    value_buffer = value_buffer.transpose(0, 1)  # (num_kv_heads, -1, head_dim)
    output = output.transpose(0, 1)  # (num_heads, query_num, head_dim)

    partial_num = block_q.shape[0]

    kv_group_num = query_states.shape[0] // key_buffer.shape[0]

    assert head_dim in {16, 32, 64, 128}

    block_num = block_q_cnts.shape[0]

    # partial_attn = torch.zeros((block_num, 32, block_len), dtype=torch.float32, device="cuda")

    partial_o = torch.zeros(
        [num_heads, partial_num, head_dim],
        dtype=torch.float32,
        device=query_states.device,
    )
    partial_lse = torch.zeros(
        [num_heads, partial_num],
        dtype=torch.float32,
        device=query_states.device,
    )

    grid = (num_heads, block_num, 1)

    scale = 1.0 / (head_dim**0.5)

    num_warps = 4
    global cached_kernel_subtree
    torch.cuda.nvtx.range_push("DeFT_fwd_stage1")
    if cached_kernel_subtree:
        cached_kernel_subtree(
            grid,
            num_warps,
            query_states,
            key_buffer,
            value_buffer,
            block_len,
            block_q,
            block_q_cnts,
            block_q_offset,
            block_bitmasks,
            # block_node_lens, block_node_cnts, block_node_offset,
            block_kv,
            block_lens,
            partial_o,
            partial_lse,
            scale,
            query_states.stride(0),
            query_states.stride(1),
            key_buffer.stride(0),
            key_buffer.stride(1),
            partial_o.stride(0),
            partial_o.stride(1),
            partial_lse.stride(0),
        )
    else:
        tree_attention_subtree_fwd_kernel2[grid](
            query_states,
            key_buffer,
            value_buffer,
            block_len,
            block_q,
            block_q_cnts,
            block_q_offset,
            block_bitmasks,
            # block_node_lens, block_node_cnts, block_node_offset,
            block_kv,
            block_lens,
            partial_o,
            partial_lse,
            scale,
            query_states.stride(0),
            query_states.stride(1),
            key_buffer.stride(0),
            key_buffer.stride(1),
            partial_o.stride(0),
            partial_o.stride(1),
            partial_lse.stride(0),
            BLOCK_N=128,
            BLOCK_M=32,
            BLOCK_D=head_dim,
            # block_len=512,
            # num_stages=2,
            kv_group_num=kv_group_num,
            num_warps=num_warps,  # type: ignore
        )
        # cached_kernel_subtree = wrap_kernel_launcher(
        #     tree_attention_subtree_fwd_kernel2
        # )
    torch.cuda.nvtx.range_pop()
    DeFT_splitBynode_Triton_stage2(block_q, partial_o, partial_lse, output)


@triton.jit
def tree_attention_subtree_fwd_kernel(  # type: ignore
    Q,
    K,
    V,
    # KV_indices, KV_incices_offset, KV_len,
    # KVMapQ_List, KVMapQ_List_Offset, KVMapQ_List_Len,
    # partial_attn,
    block_len,
    block_q,
    block_q_cnts,
    block_q_offset,
    block_bitmasks,
    block_node_lens,
    block_node_cnts,
    block_node_offset,
    block_kv,
    block_lens,
    partial_o,
    partial_lse,
    scale,
    # partial_attn_stride0, partial_attn_stride1,
    stride_qh,
    stride_qs,
    stride_kvh,
    stride_kvs,
    stride_partial_oh,
    stride_partial_os,
    stride_partial_lseh,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    kv_group_num: tl.constexpr,
):
    cur_head = tl.program_id(axis=0)
    cur_block = tl.program_id(axis=1)

    cur_kv_head = cur_head // kv_group_num

    cur_q_cnt = tl.load(block_q_cnts + cur_block)
    cur_q_offset = tl.load(block_q_offset + cur_block)
    q_loc = tl.load(
        block_q + cur_q_offset + tl.arange(0, BLOCK_M),
        tl.arange(0, BLOCK_M) < cur_q_cnt,
        other=0,
    )
    offset_q = (
        cur_head * stride_qh
        + q_loc[:, None] * stride_qs
        + tl.arange(0, BLOCK_D)[None, :]
    )
    q = tl.load(
        Q + offset_q, mask=tl.arange(0, BLOCK_M)[:, None] < cur_q_cnt, other=0.0
    )

    cur_node_cnt = tl.load(block_node_cnts + cur_block)
    cur_node_offset = tl.load(block_node_offset + cur_block)

    offset_d = tl.arange(0, BLOCK_D)

    # attn = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    cur_len_offset = 0
    cur_len_offset = cur_len_offset.to(tl.int64)  # type: ignore

    max_logic = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    sum_exp = tl.full([BLOCK_M], 0.0, dtype=tl.float32)

    one_ll = 1
    one_ll = one_ll.to(tl.int64)  # type: ignore

    for node in range(0, cur_node_cnt, 1):  # type: ignore
        cur_node_len = tl.load(block_node_lens + cur_node_offset + node)
        cur_node_bitmask = tl.load(block_bitmasks + cur_node_offset + node)
        node_steps = (cur_node_len + BLOCK_N - 1) // BLOCK_N

        # node_offset = cur_block * partial_attn_stride0 + tl.arange(0, BLOCK_M)[:, None] * partial_attn_stride1 + cur_len_offset

        for node_step in range(0, node_steps, 1):  # type: ignore
            cur_kv_offset = node_step * BLOCK_N + tl.arange(0, BLOCK_N)
            kv_loc = tl.load(
                block_kv
                + cur_block * block_len
                + cur_len_offset
                + cur_kv_offset,
                mask=cur_kv_offset < cur_node_len,
                other=0,
            )
            offset_kv = (
                cur_kv_head * stride_kvh
                + kv_loc[:, None] * stride_kvs
                + offset_d[None, :]
            )
            k = tl.load(
                K + offset_kv,
                mask=cur_kv_offset[:, None] < cur_node_len,
                other=0.0,
            )  # (BLOCK_N, BLOCK_D) # type: ignore
            v = tl.load(
                V + offset_kv,
                mask=cur_kv_offset[:, None] < cur_node_len,
                other=0.0,
            )  # (BLOCK_N, BLOCK_D) # type: ignore

            attn_value = tl.dot(q, tl.trans(k))  # (BLOCK_M, BLOCK_N)
            attn_value *= scale
            attn_value = tl.where(
                (cur_kv_offset[None, :] < cur_node_len)
                and (
                    (one_ll << tl.arange(0, BLOCK_M)[:, None])
                    & cur_node_bitmask
                )
                > 0,  # type: ignore
                attn_value,
                -float("inf"),
            )

            cur_max_logic = tl.max(attn_value, axis=1)  # (BLOCK_M)
            new_max_logic = tl.maximum(cur_max_logic, max_logic)  # (BLOCK_M)

            exp_logic = tl.exp(
                attn_value - new_max_logic[:, None]
            )  # (BLOCK_M, BLOCK_N)
            exp_logic = tl.where(
                (cur_kv_offset[None, :] < cur_node_len)
                and (
                    (one_ll << tl.arange(0, BLOCK_M)[:, None])
                    & cur_node_bitmask
                )
                > 0,  # type: ignore
                exp_logic,
                0.0,
            )
            diff = max_logic - new_max_logic
            diff = tl.where(diff != diff, 0.0, diff)
            logic_scale = tl.exp(diff)  # (BLOCK_M)
            # logic_scale = tl.where(logic_scale != logic_scale, 0.0, logic_scale)

            acc *= logic_scale[:, None]  # (BLOCK_M, BLOCK_D)

            # print('diff', diff)
            # print("exp_logic", exp_logic)

            # print('max_logic', max_logic)

            acc += tl.dot(exp_logic, v.to(tl.float32))  # (BLOCK_M, BLOCK_D)

            sum_exp = sum_exp * logic_scale + tl.sum(
                exp_logic, axis=1
            )  # (BLOCK_M)

            max_logic = new_max_logic
            # mask = tl.full([BLOCK_M, BLOCK_N], True, dtype=tl.int1)
            # mask = tl.where(((1 << tl.arange(0, BLOCK_M)[:, None]) & cur_node_bitmask) > 0, True, mask)
            # mask = tl.where(node_step * BLOCK_N + tl.arange(0, BLOCK_N)[None, :] >= cur_node_len, False, mask)
            # attn = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            # attn = tl.where(mask, float("-inf"), attn)
            # partial_attn_offset = node_offset + node_step * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
            # tl.store(partial_attn + partial_attn_offset, 1.0, mask=(cur_kv_offset[None, :] < cur_node_len) and ((1 << tl.arange(0, BLOCK_M)[:, None]) & cur_node_bitmask) > 0)

            # attn = tl.where()
        #
        cur_len_offset += cur_node_len

        # mask = tl.where()
        # mask =

    # cur_node_lens = tl.load(block_node_lens + cur_node_offset)
    partial_o_offset = (
        cur_head * stride_partial_oh
        + (cur_q_offset + tl.arange(0, BLOCK_M))[:, None] * stride_partial_os
        + offset_d[None, :]
    )
    tl.store(
        partial_o + partial_o_offset,
        acc / sum_exp[:, None],
        mask=tl.arange(0, BLOCK_M)[:, None] < cur_q_cnt,
    )
    partial_lse_offset = (
        cur_head * stride_partial_lseh + cur_q_offset + tl.arange(0, BLOCK_M)
    )
    tl.store(
        partial_lse + partial_lse_offset,
        max_logic + tl.log(sum_exp),
        mask=tl.arange(0, BLOCK_M) < cur_q_cnt,
    )


@triton.jit
def tree_attention_subtree_fwd_kernel2(  # type: ignore
    Q,
    K,
    V,
    # KV_indices, KV_incices_offset, KV_len,
    # KVMapQ_List, KVMapQ_List_Offset, KVMapQ_List_Len,
    # partial_attn,
    block_len,
    block_q,
    block_q_cnts,
    block_q_offset,
    block_bitmasks,
    # block_node_lens, block_node_cnts, block_node_offset,
    block_kv,
    block_lens,
    partial_o,
    partial_lse,
    scale,
    # partial_attn_stride0, partial_attn_stride1,
    stride_qh,
    stride_qs,
    stride_kvh,
    stride_kvs,
    stride_partial_oh,
    stride_partial_os,
    stride_partial_lseh,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    kv_group_num: tl.constexpr,
):
    cur_head = tl.program_id(axis=0)
    cur_block = tl.program_id(axis=1)

    cur_kv_head = cur_head // kv_group_num

    cur_q_cnt = tl.load(block_q_cnts + cur_block)
    cur_q_offset = tl.load(block_q_offset + cur_block)
    q_loc = tl.load(
        block_q + cur_q_offset + tl.arange(0, BLOCK_M),
        tl.arange(0, BLOCK_M) < cur_q_cnt,
        other=0,
    )
    offset_q = (
        cur_head * stride_qh
        + q_loc[:, None] * stride_qs
        + tl.arange(0, BLOCK_D)[None, :]
    )
    q = tl.load(
        Q + offset_q, mask=tl.arange(0, BLOCK_M)[:, None] < cur_q_cnt, other=0.0
    )

    # cur_node_cnt = tl.load(block_node_cnts + cur_block)
    # cur_node_offset = tl.load(block_node_offset + cur_block)
    cur_len = tl.load(block_lens + cur_block)

    offset_d = tl.arange(0, BLOCK_D)

    one_ll = 1
    one_ll = one_ll.to(tl.int64)  # type: ignore

    cur_bitmask = tl.load(
        block_bitmasks + cur_block * BLOCK_N + tl.arange(0, BLOCK_N),
        mask=tl.arange(0, BLOCK_N) < cur_len,
        other=0,
    )
    kv_loc = tl.load(
        block_kv + cur_block * BLOCK_N + tl.arange(0, BLOCK_N),
        mask=tl.arange(0, BLOCK_N) < cur_len,
        other=0,
    )

    offset_kv = (
        cur_kv_head * stride_kvh
        + kv_loc[:, None] * stride_kvs
        + offset_d[None, :]
    )
    k = tl.load(
        K + offset_kv, mask=tl.arange(0, BLOCK_N)[:, None] < cur_len, other=0.0
    )  # (BLOCK_N, BLOCK_D)
    v = tl.load(
        V + offset_kv, mask=tl.arange(0, BLOCK_N)[:, None] < cur_len, other=0.0
    )  # (BLOCK_N, BLOCK_D)

    attn_value = tl.dot(q, tl.trans(k))  # (BLOCK_M, BLOCK_N)
    attn_value *= scale
    attn_value = tl.where(
        ((one_ll << tl.arange(0, BLOCK_M)[:, None]) & cur_bitmask[None, :]),
        attn_value,
        -float("inf"),
    )

    max_logic = tl.max(attn_value, axis=1)  # (BLOCK_M)
    exp_logic = tl.exp(attn_value - max_logic[:, None])

    acc = tl.dot(exp_logic, v.to(tl.float32))

    sum_exp = tl.sum(exp_logic, axis=1)  # (BLOCK_M)

    partial_o_offset = (
        cur_head * stride_partial_oh
        + (cur_q_offset + tl.arange(0, BLOCK_M))[:, None] * stride_partial_os
        + offset_d[None, :]
    )
    tl.store(
        partial_o + partial_o_offset,
        acc / sum_exp[:, None],
        mask=tl.arange(0, BLOCK_M)[:, None] < cur_q_cnt,
    )
    partial_lse_offset = (
        cur_head * stride_partial_lseh + cur_q_offset + tl.arange(0, BLOCK_M)
    )
    tl.store(
        partial_lse + partial_lse_offset,
        max_logic + tl.log(sum_exp),
        mask=tl.arange(0, BLOCK_M) < cur_q_cnt,
    )


@triton.jit
def tree_attention_subtree_fwd_kernel3(  # type: ignore
    Q,
    K,
    V,
    # KV_indices, KV_incices_offset, KV_len,
    # KVMapQ_List, KVMapQ_List_Offset, KVMapQ_List_Len,
    # partial_attn,
    # block_len,
    block_q,
    block_q_cnts,
    block_q_offset,
    block_bitmasks,
    # block_node_lens, block_node_cnts, block_node_offset,
    block_kv,
    block_lens,
    partial_o,
    partial_lse,
    scale,
    # partial_attn_stride0, partial_attn_stride1,
    stride_qh,
    stride_qs,
    stride_kvh,
    stride_kvs,
    stride_partial_oh,
    stride_partial_os,
    stride_partial_lseh,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    block_len: tl.constexpr,
    kv_group_num: tl.constexpr,
):
    cur_head = tl.program_id(axis=0)
    cur_block = tl.program_id(axis=1)

    cur_kv_head = cur_head // kv_group_num

    cur_q_cnt = tl.load(block_q_cnts + cur_block)
    cur_q_offset = tl.load(block_q_offset + cur_block)
    q_loc = tl.load(
        block_q + cur_q_offset + tl.arange(0, BLOCK_M),
        tl.arange(0, BLOCK_M) < cur_q_cnt,
        other=0,
    )
    offset_q = (
        cur_head * stride_qh
        + q_loc[:, None] * stride_qs
        + tl.arange(0, BLOCK_D)[None, :]
    )
    q = tl.load(
        Q + offset_q, mask=tl.arange(0, BLOCK_M)[:, None] < cur_q_cnt, other=0.0
    )

    # cur_node_cnt = tl.load(block_node_cnts + cur_block)
    # cur_node_offset = tl.load(block_node_offset + cur_block)
    cur_len = tl.load(block_lens + cur_block)
    kv_block_n_size = (cur_len + BLOCK_N - 1) // BLOCK_N

    offset_d = tl.arange(0, BLOCK_D)

    one_ll = 1
    one_ll = one_ll.to(tl.int64)  # type: ignore

    max_logic = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    sum_exp = tl.full([BLOCK_M], 0.0, dtype=tl.float32)

    for start_n_kv in range(0, kv_block_n_size, 1):
        cur_kv_offset_n = start_n_kv * BLOCK_N + tl.arange(0, BLOCK_N)
        cur_bitmask = tl.load(
            block_bitmasks + cur_block * block_len + cur_kv_offset_n,
            mask=cur_kv_offset_n < cur_len,
            other=0,
        )  # (BLOCK_N)
        if tl.sum(cur_bitmask) > 0:
            # mask = tl.where(((one_ll << tl.arange(0, BLOCK_M)[:, None]) & cur_bitmask[None, :]), True, False) # (BLOCK_M, BLOCK_N)
            # allmask = tl.max(tl.where(((one_ll << tl.arange(0, BLOCK_M)[:, None]) & cur_bitmask[None, :]), True, False), axis=1) # (BLOCK_M)
            kv_loc = tl.load(
                block_kv + cur_block * block_len + cur_kv_offset_n,
                mask=cur_kv_offset_n < cur_len,
                other=0,
            )
            offset_kv = (
                cur_kv_head * stride_kvh
                + kv_loc[:, None] * stride_kvs
                + offset_d[None, :]
            )
            k = tl.load(
                K + offset_kv,
                mask=cur_kv_offset_n[:, None] < cur_len,
                other=0.0,
            )  # (BLOCK_N, BLOCK_D)
            v = tl.load(
                V + offset_kv,
                mask=cur_kv_offset_n[:, None] < cur_len,
                other=0.0,
            )  # (BLOCK_N, BLOCK_D)

            attn_value = tl.dot(q, tl.trans(k))  # (BLOCK_M, BLOCK_N)
            attn_value *= scale
            attn_value = tl.where(
                (
                    (one_ll << tl.arange(0, BLOCK_M)[:, None])
                    & cur_bitmask[None, :]
                ),
                attn_value,
                -float("inf"),
            )
            # print("attn", attn_value)
            cur_max_logic = tl.max(attn_value, axis=1)  # (BLOCK_M)
            new_max_logic = tl.maximum(cur_max_logic, max_logic)

            exp_logic = tl.exp(attn_value - new_max_logic[:, None])
            logic_scale = tl.exp(max_logic - new_max_logic)
            # print("exp_logic", exp_logic)
            acc = tl.where(
                tl.max(
                    tl.where(
                        (
                            (one_ll << tl.arange(0, BLOCK_M)[:, None])
                            & cur_bitmask[None, :]
                        ),
                        True,
                        False,
                    ),
                    axis=1,
                )[:, None],
                acc * logic_scale[:, None]
                + tl.dot(exp_logic, v.to(tl.float32)),
                acc,
            )
            # acc *= logic_scale[:, None]

            # acc += tl.dot(exp_logic, v.to(tl.float32))
            # print("acc", acc)

            sum_exp = tl.where(
                tl.max(
                    tl.where(
                        (
                            (one_ll << tl.arange(0, BLOCK_M)[:, None])
                            & cur_bitmask[None, :]
                        ),
                        True,
                        False,
                    ),
                    axis=1,
                ),
                sum_exp * logic_scale + tl.sum(exp_logic, axis=1),
                sum_exp,
            )  # (BLOCK_M)
            # sum_exp = sum_exp * logic_scale + tl.sum(exp_logic, axis=1)

            max_logic = new_max_logic

    # result_mask = tl.where(sum_exp != 0, True, False)
    len_mask = tl.arange(0, BLOCK_M) < cur_q_cnt
    final_mask = len_mask
    partial_o_offset = (
        cur_head * stride_partial_oh
        + (cur_q_offset + tl.arange(0, BLOCK_M))[:, None] * stride_partial_os
        + offset_d[None, :]
    )
    tl.store(
        partial_o + partial_o_offset,
        acc / sum_exp[:, None],
        mask=final_mask[:, None],
    )
    partial_lse_offset = (
        cur_head * stride_partial_lseh + cur_q_offset + tl.arange(0, BLOCK_M)
    )
    tl.store(
        partial_lse + partial_lse_offset,
        max_logic + tl.log(sum_exp),
        mask=final_mask,
    )
