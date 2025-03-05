# DeFT

## Folder Structure

### `/deft`
A library of DeFT frameworks, adapted from [SGLang](https://github.com/sgl-project/sglang/blob/4cb9aaedf3dfe4f876ba447ab2ac1ac9c75da911).

---

### `/examples`
- **`run_DeFT_llama_paged.py`**: Script to run LLaMA models with DeFT, supporting paged memory management.

---

### `/experiments`
Contains scripts and results for various experimental setups:
1. **`/ablation`(old and might be out of date)**:
   Ablation studies exploring the effects of different GPUs, models, and prompt lengths on speedups.
2. **`/few_shot_prompting`**:
   Scripts and results for few-shot prompting experiments.
3. **`/reasoning`**:
   Scripts and results for multi-step reasoning tasks.
4. **`/speculative_decoding`**:
   Scripts and results for speculative decoding experiments.


<!-- 
## Install
```bash
uv sync --dev
pre-commit install
. .venv/bin/activate # choose your own bash
```

## Run demos
```bash
export CUDA_VISIBLE_DEVICES=2 # chose your own GPUs
# export model="meta-llama/Meta-Llama-3-8B" # support llama models
export model="meta-llama/Meta-Llama-3.1-8B" # support llama models
export mode="flatten" # DeFT-Flatten. "seq" for Radix Attention if mem is "paged"
# export mode="seq" # for Radix Attention if mem is "paged"
export mem="paged" # "paged":paged memory management
```
Different combinations of mode and mem correspond to different baselines and DeFT variants. Refer to [Table: Attention Operators and Memory Management](#table-attention-operators-and-memory-management) for more details about supported combinations.

Example for Speculative Decoding:
```bash
export task="Speculative_Decoding"
export dataset="../dataset/generation/Speculative_Decoding/APPS_tree_size64.json" # select tree size =64 tokens for token candidates
export prompt_len=6000 # set the prompt_len(if > original prompt len, we will pad it)
export maxseq=7000
export tree_idx=0 # only select the first tree
python examples/run_DeFT_llama_paged.py --model $model --max_seq_len $maxseq  --mode $mode --Branch_controller $task --dataset $dataset --mem $mem --tree_idx $tree_idx --prompt_len $prompt_len
```

Example for Multi-step Reasoning:
```bash
export task="Practical_Tree"
export workload="sorting128ToT" #("docmergeToT" "sorting128ToT" "set128ToT" "keywordToT")
export dataset="../dataset/generation/Reasoning/$workload.json" # select tree size =128 tokens for token candidates
export tree_idx=0 # only select the first tree
export prompt_len=4000 # pad the prompt to 4000, if you want to adopt original prompt len, don't export it.
export maxseq=7000 # set it to prompt_len+3000(for generated tokens)
python examples/run_DeFT_llama_paged.py --model $model --max_seq_len $maxseq  --mode $mode --Branch_controller $task --dataset $dataset --mem $mem --tree_idx $tree_idx --prompt_len $prompt_len
```

Example for Few-shot Prompting:
```bash
export task="Simple_Tree"
export prompt_len=4000 # pad the prompt to 4000
export maxseq=4400
export width=50 #set tree width to 50
python examples/run_DeFT_llama_paged.py --model $model --max_seq_len $maxseq  --mode $mode --Branch_controller $task --dataset $dataset --mem $mem --prompt_len $prompt_len --max_width $width
``` -->

### Argument Explained
### Model (`--model`)
We support LLaMA models, and most of our experiments are conducted using [Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B).

---

### Attention Operators (`--mode`) and Memory Management (`--mem`)
We provide support for various baseline methods and DeFT variants with different combinations of attention operators and memory management strategies.

#### **Supported Methods**
- **Baselines:**
  - **Flash-Decoding**: Sequential attention with unpaged memory.
  - **Tree Attention Medusa**: Tree-structured attention with unpaged memory.
  - **Radix Attention**: Sequential attention with paged memory.
- **DeFT Variants:**
  - **DeFT-Node**: Node-based attention with paged memory.
  - **DeFT-Node-Chunk**: Chunked node-based attention with paged memory.
  - **DeFT-Flatten**: Flattened attention with paged memory.
  - **DeFT-Tree-Index**: (WIP)DeFT-Node attention with paged memory in a tree-indexed manner. It constructs metadata with TreeIndexPool, leveraging node indices for efficient memory access.
#### **Table: Attention Operators and Memory Management**
| Mode           | Memory Management | Method                  |
|----------------|-------------------|-------------------------|
| `seq`          | `unpaged`         | Flash-Decoding          |
| `tree`         | `unpaged`         | Tree Attention Medusa   |
| `seq`          | `paged`           | Radix Attention         |
| `flatten`      | `paged`           | DeFT-Flatten            |
| `node`         | `paged`           | DeFT-Node               |
| `node_chunk`   | `paged`           | DeFT-Node-Chunk         |
| `tree_index`   | `paged`           | DeFT-Tree-Index         |
---


### Task (`--Branch_controller`)

We provide implementations for the following tasks in `deft/managers/Tree_Decoding/generation/branch_func_example.py`. You can also implement your own custom tasks if needed.

#### **Available Tasks**
- **`Simple_Tree`**: Few-shot prompting.
- **`Beam_Search`**: Beam search (note: still contains some bugs).
- **`Practical_Tree`**: Multi-step reasoning tasks. Corresponding datasets of tree templates are located in `dataset/generation/Reasoning`.
- **`Speculative_Decoding`**: Mock implementation of speculative decoding. Corresponding datasets of tree templates are located in `dataset/generation/Speculative_Decoding`.

---

### **Note on Potential Output Issues**

For certain settings, the output might consist of random or unmeaningful words. Below are two specific scenarios to be aware of:

1. **When setting `--prompt_len` longer than the actual prompt length:**
   - If the provided prompt length is shorter than the value set by `--prompt_len`, the prompt will be padded to meet the specified length. However, the additional content may result in unmeaningful output.
   - **Example**:
     When the task is set to `Simple_Tree`, with `--prompt_len` set to `4000` and `--max_seq_len` set to `4400`, the prompt will be padded to `4000` tokens, and the model will generate exactly `400` tokens. The generated content may lack coherence.

2. **When using `Speculative_Decoding`:**
   - This task is a mocked version designed to verify token handling. Token candidates are selected from the top-k logits, which means the output does not carry meaningful content.

---
### Datasets of Tree Templates (`--dataset`)

Specifies the path to tree templates, which define when to branch and prune during decoding. Below are the dataset paths for different tasks:

- **Few-shot prompting**: Tree templates are not required.
- **Multi-step reasoning**: Use the path: `dataset/generation/Reasoning`.
- **Speculative decoding**: Use the path: `dataset/generation/Speculative_Decoding`.

For more details, refer to the documentation in [`/dataset/generation/TreeTemplate_readme.md`](./dataset/generation/TreeTemplate_readme.md).

### Tree Shape Settings

#### **Parameters**
- **`--max_depth`**:
  Specifies the maximum levels or cascades of the tree. Default: `10`.
  **Note**: If tree templates are already provided, this parameter will be ignored.

- **`--max_width`**:
  Defines the maximum number of branches that can exist simultaneously within a single batch. Default: `50`.
  **Note**: If tree templates are already provided, this parameter will be ignored.

- **`--prompt_len`**:
  Default: `None`.
  If set, the prompt length will be padded to the exact number of tokens specified.
  **Warning**: Padding may result in random or unmeaningful words in the output.

- **`--max_seq_len`**:
  Default: `500`.
  If the `max_seq_len` value is smaller than the prompt length, only prefill will occur, and no decoding will take place. The number of generated tokens is calculated as:
 $$
\text{Generated Tokens} = \text{max\_seq\_len} - \text{prompt\_len}
$$
