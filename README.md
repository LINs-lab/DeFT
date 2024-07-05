<div align="center">
<img src="assets/DeFT.jpeg" alt="logo" width="200"></img>
</div>

--------------------------------------------------------------------------------

# DeFT: Decoding with Flash Tree-attention for Efficient Tree-structured LLM Inference

[[paper](https://arxiv.org/abs/2404.00242)] [[slides](placeholder)][[video](placeholder)]

<!-- ![schemes](assets/DeFT.jpeg) -->


[Placeholder for video link]

## TL;DR
We propose DeFT, an IO-aware attention algorithm for efficient tree-structured interactions with LLMs by optimizing QKV grouping and attention calculation.

## News

- [2024/05] We update the second version of DeFT paper with a better algorithm for general tree-structured LLM inference: [DeFT: Decoding with Flash Tree-attention for Efficient Tree-structured LLM Inference](https://arxiv.org/abs/2404.00242)!
- [2024/03] [DeFT: Flash Tree-Attention With IO-Awareness for Efficient Tree-Search-Based LLM Inference](https://openreview.net/pdf?id=HqfLHoX8bR) has been accepted as Oral presentation in [ICLR'24 AGI Workshop](https://iclr.cc/virtual/2024/23126)!


## Abstract
Given the increasing demand for tree-structured interactions with LLMs, we introduce DeFT (Decoding with Flash Tree-Attention), an IO-aware tree attention algorithm tailored for tree-structured inference. Unlike traditional sequence-based decoding, tree-structured decoding better accommodates modern task requirements, including self-consistency, few-shot prompting, multi-step reasoning, and multi-model/head coordination. However, existing sequence-based inference systems are ill-suited for tree-structured decoding, resulting in redundancy in computation, memory footprints, and memory access, thereby undermining inference efficiency. To address this challenge, DeFT maintains memory-efficient attention calculation with low memory footprints through two key stages: (1) QKV Preparation: We propose a KV-Guided Grouping Strategy with Tree Split to intelligently group QKV, optimizing GPU resource utilization while minimizing memory reads/writes for KV cache between GPU global memory and on-chip shared memory; (2)Attention Calculation: We compute partial attention of each QKV group in a fused kernel and employ a Tree-topology-aware Global Reduction strategy to obtain final attention. By reducing 73-99% KV cache IO and nearly 100% IO for partial results during attention calculation (e.g., Softmax), DeFT achieves up to 2.52/3.82x speedup in the end-to-end/attention latency across three practical tree-based workloads: namely, few-shot prompting, multi-step reasoning, and speculative decoding, over state-of-the-art attention algorithms.

## Usage

### Environment Setup

```bash
pip install poetry
poetry lock --no-update
poetry install

```

### Run Demos

```bash
CUDA_VISIBLE_DEVICES=0 python examples/
```

## FAQ

1. **What is the difference between two versions of DeFT papers in arXiv?**

    DeFT-v1





## TODOs
We will release the code and data in the following order, please stay tuned!

- [ ] Release core code of DeFT, including kernels and examples.
- [ ] Release efficiency evaluation code
- [ ] Release real speculative decoding and multi-step reasoning demo with DeFT.



## Citation

If you find DeFT useful or relevant to your project and research, please kindly cite our paper:

```bibtex
@article{yao2024deft,
  title={DeFT: Flash Tree-attention with IO-Awareness for Efficient Tree-search-based LLM Inference},
  author={Yao, Jinwei and Chen, Kaiqi and Zhang, Kexun and You, Jiaxuan and Yuan, Binhang and Wang, Zeke and Lin, Tao},
  journal={arXiv preprint arXiv:2404.00242},
  year={2024}
}
```
