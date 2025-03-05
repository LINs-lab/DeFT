## Tree Templates

We provide tree templates for **Reasoning** and **Speculative Decoding** tasks, located in the `/Reasoning` and `/Speculative_Decoding` directories, respectively.

During runtime, the branch controller in DeFT manages the branching and pruning of trees based on these templates.

---

## Decoding Tree Generation Format

### **Reasoning**
Tree templates for four reasoning tasks collected from **Graph-of-Thoughts** follow this format:
1. **Prompt**: The input prompt for the reasoning task.
2. **Size and Children**: The size and child nodes for each tree node.

---

### **Speculative Decoding**
We provide speculative decoding records collected from **Medusa**, with tree sizes of 5, 10, 32, 64, 96, 128, and 256. The format is as follows:
1. **`Tree_Structure`**: The structure and topology of the token candidates.
2. **`Prompt`**: The input problem, sourced from the **APPS dataset**.
3. **`Accept_length`**: The accepted token length at each decoding step.
