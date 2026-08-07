# System Architecture & Technical Specifications

## PagedAttention Theory & KV Cache Complexity
In eager LLM inference, contiguous CUDA allocation for Key-Value (KV) cache wastes 60-80% memory due to internal fragmentation.
For an 8B model (e.g. Llama-3.1-8B with $n_{\text{layers}}=32, n_{\text{kv\_heads}}=8, d_{\text{head}}=128$), memory per token is:
$$\text{Memory}_{\text{token}} = 2 \times 32 \times 8 \times 128 \times 2\text{ bytes} = 128\text{ KB/token}$$

With virtual memory pages (block size $K=16$), PagedAttention drops fragmentation below $4\%$, enabling high concurrency.

## Guided Logit Masking (FSA/CFG)
To guarantee structural JSON compliance:
1. Pydantic v2 schemas compile into Deterministic Finite Automata (DFA) state machine $\mathcal{M} = (Q, \Sigma, \delta, q_0, F)$.
2. At step $t$, valid vocabulary token set $A_t$ is computed.
3. Logit vector $z_t$ is masked:
   $$\tilde{z}_{t, i} = \begin{cases} z_{t, i} & \text{if } i \in A_t \\ -\infty & \text{if } i \notin A_t \end{cases}$$
4. Softmax probability over allowed tokens guarantees zero schema/type validation errors.

## High-Dimensional Vector Cosine Caching
Prompt vectors $E(q) \in \mathbb{R}^{d}$ ($d=384$ or $1536$) are indexed in Redis VSS HNSW index:
$$S_c(E(q), v_k) = \frac{E(q) \cdot v_k}{\|E(q)\|_2 \|v_k\|_2}$$

For similarity $S_c \ge 0.92$, cached JSON is returned in $<8\text{ms}$.
