# e10 — Learnable Relative Positional Encodings (attention-mask form)

Companion to `notebooks/2026-06-30 dev_relative_pe_impl.ipynb`. Code:
`src/prism/models/gnn_llm.py` (`LearnableGraphMaskLLM`, `GraphMaskLLM`), factory
`src/prism/models/architectures.py`, runner `scripts/dev_e10_test_relative_learnable.sbatch`.

> Status: implementation done + GPU-validated; **training-accuracy results PENDING** (runs in
> progress / queued). Sections 1–3 are final; §5 records verified methodological findings; §6 is a
> results skeleton to fill once the sweep lands. Interpretation of the accuracy numbers is the
> author's, not written here.

---

## 1. Motivation

- **e9 multistage was not enough** on its own.
- Empirically, a **non-learnable, adjacency-based attention mask** (`graph_mask_llm`) looked like it
  *beat* the additive-PE approach (RPEARL / `GraphAugmentedLLM`). That motivates keeping the *masking
  form* but making it **learnable** — a step toward "relative positional encodings": a learned
  function of relative information between nodes, folded into the attention logits.
- **Hard design constraint:** the architecture's parameters must be **independent of graph size**.
  We therefore do not store an `N×N` object; we *induce* it by an **outer product** of per-node
  embeddings produced by a shared (size-independent) function.
- **Caveat that reframes the premise (see §5):** the historical "graph-mask beats RPEARL" number was
  confounded by (a) an eval-set change and (b) an attention-mask leak. e10's first job is therefore a
  **clean baseline on the full eval set with correct masking**, then the learnable sweep.

## 2. Formulation

**Notation.** Let $G=(V,E)$ be a scene graph with $N=\lvert V\rvert$ nodes and raw edge adjacency
$A_{0}\in\{0,1\}^{N\times N}$. Define the **may-attend relation** as its symmetrized, self-looped,
$k$-hop reachability closure,

$$
A \;=\; \mathbf{1}\!\left[\big(\operatorname{sym}(A_{0})\vee I_{N}\big)^{k}>0\right]\in\{0,1\}^{N\times N},
\qquad
\mathcal{A}=\{(i,j):A_{ij}=1\},
$$

where $\mathbf{1}[\cdot]$ is the entrywise indicator. A prompt is a token sequence of length $L$; an
**injection map** $\pi:\{1,\dots,L\}\to\{0,1,\dots,N\}$ assigns each token to a node ($\pi(t)=n$) or
marks it non-node ($\pi(t)=0$), with disjoint node spans.

**Per-node encoder (size-independent).** A standalone graph transformer
$\mathrm{GT}_{\theta}:\mathcal{G}\to\mathbb{R}^{N\times D}$ (R-PEARL random probes + sparse attention)
produces per-node embeddings

$$
\Psi=\mathrm{GT}_{\theta}(G)\in\mathbb{R}^{N\times D},
\qquad
\psi_{i}=\Psi_{i,:}\in\mathbb{R}^{D},
$$

with $\theta$ independent of $N$ and $\mathrm{GT}_{\theta}$ permutation-equivariant,
$\mathrm{GT}_{\theta}(P\!\cdot\!G)=P\,\mathrm{GT}_{\theta}(G)$ for any node permutation $P$.

**Scaled similarity.** Define $s:\mathbb{R}^{D}\times\mathbb{R}^{D}\to\mathbb{R}$ and the matrix
$S\in\mathbb{R}^{N\times N}$, $S_{ij}=s(\psi_{i},\psi_{j})$:

$$
s(\psi_{i},\psi_{j})=
\begin{cases}
\dfrac{\langle\psi_{i},\psi_{j}\rangle}{\lVert\psi_{i}\rVert\,\lVert\psi_{j}\rVert}, & (\texttt{cosine}),\\[2.4ex]
\dfrac{\langle\psi_{i},\psi_{j}\rangle}{\sqrt{D}}, & (\texttt{inv\_sqrt\_d}).
\end{cases}
$$

The raw Gram matrix $\Psi\Psi^{\top}$ has entries of order $D$ ($\approx 10^{3}$ at $D=1024$) and would
saturate the softmax; both choices rescale it.

**Node-level structural mask.** For a fixed mixing coefficient $\alpha\in[0,1)$, define
$M\in(\mathbb{R}\cup\{-\infty\})^{N\times N}$ by

$$
M_{ij}=
\begin{cases}
\alpha+(1-\alpha)\,S_{ij}, & (i,j)\in\mathcal{A},\\[0.5ex]
-\infty, & (i,j)\notin\mathcal{A}.
\end{cases}
$$

On the allowed set, $S=\hat{\Psi}\hat{\Psi}^{\top}$ (with $\hat{\psi}_{i}=\psi_{i}/\lVert\psi_{i}\rVert$
for `cosine`) is the **outer-product / Gram** matrix of the per-node embeddings — the size-independent
source of the $N\times N$ interaction — while non-edges are **hard-blocked** ($-\infty$), so the
learned term only re-weights already-adjacent pairs.

**Token-level bias and attention.** Lift $M$ through $\pi$, biasing only node–node token pairs:

$$
B_{ts}=
\begin{cases}
M_{\pi(t)\,\pi(s)}, & \pi(t)\neq 0\ \wedge\ \pi(s)\neq 0,\\
0, & \text{otherwise},
\end{cases}
\qquad B\in(\mathbb{R}\cup\{-\infty\})^{L\times L}.
$$

$B$ is added to the pre-softmax logits alongside the model's causal/sliding mask $C$ (with $C_{ts}=0$
if key $s$ is visible to query $t$ under causality and the layer's sliding window, and $-\infty$
otherwise):

$$
\operatorname{Attn}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_{k}}}+C+B\right)V .
$$

Queries, keys, and values are untouched; only decoder layers in the chosen scope receive $B$
(`mask_layer_scope`: `dense` $=$ Gemma-4 `full_attention` layers, `all` $=$ every self-attention layer).

**Properties.**

- **Reduction.** $S\equiv 0$ (or the hard $\{0,-\infty\}$ form) recovers the parameter-free
  `graph_mask_llm`. The limit $\alpha\to 1$ gives $M_{ij}\to\alpha$ on $\mathcal{A}$ (pure adjacency)
  with $\partial M/\partial\theta\to 0$; hence $\alpha\in[0,1)$ is enforced.
- **Boundedness (cosine).** $S_{ij}\in[-1,1]\;\Rightarrow\; M_{ij}\in[\,2\alpha-1,\,1\,]$ on
  $\mathcal{A}$, so the softmax cannot saturate.
- **Well-posedness.** $(i,i)\in\mathcal{A}$ and every non-node column keeps $B=0$, so each node-token
  row retains at least one finite logit (no degenerate/NaN softmax).
- **Equivariance.** Equivariance of $\mathrm{GT}_{\theta}$ propagates to $S$ and $M$; given $\pi$, the
  token bias $B$ is invariant to node relabeling.
- **Gradient / cold start.** The loss gradient flows
  $\mathcal{L}\to B\to S\to\Psi\to\theta$. The diagonal $s(\psi_{i},\psi_{i})\equiv 1$ (cosine) is
  constant, contributing no self-loop gradient; at initialization near-orthogonal $\psi$ give
  $S\approx 0$ and $\lVert\nabla_{\theta}\mathcal{L}\rVert$ small — empirically
  $\approx 8\times10^{-5}$ of the LoRA gradient norm (§5.3) — which motivates the boosted
  `structural_lr_mult`.

## 3. Experiment setup

- **Base model:** `google/gemma-4-12B-it`, LoRA (r=16, targets q/k/v/o/gate/up/down), bf16, from
  scratch (not multistage), `epochs=3`, `lr=2.5e-4`, `per_device_train_batch_size=1`.
- **Data:** `nav100_n30_gemma_data` (revised). **Eval on the FULL held-out set** — all 10
  `test_graphs/*.json` (`eval.num_graphs=-1`), micro-averaged `eval/accuracy`. This is the fix for the
  §5.1 artifact; every e10 arm uses it.
- **Standalone GT ($\Psi$):** `GraphTransformer` L2, `d_model=1024`, `pe_hidden_channels=256`,
  `num_samples=40`, `k_pe=2`, `k_gt=2`, `heads=8`, random probes.
- **Runner:** `scripts/dev_e10_test_relative_learnable.sbatch` (SLURM array).

**Arms.**

*(a) Masking A/B + adjacency control* — isolates the mask leak and the value of the learnable term
(all `data.text_edge_list=none`, connectivity from the mask only):

| arm | arch | scope | notes |
|---|---|---|---|
| graph_mask_fixed | graph_mask_llm | all | correct causal/sliding masking |
| graph_mask_buggy | graph_mask_llm | all | `mask_buggy_causal_fold=true` — the historical leak |
| adjacency_dense | graph_mask_llm | dense | pure adjacency, matches learnable scope |
| learnable | learnable_graph_mask | dense | $\alpha A+(1-\alpha)S$, $\alpha=0.7$, cosine, lr_mult=5 |

*(b) mask_alpha sweep* — current `scripts/dev_e10_test_relative_learnable.sbatch`, **edges in the
prompt** (`data.text_edge_list=present`), dense, cosine, `structural_lr_mult=5`, full eval:

| arm | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| $\alpha$ | 0.01 | 0.05 | 0.10 | 0.50 | 0.90 | 1.00 |
| arch | learnable | learnable | learnable | learnable | learnable | graph_mask (pure adj) |

Run: `sbatch scripts/dev_e10_test_relative_learnable.sbatch`. W&B tag `e10_alpha_sweep`.

## 4. Config knobs (base_config.yaml → `gnn.*`)

`arch=learnable_graph_mask`; `mask_alpha` ($\alpha$, in $[0,1)$); `mask_psi_scale`
(`cosine`|`inv_sqrt_d`); `mask_layer_scope` (`all`|`dense`);
`mask_k_hops`/`mask_symmetrize`/`mask_use_edges` (adjacency $A$); `mask_buggy_causal_fold` (A/B leak
reproduction); `structural_lr_mult` (GT LR boost); GT block reused from `gt_*`/`pe_*`.
Eval-from-checkpoint reconstructs all of these via `loaders.py`.

## 5. Verified findings this session (methodological)

These are GPU-verified facts, not the training-accuracy results.

**5.1 The "30% → 8%" was an eval-set artifact, not a regression.** The historical graph-mask number
(~0.7 on `eval/acc/data_gen_023`) came from evaluating a **single graph**, `data_gen_023.json`. After
the refactor, eval defaults to `num_graphs=5` over the *sorted* `test_graphs/` directory, keeping
`{004, 009, 011, 016, 019}` and **dropping `data_gen_023` (the 6th file)** — which is exactly why 023
vanished from the new logs and the numbers looked like a collapse (observed per-graph `~0.1`). It
prints `[eval] capping train-time eval to 5 of 10 graphs (5 dropped)`. All graphs are ~26–33 nodes ×
10 tasks, so per-graph accuracy is ±0.1-grained (noisy). **Fix:** `eval.num_graphs=-1` everywhere.

**5.2 Graph-mask attention leak (sliding layers).** `_graph_mask_attention_forward` cast SDPA's
*boolean* mask to float, mapping blocked positions to additive $0$ (attendable). It fires even at
`batch=1` via the **sliding-window layers** (they get an explicit mask; cannot use `is_causal`), so on
Gemma-4 (~5/6 sliding layers) the "graph mask" ran **near-bidirectional** attention. Historical
graph-mask runs trained/evaluated with this. Empirically (tiny Gemma-3, B=1): fixed vs buggy differ by
~0.30–0.36 in logits; on Llama (no sliding layers) they are identical. Correct fold now default; the
old behavior is reproducible via `mask_buggy_causal_fold=true` (arm `graph_mask_buggy`). **Implication:
the "graph-mask beats RPEARL" premise must be re-checked with correct masking** — that is what the A/B
arms test. (RPEARL/`gt` are on a separate PE-injection path, untouched by this bug.)

**5.3 GT cold-start.** Cosine scaling is scale-invariant, so at init the GT gradient is tiny
($\lVert\nabla_\theta\rVert$ grad-norm $\approx 1.7\times10^{-5}$ vs LoRA $\approx 2.1\times10^{-1}$,
ratio $\approx 8\times10^{-5}$). AdamW normalizes by gradient RMS, so this overstates the effect, but
the GT is a deep module with noisy probe gradients → boosted with `structural_lr_mult=5` (precedent:
e7 composite used 3–5×). Watch `debug/grad_norm_gnn`; if flat, raise the mult or switch to
`inv_sqrt_d`.

## 6. Results — PENDING (skeleton to fill from W&B)

I can't pull W&B directly; paste `eval/accuracy` (full 10-graph micro-average) per run — tags
`e10_alpha_sweep` and `e10_relative_pe` — and I'll complete these tables and the summary.

**Masking A/B (edges from mask only):**

| arm | eval/accuracy (full set) |
|---|---|
| graph_mask_fixed | _pending_ |
| graph_mask_buggy | _pending_ |
| adjacency_dense | _pending_ |
| learnable ($\alpha=0.7$) | _pending_ |

**mask_alpha sweep (edges in prompt):**

| $\alpha$ | 0.01 | 0.05 | 0.10 | 0.50 | 0.90 | 1.00 |
|---|---|---|---|---|---|---|
| eval/accuracy | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |
| `debug/grad_norm_gnn` (final) | | | | | | n/a |

Compare arms by the **micro-averaged `eval/accuracy`**, not individual per-graph rates (±0.1-grained).

## 7. Open questions

- Does "graph-mask > RPEARL" survive on the full eval set with correct (non-leaky) masking?
- Where does the $\alpha$ sweep peak — is the learnable term ($\alpha$ small) worth its cost over pure
  adjacency?
- Is the cold-start starving the GT (grad-norm), and does `inv_sqrt_d` help?
- $\alpha=1.0$ is run as `graph_mask` (edge bias $0$) vs the learnable $\alpha\to 1$ limit (edge bias
  $\alpha$) — minor, but the two are not bit-identical.
