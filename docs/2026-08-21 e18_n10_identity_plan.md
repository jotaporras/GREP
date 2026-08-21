# e18 — n10 node-identity dev loop: plan + runbook

Date: 2026-08-21. Companion to `docs/2026-08-21 e18_direction_discussion.md`
(the diagnosis and the candidate list). This note is the **executable plan**: what
was built, which jobs to run, in which order, what to read off each, and what
decides the next step. Written so an Opus session with `/monitor-job` can run it
without re-deriving anything.

## 0. TL;DR

- Goal: make the mask LLM **name a node's neighbours** on tiny graphs (n≈10),
  the capability the e17 n60 failure analysis says is missing (76% of errors are
  2-hop-shortcut edge hallucinations, 24% sibling/endpoint confusion). 1-hop
  accuracy ≈100% on n10 and sibling confusion → 0 is the bar; then n15, then 2 hops.
- Three composable flags were added to `learnable_graph_mask` (all default off,
  all zero-init-able no-ops, all checkpoint-round-tripped): **A** `decision_gating`
  (diagnostic), **B** `struct_keys` (the candidate), **binding_head** (auxiliary
  loss). Plus a standalone **neighbour-naming probe**.
- Fleet: 7 arms × ≤30 min training on the n10 corpus via `scripts/e18_n10_sft.sbatch`.
  Calibrate s/step first (one 40-step job), then submit the arms.
- Prior to lower expectations for A: e13b's `decode_trail` (hard row trailing
  inter-mention tokens, zero-shot at eval, not trained) changed nothing
  (0.838 vs 0.840 decision-token acc). A differs (soft, mid-name positions,
  trained) but is kept only because it costs one flag.

## 1. What was built (commit on `feature/final_experiments`)

### 1.1 Model — `src/prism/models/gnn_llm.py` (`LearnableGraphMaskLLM`)

| flag | config keys | what it does | parameters |
|---|---|---|---|
| **A** decision gating | `gnn.decision_gating`, `gnn.decision_gain_init` | At every *untagged* answer-side position (separator after a mention, mid-name tokens) the attention bias row gets `decision_gain · gate(current, j)` on the name tokens of the current node's neighbours (gate = α+(1−α)cos ∈ [ε,1]), 0 elsewhere — goal mention stays visible. "Current node" = last completed mention. Teacher-forced via `decision_query_map` (collator), decode via `MaskDecodeInjector`/`_MaskDecodeRowState.current_node`. | scalar `decision_gain` (base LR) |
| **B** structural keys | `gnn.struct_keys`, `struct_keys_dim`, `struct_keys_layer_scope`, `struct_keys_gain_init` | At every in-scope (dense) attention layer, every query position t and node-token key j: `logits += tanh(sk_gain_ℓ)·(W_q^ℓ h_t · W_k Ψ_j)/√d_s`. h = post-input-layernorm attention input (pre-hook stash). Values/residual untouched. The LLM computes its own structural query (the mask is the id-lookup special case). | `sk_k` (d_gt→d_s, shared), `sk_q[ℓ]` (hidden→d_s), `sk_gain[ℓ]` — all base LR |
| binding head | `gnn.binding_head`, `binding_temperature`, `binding_loss_weight` | InfoNCE at the final token of every node mention over the graph's nodes: `CE(softmax_n cos(W_b h_p, Ψ_n)/τ, node(p))`, added to the LM loss × weight. Logged as `binding_loss` (wandb). Inert at generation. | `bind_proj` (hidden→d_gt), base LR |
| **D** soft edge tokens | `gnn.soft_edges` | One learned embedding per *directed* edge (u→v), `MLP([emb(u); emb(v); Ψ_u; Ψ_v])` (emb = mean input embedding of the node's first name mention), rescaled to the name-token embedding norm, spliced into `inputs_embeds` right after BOS. Every position map / label / attention mask is shifted by E = #directed edges; logits are sliced back so callers see the original frame. The edge list is thus *in context* as graph-side tokens — the graph-side upper bound the direction note asked for. Batch size 1 only (prefix length varies); decode = `generate(inputs_embeds=…)` prefill (`inference.py`). | `se_mlp` (2·hidden+2·d_gt → hidden → hidden), base LR |

All four: fail-loud loaders (`loaders.py` requires the weights when the flag is
on), saved by `run_dir.py`, recorded in `train_config.json` `gnn` block
(`train_v3.py` provenance now also records post_fusion/graph_lora/pointer/cross
flags — closes the e17 hole where pf runs reloaded as plain masks).

Tests: `tests/test_e18_identity.py` (23 tests: decision map, soft-row placement,
gain-0 bitwise no-ops for A and B, gradient reach, **decode parity of A, B, A+B,
D, D+A+B** against teacher forcing, bf16 autocast, binding loss = LM + w·aux,
inert without labels, D no-edge identity / batch>1 rejection / HF
`generate(inputs_embeds=…)` returns new tokens only, save keys). `tests/test_neighbour_probe.py` (scoring contract).

### 1.2 Probe — `scripts/neighbour_probe.py`

For every region r of every graph file: prompt the checkpoint through the SAME
client stack as the navigation eval (compact prompt, no edge list for graph
archs, `present` for the text control) with *"You are at r. Which regions are
directly connected to r? List EVERY region that shares an edge with r …"*, parse
the plan, score as a set. Metrics: `first_ok` (first named is a neighbour — the
decision-step quantity), `exact`, `precision`, `recall`, `sibling_err/q`
(named non-neighbour sharing a type prefix with a true neighbour),
`hallucinated/q`. Standalone because `path_validator.derive_targets` would
grade a neighbour list as a route to its last id.

### 1.3 Corpus — `scripts/nav_small_vllm_generate.sbatch`

`SIZE=10` → 2 communities × 4 regions (+objects ⇒ ~10 nodes), seeds 301+, 45
graphs × 10 tasks, inter-community edge prob 0.25 (0.05 never connects at this
size), ids 040–044 held out. `SIZE=15` → 3×4, seeds 401+. Jobs **7730744 (n10)**
and **7730745 (n15)** submitted 2026-08-21, pending on priority at the time of
writing. Outputs:
`$ALELAB_DRIVE/GREP-PRISM/data/n_{10,15}_vllm/gen/nav_n{10,15}_gemma_data/split/{formatted_all_new__train.json, formatted_all_new__val.json, train_graphs/, test_graphs/}`.

### 1.4 Runner — `scripts/e18_n10_sft.sbatch`

Arm table (flags per ARM) lives in `scripts/e18_arms.sh`, shared with the
plaza twin `scripts/e18_n10_plaza.sh` (same recipe but **4-bit** base —
the 31B bf16 does not fit a 48 GB A6000 — so plaza runs are a pipeline/probe
smoke test, not an s/step calibration for the B200 fleet; `GPU=<0|1> ARM=…
MAX_STEPS=… RUN_NAME=…_plaza4b nohup scripts/e18_n10_plaza.sh > logs/… &`).

`ARM=<arm> sbatch scripts/e18_n10_sft.sbatch`. e17 mask recipe verbatim
(`e17_pf_sft.sbatch`: 31B-it, stage-1 LoRA warm start, navigator-GT tower,
mask_alpha 0, binary, decode_consistent, slr 0.012, lr 2.5e-4) + arm flags,
`trainer.max_steps=$MAX_STEPS` (default 300), no in-train generation eval,
post-train held-out navigation eval on `test_graphs`, then the probe on
`test_graphs` and 3 `train_graphs`. Outputs under
`$ALELAB_DRIVE/GREP-PRISM/outputs/e18_identity/<RUN_NAME>_<wandb_id>/` and
`$ALELAB_DRIVE/GREP-PRISM/results/e18_probe/<RUN_NAME>_{test,train}.json`.
wandb tag `e18_identity`.

| ARM | flags | role |
|---|---|---|
| `mask` | — | control (e17 mask) |
| `mask_a` | A, gain init 3.0 | diagnostic: does a trained soft "neighbours of the current node" row at the choosing steps fix first-neighbour identity? |
| `mask_b` | B, d_s 64, gain init 1.0 | **the candidate** |
| `mask_ab` | A + B | |
| `mask_bind` | binding w 0.1 | does explicit name↔Ψ supervision alone help? |
| `mask_b_bind` | B + binding | |
| `mask_d` | D (soft edge tokens) | graph-side upper bound: if even this doesn't beat `mask`, the bottleneck is not the pathway |
| `text_edges` | `gnn.arch=llm`, edge list in text, fresh LoRA | upper bound / sanity: the probe must be ≈100% here |

## 2. Runbook (in order)

### Step 0 — preflight (login node, no GPU)

```
ssh betty 'bash -lc "cd ~/sourcecode/GREP && git pull && source /vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v3/bin/activate 2>/dev/null || export PATH=/vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v3/bin:\$PATH; python -m pytest tests/test_e18_identity.py tests/test_neighbour_probe.py tests/test_decode_style_mask.py tests/test_post_fusion.py tests/test_injection_scope.py tests/test_edge_weights.py -q"'
```
Expect all green. If `test_e18_identity.py` fails on betty but passed locally
(it did: 15 passed on macOS CPU), it is a transformers-version difference in
the attention pre-hook kwargs — escalate with the traceback.

### Step 1 — corpus ready?

```
ssh betty 'bash -lc "sacct -j 7730744,7730745 --format=JobID,State,Elapsed -n | grep -v batch | grep -v extern; ls /vast/projects/aribeiro/alelab/jporras/GREP-PRISM/data/n_10_vllm/gen/nav_n10_gemma_data/split/"'
```
Need `formatted_all_new__train.json`, `__val.json`, `train_graphs/` (40 files),
`test_graphs/` (5 files). Checks once it exists:

```
ssh betty 'bash -lc "cd ~/sourcecode/GREP && python - <<EOF
import json,glob
S=\"/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/data/n_10_vllm/gen/nav_n10_gemma_data/split\"
tr=json.load(open(S+\"/formatted_all_new__train.json\")); print(\"train examples\", len(tr))
import collections
degs=[]; sib=0; n=0
for f in sorted(glob.glob(S+\"/train_graphs/*.json\"))+sorted(glob.glob(S+\"/test_graphs/*.json\")):
    g=json.load(open(f))[\"graph\"]; R=[r[\"name\"] for r in g[\"regions\"]]
    d=collections.Counter(); [d.update(e) for e in g[\"region_connections\"]]
    degs+= [d[r] for r in R]; n+=1
    pre=collections.Counter(r.rsplit(\"_\",1)[0] for r in R); sib+= sum(v>1 for v in pre.values())
print(\"graphs\",n,\"mean region degree\",sum(degs)/len(degs),\"graphs with sibling prefixes\",sib)
EOF"'
```
Expect ~400 train examples, mean degree ≈3, siblings present in most graphs
(if `sib` is ~0 the probe cannot measure sibling confusion — report it, still run).

Tail of the generation log: `$ALELAB_DRIVE/GREP-PRISM/slurm-nav-small-gemma31b-vllm-7730744.out`
— look for `connectivity` retries or `0 graphs` (the INTER_PROB comment in the
sbatch explains why). Escalate if the split dir never appears after the job
ends.

### Step 2 — calibration (one short job)

```
ssh betty 'bash -lc "cd ~/sourcecode/GREP && ARM=mask MAX_STEPS=40 RUN_NAME=e18_n10_calib sbatch scripts/e18_n10_sft.sbatch"'
```
From its log read the s/step (`train/` progress bar or the wandb `train/
samples_per_second`) and the wall time of the post-train eval + the two probes.
Set `MAX_STEPS` for the fleet so that **training ≤ 30 min**: `MAX_STEPS =
floor(1800 / s_per_step)`, rounded down to a multiple of 50, capped at 400
(n10 has 400 train examples ⇒ 200 steps/epoch at bs1×ga2; 300 = 1.5 epochs).
n60 was 14.7 s/step on ~3k-token sequences; n10 sequences are ~4× shorter, so
expect 4–6 s/step ⇒ 300 steps is the likely answer. If the held-out eval + probes
take > 1 h, lower `PROBE_TRAIN_GRAPHS` to 1 — don't cut the test-graph probe.

Sanity from the calibration run (it is a real, if short, `mask` arm):
- the log prints `gnn_config` with the three e18 flags false;
- the probe JSONs exist and `first_ok` is a number (not NaN) — the client
  stack works end-to-end on n10 prompts;
- the post-train eval wrote `eval_logs/cross_eval/*.json`.

### Step 3 — the fleet

Submit all eight (they queue; 1 GPU each):
```
ssh betty 'bash -lc "cd ~/sourcecode/GREP && for a in mask mask_a mask_b mask_ab mask_bind mask_b_bind mask_d text_edges; do ARM=\$a MAX_STEPS=<calibrated> sbatch scripts/e18_n10_sft.sbatch; done"'
```
Then `/monitor-job all`. Per-arm watch list:

| signal | where | healthy | escalate if |
|---|---|---|---|
| `train/loss` | wandb / log | falls from ~1.x to <0.3 within 150 steps (n10 is easy) | flat, or NaN |
| `binding_loss` (bind arms) | wandb | falls well below `log(N)≈2.3` (N≈10 nodes) — ideally <0.3 | stays ≈2.3 (head not learning ⇒ Ψ and h not bindable) |
| `e18/decision_gain` (A arms) | wandb (GradientDebugCallback) | moves from 3.0 (either way); `e18/grad_norm_decision_gain` > 0 | frozen at exactly 3.0 ⇒ not in the optimizer (param-group bug) |
| `e18/sk_gain_mean`, `e18/sk_gain_absmax`, `e18/grad_norm_sk` (B arms) | wandb | gains move from 1.0; grad norm > 0 | frozen ⇒ same bug |
| `e18/grad_norm_bind_proj` (bind arms) | wandb | > 0 | 0 ⇒ head not in the loss graph |
| `e18/grad_norm_se_mlp` (`mask_d`) | wandb | > 0 | 0 ⇒ soft tokens not in the loss graph |
| post-train eval | `eval_logs/cross_eval/` + log "accuracy" | `text_edges` ≳ 95% on n10; `mask` should be high too (n10 is tiny) | any arm < `mask` by > 10 pts ⇒ the flag hurts, note it |
| probe | `results/e18_probe/<run>_test.json` `aggregate` | see §3 | NaN / 0 queries |
| exit | sacct | COMPLETED | FAILED/OOM ⇒ handoff to fable-debugger with the last 80 log lines |

Do **not** resubmit or kill on your own overnight (memory: monitoring only);
queue proposals for the morning.

### Step 4 — readout table

Fill this for every arm (test-graph probe unless noted):

| arm | nav acc (held-out) | probe first_ok | probe exact | sibling_err/q | hallucinated/q | train-graph probe exact | binding_loss final |
|---|---|---|---|---|---|---|---|

Script: each probe JSON has `aggregate` with exactly these keys; nav accuracy is
the `accuracy` field in `eval_logs/cross_eval/*.json` (sample-weighted mean over
the 5 test graphs; the log prints it as the summary table).

## 3. Decision rules

1. **`text_edges` probe `exact` < 0.9** ⇒ the probe prompt is the problem (the
   SFT'd model doesn't answer list questions), not the architectures. Fix the
   prompt / parser before reading anything else.
2. **`mask` probe `first_ok` already ≈ 1.0 on n10** ⇒ neighbour identity is not
   the bottleneck at this size; go straight to n15 (`DATA_SPLIT=.../n_15_vllm/...`)
   and, if still ≈1.0, to the 2-hop question (the n60 cliff). Report before
   launching.
3. **`mask_b` probe `exact` ≥ `mask` + 0.15 and sibling_err/q ≈ 0** ⇒ B works;
   next: n15 with `mask` vs `mask_b` vs `mask_b_bind`, then re-run e17's n60
   eval suite on a B checkpoint (memory `e17-eval-suite-jobs`).
4. **`mask_a` ≥ `mask_b`** ⇒ the decision-step *row* suffices and B's learned
   query isn't buying anything: keep A as the cheap mechanism, report the
   `decode_trail` contrast (trained soft ≠ zero-shot hard).
5. **Nothing beats `mask`, `text_edges` ≈ 1.0** ⇒ the pathways reach the
   decision step but the LLM doesn't exploit them in 300 SFT steps. Before
   concluding: (a) check `sk_gain`/`decision_gain` actually moved; (b) one
   longer run (`MAX_STEPS=600`) of `mask_b_bind`; (c) read `mask_d`: if the
   soft-edge upper bound beats `mask` on the probe, the LLM *can* use graph-side
   edge information and A/B are the wrong pathway (keep iterating on pathways);
   if `mask_d` ≈ `mask` too while `text_edges` ≈ 1.0, the graph-side route
   itself is the problem at this SFT budget — report, don't iterate.
6. **`mask_bind` alone helps** ⇒ binding was the missing supervision; combine
   with whichever pathway is best and carry the loss forward.

## 4. What I'm unsure about (verify on the calibration run)

- **s/step on n10** — guessed 4–6 s; MAX_STEPS=300 default may need changing.
- **Probe prompt** — the SFT'd models were never asked a list question; they may
  answer with a route anyway. The `plan` field is scored; if plans are routes, fall
  back to scoring `full_response` (one-line change in `neighbour_probe.plan_text`).
- **B under gradient checkpointing** — `_sk_keys` is kept armed across the
  recompute (same rule as `_pf_signal`); `_sk_h` is stashed by a pre-hook on
  `self_attn`, which fires during recompute too. Tested only on the CPU tiny
  model without checkpointing.
- **Binding head under gradient checkpointing** — the lm_head pre-hook captures
  the final hidden state outside the checkpointed blocks, so it should be live;
  if `binding_loss` logs NaN or the grad-debug shows `bind_proj` grad 0, that's
  where to look.
- **Decision gain init 3.0 / sk gain init 1.0** — chosen so the channels are open
  from step 0 (the e17 pf lesson: zero-init gates at structural LR never opened;
  these are at base LR, but 300 steps is short). Not tuned.
- **D scale** — soft tokens are normalised to the mean name-token embedding
  norm (Gemma's embeddings are pre-scaled by √hidden, so this is the right
  frame); not tuned. D at batch 1 only — the e17 recipe already trains at
  batch 1, but if `trainer.sft.per_device_train_batch_size` > 1 the run fails
  loud at the first step.
- **D + gradient checkpointing** — `build_soft_edges` runs outside the
  checkpointed blocks, so the splice happens once; untested on GPU.
- **RL path**: `BatchedMaskDecodeInjector` carries A/B state, but the RL prefill
  (`rl/` rollouts) does not build `decision_maps` — A/B are SFT-only until that
  is wired. Not needed for this loop.
