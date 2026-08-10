# Status summary — August 10, 2026

Two topics: (1) the e16 RL implementation (GREP-T3), (2) the SPINE eval results
for the e14 models, plus a plain-language explanation of the plaza setup work.

---

## 1. State of the RL implementation (e16_rl_training / GREP-T3)

### What this is

The approved plan turns the unvetted notebook (`notebooks/2026-08-07
fable-vllm-graph-demo.ipynb`) into a production path for RL training on the GT
architecture: a vLLM engine that carries the graph signal Ψ, a vLLM-backed eval
backend, a trl GRPO trainer that uses that engine for rollouts, a proof-of-concept
run on plaza, and finally a betty sbatch. Milestones M0–M6; the campaign name and
wandb tag are locked as `e16_rl_training` (e15 was already taken by WIRE pe_pool).

### Done (all committed and pushed on `feature/final_experiments`)

| commit | what |
|---|---|
| `1db0863` | Housekeeping: notebooks moved into `notebooks/`, git tag `e14_scaling` on the pre-e16 state |
| `b628726` | **M1** — the vLLM graph engine, ported Qwen→Gemma-4, with parity tests |
| `9b39865` | **M2** — `--backend vllm` for eval (code complete; real-checkpoint parity run still pending) |
| `eab36e3` | **M3** — GRPO RL trainer with graph-conditioned vLLM rollouts |

The full local test battery (21 tests) is green on the Mac (vLLM CPU backend):
`tests/test_vllm_graph_{psi,noop,parity}.py`, `tests/test_vllm_eval_backend.py`,
`tests/test_rl_{rewards,savedir,grpo_smoke}.py`.

### How the pieces fit — files and load-bearing code

**The engine (M1), `src/prism/models/vllm_graph/`:**

- `psi.py` — `build_psi_transport()` builds a `[seq_len, hidden+1]` tensor per
  prompt: columns `[:hidden]` are Ψ from the checkpoint's own
  `build_pe_signal` chain (so trained weights, permutations, and node features
  are honored exactly); the last column marks which token positions are inside
  node-name mentions. It fails loudly if an identity-RoPE checkpoint's prompt
  ends inside a node mention (same check as `inference._identity_rope_kwargs`).
- `processor.py` — vLLM multimodal plumbing. The Ψ tensor rides under vLLM's
  "image" modality (vLLM's parser only speaks image/video/audio). The
  `GraphMultiModalProcessor.apply()` override is pinned to vLLM 0.26's
  signature — revisit on any vLLM version bump.
- `model.py` — `GraphGemma4ForCausalLM`, a registered wrapper around vLLM's
  stock `Gemma4ForCausalLM`. `embed_input_ids()` receives the transport rows
  batch-aligned from the runner and splits them into `_psi_packed` (the Ψ) and
  `_span_mask_packed` (the mention-position mask). Registration is runtime
  (`register_graph_gemma4()`), which requires a single-process engine
  (`VLLM_ENABLE_V1_MULTIPROCESSING=0`).
- `attention.py` — the core of the port. Every `Gemma4Attention.forward` is
  replaced with a version that projects Ψ through the layer's own fused qkv
  weight and adds the un-normed, un-rotated contributions AFTER q/k-norm and
  RoPE — exactly mirroring `gnn_llm._prism_pe_attention_forward`
  (gnn_llm.py:19–49). Layers with `use_k_eq_v` (no `v_proj` in HF) skip the
  v-injection like HF does. KV-shared configs are REFUSED at engine build
  (see "findings" below). Identity-RoPE is implemented by zeroing `positions`
  at masked spans before `rotary_emb`.
- `engine.py` — `build_graph_llm()` (engine constraints: `enforce_eager=True`,
  `enable_prefix_caching=False`, `enable_mm_embeds=True`),
  `checkpoint_engine_policy()` (reads the recorded config, refuses non-additive
  architectures — mask archs have no vLLM analog for their decode-time
  injection), `materialize_serving_dir()` (one-time bf16 LoRA merge for
  serving; never merges into nf4 — that was the e14 reload bug),
  `build_plain_llm()` for plain-LLM checkpoints.
- `psi_producer.py` — for eval, rebuilds ONLY the Ψ tower + the base model's
  embedding table (an "embeddings-only shim") instead of loading the whole 31B
  HF model next to the engine. Uses the same rebuild code as the eval loader:
  `loaders.additive_model_from_config()` (extracted in `loaders.py` around
  line 167, used by both the HF loader and this producer so they can't drift).
- `spine_client.py` — `VLLMInMemoryLLM` / `VLLMGraphInMemoryLLM` subclass the
  HF eval clients and override ONLY `_generate_tokens`, so prompts, the SPINE
  simulator loop, and scoring are identical between backends by construction.

**The eval backend (M2):**

- `src/prism/eval/scalability_evaluation.py` — new `--backend {hf,vllm}` flag
  (env mirror `PRISM_EVAL_BACKEND`), default `hf` = zero behavior change.
- `src/prism/eval/evaluate.py` — `eval_model_single_graph` /
  `eval_model_multiple_graphs` accept a pre-built `client=`; everything else
  untouched.
- Still pending for M2 acceptance: run the same e14 checkpoint under both
  backends on real hardware and compare (threshold: ≤2 points accuracy delta,
  ≥95% per-sample agreement) — see the "important finding" below for why this
  comparison is now scientifically interesting, not just a port check.

**The RL trainer (M3):**

- `src/prism/training/train_rl.py` — hydra entrypoint
  (`python -m prism.training.train_rl --config-name=e16_rl_config`). Two init
  modes: from an SFT run dir (`trainer.rl.init_checkpoint`; the nf4 reload path
  keeps the LoRA adapter unmerged, which is exactly right for continued
  training) or from scratch (base LLM + fresh LoRA + `gnn.pe_gt_from` navigator
  carry, as in e14).
- `src/prism/training/trainers_rl.py` — `GraphGRPOTrainer(trl.GRPOTrainer)`.
  Three seams:
  1. `_generate_single_turn()` — rollouts through OUR engine (trl's stock vLLM
     integration can't know about the registered model or the mm transport).
     Ψ is built once per unique prompt and cached (`_transport_for_prompt`,
     valid because the Ψ tower is frozen in v1); the scene graph is parsed from
     the prompt text itself, the same way eval parses it.
  2. `_get_per_token_logps_and_entropies()` — the loss-side forward arms the
     SAME prompt-side Ψ on the policy model, so log-probs are computed under
     the same semantics the rollout sampled from. It is chunk-aware, and it
     REFUSES the combination of chunked logps + gradient checkpointing (the
     backward would recompute earlier chunks' attention against the last
     chunk's Ψ — silently wrong gradients).
  3. `_sync_policy_to_engine()` — after each optimizer step, every LoRA layer's
     merged weight (`W0 + scaling·B·A`) is pushed into the engine via
     `load_weights` (vLLM's own name mapping handles the fused-qkv layout).
- `src/prism/training/rewards.py` — verifiable rewards, no LLM judge:
  completions are parsed with `compact_prompt.compact_output_to_spine_json`
  and graded with `path_validator.validate_path` — i.e., the reward IS the eval
  metric. One trl reward function per component (path validity, edge validity,
  node existence, cost optimality, format, keyword) so wandb logs each raw
  component next to the weighted sum. Weights configurable via
  `trainer.rl.reward_weights`.
- `src/prism/data/rl_dataset.py` — one row per (graph, task) pair from the
  `data_gen_NNN.json` files, rendered as the exact prompt the eval harness
  would show the model (via `evaluate._fixed_get_base_prompt` +
  `compact_prompt.spine_to_compact_messages`).
- `src/prism/training/run_dir.py` — `save_run_dir()` extracted from
  `GraphSFTTrainer.save_model` so SFT and RL write the identical run-dir layout
  (`train_config.json` + `gnn_weights.pt` + adapter). This one refactor is why
  `checkpoint.load_checkpoint` — and therefore both eval backends — work on RL
  outputs with no further changes (verified by `tests/test_rl_savedir.py`).
- `src/prism/models/gnn_llm.py` — `GraphAugmentedLLM.forward` now accepts an
  externally-armed Ψ (`graphs=None` with `_pe_signal` set) for the RL loss
  path; a graphs-less forward with no armed signal still fails loudly.
  `core_graph_model` (the PEFT unwrapper) moved here from `inference.py` so
  spine-free code can import it.
- `src/prism/training/_trl_compat.py` — see finding 3 below. Must be imported
  before any `trl.trainer` module (both `trainers_rl.py` and `train_rl.py` do).
- `experiments/e16_rl_config.yaml` — the RL config; `trainer.rl.grpo` is a
  free-form dict merged last into `GRPOConfig` (same extension pattern as
  `trainer.sft`).

**v1 restrictions (all fail loudly, all deliberate):** additive architectures
only (`gt_llm` / `rpearl_gt_llm` / `rpearl_llm`); `disable_graph_token_rope`
checkpoints unsupported in the RL loss path (trl passes no position_ids; the
e14 runs trained with it false, so no practical impact); `beta` must be 0 (a
trl reference model would compute Ψ-free log-probs); the Ψ tower is frozen
(enables the per-prompt Ψ cache and LoRA-only weight sync; unfreezing is v2).

### Three findings made during this work

1. **The HF eval path has a train/decode inconsistency for the GT (additive)
   architectures.** In HF's `Gemma4TextAttention.forward`, k/v are written into
   the KV cache BEFORE the Ψ injection runs. So during cached decoding — which
   is how all e14 evals generate — every generated token attends over Ψ-FREE
   keys, while during training (no cache) answer tokens attend over Ψ-carrying
   prompt keys. The vLLM engine cannot reproduce this: its cache is written
   inside the attention op, so Ψ persists — which matches the TRAINING
   semantics. Verified empirically: vLLM output is token-for-token identical to
   a no-cache HF decode loop and diverges from the cached HF path from the
   first generated token. Consequences: (a) the parity tests use the
   training-consistent forward as the referee (with the first generated token
   additionally pinned to the HF eval path, since prefill is identical);
   (b) all e14 GT eval numbers were measured under this handicap, so the M2
   HF-vs-vLLM comparison on a real checkpoint will now MEASURE the size of
   that handicap rather than just validate the port.
2. **KV-shared layers can't carry Ψ under vLLM at all** (HF captures the shared
   k/v before Ψ is added; vLLM's single paged cache can't express
   "Ψ for the owning layer, Ψ-free for sharers"). Refused at engine build.
   The 31B config has zero KV-shared layers, so no practical impact.
3. **trl 0.27 is broken under transformers ≥5.12**: transformers'
   `_is_package_available` now always returns a `(bool, version)` tuple, and
   trl's `is_*_available()` helpers pass it through — a MISSING package yields
   the truthy `(False, None)`, so trl tries to import absent optionals (weave,
   vllm_ascend) and crashes at import. `_trl_compat.py` normalizes the helpers.
   This will also be needed in the betty RL env.

### Pending

- **M4 — plaza PoC.** Plaza is provisioned (details in section 2). Immediate
  blocker at time of writing: the test battery run ON PLAZA stopped with
  6 test-collection errors (not yet diagnosed — the same battery is green on
  the Mac at the same commit `eab36e3`; likely an environment difference such
  as a missing test dependency or a network-restricted HF tokenizer fetch,
  since the fixture tests load the gemma tokenizer). Diagnose that first.
  Then: the M2 parity run (`scalability_evaluation --backend hf` vs
  `--backend vllm` on `~/checkpoints/e14v3_n60_gt_6ep_n6dz4zlq`), then the
  N=30 PoC: `python -m prism.training.train_rl --config-name=e16_rl_config`
  with `trainer.rl.init_checkpoint` pointed at that checkpoint, 200–500 GRPO
  steps. Success criteria: shaped reward rising, `full_path_valid` above its
  step-0 value, rollout throughput recorded, mid-run checkpoint reloads and
  evaluates under both backends.
- **Open question Q2 (GPU memory):** 31B bf16 (~62 GB) does not fit one 48 GB
  A6000, so plaza rollouts must be quantized (whether vLLM's bitsandbytes mode
  composes with our registered class is untested) — OR the PoC uses the planned
  fallback: the engine serves the BASE model weights straight from plaza's HF
  cache and the trainer's first weight-sync pushes the SFT LoRA into it, which
  avoids materializing a merged copy entirely. GPU placement of engine vs
  trainer across the two A6000s is also settled empirically at PoC time.
- **M5** — the architecture summary doc (`docs/`), including the compute
  optimization plan seeded with PoC throughput numbers.
- **M6** — the betty conda env `GREP-PRISM-rl` (needs vllm + trl together;
  note the torch deviation: vllm 0.26 pairs with torch 2.11, while
  requirements.txt pins 2.10 — document or repin) and
  `scripts/e16_rl_gt.sbatch` (not yet written).

---

## 2. SPINE eval results for the e14 models

### What "SPINE evals" means here

All the e14 headline numbers were measured with SPINE **tool-calling disabled**
(`PRISM_DISABLE_SPINE_TOOLS=1`): the model writes a route directly, nothing
executes. The SPINE evals are the tools-ENABLED counterpart
(`scripts/e14_transferability_spine.sbatch`): the model may call the SPINE API
actions (goto, map_region, ...) against a live graph simulator during planning.
This batch is the first time any post-v1 checkpoint was evaluated in that mode.
One script fix was needed first: the sbatch hardcoded `--text-edge-list none`,
which would have evaluated the edges-arm baselines without the edge list they
were trained with; commit `fcbdd5b` made it overridable, and the monitor
verified from the output JSONs that each arm ran under its correct condition.

### Results (all four jobs COMPLETED 0:0; per-graph JSONs under `outputs/e14_transferability/*_spine/` on betty)

| job | model | accuracy | errors (OOM) | formatted | keyword |
|---|---|---|---|---|---|
| 7475035 | n30 edges baseline (krktnvkr) | **98.6%** (69/70) | 0 | 70/70 | 70/70 |
| 7475036 | n30 GT (8g2o6yzd) | **71.4%** (50/70) | 2 | 68/70 | 62/70 |
| 7475037 | n60 edges baseline (zqvzaab6) | **96.4%** (81/84) | 0 | 84/84 | 84/84 |
| 7475039 | n60 GT 6-epoch (n6dz4zlq) | **47.6%** (40/84) | 2 | 82/84 | 61/84 |

**IMPORTANT — audit finding (added after the first version of this summary):
these runs are effectively SINGLE-SHOT and did not test interactive planning.**
The rollout JSONs were pulled to `results/e14_spine_2026-08-10/` and inspected:
**307 of 308 samples used exactly one planner call.** The models emit
`[goto(X), answer(route)]` as a single response — the answer is written before
any tool result exists — and `goto` is a pure location update that reveals
nothing by design (only `map_region` / `explore_region` / `inspect` return
information). So the executed call returns no updates, the planning loop
terminates on the `answer` in that same response, and the graded output is one
greedy generation, exactly as in the tools-off eval. Exactly one sample in the
batch (n30 edges, `data_gen_011` idx 4) called `inspect`, got real feedback,
and produced a genuine second turn. Treat this batch as a **prompt-format
ablation** (does adding the SPINE tool documentation change accuracy?), not as
evidence about whether tool use would help the graph channel. The likely reason
the models never explore: the prompt already contains the complete scene graph,
so there is nothing to discover.

How to read this:

- **The tool-documented prompt is close to free.** Each arm lands near its own tools-off number
  (n30 edges 0.986→98.6%, n60 edges 0.952→96.4%, n30 GT 0.700→71.4%,
  n60 GT 0.560→47.6%). Tools neither rescue the graph channel nor break the
  baselines. The n60 GT drop (~8 points) is the largest movement.
- **The edges-over-GT gap persists under tools and widens with scale:**
  27 points at n30, 49 points at n60 — the same ordering and widening seen
  without tools.
- **Two GT-specific defects surfaced.** First, the GT arms lose 11%/27% of
  samples to missing answer keywords (62/70, 61/84) while both edges arms are
  perfect — a formatting/keyword failure mode separate from the crashes.
  Second, each GT arm had exactly 2 eval OOM crashes (scored incorrect,
  ~0.03 accuracy handicap each) while the edges arms had zero, and the failing
  allocation sits in a 62–70 GiB band REGARDLESS of graph size — which points
  at a fixed-shape (padded) tensor in the graph-channel attention path, not at
  prompt length. That is a concrete lead for a later fix.
- Crash-adjusted GT numbers: n30 50/68 = 73.5%, n60 40/82 = 48.8%. Neither
  changes the picture.

### What the "plaza provisioning" paragraph meant, in plain terms

Plaza is the lab workstation (2× RTX A6000 GPUs, 48 GB each) where the RL
proof-of-concept is supposed to run. It had no usable setup for this project,
so I prepared it. Concretely, a background job did the following:

1. **Repo sync:** plaza's clone of this repo (at `~/sourcecode/GREP-PRISM`,
   which pulls from betty, not GitHub) was updated to the `feature/final_experiments`
   branch at the latest commit, so all the new e16 code is there.
2. **SPINE clone:** the SPINE package (a dependency of the eval stack) was
   cloned from betty to `~/sourcecode/SPINE`, since plaza didn't have it.
3. **A new Python environment** at `~/venvs/grep-rl` (built with `uv`,
   Python 3.12) with the full RL stack installed: vLLM pinned to 0.26.0 (the
   version the engine plugin was written and tested against), trl 0.27.2,
   peft, torch-geometric, and the rest, plus this repo and SPINE installed as
   editable packages.
4. **Data and checkpoint transfer:** the N=30 training corpus
   (`data/n_30/.../split`, the RL prompt source) and the 6-epoch GT checkpoint
   (`e14v3_n60_gt_6ep_n6dz4zlq`, the intended RL starting point) were copied
   from betty to plaza.

Status of that setup: everything above completed. One follow-up fix was
required — the default torch wheel targets CUDA 13.0 and plaza's driver only
supports 12.8, so torch was swapped for the `+cu128` build; CUDA now works
(`torch.cuda.is_available()` = True, A6000 visible). Two useful discoveries
along the way: the full **gemma-4-31B-it weights (59 GB) were already cached on
plaza** (only the smaller SKUs had been stripped), so no download is needed;
and consequently **nothing needs to be deleted** from plaza's tight disk
(45 GB free) — the earlier question about freeing space is moot unless you want
headroom (candidates, untouched: `~/miniconda3` 52 GB — conda is no longer on
PATH there; `~/utils-from-mac-may-26-2026` 7.8 GB; the `PKU-ML--G1-7B` HF cache
entry 15 GB).

The last action attempted was running the 21-test battery on plaza's GPUs; it
stopped with 6 test-collection errors that are not yet diagnosed (same commit
is green on the Mac). That diagnosis is the first step when work resumes.
