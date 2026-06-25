# e9_hydra_training — Unified Hydra Config Tree

## Overview

This directory is a [Hydra](https://hydra.cc) config tree that replaces the per-run
monolithic YAMLs of three earlier experiment series — `e7_architecture_improvements/`
(3 files), `e8_new_base_models/` (28 files), and `e9_multistage_training/` (5 files).
Those 36 configs each copied the same ~75 keys with only a handful of axes actually
varying (base model, corpus, architecture, GT depth/width, freeze regime, edges on/off).

Here that surface is factored into reusable, composable **config groups**. You no longer
write a whole YAML per run; instead you compose a run from group options on the command
line. The root `config.yaml` declares a `defaults:` list that composes a single, sensible
**baseline**, and every individual run is expressed as a small set of overrides on top of
that baseline.

> **Status — config only.** This tree is the configuration surface. Wiring `TrainConfig`
> (`src/prism/training/train_v3.py`) to read it through a `@hydra.main` entrypoint is a
> separate, deferred step. The commands below show the *intended* interface once that
> entrypoint lands; until then the legacy launcher is
> `python -m prism.training.train_v3 <yaml> --key value`.

## Core Concepts

### Flat-global packaging
Every option file begins with `# @package _global_`. In Hydra this means the file's keys
are merged into the **root** of the config, not nested under their group name. The result
is one flat namespace that mirrors the flat `TrainConfig` dataclass — so a group file can
set or override *any* field, and CLI overrides are always written as bare `key=value`
(never `group.key=value`).

### The defaults list and conflict resolution
`config.yaml` lists exactly one default option per group. Those defaults compose the
**merged, most-recent-wins baseline** (e9 > e8 > e7): Gemma-4-12B, the revised corpus,
bf16 (`bit4: false`), multi-graph eval, assistant-only loss, and `architecture:
rpearl_llm`. Cross-series conflicts are resolved simply by *which option each group
defaults to* — the e9 values are the defaults; e7/e8 values are reachable only by
explicitly selecting their options (e.g. `overview=e7`, `data=legacy`).

The defaults list is **ordered**, and later entries win on key collisions. The order is:

```
overview → data → training → lora → multistage → llm → structural
→ r_pearl → gt → graph_mask → composite_graphs → eval → device → _self_
```

Two consequences worth internalizing:
- **`composite_graphs` is composed after `structural`/`r_pearl`/`gt`.** This is deliberate:
  the named composite designs (`centered`, `c_bias`, `c_per_layer`) carry their own
  structural overrides (e.g. `d_model: 4096`, `pe_readout: second_moment`) and must win.
  The default `composite_graphs/default.yaml` is **inert** — it sets only
  composite-specific keys and never structural ones, so it cannot clobber the baseline
  even though it is always composed.
- **`_self_` is last**, so the keys written directly in `config.yaml` (notably
  `architecture`) override the groups, and CLI overrides override everything.

### The architecture switch
`architecture` is the master key. It selects which model wrapper is built and therefore
which structural groups are actually read at runtime:

| `architecture` | Reads | Description |
|---|---|---|
| `llm` | — | Plain LoRA-SFT baseline, no graph module |
| `rpearl_llm` | `r_pearl`, `structural` | R-PEARL probe PE injected into the LLM |
| `rpearl_gt_llm` | `r_pearl`, `gt`, `structural` | R-PEARL PE refined by a Graph Transformer |
| `gt_llm` | `gt`, `structural` | Graph Transformer over word-embedding node features (no probes; **requires** `pe_node_features=word_embeddings`) |
| `graph_mask_llm` | `graph_mask` | Parameter-free structural attention mask |
| `composite_graph_gt` | `composite_graphs`, `r_pearl`, `gt`, `structural` | Composite-graph construction + C-in-attention injection |

Groups not read by the chosen architecture still compose, but their keys are simply
unused — harmless, because `TrainConfig` defines every field with a default.

## Config Groups

Thirteen groups. **Bold** marks the option selected by `config.yaml`'s defaults list.

| Group | Options | Responsibility |
|---|---|---|
| `overview` | **e9**, e8, e7 | Run identity: `name`, wandb tag, `checkpoint_dir`, `debug`, viz flags |
| `data` | **revised**, legacy | Train/val corpus, `max_seq_length`, `text_edge_list` |
| `training` | **default**, stage2_pe, stage3_joint, zeroshot | Precision (`bit4`), optimizer, LR, epochs, `no_train` |
| `lora` | **default** | Adapter shape: `r`, `lora_alpha`, `lora_dropout`, `target_modules` |
| `multistage` | **none**, stage1_sft, stage2_pe, stage3_joint | Freeze regime, `init_lora_from`/`init_pe_from`, `loss_target` |
| `llm` | **gemma4_12b**, gemma4_31b, llama31_8b | `base_model` + RoPE controls |
| `structural` | **default** | Shared `d_model`/`dropout`/`eps`/`use_layer_norm` + PE-injection gate |
| `r_pearl` | **default** | Probe encoder: hidden/layers/`num_samples`/`k_pe`/readout/node features |
| `gt` | **default**, L3, L4, L5, L5_d2048, L5_d3072, L5_d4096 | `gt_num_layers`/`gt_heads`/`k_gt` (+ `d_model` on the width sweep) |
| `graph_mask` | **default** | `mask_k_hops`/`mask_symmetrize`/`mask_use_edges` |
| `composite_graphs` | **default** (inert), centered, c_bias, c_per_layer | Composite graph build + C-injection design |
| `eval` | **multi_graph**, single_graph | Eval data, cadence, ICL |
| `device` | **auto** (-1), cuda0 (0) | GPU placement |

### Axes that are plain key overrides (not their own group)
Some axes vary independently of any group and are toggled by overriding a single field:

- **Edges in the prompt:** `text_edge_list=present|none` (the `*_no_edges` axis).
- **4-bit quantization:** `bit4=true` (e7/e8 trained 4-bit; e9 default is bf16).
- **Frozen-base PE ablation:** `freeze_llm=true pe_gain_init=1.0` (e8 frozen_minimal/frozen_pe1).
- **Semantic node features:** `pe_node_features=word_embeddings` (required by `gt_llm`; also the rpearl_gt "semantic" run).
- **No-RoPE on graph tokens:** `disable_graph_token_rope=true` (the norope_graph run).
- **Carried-weight init:** `init_lora_from=<dir>` / `init_pe_from=<dir>` (multistage; normally filled by the driver script).

## Command-Line Interface

The intended entrypoint is `python -m prism.training.train_v3` with Hydra overrides:

- **Swap a group option:** `group=option` (e.g. `gt=L5_d4096`).
- **Override a flat field:** `key=value` (e.g. `text_edge_list=none`, `bit4=true`).
- **Sweep (multirun):** add `-m` and pass comma-separated values
  (e.g. `-m gt=L3,L4,L5`) to launch one job per combination.
- **Inspect without training:** `--cfg job` prints the fully composed flat config;
  `--help` lists every group and its available options.

> **Null-field note.** Fields like `init_lora_from` default to `null`, so they override
> cleanly with the plain form `init_lora_from=<dir>`. If a future change makes any of them
> Hydra-mandatory (`???`) instead of `null`, switch to the append form `+init_lora_from=<dir>`.

## Example Commands

### A. Default config per architecture
Each line is the pure baseline composition (e9 merged defaults) with only the architecture
switched — the fastest way to see what a given model wrapper does out of the box.

```bash
python -m prism.training.train_v3 architecture=llm
python -m prism.training.train_v3 architecture=rpearl_llm            # == config.yaml default
python -m prism.training.train_v3 architecture=rpearl_gt_llm
python -m prism.training.train_v3 architecture=gt_llm pe_node_features=word_embeddings   # gt_llm requires semantic features
python -m prism.training.train_v3 architecture=graph_mask_llm
python -m prism.training.train_v3 architecture=composite_graph_gt composite_graphs=centered
```

### B. Settings changed per architecture
Each example overrides several groups and keys at once. Collectively they exercise the
e7/e8/e9 bookkeeping, both corpora, 4-bit, the staged regimes, alternate base models, and
the structural ablation axes.

```bash
# llm → zero-shot eval of base Gemma; e8 bookkeeping, 4-bit, single-graph eval, no edges
python -m prism.training.train_v3 \
  architecture=llm training=zeroshot overview=e8 eval=single_graph bit4=true text_edge_list=none

# rpearl_llm → e9 Stage-2 PE-only (frozen carried adapter; driver normally fills init_lora_from)
python -m prism.training.train_v3 \
  architecture=rpearl_llm multistage=stage2_pe training=stage2_pe \
  init_lora_from=outputs/e9_multistage_training/e9_ms_s1_ab12cd34

# rpearl_gt_llm → Gemma-31B, semantic node features, no-RoPE on graph tokens, e8, 4-bit
python -m prism.training.train_v3 \
  architecture=rpearl_gt_llm llm=gemma4_31b overview=e8 eval=single_graph bit4=true \
  pe_node_features=word_embeddings disable_graph_token_rope=true

# gt_llm → width-sweep point L5/d3072, warm gate, no edges, e8, 4-bit
python -m prism.training.train_v3 \
  architecture=gt_llm gt=L5_d3072 overview=e8 eval=single_graph bit4=true \
  pe_node_features=word_embeddings pe_gain_init=0.5 text_edge_list=none

# graph_mask_llm → floor ablation: 2-hop mask, self-loops only, frozen base
python -m prism.training.train_v3 \
  architecture=graph_mask_llm overview=e8 eval=single_graph bit4=true \
  mask_k_hops=2 mask_use_edges=false freeze_llm=true

# composite_graph_gt → e7 c_per_layer design, legacy corpus, Llama-3.1-8B, pinned to cuda0
python -m prism.training.train_v3 \
  architecture=composite_graph_gt composite_graphs=c_per_layer data=legacy llm=llama31_8b \
  overview=e7 eval=single_graph bit4=true device=cuda0 text_edge_list=none
```

### C. The e9 multistage chain
The three-stage curriculum is three separate invocations; weights carry forward via
`init_lora_from`/`init_pe_from` (resolved checkpoint dirs — normally supplied by
`scripts/e9_multistage_training.sh`). Stage 2's command appears in section B above.

```bash
# Stage 1 — SFT the LoRA, PE frozen and gated off, edges in text
python -m prism.training.train_v3 architecture=rpearl_llm multistage=stage1_sft training=default

# Stage 3 — joint PE + LoRA, edges removed, init both from the Stage-2 checkpoint
python -m prism.training.train_v3 \
  architecture=rpearl_llm multistage=stage3_joint training=stage3_joint text_edge_list=none \
  init_lora_from=outputs/e9_multistage_training/e9_ms_s2_ef56ab78 \
  init_pe_from=outputs/e9_multistage_training/e9_ms_s2_ef56ab78
```

### D. Sweeps (Hydra multirun)
`-m` launches one job per value combination — ideal for the GT depth/width sweep and the
composite-design comparison, and for small grids.

```bash
# GT depth/width sweep — one job per option
python -m prism.training.train_v3 -m \
  architecture=gt_llm pe_node_features=word_embeddings overview=e8 bit4=true eval=single_graph \
  text_edge_list=none gt=L3,L4,L5,L5_d2048,L5_d4096

# Composite-design comparison
python -m prism.training.train_v3 -m \
  architecture=composite_graph_gt data=legacy llm=llama31_8b overview=e7 bit4=true \
  eval=single_graph text_edge_list=none composite_graphs=centered,c_bias,c_per_layer

# Cross-product: 3 base models × edges on/off = 6 runs
python -m prism.training.train_v3 -m \
  architecture=rpearl_llm llm=gemma4_12b,gemma4_31b,llama31_8b text_edge_list=present,none
```

### E. Inspect without training
```bash
# Print the fully composed flat config (sanity-check overrides before launching)
python -m prism.training.train_v3 architecture=gt_llm gt=L5_d4096 pe_node_features=word_embeddings --cfg job

# List every group and its available options
python -m prism.training.train_v3 --help
```

## Mapping Back to the Original Runs

| Original run | Reconstruction |
|---|---|
| e9 baseline (edges) | `architecture=llm` |
| e9 baseline (no edges) | `architecture=llm text_edge_list=none` |
| e9 Stage 1 / 2 / 3 | section C + B above |
| e8 plain LLM | `architecture=llm overview=e8 bit4=true eval=single_graph` |
| e8 zero-shot | `architecture=llm training=zeroshot overview=e8 bit4=true eval=single_graph` |
| e8 rpearl (no edges) | `architecture=rpearl_llm overview=e8 bit4=true eval=single_graph text_edge_list=none` |
| e8 rpearl+GT | `architecture=rpearl_gt_llm overview=e8 bit4=true eval=single_graph` |
| e8 rpearl+GT, Gemma-31B | add `llm=gemma4_31b` |
| e8 rpearl+GT, semantic | add `pe_node_features=word_embeddings` |
| e8 rpearl+GT, norope_graph | add `disable_graph_token_rope=true` |
| e8 rpearl+GT, frozen | add `freeze_llm=true pe_gain_init=1.0` |
| e8 GT depth/width sweep | `architecture=gt_llm overview=e8 bit4=true eval=single_graph pe_node_features=word_embeddings text_edge_list=none gt=<L3|L4|L5|L5_d2048|L5_d3072|L5_d4096>` |
| e8 graph mask | `architecture=graph_mask_llm overview=e8 bit4=true eval=single_graph` |
| e7 centered / c_bias / c_per_layer | `architecture=composite_graph_gt overview=e7 data=legacy llm=llama31_8b bit4=true eval=single_graph text_edge_list=none composite_graphs=<design>` (c_bias adds `device=cuda0 max_seq_length=2048`) |

## Directory Layout

```
e9_hydra_training/
├── config.yaml                 # defaults list + architecture switch
├── README.md                   # this file
├── overview/    { e9, e8, e7 }
├── data/        { revised, legacy }
├── training/    { default, stage2_pe, stage3_joint, zeroshot }
├── lora/        { default }
├── multistage/  { none, stage1_sft, stage2_pe, stage3_joint }
├── llm/         { gemma4_12b, gemma4_31b, llama31_8b }
├── structural/  { default }
├── r_pearl/     { default }
├── gt/          { default, L3, L4, L5, L5_d2048, L5_d3072, L5_d4096 }
├── graph_mask/  { default }
├── composite_graphs/ { default, centered, c_bias, c_per_layer }
├── eval/        { multi_graph, single_graph }
└── device/      { auto, cuda0 }
```
