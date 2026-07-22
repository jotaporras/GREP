# e13: navigation-pretrained PE (suite4 GT/AGT) as mask Ψ — setup log

Setup by Claude (autonomous session, user away), 2026-07-21. Status when written:
smoke on cluster, real arms BLOCKED on a missing weights file (see §Blocker).

## Goal

Replace the e10/e12 edge-prediction PE pretraining with the collaborator's
navigation-pretrained suite (branch `feature/gnn_nav_integration`, notebook
`e9_gnn_navigation.ipynb`, suite4): a 72-key GraphTransformer
(`path_navigator_gt.pt`) and a Semantic GT / AGT head (`path_navigator_agt.pt`,
38 keys, composed as `NavigatorPE: Ψ = SemanticGT(PE_GT(graph))`). Two questions:

1. Does the nav-pretrained Ψ beat the edge-detector Ψ in stage-3 joint training
   (no edge text, prompt_only)? — arms `e13_gt_joint3`, `e13_agt_joint3`.
2. Does a freeze-then-joint schedule (PE-only 1 epoch with LoRA frozen, then
   joint 2 epochs) beat direct joint? — arms `e13_{gt,agt}_pe1only` →
   `e13_{gt,agt}_joint2` (stage B).

All arms: `learnable_graph_mask`, gemma-4-31B, `data.text_edge_list=none`,
`data.injection_scope=prompt_only`, LoRA warm-start from the SAME stage-1
adapter as e11/e12 (`e9_ms_stage1_sqgk4o3j/checkpoint-100`), lr 2.5e-4,
in-config 10-graph generation eval, wandb tag `e13_nav_pe` (one run per stage).
GT LR via `structural_lr_mult`: joint 0.012 (→3e-6), PE-only 0.12 (→3e-5) —
the collaborator's validated GNN LRs (3e-5 standalone, ÷10 when the LLM co-trains).

## Code (branch `e13_nav_pe`, from `fable_experiments`)

- `9924b8f` ports NavigatorPE from `feature/gnn_nav_integration`: `NavigatorPE`
  class + sparse-attention device fix (`gt.py`), navigator branches in the
  builder (`architectures.py`) and checkpoint reload (`loaders.py`),
  `load_navigator_pe_into`, `gnn.pe_gt_from`/`gnn.semantic_gt_from` config keys,
  train_v3 wiring (navigator load BEFORE `init_pe_from` carry; keys recorded in
  `train_config.json`; validation). Deviations from their branch, both
  deliberate: navigator-mode state-dict loads are strict=True (theirs
  strict=False — silent-mismatch risk), no getattr guard on `pe_model`.
- `de7927b`+ sbatch suite: `scripts/e13_nav_pe_smoke.sbatch` (4-step wiring
  smoke: RANDOMINIT stand-in GT + real AGT, 1-graph in-config eval + from-disk
  scalability_evaluation reload), `scripts/e13_nav_pe.sbatch` (array 0-3 = the
  4 arms), `scripts/e13_nav_pe_stageb.sbatch` (array 0-1, submit
  `--dependency=afterok:<arms jobid>`).
- Verified before cluster: real AGT strict-loads into our
  `SemanticGraphTransformer` (19.9M params); our GT hyperparams reproduce the
  72-key shape; NavigatorPE forward finite; 22 unit tests pass
  (test_learnable_graph_mask + test_edge_weights); all 6 arm override sets
  compose through `_validate_config`; residual diff vs collaborator branch is
  only the intended deviations. `fable_backup` branch = pre-session state.

## Blocker: path_navigator_gt.pt is missing

Only the AGT (77M, = `navigator.head`) was in the repo dir. The GT
(= `navigator.gnn`, saved by the notebook to
`outputs/e9_multistage_training/suite4/path_navigator_gt.pt`) is NOWHERE
reachable: not on plaza, not in `~/Downloads`, not on betty
(`$ALELAB_DRIVE` searched), no wandb artifact. The user's shell history shows
it lives under `/home/shared/GREP-PRISM/...` on a machine that is not plaza and
not betty — likely the collaborator's box (lb1 unreachable / siblings deny
auth). ALL arms need the GT (navigator composes AGT on top of it). A watcher
polls for the file locally + on cluster; arms fire when it lands in
`$ALELAB_DRIVE/GREP-PRISM/pretrained_pe/`. Alternative if only the full
`path_navigator.pt` is available: extract `gnn.*` keys → the GT state dict.

## Infra dead ends (so nobody repeats them)

- Local A/B smoke impossible: plaza's HF caches are weight-stripped stubs
  (12B/31B/E4B ~40K each; 14G disk free) — someone purged model weights.
  Cluster smoke replaces it (more faithful anyway).
- betty: only `login02` accepts our auth (GSSAPI/ControlMaster); login01/03 and
  all lab siblings (lc1/lc2/lb2/ld2/sa1) deny. `sbatch` needs a LOGIN shell
  (`bash -lc`) — bare ssh exec lacks SLURM_CONF (DNS SRV fatal). Login banner
  pollutes stdout of `bash -lc` (breaks naive output parsing).
- Cluster repo can't fetch GitHub in batch mode (no agent); direct
  `git push betty:sourcecode/GREP <branch>` works (can't push the checked-out
  branch, so cluster now sits on `e13_nav_pe`).
- alelab vast quota at 4.81/5.00 TB — arms use `+trainer.sft.save_total_limit=1`.

## Known latent issue (harmless for e13, flagged by review)

`NavigatorPE.forward` applies `permutation` only to the PE GT; the Semantic GT
gets the unpermuted `edge_index` (and would raise NotImplementedError if handed
a permutation). None of the e13 runs pass `--permutation-seed`, so this cannot
affect them — but a future permutation/transferability sweep on this arch must
fix it first.

## RESULTS (all 6 arms complete, 10-graph free-generation eval, n=100 tasks)

| arm | schedule | acc | halluc | vs e11 edge-detector Ψ (0.48) |
|---|---|---|---|---|
| e13_gt_pe1only | PE-only 1ep (LoRA frozen) | 0.08 | 0.42 | floor, expected pre-joint |
| e13_agt_pe1only | PE-only 1ep | 0.11 | 0.44 | floor, expected pre-joint |
| e13_gt_joint3 | direct joint 3ep | 0.44 | 0.21 | ≈ (−0.04) |
| **e13_agt_joint3** | **direct joint 3ep** | **0.58** | **0.13** | **+0.10 — best mask-family number to date** |
| e13_gt_joint2 | freeze→joint (1+2ep) | 0.47 | 0.16 | ≈ (−0.01) |
| e13_agt_joint2 | freeze→joint (1+2ep) | 0.48 | 0.17 | ≈ (0.00) |

Observations (single seed, n=100 ⇒ ±0.05 s.e.; interpretation is Javier's):
- Nav-pretrained **GT alone ≈ edge-detector GT** (0.44–0.47 vs 0.48) — consistent
  with the e13b diagnostic's "wall is channel content; both pretrainings deliver
  similar content" reading.
- The **NavigatorPE (GT+AGT) direct-joint arm is the outlier**: 0.58, lowest
  hallucination (0.128), ~2 s.e. above the 0.48 reference. The AGT's extra
  pretrained depth carries real content after joint adaptation — contra the
  session's stated prior that the null-episode AGT would add nothing.
- Schedule × Ψ interaction: freeze-then-joint helped the GT slightly
  (0.44→0.47) but the AGT lost its edge under it (0.58→0.48). Single-seed;
  needs a replication before believing the interaction.
- W&B tag `e13_nav_pe`; run dirs under `$ALELAB/GREP-PRISM/outputs/e13_nav_pe/`.

## e13c: decode-consistent injection (same session, user-approved)

Gate evidence in `notebooks/2026-07-21 e13b_decode_style_diagnostic.ipynb`:
decode-realizable wiring retains ~97% of the leak circuit's decision channel.
Build (`c07d086`): `injection_scope=decode_consistent` — collator emits asymmetric
q/k maps (answer mentions: keys full-span, queries span-final only;
`decode_style_query_map`); `MaskDecodeInjector` extends the mask to generated
mentions at decode (suffix re-matching with partial-mention deferral, per-step
bias row through the attention patch's former fall-through); scope recorded in
`train_config.json` and threaded through every eval path (mask archs only;
requires `mask_layer_scope=dense` — sliding layers crop their cache and cannot
carry the row, validated fail-loud). Parity tests (design note §4.2): per-step
decode rows == teacher-forced asymmetric bias exactly; step-wise cached decode
logits == teacher-forced logits (atol 1e-4); partial-mention deferral covered.
78 tests green. Arms (`scripts/e13c_decode_consistent.sbatch`): single-flag A/Bs
vs e11 — `e13c_rpe_decode` vs 0.48 (22lq43i6), `e13c_gmask_decode` vs 0.39
(bznw3x9p). Open question: do prompt-side + answer-side channels stack above the
~0.83/decision convergence?

**e13c RESULTS**: `e13c_rpe_decode` **0.73** (paired e11 prompt_only: 0.48;
single flag changed; hallucination 0.06 vs ~0.17-0.22 band) — the largest
single-intervention jump in the project and the best no-edge-text number to
date, 0.19-0.22 below the with-edges band (0.92-0.95 per-decision / 0.87-0.95
task). `e13c_gmask_decode` pending final (interim eval 0.61 vs paired 0.39).
The e13b teacher-forced convergence at ~0.83/decision did NOT bound joint
training: trained WITH the decode wiring, prompt-side and answer-side channels
stack. (Caveats: single seed, n=100; free-gen per-decision profile not yet
diagnosed — worth running the decision diagnostic on this checkpoint.)

Review (independent agent) caught a critical latent bug in the first injector:
prefix-ambiguous node names (`region_1` vs `region_10` under digit-split BPE)
never received their query tag (deferred-commit landed one step too late). Fixed
in `b9a46ba` with the resolve-position rule (tag at `e` instead of `e-1` for
extendable names) on BOTH the training map and the injector; parity re-proven
incl. an ambiguous-name fixture. On the CURRENT n30 val set the fix is a
verified no-op (nodes numbered 1-3 per type — corrected diagnostic reproduces
every number bit-for-bit), but it is load-bearing for the 100/1000-node
transferability graphs where double-digit names make prefix collisions routine.

## Smoke (job 7156517 → crashed on a pre-existing bug; fixed; job 7156540)

First smoke run validated the navigator load + 4 training steps on a b200, then
crashed in the post-eval logging: `EvalCallback` line 106 called `wandb.log`
unguarded while `report_to=none` leaves `wandb.run` None (guard missed in the
a12a622 "gate wandb init" refactor — every OTHER call site has it). Fixed in
`b16b282`; smoke resubmitted as 7156540.

**7156540 PASSED end-to-end**: navigator strict-load on cluster, 4 training
steps, save, in-config 1-graph generation eval, and the from-disk
`scalability_evaluation` reload (NavigatorPE rebuilt from `train_config.json`'s
`pe_gt_from`/`semantic_gt_from` keys + `gnn_weights.pt`). Smoke run dirs
deleted (6.3G freed). The e13 pipeline is fully validated; arms fire as soon as
`path_navigator_gt.pt` lands in `$ALELAB_DRIVE/GREP-PRISM/pretrained_pe/`.

Navigator-mode wiring check on a b200: 4 steps + save + 1-graph eval +
from-disk reload eval. RANDOMINIT GT stand-in — NOT science, report_to=none.
