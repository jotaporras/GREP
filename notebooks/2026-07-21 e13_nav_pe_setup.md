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
