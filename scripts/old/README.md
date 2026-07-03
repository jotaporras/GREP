# scripts/old — legacy launchers (e1–e10 era)

Frozen provenance for completed experiments; not maintained and not expected to run.

- The e1–e8 runners target the legacy `prism.training.train_v2` entrypoint, which has
  been **removed from src** (see git history if you need it). The current entrypoint
  is `python -m prism.training.train_v3 --config-name=<experiment>` (Hydra).
- `refactor_verify.sbatch` / `dev_e10_*` / `verify_loss_masking.*` are train_v3-era
  but their experiments are concluded (results recorded in `notebooks/`).
- `*composite*` / `smoke_qk_injection.py` target the removed `composite_graph_gt`
  architecture (e7, Llama-era).
- Matching legacy experiment configs live in `experiments/old/`.

Current launchers stay in `scripts/`: the e9 multistage chain, e11/e12 injection-scope
experiments, and the injection diagnostics.
