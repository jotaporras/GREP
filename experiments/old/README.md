# experiments/old — legacy configs (e1–e9-gemma31b era)

Configs for completed experiment series, frozen for provenance. The e1–e8 sets (and
`e2_qwen05b_smoke.yaml`) target the removed legacy `train_v2` entrypoint and its
positional-YAML config format; `e8_new_base_models/` and `e9_gemma31b/` were the
Gemma migration sweeps that the (kept) `refactor_verify/` configs reproduce under
the current Hydra/train_v3 stack. Matching launchers live in `scripts/old/`.
