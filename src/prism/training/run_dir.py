"""The run-dir save contract shared by SFT and RL trainers.

Kept spine-free (os/json/torch only) so the RL stack and tests can import it
without dragging in the eval/SPINE dependency chain that ``trainers.py``
carries via its callbacks.
"""
import json
import os

import torch


# Shared run-metadata keys stored at the TOP LEVEL of train_config.json; every
# other gnn_config entry (the arch hyperparameters) nests under "gnn". Loaders
# flatten this back (and still read legacy flat gnn_config.json checkpoints).
# spine_tools / icl_examples belong here (not under "gnn"): they are PROMPT policy,
# and eval.checkpoint.resolve_prompt_policy reads them at the top level. Nesting them
# made that resolver always return the "predates the knob" fallback ("none", 0), so an
# ICL/tool-trained graph checkpoint was silently re-evaluated zero-shot and tool-free.
_RUN_META_KEYS = ("architecture", "base_model", "text_edge_list", "injection_scope",
                  "edge_weights", "spine_tools", "icl_examples")


def save_run_dir(model, gnn_config: dict, output_dir: str) -> None:
    """The run-dir contract every trainer writes: ``train_config.json`` +
    per-arch ``gnn_weights.pt``.

    Extracted from ``GraphSFTTrainer.save_model`` so the RL trainer
    (``trainers_rl.GraphGRPOTrainer``) saves the IDENTICAL layout and
    ``checkpoint.load_checkpoint`` works on RL outputs unchanged. The LoRA
    adapter is NOT saved here — each trainer's ``save_model`` owns that (it
    goes through the HF ``Trainer`` machinery).
    """
    os.makedirs(output_dir, exist_ok=True)
    run_config = {k: gnn_config[k] for k in _RUN_META_KEYS if k in gnn_config}
    run_config["gnn"] = {k: v for k, v in gnn_config.items()
                         if k not in _RUN_META_KEYS}
    with open(os.path.join(output_dir, "train_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)
    if gnn_config.get("architecture") == "graph_mask_llm":
        # Parameter-free: mask rebuilt from config; train_config.json + LoRA adapter suffice.
        pass
    elif gnn_config.get("architecture") == "learnable_graph_mask":
        # Save the standalone GraphTransformer (Psi producer); the mask + adjacency
        # rebuild from gnn_config and the LoRA adapter is saved by the trainer.
        # mask_composite (MagCompGraphLLM) additionally carries `beta`, the ONE scalar
        # scaling beta * C_tok — reloading without it evaluates the base LLM at 0.
        torch.save(
            {"pe_model": model.pe_model.state_dict(),
             **({"mask_beta": model.beta.data} if hasattr(model, "beta") else {})},
            os.path.join(output_dir, "gnn_weights.pt"),
        )
    elif gnn_config.get("architecture") == "wire_llm":
        # WIRE: the Ψ producer, the angle gate, and the frequency store. Which store
        # is populated depends on gnn.wire_vanilla — the learnable ω table
        # (wire_vanilla=true, the paper's form) or the frozen ε directions plus the
        # learned per-layer σ (the expectation arm). BOTH key sets are always
        # written (the unused one is an empty dict) so the checkpoint key set does
        # not depend on the mode: a key present in one mode and absent in the other
        # is exactly the silent-corruption case loaders.py guards against.
        # ε/ω are SAVED rather than reconstructed from wire_omega_seed: regenerating
        # them would make the checkpoint depend on torch RNG determinism across
        # versions/devices, which is exactly the silent-drift failure mode. (The seed
        # is still recorded in train_config.json.)
        torch.save(
            {
                "pe_model": model.pe_model.state_dict(),
                "pe_gain": model.pe_gain.data,
                "wire_eps": model._wire_eps.state_dict(),
                "wire_sigma": model._wire_sigma.state_dict(),
                "wire_omega": model._wire_omega.state_dict(),
            },
            os.path.join(output_dir, "gnn_weights.pt"),
        )
    elif gnn_config.get("architecture") == "rpearl_gt_llm":
        # Full GT: save the whole GraphTransformer (includes R-PEARL inside) + projection head.
        torch.save(
            {
                "gt_model": model.pe_model.state_dict(),
                "pe_proj": model.pe_proj.state_dict(),
                "pe_gain": model.pe_gain.data,
                **(
                    {"pe_norm": model.pe_norm.state_dict()}
                    if model.pe_norm is not None
                    else {}
                ),
            },
            os.path.join(output_dir, "gnn_weights.pt"),
        )
        # Also save the inner R-PEARL separately for analysis / reuse.
        torch.save(
            {
                "rpearl": model.pe_model.pe_model.state_dict(),
            },
            os.path.join(output_dir, "rpearl_weights.pt"),
        )
    else:
        torch.save(
            {
                "pe_model": model.pe_model.state_dict(),
                "pe_proj": model.pe_proj.state_dict(),
                "pe_gain": model.pe_gain.data,
                **(
                    {"pe_norm": model.pe_norm.state_dict()}
                    if model.pe_norm is not None
                    else {}
                ),
            },
            os.path.join(output_dir, "gnn_weights.pt"),
        )
