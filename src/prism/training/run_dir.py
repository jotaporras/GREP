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
        # Post-fusion (e17): the residual-injection modules ride alongside; the
        # loader fails loud if post_fusion=true is recorded without them.
        weights = {"pe_model": model.pe_model.state_dict()}
        if getattr(model, "_post_fusion", False):
            weights["pf_proj"] = model.pf_proj.state_dict()
            weights["pf_norm"] = model.pf_norm.state_dict()
            weights["pf_gain"] = model.pf_gain.data
        # e17 candidates D/E/C ride alongside the tower the same way; the
        # loader fails loud if a recorded flag's weights are absent.
        if getattr(model, "_graph_lora", False):
            weights["glora_gen"] = model.glora_gen.state_dict()
            weights["glora_B"] = model.glora_B.state_dict()
        if getattr(model, "_pointer_fusion", False):
            weights["ptr_q"] = model.ptr_q.state_dict()
            weights["ptr_gate"] = model.ptr_gate.state_dict()
            weights["ptr_gain"] = model.ptr_gain.data
            weights["ptr_scale"] = model.ptr_scale.data
        if getattr(model, "_cross_fusion", False):
            for name in ("xf_ln", "xf_q", "xf_k", "xf_v", "xf_o"):
                weights[name] = getattr(model, name).state_dict()
            weights["xf_gain"] = model.xf_gain.data
        # e18 node-identity pathways (A decision gating, B structural keys,
        # binding head) — same fail-loud contract in loaders.
        if getattr(model, "_decision_gating", False):
            weights["decision_gain"] = model.decision_gain.data
        if getattr(model, "_struct_keys", False):
            weights["sk_k"] = model.sk_k.state_dict()
            weights["sk_q"] = model.sk_q.state_dict()
            weights["sk_gain"] = model.sk_gain.data
        if getattr(model, "_binding_head", False):
            weights["bind_proj"] = model.bind_proj.state_dict()
        if getattr(model, "_soft_edges", False):
            weights["se_mlp"] = model.se_mlp.state_dict()
        torch.save(weights, os.path.join(output_dir, "gnn_weights.pt"))
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
