"""e21 activation/attention probe: WHERE does the v2c model read the graph channel?

For a trained learnable_graph_mask checkpoint (the e21 v2c hop_depth cell), this
script (1) re-runs the REAL eval loop on a hand-picked subset of the frozen n60
test questions with a capturing client — so prompts, injection wiring, and greedy
decoding are byte-identical to the reported eval — and (2) replays each captured
prompt+generation teacher-forced under channel-ablation conditions, recording per
layer/position hidden states, attention rows, and next-token probability shifts.

Conditions (decode_consistent wiring, exactly what training and decode saw):
  on           query map = decode_style_query_map, key map = full map,
               decision map = decision_query_map; struct bias + post-fusion armed.
  no_mask      post-fusion only (struct bias disarmed).
  no_pf        struct bias only (post-fusion disarmed).
  no_decision  struct bias without the e18-A decision-gating rows; post-fusion on.
  off          nothing armed, natural RoPE — the plain-LLM fallback path.

Per sample the probe stores (out dir, one .npz + .meta.json per sample):
  - logp of the realized token at every position, per condition;
  - entropy over the answer region, per condition;
  - top-10 candidates + per-node first-token logp at answer node-mention
    positions (probability-shift statistics on node-dependent predictions);
  - per-layer per-position ||h_cond - h_off||, cos(h_cond, h_off), ||h||;
  - raw hidden vectors at decision-adjacent positions (on/off only);
  - head-mean attention rows for decision-adjacent queries at every layer
    (on/no_mask/off) and per-head rows at 4 featured layers (on/off).

Usage (betty, one GPU; see scripts/e21_activation_probe.sbatch):
    python scripts/e21_activation_probe.py \
        --checkpoint $PROJ/outputs/e20_path_only_pred/e20_oracle_v2c_hop_depth_route_only_4ghta3jm \
        --graphs $PROJ/data/n_60_vllm_v3/gen/nav_n60_gemma_data/split/test_graphs \
        --samples data_gen_052:1,data_gen_052:7,data_gen_052:10,data_gen_052:11,... \
        --out $PROJ/outputs/e21_probe/v2c_final
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import torch

from prism.data import data as data_mod
from prism.eval import checkpoint as ckpt_mod
from prism.eval import evaluate
from prism.eval import injection_diag
from prism.models import inference
from prism.models import gnn_llm
from prism.models.gnn_llm import (
    build_injection_map,
    core_graph_model,
    decision_query_map,
    decode_style_query_map,
    find_last_graph_scope,
    node_token_variants,
    resolve_mask_active_flags,
)

CONDITIONS = ("on", "no_mask", "no_pf", "no_decision", "off")
ATTN_CONDITIONS = ("on", "no_mask", "off")      # head-mean rows stored for these
HSEL_CONDITIONS = ("on", "off")                 # raw hidden vectors stored for these


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True,
                   help="Trained run dir (train_config.json present).")
    p.add_argument("--graphs", required=True,
                   help="Frozen eval graph dir (test_graphs) — same as eval.data.")
    p.add_argument("--samples", required=True,
                   help="Comma list of <graph_stem>:<idx> to probe "
                        "(idx = position in the graph's task list, as in eval logs).")
    p.add_argument("--out", required=True, help="Output directory.")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--four-bit", action="store_true")
    p.add_argument("--seed", type=int, default=1234,
                   help="Re-seeded before every condition's Ψ build so the R-PEARL "
                        "probes are identical across conditions of one example.")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    return p.parse_args()


class CapturingClient(inference.GraphAugmentedInMemoryLLM):
    """The stock eval client, additionally recording (prompt, generation, graph)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.captures = []

    def _generate_tokens(self, input_ids, attention_mask, pyg_graphs, max_new_tokens):
        out = super()._generate_tokens(input_ids, attention_mask, pyg_graphs,
                                       max_new_tokens)
        self.captures.append({
            "prompt_ids": input_ids[0].tolist(),
            "gen_ids": out[0].tolist(),
            "pyg_graph": pyg_graphs[-1] if pyg_graphs else None,
        })
        return out


class AttnCollector:
    """Forward hooks on every self_attn capturing (reduced) attention rows.

    Eager attention materializes per-head probabilities and the attention module
    returns them as output[1] regardless of ``output_attentions`` — the hook
    slices the selected query rows and immediately moves them to CPU f16.
    """

    def __init__(self, layers, featured: set[int]):
        self.featured = featured
        self.handles = []
        self.qsel: list[int] | None = None          # head-mean rows for these
        self.qsel_heads: list[int] | None = None    # per-head rows (featured layers)
        self.active = False
        self.keep_heads = False
        self.rows: dict[int, torch.Tensor] = {}
        self.head_rows: dict[int, torch.Tensor] = {}
        for li, layer in enumerate(layers):
            self.handles.append(layer.self_attn.register_forward_hook(
                self._make_hook(li)))

    def _make_hook(self, li):
        def hook(module, args, output):
            if not self.active or self.qsel is None:
                return
            w = None
            if isinstance(output, tuple) and len(output) >= 2:
                w = output[1]
            if w is None or not torch.is_tensor(w) or w.dim() != 4:
                return
            rows = w[0, :, self.qsel, :]                       # [H, nq, S]
            self.rows[li] = rows.mean(dim=0).to("cpu", torch.float16)
            if self.keep_heads and li in self.featured and self.qsel_heads:
                self.head_rows[li] = w[0, :, self.qsel_heads, :].to(
                    "cpu", torch.float16)
        return hook

    def start(self, qsel, qsel_heads, keep_heads):
        self.qsel, self.qsel_heads = qsel, qsel_heads
        self.keep_heads = keep_heads
        self.rows, self.head_rows = {}, {}
        self.active = True

    def stop(self):
        self.active = False

    def remove(self):
        for h in self.handles:
            h.remove()


def parse_sample_spec(spec: str) -> list[tuple[str, int]]:
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        stem, idx = tok.rsplit(":", 1)
        out.append((stem, int(idx)))
    return out


def chunked_logprob_stats(logits, seq_ids, answer_start):
    """Per-position realized-token logp (full seq) + entropy over the answer region.

    Target frame: logits[p-1] predicts token p (grade_positions convention).
    Returns (logp_realized [S-1] f32, entropy_ans [S-answer_start] f32).
    """
    S = seq_ids.shape[1]
    logp_realized = np.zeros(S - 1, dtype=np.float32)
    ent = np.zeros(S - answer_start, dtype=np.float32)
    for lo in range(0, S - 1, 256):
        hi = min(lo + 256, S - 1)
        rows = logits[0, lo:hi].float()
        lp = torch.log_softmax(rows, dim=-1)
        tgt = seq_ids[0, lo + 1:hi + 1]
        logp_realized[lo:hi] = lp[torch.arange(hi - lo, device=lp.device), tgt] \
            .cpu().numpy()
        # entropy of the prediction AT position j+1 lands at index j+1-answer_start
        ent_chunk = -(lp.exp() * lp).sum(dim=-1).cpu().numpy()
        for j in range(lo, hi):
            if j + 1 >= answer_start:
                ent[j + 1 - answer_start] = ent_chunk[j - lo]
    return logp_realized, ent


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    wanted = parse_sample_spec(args.samples)

    is_gnn = ckpt_mod.is_gnn_checkpoint(args.checkpoint)
    if not is_gnn:
        raise SystemExit("probe requires a graph-augmented checkpoint")
    arch = ckpt_mod.load_gnn_config(args.checkpoint)["architecture"]
    text_edge_list = ckpt_mod.resolve_text_edge_list(args.checkpoint, is_gnn, None)
    edge_weights = ckpt_mod.resolve_edge_weights(args.checkpoint)
    injection_scope = ckpt_mod.resolve_injection_scope(args.checkpoint)
    response_format = ckpt_mod.resolve_response_format(args.checkpoint)
    print(f"[probe] arch={arch} text_edge_list={text_edge_list} "
          f"edge_weights={edge_weights} scope={injection_scope} fmt={response_format}")
    if arch != "learnable_graph_mask":
        raise SystemExit(f"probe is wired for learnable_graph_mask, got {arch!r}")
    if injection_scope != "decode_consistent":
        raise SystemExit(f"probe assumes decode_consistent wiring, got {injection_scope!r}")

    model, tokenizer, _ = ckpt_mod.load_checkpoint(
        args.checkpoint, four_bit=args.four_bit, device=args.device)
    model.eval()
    graph_model = core_graph_model(model)
    device = next(model.parameters()).device
    identity_rope = bool(getattr(graph_model, "_disable_graph_token_rope", False))

    layers = graph_model._decoder_layers()
    L = len(layers)
    mask_flags = resolve_mask_active_flags(layers, graph_model._mask_layer_scope)
    pf_scope = getattr(graph_model, "_pf_layer_scope", None)
    pf_flags = resolve_mask_active_flags(layers, pf_scope) if pf_scope else [False] * L
    dense_idx = [i for i, f in enumerate(mask_flags) if f]
    pf_idx = [i for i, f in enumerate(pf_flags) if f]
    featured = sorted({dense_idx[0], dense_idx[len(dense_idx) // 2],
                       (pf_idx[0] if pf_idx else L - 2), L - 1})
    print(f"[probe] {L} layers; mask-active {len(dense_idx)}, pf-active {len(pf_idx)}; "
          f"featured layers {featured}; identity_rope={identity_rope}")

    # ---- stage 1: capture prompts + generations through the real eval loop ----
    samples_by_graph, _files = data_mod.load_samples_by_graph(args.graphs)
    client = CapturingClient(
        model=model, tokenizer=tokenizer,
        include_edges=(text_edge_list == "present"), include_tools=False,
        icl_examples=0, permutation=None, edge_weights=edge_weights,
        injection_scope=injection_scope, response_format=response_format)
    client.max_new_tokens = args.max_new_tokens

    probe_records = []       # (graph_stem, orig_idx, sample_result, capture)
    for stem in sorted({s for s, _ in wanted}):
        idxs = [i for s, i in wanted if s == stem]
        all_samples = samples_by_graph[stem]
        sel = [all_samples[i] for i in idxs]
        n_before = len(client.captures)
        _acc, sample_results = evaluate.eval_model_single_graph(
            model, tokenizer, sel,
            include_edge_list=(text_edge_list == "present"),
            use_icl=False, permutation=None, edge_weights=edge_weights,
            injection_scope=injection_scope, response_format=response_format,
            client=client)
        got = client.captures[n_before:]
        if len(got) != len(sel):
            raise SystemExit(
                f"{stem}: {len(got)} generate calls for {len(sel)} samples — "
                "cannot align captures to samples (multi-iteration planning?)")
        for orig_idx, res, cap in zip(idxs, sample_results, got):
            probe_records.append((stem, orig_idx, res, cap))
            print(f"[probe] captured {stem}#{orig_idx} correct={res['correct']} "
                  f"prompt={len(cap['prompt_ids'])} gen={len(cap['gen_ids'])}")

    # ---- stage 2: teacher-forced condition forwards with hooks ----
    collector = AttnCollector(layers, set(featured))
    summary = {"checkpoint": args.checkpoint, "seed": args.seed,
               "conditions": list(CONDITIONS), "featured_layers": featured,
               "n_layers": L, "mask_active": mask_flags, "pf_active": pf_flags,
               "identity_rope": identity_rope, "samples": []}

    for stem, orig_idx, res, cap in probe_records:
        tag = f"{stem}_q{orig_idx:02d}"
        g = cap["pyg_graph"]
        prompt_len = len(cap["prompt_ids"])
        seq_list = cap["prompt_ids"] + cap["gen_ids"]
        seq_len = len(seq_list)
        seq_ids = torch.tensor([seq_list], dtype=torch.long, device=device)
        attn_mask = torch.ones_like(seq_ids)
        answer_start = prompt_len

        node_token_seqs = node_token_variants(g.node_names, tokenizer)
        scope_start = find_last_graph_scope(seq_list, tokenizer)
        full_map = build_injection_map(seq_list, node_token_seqs,
                                       scope_start=scope_start)
        query_map = decode_style_query_map(full_map, answer_start, seq_list,
                                           node_token_seqs)
        dec_map = decision_query_map(query_map, answer_start, seq_len)
        pos_sets = injection_diag.partition_answer_node_positions(
            full_map, answer_start)
        node_pos_all = sorted({p for spans in full_map.values()
                               for a, b in spans for p in range(a, b)})
        scene_node_pos = [p for p in node_pos_all if scope_start <= p < answer_start]
        ans_node_pos = pos_sets["all_answer_nodes"]
        decision_pos = pos_sets["decision"]
        nonnode_ans = [p for p in range(answer_start, seq_len)
                       if p not in set(ans_node_pos)]
        nonnode_pick = nonnode_ans[:: max(1, len(nonnode_ans) // 4)][:4]

        # Query rows probed for attention/hidden capture: the position CHOOSING each
        # decision token (p-1), the decision token itself, the last prompt token,
        # and a few non-node answer tokens.
        qsel_heads = sorted({p - 1 for p in decision_pos if p >= 1})
        qsel = sorted(set(qsel_heads) | set(decision_pos) | {prompt_len - 1}
                      | set(nonnode_pick))

        # First-token ids per node (all tokenization variants, padded with -1).
        max_var = max(len(v) for v in node_token_seqs)
        node_first = np.full((len(node_token_seqs), max_var), -1, dtype=np.int64)
        for n, variants in enumerate(node_token_seqs):
            for vi, v in enumerate(variants):
                node_first[n, vi] = v[0]

        arrays: dict[str, np.ndarray] = {"node_first_tokens": node_first}
        hs_by_cond: dict[str, list[torch.Tensor]] = {}
        cond_summ = {}

        for cond in CONDITIONS:
            torch.manual_seed(args.seed)
            try:
                with torch.no_grad():
                    if cond == "off":
                        graph_model._struct_bias = None
                        graph_model._pf_signal = None
                        pos_kwargs = {}
                    else:
                        dm = None if cond == "no_decision" else [dec_map]
                        # Re-seed before EACH Ψ build: both consume RNG (R-PEARL
                        # probes), and the pf signal must be bitwise identical
                        # across every condition that arms it.
                        torch.manual_seed(args.seed)
                        bias = None if cond == "no_mask" else \
                            graph_model.build_structural_mask(
                                seq_len, [g], [query_map], device,
                                key_injection_maps=[full_map], decision_maps=dm)
                        torch.manual_seed(args.seed + 1)
                        pf = None if cond == "no_pf" else \
                            graph_model.build_pf_signal(
                                seq_len, [g], [query_map], device)
                        graph_model._struct_bias = bias
                        graph_model._pf_signal = pf
                        pos_kwargs = {}
                        if identity_rope:
                            pos_kwargs["position_ids"] = \
                                graph_model.graph_token_position_ids(
                                    [query_map], seq_len, device)
                    collector.start(qsel, qsel_heads,
                                    keep_heads=cond in HSEL_CONDITIONS)
                    out = graph_model.llm(
                        input_ids=seq_ids, attention_mask=attn_mask,
                        output_hidden_states=True, **pos_kwargs)
                    collector.stop()
            finally:
                graph_model._struct_bias = None
                graph_model._pf_signal = None

            # hidden states -> CPU f16 (kept per condition for delta computation)
            hs = [h[0].to("cpu", torch.float16) for h in out.hidden_states]
            hs_by_cond[cond] = hs

            logp, ent = chunked_logprob_stats(out.logits, seq_ids, answer_start)
            arrays[f"{cond}/logp_realized"] = logp
            arrays[f"{cond}/entropy_ans"] = ent

            # top-10 + per-node first-token logp at answer node-mention predictions
            if ans_node_pos:
                pred_rows = torch.tensor([p - 1 for p in ans_node_pos],
                                         device=device)
                rows = out.logits[0, pred_rows].float()
                lp = torch.log_softmax(rows, dim=-1)
                top = lp.topk(10, dim=-1)
                arrays[f"{cond}/ansnode_top10_ids"] = \
                    top.indices.cpu().numpy().astype(np.int32)
                arrays[f"{cond}/ansnode_top10_logp"] = \
                    top.values.cpu().numpy().astype(np.float32)
                nf = torch.tensor(node_first, device=device)      # [N, V]
                nf_safe = nf.clamp(min=0)
                gathered = lp[:, nf_safe.reshape(-1)].reshape(
                    len(ans_node_pos), *nf.shape)                  # [P, N, V]
                gathered = gathered.masked_fill(
                    (nf < 0).unsqueeze(0), float("-inf")).max(dim=-1).values
                arrays[f"{cond}/ansnode_node_logp"] = \
                    gathered.cpu().numpy().astype(np.float32)

            if cond in ATTN_CONDITIONS:
                got_layers = sorted(collector.rows)
                arrays[f"{cond}/attn_rows"] = torch.stack(
                    [collector.rows[li] for li in got_layers]).numpy()
                arrays[f"{cond}/attn_rows_layers"] = np.array(got_layers,
                                                              dtype=np.int32)
            if cond in HSEL_CONDITIONS and collector.head_rows:
                got_f = sorted(collector.head_rows)
                arrays[f"{cond}/attn_heads"] = torch.stack(
                    [collector.head_rows[li] for li in got_f]).numpy()
                arrays[f"{cond}/attn_heads_layers"] = np.array(got_f,
                                                               dtype=np.int32)
            del out
            torch.cuda.empty_cache()

        # deltas vs off + raw hidden at qsel for on/off
        hs_off = hs_by_cond["off"]
        for cond in CONDITIONS:
            hs = hs_by_cond[cond]
            dn = np.zeros((len(hs), seq_len), dtype=np.float32)
            hn = np.zeros_like(dn)
            cs = np.zeros_like(dn)
            for li, (a, b) in enumerate(zip(hs, hs_off)):
                af, bf = a.float(), b.float()
                dn[li] = (af - bf).norm(dim=-1).numpy()
                hn[li] = af.norm(dim=-1).numpy()
                cs[li] = torch.nn.functional.cosine_similarity(
                    af, bf, dim=-1).numpy()
            arrays[f"{cond}/dnorm_vs_off"] = dn
            arrays[f"{cond}/h_norm"] = hn
            arrays[f"{cond}/cos_vs_off"] = cs
            if cond in HSEL_CONDITIONS:
                arrays[f"{cond}/h_sel"] = torch.stack(
                    [h[qsel] for h in hs]).numpy()          # [L+1, nq, H]
        del hs_by_cond

        # id -> string map for every vocab id the notebook may need to render
        # (top-10 candidates under any condition + node first tokens).
        vocab_ids = set(int(i) for i in node_first.flatten() if i >= 0)
        for cond in CONDITIONS:
            key = f"{cond}/ansnode_top10_ids"
            if key in arrays:
                vocab_ids.update(int(i) for i in arrays[key].flatten())
        tok_str = {i: tokenizer.convert_ids_to_tokens(i) for i in sorted(vocab_ids)}

        meta = {
            "graph_name": stem, "idx": orig_idx, "tag": tag,
            "tok_str": tok_str,
            "task": res["task"], "answer_key": res["answer_key"],
            "correct": bool(res["correct"]),
            "path_metrics": res.get("path_metrics"),
            "response": res.get("response"),
            "prompt_len": prompt_len, "seq_len": seq_len,
            "answer_start": answer_start, "scope_start": scope_start,
            "gen_text": tokenizer.decode(cap["gen_ids"], skip_special_tokens=True),
            "tokens": tokenizer.convert_ids_to_tokens(seq_list),
            "node_names": list(g.node_names),
            "edges": g.edge_index.t().tolist(),
            "robot_location": getattr(g, "robot_location", None),
            "full_map": {str(k): v for k, v in full_map.items()},
            "query_map": {str(k): v for k, v in query_map.items()},
            "decision_map": {str(k): int(v) for k, v in dec_map.items()},
            "decision_pos": decision_pos,
            "completion_pos": pos_sets["completion"],
            "repeat_pos": pos_sets["repeat"],
            "ans_node_pos": ans_node_pos,
            "scene_node_pos": scene_node_pos,
            "nonnode_ans_pos": nonnode_pick,
            "qsel": qsel, "qsel_heads": qsel_heads,
        }
        np.savez_compressed(os.path.join(args.out, f"{tag}.npz"), **arrays)
        with open(os.path.join(args.out, f"{tag}.meta.json"), "w") as f:
            json.dump(meta, f)

        # quick per-class Δlogp summary (on vs each ablation) for the job log
        base = arrays["on/logp_realized"]
        for cond in CONDITIONS:
            d = arrays[f"{cond}/logp_realized"] - base
            cond_summ[cond] = {
                name: float(np.mean([d[p - 1] for p in pos])) if pos else None
                for name, pos in (("decision", decision_pos),
                                  ("completion", pos_sets["completion"]),
                                  ("repeat", pos_sets["repeat"]),
                                  ("nonnode", nonnode_ans))}
        summary["samples"].append({"tag": tag, "correct": bool(res["correct"]),
                                   "delta_logp_vs_on": cond_summ})
        print(f"[probe] {tag}: wrote npz "
              f"(decision Δlogp off-on = {cond_summ['off']['decision']})")

    collector.remove()
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[probe] done — {len(probe_records)} samples -> {args.out}")


if __name__ == "__main__":
    main()
