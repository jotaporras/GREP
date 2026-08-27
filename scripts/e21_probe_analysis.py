"""Analysis + figures for the e21 activation probe (scripts/e21_activation_probe.py).

Reads the probe output dir (one .npz + .meta.json per sample, plus summary.json)
and produces the four probe figures + a printed stats block. The notebook
``notebooks/2026-08-26 e21_v2c_activation_probe.ipynb`` mirrors these cells; this
script exists so the numbers/figures can be produced headless on betty first.

Usage:
    python scripts/e21_probe_analysis.py --probe-dir <dir> --figs <dir>
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONDITIONS = ("on", "no_mask", "no_pf", "no_decision", "off")
COND_LABEL = {"on": "full graph channel", "no_mask": "− Ψ-mask", "no_pf": "− post-fusion",
              "no_decision": "− decision gating", "off": "plain LLM (all off)"}
CLASSES = ("decision", "completion", "repeat", "nonnode")


def load_sample(probe_dir, tag):
    npz = np.load(os.path.join(probe_dir, f"{tag}.npz"))
    with open(os.path.join(probe_dir, f"{tag}.meta.json")) as f:
        meta = json.load(f)
    return npz, meta


def class_positions(meta):
    nonnode = [p for p in range(meta["answer_start"], meta["seq_len"])
               if p not in set(meta["ans_node_pos"])]
    return {"decision": meta["decision_pos"], "completion": meta["completion_pos"],
            "repeat": meta["repeat_pos"], "nonnode": nonnode}


def delta_logp(npz, cond, positions):
    """Δ logp(realized) cond − on, at ``positions`` (token frame -> row p-1)."""
    d = npz[f"{cond}/logp_realized"] - npz["on/logp_realized"]
    return np.array([d[p - 1] for p in positions if p >= 1])


# --------------------------------------------------------------------------- fig 1
def fig_prob_shift(samples, figs):
    """Per-class Δlogp distributions for each ablation (the probability-shift stat)."""
    agg = {c: {cl: [] for cl in CLASSES} for c in CONDITIONS if c != "on"}
    for npz, meta in samples:
        pos = class_positions(meta)
        for c in agg:
            for cl in CLASSES:
                agg[c][cl].extend(delta_logp(npz, c, pos[cl]).tolist())
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True)
    for ax, c in zip(axes, [c for c in CONDITIONS if c != "on"]):
        data = [agg[c][cl] for cl in CLASSES]
        bp = ax.boxplot(data, labels=CLASSES, showmeans=True, showfliers=False)
        for i, d in enumerate(data):
            ax.text(i + 1, ax.get_ylim()[0], f"μ={np.mean(d):+.2f}\nn={len(d)}",
                    ha="center", va="bottom", fontsize=8)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(COND_LABEL[c])
        ax.tick_params(axis="x", rotation=20)
    axes[0].set_ylabel("Δ logp(realized token) vs full channel")
    fig.suptitle("Ablating the graph channel: probability shift by token class "
                 "(12 eval samples, teacher-forced)")
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "e21_probe_prob_shift.png"), dpi=150)
    plt.close(fig)
    return {c: {cl: (float(np.mean(v)), len(v)) for cl, v in cls.items()}
            for c, cls in agg.items()}


# --------------------------------------------------------------------------- fig 2
def fig_delta_norm(samples, figs, summary):
    """Layer × position-class map of ||h_on − h_off||: where the signal enters."""
    mask_active = np.array(summary["mask_active"], bool)
    pf_active = np.array(summary["pf_active"], bool)
    Lp1 = len(mask_active) + 1
    groups = ("scene_node_pos", "ans_node_pos", "nonnode")
    acc = {g: np.zeros(Lp1) for g in groups}
    cnt = {g: 0 for g in groups}
    ch_acc = {c: np.zeros(Lp1) for c in ("no_mask", "no_pf", "no_decision", "off")}
    for npz, meta in samples:
        dn = npz["on/dnorm_vs_off"]                     # [L+1, S]
        hn = npz["on/h_norm"]
        rel = dn / (hn + 1e-6)
        pos = class_positions(meta)
        sel = {"scene_node_pos": meta["scene_node_pos"],
               "ans_node_pos": meta["ans_node_pos"], "nonnode": pos["nonnode"]}
        for g in groups:
            if sel[g]:
                acc[g] += rel[:, sel[g]].mean(axis=1)
                cnt[g] += 1
        node_all = meta["scene_node_pos"] + meta["ans_node_pos"]
        for c in ch_acc:
            ch_acc[c] += (npz[f"{c}/dnorm_vs_off"] / (hn + 1e-6))[:, node_all].mean(axis=1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.6))
    for g, lab in zip(groups, ("scene-graph node tokens", "answer node tokens",
                               "non-node answer tokens")):
        ax1.plot(acc[g] / max(cnt[g], 1), label=lab)
    for li, f in enumerate(mask_active):
        if f:
            ax1.axvspan(li + 0.5, li + 1.5, color="tab:blue", alpha=0.05)
    for li, f in enumerate(pf_active):
        if f:
            ax1.axvspan(li + 0.5, li + 1.5, color="tab:red", alpha=0.05)
    ax1.set_xlabel("layer (blue band = Ψ-mask active, red = post-fusion active)")
    ax1.set_ylabel("relative ||h_on − h_off|| (mean)")
    ax1.set_title("Where the graph channel perturbs the stream")
    ax1.legend(fontsize=8)
    n = len(samples)
    for c in ch_acc:
        ax2.plot(ch_acc[c] / n, label=COND_LABEL[c])
    ax2.set_xlabel("layer")
    ax2.set_ylabel("relative ||h_cond − h_off|| at node tokens")
    ax2.set_title("Channel attribution (distance that REMAINS with each ablation)")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "e21_probe_delta_norm.png"), dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- fig 3
def attn_mass_groups(npz, meta, cond):
    """Attention-mass decomposition at the rows CHOOSING each decision token.

    For decision position p (first token of an answer node mention), the choosing
    row is q = p-1; its current node comes from the decision map (every untagged
    answer row carries "the node you are standing on"). Groups: current-node
    tokens, neighbour-node tokens, non-neighbour-node tokens, non-node tokens.
    Returns [n_layers, n_valid_decisions, 4] (skips rows outside qsel/map).
    """
    rows = npz[f"{cond}/attn_rows"].astype(np.float32)   # [L, nq, S]
    layers = npz[f"{cond}/attn_rows_layers"].tolist()
    qsel = meta["qsel"]
    edges = {(a, b) for a, b in meta["edges"]} | {(b, a) for a, b in meta["edges"]}
    n_nodes = len(meta["node_names"])
    tok_of_node = {n: set() for n in range(n_nodes)}
    for n_str, spans in meta["full_map"].items():
        for a, b in spans:
            tok_of_node[int(n_str)].update(range(a, b))
    allnode = set().union(*tok_of_node.values()) if tok_of_node else set()
    S = rows.shape[2]
    valid, out_rows = [], []
    for p in meta["decision_pos"]:
        q = p - 1
        cur = meta["decision_map"].get(str(q))
        if q not in qsel or cur is None:
            continue
        qi = qsel.index(q)
        neigh = {m for m in range(n_nodes) if (cur, m) in edges}
        cur_toks = sorted(tok_of_node[cur])
        neigh_toks = sorted(set().union(*[tok_of_node[m] for m in neigh])
                            if neigh else set())
        other_toks = sorted(set().union(*[tok_of_node[m] for m in range(n_nodes)
                                          if m != cur and m not in neigh]))
        nonnode = [t for t in range(S) if t not in allnode]
        g4 = np.stack([rows[:, qi, toks].sum(axis=1) if toks else
                       np.zeros(rows.shape[0], np.float32)
                       for toks in (cur_toks, neigh_toks, other_toks, nonnode)],
                      axis=1)                             # [L, 4]
        out_rows.append(g4)
        valid.append(p)
    out = np.stack(out_rows, axis=1) if out_rows else \
        np.zeros((len(layers), 0, 4), np.float32)
    return out, valid, layers


def fig_attention(samples, figs, summary):
    """ON vs OFF attention-mass decomposition at the rows CHOOSING each decision."""
    conds = ("on", "no_mask", "off")
    agg = {c: [] for c in conds}
    for npz, meta in samples:
        for c in conds:
            if f"{c}/attn_rows" not in npz:
                continue
            m, dec_ps, layers = attn_mass_groups(npz, meta, c)
            if m.shape[1]:
                agg[c].append(m.mean(axis=1))            # [L, 4] per sample
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    labels = ("current node", "neighbours", "non-neighbour nodes", "non-node")
    colors = ("tab:green", "tab:blue", "tab:red", "0.7")
    for ax, c in zip(axes, conds):
        if not agg[c]:
            continue
        m = np.mean(agg[c], axis=0)                      # [L, 4]
        bottom = np.zeros(m.shape[0])
        for gi, (lab, col) in enumerate(zip(labels, colors)):
            ax.bar(range(m.shape[0]), m[:, gi], bottom=bottom, color=col,
                   label=lab, width=1.0)
            bottom += m[:, gi]
        ax.set_title(COND_LABEL[c])
        ax.set_xlabel("layer")
    axes[0].set_ylabel("attention mass from decision-choosing rows")
    axes[0].legend(fontsize=8)
    fig.suptitle("Attention of the rows that CHOOSE the next node, by key group")
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "e21_probe_attention.png"), dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- fig 4
def focus_decision(meta):
    """Index into decision_pos where the error most plausibly happened: the first
    decision whose generated node is not the answer key's required node at that
    slot (falls back to 0)."""
    import re
    req = re.findall(r"\\b([a-z][a-z0-9_]+)\\b", meta["answer_key"])
    route = (meta.get("path_metrics") or {}).get("parsed_nodes") or []
    if req and route and route[0] == req[0] and len(meta["decision_pos"]) > 1:
        return 1
    return 0


def fig_error_cases(samples, figs):
    """Error-vs-control zoom: P(node) shift at the failing decision, on vs off."""
    errs = [(npz, meta) for npz, meta in samples if not meta["correct"]]
    fig, axes = plt.subplots(1, len(errs), figsize=(3.4 * len(errs), 4.6),
                             squeeze=False)
    for ax, (npz, meta) in zip(axes[0], errs):
        names = meta["node_names"]
        ap = meta["ans_node_pos"]
        d0 = meta["decision_pos"][focus_decision(meta)]
        pi = ap.index(d0)
        on = npz["on/ansnode_node_logp"][pi]
        off = npz["off/ansnode_node_logp"][pi]
        top = np.argsort(on)[::-1][:6]
        x = np.arange(len(top))
        ax.bar(x - 0.2, on[top], width=0.4, label="on", color="tab:blue")
        ax.bar(x + 0.2, off[top], width=0.4, label="off", color="0.6")
        ax.set_xticks(x)
        ax.set_xticklabels([names[i] for i in top], rotation=60, ha="right",
                           fontsize=7)
        ax.set_title(f"{meta['tag']}\ndecision", fontsize=9)
    axes[0][0].set_ylabel("logp(node first token)")
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Residual errors: node scores at the failing decision, graph channel on vs off")
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "e21_probe_error_cases.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe-dir", required=True)
    ap.add_argument("--figs", required=True)
    args = ap.parse_args()
    os.makedirs(args.figs, exist_ok=True)
    with open(os.path.join(args.probe_dir, "summary.json")) as f:
        summary = json.load(f)
    tags = [s["tag"] for s in summary["samples"]]
    samples = [load_sample(args.probe_dir, t) for t in tags]
    print(f"[analysis] {len(samples)} samples loaded")

    stats = fig_prob_shift(samples, args.figs)
    print("\n=== mean Δlogp vs full channel (per class) ===")
    for c, cls in stats.items():
        line = "  ".join(f"{cl}={m:+.3f}(n={n})" for cl, (m, n) in cls.items())
        print(f"{c:<12} {line}")

    fig_delta_norm(samples, args.figs, summary)
    fig_attention(samples, args.figs, summary)
    fig_error_cases(samples, args.figs)
    print(f"[analysis] figures -> {args.figs}")


if __name__ == "__main__":
    main()
