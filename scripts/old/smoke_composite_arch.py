"""Thorough architecture smoke test for the composite-graph GT + second-moment path.

Covers: composite-graph assembly, GSO, R-PEARL first/second moment, the H0 =
seeded + C·seeded fusion, magnitude balance, injection modes, permutation,
determinism, dtype, gradient flow, and edge cases. CPU-only; no LLM required.
"""
import torch
from torch_geometric.data import Data

from prism.models.composite_graph import build_composite_graph
from prism.models.r_pearl import RandomGNNPositionalEncodings
from prism.models.gt import GraphTransformer
from prism.models.composite_graph_llm import GatedInjection

torch.manual_seed(0)
PASS, FAIL = [], []
def check(name, cond, info=""):
    (PASS if cond else FAIL).append(name)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {info}" if info else ""))

d_model = 64
c = 16                      # token-cycle length
n_scene = 6
EMB = 24.0                  # Llama-like embedding norm

# ---- injection map: a couple of scene nodes mentioned by token spans ----
inj = {0: [(2, 4)], 3: [(9, 11)]}
scene_ei = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
scene_ew = torch.ones(scene_ei.shape[1])

cg = build_composite_graph(c, scene_ei, scene_ew, n_scene, inj,
                           cycle_directed=True)

# ---------- 1. composite-graph structure ----------
check("num_nodes = c + n_scene", cg.num_nodes == c + n_scene, f"{cg.num_nodes}")
check("is_token mask: c True then scene False",
      cg.is_token[:c].all().item() and (~cg.is_token[c:]).all().item())
# directed cycle: i -> (i+1)%c present, reverse absent
ei = cg.edge_index
has = lambda s, d: bool(((ei[0] == s) & (ei[1] == d)).any())
check("directed cycle forward edge present", has(0, 1) and has(c - 1, 0))
check("directed cycle reverse edge absent (no crosslink there)", not has(1, 0))
# crosslink both directions token<=>scene
check("crosslink token->scene & scene->token", has(2, c + 0) and has(c + 0, 2))
# GSO: square, sparse, directed (S[1,0]==0 since reverse cycle edge absent)
gso = cg.gso.coalesce()
gidx, gval = gso.indices(), gso.values()
sval = lambda r, cc: float(gval[((gidx[0] == r) & (gidx[1] == cc))].sum())
# edge 0->1 lives at S[0,1]; reverse S[1,0] must be 0 (directed cycle, no reverse edge)
check("GSO directed: S[0,1]!=0 and S[1,0]==0",
      sval(0, 1) != 0.0 and sval(1, 0) == 0.0, f"S[0,1]={sval(0,1):.3f} S[1,0]={sval(1,0):.3f}")
check("GSO finite & coalesced", torch.isfinite(gval).all().item())

# ---------- 2. R-PEARL first vs second moment scale ----------
pe = RandomGNNPositionalEncodings(pe_hidden_channels=32, pe_num_layers=3,
                                  d_model=d_model, num_samples=8, k=2,
                                  eps=1e-6)
pe.eval()
pe_data = Data(x=torch.zeros(cg.num_nodes, 1), edge_index=cg.edge_index)
pe_data.edge_weight = cg.edge_weight
with torch.no_grad():
    psi = pe(pe_data)                                   # first moment Ψ
psi_norm = psi.norm(dim=-1).mean().item()
check("first-moment Ψ is ~unit norm (LipschitzNorm)", abs(psi_norm - 1.0) < 0.5,
      f"Ψ mean norm={psi_norm:.3f}")

# build seeded = [X ; Ψ_scene] exactly as gt.forward does
X = torch.randn(c, d_model)
X = X / X.norm(dim=-1, keepdim=True) * EMB               # force embedding scale
seeded = torch.zeros(cg.num_nodes, d_model)
seeded[cg.is_token] = X
seeded[~cg.is_token] = psi[~cg.is_token]
tok_seed = seeded[cg.is_token].norm(dim=-1).mean().item()
scene_seed = seeded[~cg.is_token].norm(dim=-1).mean().item()
print(f"    seeded token-row norm={tok_seed:.2f}  scene-row norm={scene_seed:.2f}  "
      f"ratio scene/token={scene_seed/tok_seed:.3f}")

with torch.no_grad():
    cx = pe.second_moment_apply(pe_data, seeded)
cx_tok = cx[cg.is_token].norm(dim=-1).mean().item()
check("C·seeded matched to signal scale (not unit ~4%)", cx_tok / tok_seed > 0.3,
      f"C·seeded/token ratio={cx_tok/tok_seed:.3f}")

# ---------- 3. H0 = seeded + C·seeded properties ----------
H0 = seeded + cx
# embedding retention: token rows of (H0 - C·seeded) == X exactly
check("embedding retention: H0[token] - C·seeded[token] == X",
      torch.allclose(H0[cg.is_token] - cx[cg.is_token], X, atol=1e-4))
# position present: H0[token] differs from X (operator added signal)
delta = (H0[cg.is_token] - X).norm() / X.norm()
check("relative-position signal present in H0 (not negligible)", delta > 0.1,
      f"||H0-X||/||X||={delta:.3f}")

# ---------- 4. second-moment identity (no C formed) vs explicit ----------
pe.fixed_seed_mode, pe.fixed_seed_value = True, 7
pe.center_second_moment = False                # this check tests the raw associativity
with torch.no_grad():
    cx_a = pe.second_moment_apply(pe_data, seeded)
    # explicit: Σ_s Φ_s (Φ_sᵀ seeded)/m, then the SAME magnitude match
    Q = pe._sample_probes(cg.num_nodes, pe.M, torch.device("cpu"),
                          torch.Generator().manual_seed(7))
    P = pe._batched_gcn_forward(Q, cg.edge_index.cpu(), cg.num_nodes, pe.M,
                                edge_weight=cg.edge_weight, device=torch.device("cpu"),
                                pool=False)
    acc = sum(P[s] @ (P[s].T @ seeded) for s in range(P.shape[0])) / pe.M
    ratio = seeded.float().norm(dim=-1).mean() / acc.float().norm(dim=-1).mean().clamp(min=1e-6)
    cx_b = acc * ratio * torch.tanh(pe.output_gain)        # include the R-PEARL output gate
check("C·seeded == explicit ΣΦ(Φᵀs)/m (no N×N formed)",
      torch.allclose(cx_a, cx_b, atol=1e-4),
      f"maxerr={(cx_a-cx_b).abs().max():.2e}")

# ---------- 4b. centering recovers position rank (raw E[ΦΦᵀ] is rank-1 DC) ----------
import numpy as np
ntok = int(cg.is_token.sum())
def Ctok(center):
    pe.center_second_moment = center
    Ct = torch.zeros(ntok, ntok)
    with torch.no_grad():
        for n in range(ntok):
            s1 = torch.zeros(cg.num_nodes, d_model); s1[torch.where(cg.is_token)[0][n], 0] = 1.0
            Ct[:, n] = pe.second_moment_apply(pe_data, s1)[cg.is_token, 0]
    sv = np.linalg.svd(Ct.numpy(), compute_uv=False)
    return int((sv > 1e-3 * sv[0]).sum())
rank_raw, rank_cen = Ctok(False), Ctok(True)
check("raw E[ΦΦᵀ] collapses to ~rank-1 DC; centering C−ΨΨᵀ restores rank",
      rank_raw <= 2 and rank_cen > rank_raw,
      f"raw rank={rank_raw}, centered rank={rank_cen} / {ntok}")
pe.center_second_moment = True
pe.fixed_seed_mode = False

# ---------- 5. full GT forward: magnitude pinned to embedding scale ----------
def run_gt(mode, **kw):
    gt = GraphTransformer(num_layers=3, pe_hidden_channels=32, pe_num_layers=3,
                          d_model=d_model, heads=4, num_samples=8, k_pe=2, k_gt=2,
                          eps=1e-6, spectral_norm_linears=False,
                          pe_readout=mode, **kw)
    return gt
import math
gt = run_gt("second_moment")
gt.eval()
with torch.no_grad():
    Y = gt(pe_data, token_embeddings=X, is_token=cg.is_token)
ytok = Y[cg.is_token].norm(dim=-1).mean().item()
# Output is now embedding-scale-rescaled THEN gated by g = tanh(output_gain)
# (init output_gain=1 -> g ~ 0.762), so the pinned magnitude is g*EMB.
gate = math.tanh(float(gt.output_gain))
check("GT output token rows pinned to tanh(g)*embedding scale",
      abs(ytok - EMB * gate) / (EMB * gate) < 0.05,
      f"Y token norm={ytok:.2f} vs tanh(g)*emb {EMB * gate:.2f}")
with torch.no_grad():
    gt.output_gain.data.fill_(10.0)                 # tanh(10) ~ 1 -> full embedding scale
    Yhi = gt(pe_data, token_embeddings=X, is_token=cg.is_token)
    gt.output_gain.data.fill_(1.0)
check("tanh(g) output gate scales the GT magnitude (g large -> full ‖X‖)",
      abs(Yhi[cg.is_token].norm(dim=-1).mean().item() - EMB) / EMB < 0.05)
check("GT output finite & correct shape",
      torch.isfinite(Y).all().item() and Y.shape == (cg.num_nodes, d_model))
check("last block is norm-free", len(gt.blocks[-1].norms) == 0)
check("earlier blocks keep norms", len(gt.blocks[0].norms) == 2)

# ---------- 6. injection modes ----------
Ytok = Y[cg.is_token]
for mode in ("none", "additive", "interpolate"):
    gi = GatedInjection(d_model, gate_init=0.3, injection_mode=mode)
    out = gi(X, Ytok)
    if mode == "none":
        check("injection none returns Y untouched", torch.allclose(out, Ytok))
    elif mode == "additive":
        check("injection additive = X + g·Y", torch.allclose(out, X + 0.3 * Ytok, atol=1e-4))
    else:
        check("injection interpolate = (1-g)X + gY",
              torch.allclose(out, 0.7 * X + 0.3 * Ytok, atol=1e-4))

# ---------- 7. mean-readout regression (legacy path unaffected) ----------
gtm = run_gt("mean"); gtm.eval()
with torch.no_grad():
    Ym = gtm(pe_data, token_embeddings=X, is_token=cg.is_token)
gate_m = math.tanh(float(gtm.output_gain))
check("mean-readout path finite & tanh(g)*embedding-scaled",
      torch.isfinite(Ym).all().item()
      and abs(Ym[cg.is_token].norm(dim=-1).mean().item() - EMB * gate_m) / (EMB * gate_m) < 0.05)

# ---------- 8. gradient flow (train mode) ----------
gt.train()
Yt = gt(pe_data, token_embeddings=X.requires_grad_(False), is_token=cg.is_token)
Yt.sum().backward()
peg = sum(1 for p in gt.pe_model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
nan = any(torch.isnan(p.grad).any() for p in gt.parameters() if p.grad is not None)
check("R-PEARL params receive nonzero grad", peg > 0, f"{peg} tensors")
check("no NaN gradients", not nan)

# ---------- 9. determinism with fixed seed ----------
gt2 = run_gt("second_moment"); gt2.eval()
gt2.pe_model.fixed_seed_mode = True
with torch.no_grad():
    a = gt2(pe_data, token_embeddings=X, is_token=cg.is_token)
    b = gt2(pe_data, token_embeddings=X, is_token=cg.is_token)
check("fixed_seed_mode → deterministic output", torch.allclose(a, b))

# ---------- 10. bf16 dtype path ----------
with torch.no_grad():
    Yb = gt.eval()(pe_data, token_embeddings=X.to(torch.bfloat16), is_token=cg.is_token)
check("bf16 token-embedding path runs & finite", torch.isfinite(Yb).all().item(),
      f"out dtype={Yb.dtype}")

# ---------- 11. edge cases ----------
# (a) no scene edges
cg0 = build_composite_graph(c, torch.empty(2, 0, dtype=torch.long),
                            torch.empty(0), n_scene, inj)
d0 = Data(x=torch.zeros(cg0.num_nodes, 1), edge_index=cg0.edge_index); d0.edge_weight = cg0.edge_weight
with torch.no_grad():
    Y0 = gt.eval()(d0, token_embeddings=X, is_token=cg0.is_token)
check("no-scene-edges graph runs & finite", torch.isfinite(Y0).all().item())
# (b) no crosslinks (empty injection map) -> layers disconnected
cg1 = build_composite_graph(c, scene_ei, scene_ew, n_scene, {})
check("empty injection map: only cycle+scene edges",
      cg1.edge_index.shape[1] == c + scene_ei.shape[1])
# (c) single scene node
cg2 = build_composite_graph(c, torch.empty(2, 0, dtype=torch.long), torch.empty(0),
                            1, {0: [(0, 1)]})
d2 = Data(x=torch.zeros(cg2.num_nodes, 1), edge_index=cg2.edge_index); d2.edge_weight = cg2.edge_weight
with torch.no_grad():
    Y2 = gt.eval()(d2, token_embeddings=X, is_token=cg2.is_token)
check("single scene node runs & finite", torch.isfinite(Y2).all().item())

# ---------- 12. scene→token coupling lives through gt.forward (GAP B fix) ----------
gtc = run_gt("second_moment"); gtc.eval(); gtc.pe_model.fixed_seed_mode = True
# patch second_moment_apply to capture the `seeded` it receives
seen = {}
_orig = gtc.pe_model.second_moment_apply
def _cap(data, signal):
    seen["scene"] = signal[~cg.is_token].norm(dim=-1).mean().item()
    seen["tok"] = signal[cg.is_token].norm(dim=-1).mean().item()
    return _orig(data, signal)
gtc.pe_model.second_moment_apply = _cap
with torch.no_grad():
    gtc(pe_data, token_embeddings=X, is_token=cg.is_token)
check("scene seed scaled to token scale inside gt.forward (GAP B fix)",
      abs(seen["scene"] - seen["tok"]) / seen["tok"] < 0.05,
      f"scene={seen['scene']:.1f} tok={seen['tok']:.1f}")

print("\n================  SUMMARY  ================")
print(f"PASSED {len(PASS)} / {len(PASS) + len(FAIL)}")
if FAIL:
    print("FAILED:", FAIL)
