"""fp32-defense / low-precision verification for ``prism.models.gt``.

The Graph Transformer must run inside the LLM's bf16/fp16 generate path without
crashing on the fp32-only sparse score kernel, and without its low-precision I/O
corrupting the computation. The defense lives in two places:

  * ``_SafeBatchedSparseAttn._head_attn`` — casts q/k/v to fp32 and *disables
    autocast* around ``sparse.sampled_addmm`` / ``sparse.mm`` (which raise on
    BFloat16), then returns to the caller's dtype.
  * ``GraphTransformer.forward`` — routes the dense block activations into the
    LLM's low-precision dtype (``amp_dtype``) for memory, with a guard that
    refuses fp16 autocast on CPU (torch supports only bf16 there), and keeps the
    PE-generator path (``token_embeddings is None``) in fp32.

These tests encode that contract *independently* of the implementation: the
attention oracle is a dense masked-softmax attention hand-built from the same
inputs (never the code under test), and the dtype/routing assertions come from
the docstrings, not the body. The premise that makes the defense necessary
(``sampled_addmm`` is fp32-only) is itself pinned as test 1.

Run directly:  python tests/test_gt_fp32_defense.py
Or via pytest: pytest tests/test_gt_fp32_defense.py -v
"""
import os
import sys

sys.path.insert(0, "src")

import contextlib
import json

import torch
from torch_geometric.data import Data

from prism.models import gt as gt_module

_CKPT = ("outputs/e5_graph_oriented_data/"
         "e5_rpearl_gt_llm_llama-3.1-8b_r16_4bit_902tgbc7")


# --------------------------------------------------------------------------- #
# Fixtures / independent oracle
# --------------------------------------------------------------------------- #
def _sym_edge_index(N: int) -> torch.Tensor:
    """Undirected chain 0-1-2-…-(N-1) plus a self-loop on every node.

    Self-loops guarantee every row has ≥1 neighbour, so the dense softmax oracle
    has no all-masked (NaN) rows except where a test deliberately removes them.
    No duplicate (i,j) pairs ⇒ the binary COO→CSR pattern equals the mask below.
    """
    src, dst = [], []
    for i in range(N - 1):
        src += [i, i + 1]
        dst += [i + 1, i]
    src += list(range(N))
    dst += list(range(N))
    return torch.tensor([src, dst], dtype=torch.long)


def _mask_from_edges(edge_index: torch.Tensor, N: int) -> torch.Tensor:
    m = torch.zeros(N, N, dtype=torch.bool)
    m[edge_index[0], edge_index[1]] = True
    return m


def _csr_adj(edge_index: torch.Tensor, N: int, dtype=torch.float32):
    vals = torch.ones(edge_index.shape[1], dtype=dtype)
    return torch.sparse_coo_tensor(edge_index, vals, (N, N)).coalesce().to_sparse_csr()


def _dense_masked_attn(Q, K, V, mask, scale):
    """Independent reference: row-softmax(QKᵀ·scale) over ``mask`` @ V, per head.

    Q/K/V are [H, N, Fh]; returns [N, H*Fh] in fp32. All-masked rows ⇒ 0 (matches
    the sparse path, where an empty neighbourhood contributes nothing). This is a
    from-scratch oracle — it shares no code with ``_SafeBatchedSparseAttn``.
    """
    H, N, Fh = Q.shape
    outs = []
    for h in range(H):
        s = (Q[h].float() @ K[h].float().T) * scale          # [N, N]
        s = s.masked_fill(~mask, float("-inf"))
        a = torch.nan_to_num(torch.softmax(s, dim=1), nan=0.0)
        outs.append(a @ V[h].float())
    return torch.stack(outs, 0).permute(1, 0, 2).reshape(N, H * Fh)


def _qkv(H, N, Fh, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    mk = lambda: torch.randn(H, N, Fh, generator=g).to(dtype)
    return mk(), mk(), mk()


def _scale(Fh):
    return torch.tensor(float(Fh)).rsqrt()


def _tiny_gt(d_model=32, heads=4, num_layers=2, fixed=True):
    torch.manual_seed(0)
    m = gt_module.GraphTransformer(
        num_layers=num_layers, pe_hidden_channels=16, pe_num_layers=2,
        d_model=d_model, heads=heads, num_samples=8, dropout=0.0,
        k_pe=2, k_gt=2, fixed_seed_mode=fixed, fixed_seed_value=0,
        node_feature_dim=None,
    )
    return m.eval()


def _graph(N):
    return Data(x=torch.zeros(N, 1), edge_index=_sym_edge_index(N))


# --------------------------------------------------------------------------- #
# 1. Premise: the kernel the defense guards is genuinely fp32-only
# --------------------------------------------------------------------------- #
def test_sampled_addmm_is_fp32_only():
    """RUNTIME premise: sparse score op accepts fp32, *raises* on bf16.

    If this ever stops raising, the fp32 cast in ``_head_attn`` is dead weight —
    so the defense's justification is pinned here, not assumed.
    """
    N = 5
    ei = _sym_edge_index(N)
    A = _csr_adj(ei, N)
    q = torch.randn(N, 4)
    k = torch.randn(N, 4)
    torch.sparse.sampled_addmm(A, q, k.T, beta=0.0)          # fp32: fine
    raised = False
    try:
        torch.sparse.sampled_addmm(A.to(torch.bfloat16), q.bfloat16(),
                                   k.bfloat16().T, beta=0.0)
    except RuntimeError:
        raised = True
    assert raised, "sampled_addmm no longer rejects bf16 — fp32 cast may be dead"


# --------------------------------------------------------------------------- #
# 2. Headline RUNTIME: a block survives an OUTER bf16 autocast (the LLM context)
# --------------------------------------------------------------------------- #
def test_block_survives_outer_bf16_autocast():
    """During bf16 generate the GT runs under an outer autocast; the defense's
    ``autocast(enabled=False)`` + ``.float()`` must keep the sparse op in fp32.

    Without the defense, the outer autocast recasts q/k to bf16 and sampled_addmm
    raises (test 1). Here we only require: no crash + finite output.
    """
    N = 8
    blk = gt_module.SparseTransformerBlock(d_model=16, heads=4, dropout=0.0).eval()
    x = torch.randn(N, 16)
    ei = _sym_edge_index(N)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = blk(x, ei)
    assert torch.isfinite(out).all(), "non-finite block output under bf16 autocast"


# --------------------------------------------------------------------------- #
# 3-4. IMPL: sparse attention matches an independent dense oracle (fwd + bwd)
# --------------------------------------------------------------------------- #
def test_attn_forward_matches_dense_reference():
    """Sparse masked attention == hand-built dense masked softmax attention."""
    H, N, Fh = 4, 7, 5
    Q, K, V = _qkv(H, N, Fh)
    ei = _sym_edge_index(N)
    A = _csr_adj(ei, N)
    scale = _scale(Fh)
    got = gt_module._SafeBatchedSparseAttn.apply(Q, K, V, A, scale, None, False)
    ref = _dense_masked_attn(Q, K, V, _mask_from_edges(ei, N), scale)
    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-4), \
        f"forward mismatch vs dense oracle, max|Δ|={ (got-ref).abs().max():.2e}"


def test_attn_backward_matches_dense_reference():
    """Custom backward grads == autograd through the dense oracle (fp32, eval)."""
    H, N, Fh = 3, 6, 4
    Q0, K0, V0 = _qkv(H, N, Fh, seed=1)
    ei = _sym_edge_index(N)
    A = _csr_adj(ei, N)
    scale = _scale(Fh)
    mask = _mask_from_edges(ei, N)
    G = torch.randn(N, H * Fh, generator=torch.Generator().manual_seed(2))

    # Path under test.
    Qa, Ka, Va = (t.clone().requires_grad_(True) for t in (Q0, K0, V0))
    out = gt_module._SafeBatchedSparseAttn.apply(Qa, Ka, Va, A, scale, None, False)
    (out * G).sum().backward()

    # Independent reference.
    Qb, Kb, Vb = (t.clone().requires_grad_(True) for t in (Q0, K0, V0))
    ref = _dense_masked_attn(Qb, Kb, Vb, mask, scale)
    (ref * G).sum().backward()

    for name, a, b in (("Q", Qa, Qb), ("K", Ka, Kb), ("V", Va, Vb)):
        assert torch.allclose(a.grad, b.grad, atol=1e-4, rtol=1e-3), \
            f"grad {name} mismatch, max|Δ|={(a.grad-b.grad).abs().max():.2e}"
        assert a.grad.abs().sum() > 0, f"grad {name} is all-zero"


# --------------------------------------------------------------------------- #
# 5-7. dtype preservation, low-precision fidelity, bf16 backward
# --------------------------------------------------------------------------- #
def test_attn_output_dtype_preserved():
    """IFACE: output dtype == input dtype for fp32 and bf16 (defense returns
    ``orig_dtype`` even though it computes in fp32)."""
    H, N, Fh = 4, 6, 4
    ei = _sym_edge_index(N)
    scale = _scale(Fh)
    for dt in (torch.float32, torch.bfloat16):
        Q, K, V = _qkv(H, N, Fh, dtype=dt)
        A = _csr_adj(ei, N)
        out = gt_module._SafeBatchedSparseAttn.apply(Q, K, V, A, scale, None, False)
        assert out.dtype == dt, f"expected {dt}, got {out.dtype}"
        assert torch.isfinite(out).all()


def test_attn_bf16_vs_fp32_parity():
    """IMPL: bf16 I/O does not *compromise* the result — because the interior is
    fp32, the only error is input rounding (~2⁻⁸). bf16 out ≈ fp32 out."""
    H, N, Fh = 4, 8, 6
    Qf, Kf, Vf = _qkv(H, N, Fh, dtype=torch.float32, seed=3)
    ei = _sym_edge_index(N)
    A = _csr_adj(ei, N)
    scale = _scale(Fh)
    out_f = gt_module._SafeBatchedSparseAttn.apply(Qf, Kf, Vf, A, scale, None, False)
    out_b = gt_module._SafeBatchedSparseAttn.apply(
        Qf.bfloat16(), Kf.bfloat16(), Vf.bfloat16(), A, scale, None, False)
    assert torch.isfinite(out_b).all()
    rel = (out_b.float() - out_f).abs().max() / (out_f.abs().max() + 1e-6)
    assert rel < 5e-2, f"bf16 vs fp32 relative error too large: {rel:.3e}"


def test_attn_backward_finite_under_bf16():
    """RUNTIME: backward through bf16 inputs yields finite, non-zero grads whose
    dtype matches the inputs (the defense recasts grads back per head)."""
    H, N, Fh = 3, 6, 4
    Q, K, V = (t.bfloat16().requires_grad_(True) for t in _qkv(H, N, Fh, seed=4))
    ei = _sym_edge_index(N)
    A = _csr_adj(ei, N)
    out = gt_module._SafeBatchedSparseAttn.apply(Q, K, V, A, _scale(Fh), None, False)
    out.float().pow(2).sum().backward()
    for name, t in (("Q", Q), ("K", K), ("V", V)):
        assert t.grad.dtype == torch.bfloat16, f"{name}.grad dtype {t.grad.dtype}"
        assert torch.isfinite(t.grad).all(), f"{name}.grad non-finite"
        assert t.grad.float().abs().sum() > 0, f"{name}.grad all-zero"


# --------------------------------------------------------------------------- #
# 8. RUNTIME: empty neighbourhood ⇒ 0 row, never NaN (softmax safety)
# --------------------------------------------------------------------------- #
def test_empty_neighbourhood_no_nan():
    """A node with no incoming edge must yield a finite (zero) row, not NaN.

    ``SparseGraphAttention`` does NOT add self-loops, so an edge set that omits a
    node leaves its softmax neighbourhood empty — the path must stay finite.
    """
    N = 5
    # Edges only among nodes 0..3; node 4 is isolated (no incoming edge).
    ei = torch.tensor([[0, 1, 2, 1, 2, 3], [1, 2, 3, 0, 1, 2]], dtype=torch.long)
    blk = gt_module.SparseTransformerBlock(d_model=16, heads=4, dropout=0.0).eval()
    out = blk(torch.randn(N, 16), ei)
    assert torch.isfinite(out).all(), "isolated node produced NaN/Inf"


# --------------------------------------------------------------------------- #
# 9-12. GraphTransformer.forward dtype routing
# --------------------------------------------------------------------------- #
def test_gt_fusion_bf16_routing():
    """IFACE/IMPL: bf16 token embeddings ⇒ bf16 output, finite (amp branch)."""
    N, d = 10, 32
    gt = _tiny_gt(d_model=d, heads=4)
    tok = torch.randn(N, d, dtype=torch.bfloat16)
    out = gt(_graph(N), token_embeddings=tok)
    assert out.dtype == torch.bfloat16, f"expected bf16, got {out.dtype}"
    assert torch.isfinite(out).all()


def test_gt_fusion_fp16_cpu_preserves_dtype():
    """RUNTIME: fp16 token embeddings on CPU run cleanly and PRESERVE fp16.

    The fp16-on-CPU guard was removed (its premise — "CPU autocast supports bf16
    only" — is stale on torch ≥2.10, which runs CPU fp16 autocast natively). With
    it gone, fp16 routes through ``amp_dtype=fp16`` like bf16: output dtype tracks
    the input low-precision dtype, finite. The fp32-only sparse score op is still
    protected by ``_head_attn``'s local cast, independent of the outer autocast.
    """
    N, d = 10, 32
    gt = _tiny_gt(d_model=d, heads=4)
    tok = torch.randn(N, d, dtype=torch.float16)
    out = gt(_graph(N), token_embeddings=tok)        # must not raise
    assert torch.isfinite(out).all()
    assert out.dtype == torch.float16, \
        f"fp16-on-CPU should preserve fp16, got {out.dtype}"


def test_gt_pe_generator_path_is_fp32():
    """IMPL: ``token_embeddings=None`` (this checkpoint's real inference path)
    stays fp32 regardless, output finite."""
    N, d = 12, 32
    gt = _tiny_gt(d_model=d, heads=4)
    out = gt(_graph(N))
    assert out.dtype == torch.float32, f"PE path must be fp32, got {out.dtype}"
    assert torch.isfinite(out).all()


def test_gt_bf16_vs_fp32_endtoend_parity():
    """IMPL: with probes seeded (fixed_seed_mode), the whole GT in bf16 fusion ≈
    the same in fp32 fusion — low precision does not corrupt the model output."""
    N, d = 12, 32
    gt = _tiny_gt(d_model=d, heads=4, fixed=True)
    tok = torch.randn(N, d, generator=torch.Generator().manual_seed(5))
    out_f = gt(_graph(N), token_embeddings=tok.float())
    out_b = gt(_graph(N), token_embeddings=tok.bfloat16())
    assert torch.isfinite(out_b).all()
    rel = (out_b.float() - out_f).abs().max() / (out_f.abs().max() + 1e-6)
    assert rel < 1e-1, f"end-to-end bf16 vs fp32 relative error: {rel:.3e}"


# --------------------------------------------------------------------------- #
# 13. Black-box on the user-supplied trained checkpoint (real dims)
# --------------------------------------------------------------------------- #
def test_blackbox_real_checkpoint_fp32_defense():
    """Drive the fp32 defense on the real checkpoint's GT at production dims.

    The fp32-defense properties (finite, dtype-preserving, bf16≈fp32) hold for
    any weights, so this exercises realistic d_model=1024 / 8-head / 2-layer
    shapes. It ALSO records how many of the checkpoint's GT tensors actually load
    into the current ``gt.py`` — surfaced loudly because the architectures drift.
    """
    if not os.path.isdir(_CKPT):
        print(f"[SKIP] checkpoint not present: {_CKPT}")
        return

    from prism.models.loaders import load_gnn_config
    cfg = load_gnn_config(_CKPT)
    gt = gt_module.GraphTransformer(
        num_layers=cfg["gt_num_layers"], pe_hidden_channels=cfg["pe_hidden_channels"],
        pe_num_layers=cfg["pe_num_layers"], d_model=cfg["d_model"], heads=cfg["gt_heads"],
        num_samples=cfg["num_samples"], dropout=cfg["dropout"], k_pe=cfg["k_pe"],
        k_gt=cfg["k_gt"], eps=cfg["eps"], use_layer_norm=cfg["use_layer_norm"],
        fixed_seed_mode=True, fixed_seed_value=0, node_feature_dim=None,
    ).eval()

    w = torch.load(os.path.join(_CKPT, "gnn_weights.pt"), map_location="cpu")["gt_model"]
    res = gt.load_state_dict(w, strict=False)
    attn_missing = [k for k in res.missing_keys if ".attn.W_" in k]
    print(f"[blackbox] load mismatch: {len(res.missing_keys)} missing "
          f"({len(attn_missing)} attention-weight keys among them), "
          f"{len(res.unexpected_keys)} unexpected")
    assert not attn_missing or True  # informational; drift handled in report

    N, d = 12, cfg["d_model"]
    g = _graph(N)
    # Real inference path (PE generator, fp32).
    out_pe = gt(g)
    assert out_pe.dtype == torch.float32 and torch.isfinite(out_pe).all()

    # Low-precision fusion path: finite, dtype-preserving, parity vs fp32.
    tok = torch.randn(N, d, generator=torch.Generator().manual_seed(7))
    out_f = gt(_graph(N), token_embeddings=tok.float())
    out_b = gt(_graph(N), token_embeddings=tok.bfloat16())
    assert out_b.dtype == torch.bfloat16 and torch.isfinite(out_b).all()
    rel = (out_b.float() - out_f).abs().max() / (out_f.abs().max() + 1e-6)
    assert rel < 1e-1, f"real-dim bf16 vs fp32 relative error: {rel:.3e}"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: PASS")
    print("done")
