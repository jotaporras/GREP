"""Black-box audit of the fp32 / low-precision defense in ``prism.models.gt``.

The Graph Transformer runs its dense transformer blocks in the LLM's low-precision
dtype (bf16/fp16) to avoid OOM on the composite graph, while the fp32-only sparse
score op (``torch.sparse.sampled_addmm``) is cast to fp32 *locally* with autocast
disabled (see ``_SafeBatchedSparseAttn._head_attn`` and ``GraphTransformer.forward``).

Every test is driven through a fully-initialized (all submodules constructed: R-PEARL +
stacked blocks + output gate), UNTRAINED random-init ``GraphTransformer`` on CPU — the
attention/block fixtures and the Q/K/V fed to the core autograd Function are pulled from
that real model's weights, not from bare standalone layers.

This suite verifies, in black-box mode, that low precision does not compromise the model:

  * dtype contract  — low-precision in ⇒ low-precision out; PE-generator path stays fp32;
  * defense holds   — the sparse op is fp32 even inside a bf16 autocast (no BFloat16 crash);
  * numerics        — bf16 output tracks the fp32 reference to bf16 rounding, not garbage;
  * runtime safety  — finite output / finite-nonzero grads in low precision, incl. degenerate
                      graphs (isolated nodes, single node).

Oracle is independent of the implementation: references are the fp32 path, hand-built
small graphs, and the documented contract — never a re-implementation of the code.
"""
import sys

sys.path.insert(0, "src")

import torch
from torch_geometric.data import Data

from prism.models import gt


# ---------------------------------------------------------------------------
# fixtures (tiny, random-init, CPU)
# ---------------------------------------------------------------------------
def _ring_edge_index(n: int) -> torch.Tensor:
    """Undirected ring + self-loops on n nodes — every node has >=1 in-neighbor.

    Self-loops mirror the real model (``_expand_edge_index`` adds them), so no
    softmax row is ever empty.
    """
    src, dst = [], []
    for i in range(n):
        src += [i, i, i]
        dst += [i, (i + 1) % n, (i - 1) % n]
    return torch.tensor([src, dst], dtype=torch.long)


def _csr_adj(edge_index: torch.Tensor, n: int, dtype=torch.float32) -> torch.Tensor:
    vals = torch.ones(edge_index.shape[1], dtype=dtype)
    return torch.sparse_coo_tensor(edge_index, vals, (n, n)).coalesce().to_sparse_csr()


def _gt(num_layers=2, d_model=16, heads=4) -> gt.GraphTransformer:
    """A fully-initialized (R-PEARL + blocks + gate), untrained random-init GT, eval mode."""
    torch.manual_seed(0)
    model = gt.GraphTransformer(
        num_layers=num_layers, pe_hidden_channels=8, pe_num_layers=2,
        d_model=d_model, heads=heads, num_samples=4, dropout=0.0,
        k_pe=2, k_gt=2, fixed_seed_mode=True, fixed_seed_value=0,
    )
    model.eval()
    return model


def _attn(d_model=16, heads=4) -> gt.SparseGraphAttention:
    """The first block's attention, taken from a fully-initialized GT (not a bare layer)."""
    return _gt(num_layers=2, d_model=d_model, heads=heads).blocks[0].attn


def _block(d_model=16, heads=4) -> gt.SparseTransformerBlock:
    """The first (LayerNorm'd) block of a fully-initialized GT."""
    return _gt(num_layers=2, d_model=d_model, heads=heads).blocks[0]


def _qkv_from_attn(attn: gt.SparseGraphAttention, x: torch.Tensor):
    """Project x through the GT attention's real W_Q/W_K/W_V -> (H, N, F_head) heads.

    Mirrors ``SparseGraphAttention.forward``; gives the core Function the model's
    actual activations instead of synthetic random tensors.
    """
    N = x.shape[0]
    q = attn.W_Q(x).view(N, attn.heads, attn.head_dim).permute(1, 0, 2)
    k = attn.W_K(x).view(N, attn.heads, attn.head_dim).permute(1, 0, 2)
    v = attn.W_V(x).view(N, attn.heads, attn.head_dim).permute(1, 0, 2)
    return q, k, v


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative L2 error ‖a-b‖ / ‖b‖ with a/b upcast to fp32."""
    a32, b32 = a.float(), b.float()
    return (a32 - b32).norm().item() / (b32.norm().item() + 1e-12)


# ---------------------------------------------------------------------------
# A. Core defense: _SafeBatchedSparseAttn dtype contract + numerics
# ---------------------------------------------------------------------------
def test_safebatched_output_dtype_follows_input():
    """orig_dtype contract: output dtype == input QX dtype (fp32->fp32, bf16->bf16)."""
    N, d = 6, 16
    attn = _attn(d_model=d, heads=4)
    H, F = attn.heads, attn.head_dim
    A = _csr_adj(_ring_edge_index(N), N)
    torch.manual_seed(1)
    q, k, v = _qkv_from_attn(attn, torch.randn(N, d))

    out32 = gt._SafeBatchedSparseAttn.apply(q, k, v, A, attn.scale, None, False)
    out16 = gt._SafeBatchedSparseAttn.apply(
        q.bfloat16(), k.bfloat16(), v.bfloat16(), A, attn.scale, None, False)

    assert out32.dtype == torch.float32, out32.dtype
    assert out16.dtype == torch.bfloat16, out16.dtype          # saved-dtype round-trip
    assert out32.shape == (N, H * F) and out16.shape == (N, H * F)


def test_safebatched_bf16_tracks_fp32():
    """Low precision must not corrupt: bf16 output ~= fp32 output to bf16 rounding."""
    N, d = 8, 16
    attn = _attn(d_model=d, heads=4)
    A = _csr_adj(_ring_edge_index(N), N)
    torch.manual_seed(2)
    q, k, v = _qkv_from_attn(attn, torch.randn(N, d))

    out32 = gt._SafeBatchedSparseAttn.apply(q, k, v, A, attn.scale, None, False)
    out16 = gt._SafeBatchedSparseAttn.apply(
        q.bfloat16(), k.bfloat16(), v.bfloat16(), A, attn.scale, None, False)

    assert torch.isfinite(out16).all()
    err = _rel_err(out16, out32)
    # bf16 has ~8 mantissa bits (eps ~ 4e-3); a single attention op should agree
    # with fp32 well under 5%. A failure here means low precision corrupts output.
    assert err < 0.05, f"bf16 diverges from fp32: rel_err={err:.4f}"


def test_safebatched_backward_bf16_finite_nonzero():
    """Backward through the per-head fp32 recompute gives finite, nonzero bf16 grads."""
    N, d = 6, 16
    attn = _attn(d_model=d, heads=4)
    A = _csr_adj(_ring_edge_index(N), N)
    torch.manual_seed(3)
    q0, k0, v0 = _qkv_from_attn(attn, torch.randn(N, d))
    q = q0.detach().bfloat16().requires_grad_(True)
    k = k0.detach().bfloat16().requires_grad_(True)
    v = v0.detach().bfloat16().requires_grad_(True)

    out = gt._SafeBatchedSparseAttn.apply(q, k, v, A, attn.scale, None, False)
    out.float().pow(2).sum().backward()

    for name, g in (("q", q.grad), ("k", k.grad), ("v", v.grad)):
        assert g is not None, name
        assert g.dtype == torch.bfloat16, (name, g.dtype)
        assert torch.isfinite(g).all(), name
        assert g.abs().sum().item() > 0, f"{name} grad all-zero"


# ---------------------------------------------------------------------------
# B. The autocast-disable defense (headline)
# ---------------------------------------------------------------------------
def test_failure_mode_bf16_sampled_addmm_raises():
    """Independent oracle: sampled_addmm with bf16 operands RAISES -> fp32 cast is load-bearing."""
    N, F = 5, 4
    A = _csr_adj(_ring_edge_index(N), N)
    q = torch.randn(N, F).bfloat16(); k = torch.randn(N, F).bfloat16()
    raised = False
    try:
        torch.sparse.sampled_addmm(A, q, k.T, beta=0.0)
    except RuntimeError:
        raised = True
    assert raised, "bf16 sampled_addmm unexpectedly succeeded; defense premise is stale"


def test_attention_survives_bf16_autocast():
    """The whole point: SparseGraphAttention runs inside a bf16 autocast without crashing."""
    N, d = 8, 16
    attn = _attn(d_model=d, heads=4)
    ei = _ring_edge_index(N)
    torch.manual_seed(4)
    x = torch.randn(N, d).bfloat16()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = attn(x, ei)
    assert torch.isfinite(out).all()
    assert out.shape == (N, d)
    assert out.dtype == torch.bfloat16, out.dtype


def test_attention_autocast_matches_eager_fp32():
    """Defense preserves the computation: bf16-autocast output ~= fp32 eager output."""
    N, d = 10, 16
    attn = _attn(d_model=d, heads=4)
    ei = _ring_edge_index(N)
    torch.manual_seed(5)
    x = torch.randn(N, d)

    out32 = attn(x, ei)                                         # fp32 eager reference
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out16 = attn(x.bfloat16(), ei)
    err = _rel_err(out16, out32)
    assert err < 0.06, f"autocast path diverges from fp32 eager: rel_err={err:.4f}"


def test_block_backward_in_bf16_autocast():
    """A full transformer block trains in bf16 autocast: finite, nonzero param grads."""
    N, d = 8, 16
    block = _block(d_model=d, heads=4)
    block.train()
    ei = _ring_edge_index(N)
    torch.manual_seed(6)
    x = torch.randn(N, d).bfloat16().requires_grad_(True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = block(x, ei)
    out.float().pow(2).sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    grad_params = [p for p in block.parameters() if p.grad is not None]
    assert grad_params, "no param received grad"
    for p in grad_params:
        assert torch.isfinite(p.grad).all()


# ---------------------------------------------------------------------------
# C. End-to-end GraphTransformer OOM-defense (amp_dtype) path
# ---------------------------------------------------------------------------
def _gt_inputs(N, d_model, dtype):
    data = Data(x=torch.zeros(N, 1), edge_index=_ring_edge_index(N))
    torch.manual_seed(7)
    tok = torch.randn(N, d_model).to(dtype)
    return data, tok


def test_gt_lowprec_token_embeddings_output_dtype():
    """amp path: bf16/fp16 token embeddings -> low-precision output; fp32 -> fp32 output."""
    N, d = 7, 16
    model = _gt(num_layers=2, d_model=d)
    for dtype in (torch.bfloat16, torch.float32):
        data, tok = _gt_inputs(N, d, dtype)
        out = model(data, token_embeddings=tok)
        assert out.shape == (N, d)
        assert out.dtype == dtype, (dtype, out.dtype)
        assert torch.isfinite(out).all()


def test_gt_pe_generator_path_stays_fp32():
    """Documented invariant: with token_embeddings=None the amp defense is NOT applied."""
    N, d = 7, 16
    model = _gt(num_layers=2, d_model=d)
    data = Data(x=torch.zeros(N, 1), edge_index=_ring_edge_index(N))
    out = model(data, token_embeddings=None)
    assert out.dtype == torch.float32, out.dtype
    assert torch.isfinite(out).all()


def test_gt_bf16_tracks_fp32_endtoend():
    """OOM low-precision blocks don't compromise the model: bf16 output ~= fp32 output.

    fixed_seed_mode + eval + dropout=0 makes R-PEARL probes identical across calls,
    so the ONLY difference is the transformer-block precision.
    """
    N, d = 9, 16
    model = _gt(num_layers=2, d_model=d)

    data32, tok32 = _gt_inputs(N, d, torch.float32)
    out32 = model(data32, token_embeddings=tok32)
    data16, tok16 = _gt_inputs(N, d, torch.bfloat16)
    out16 = model(data16, token_embeddings=tok16)

    assert torch.isfinite(out16).all()
    err = _rel_err(out16, out32)
    # Two normed blocks + gate; bf16 rounding accumulates but must stay small.
    assert err < 0.10, f"end-to-end bf16 diverges from fp32: rel_err={err:.4f}"


def test_gt_fp16_path_runs():
    """fp16 token embeddings also trigger the amp defense and produce finite fp16 output."""
    N, d = 7, 16
    model = _gt(num_layers=2, d_model=d)
    data, tok = _gt_inputs(N, d, torch.float16)
    out = model(data, token_embeddings=tok)
    assert out.dtype == torch.float16, out.dtype
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# D. Degenerate graphs — softmax safety in low precision
# ---------------------------------------------------------------------------
def test_attention_self_loops_only_no_nan_bf16():
    """Each node attends only to itself: single-element softmax must be finite (no NaN)."""
    N, d = 5, 16
    attn = _attn(d_model=d, heads=4)
    ei = torch.tensor([[i for i in range(N)], [i for i in range(N)]], dtype=torch.long)
    torch.manual_seed(8)
    x = torch.randn(N, d).bfloat16()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = attn(x, ei)
    assert torch.isfinite(out).all(), "NaN/Inf on self-loop-only graph"


def test_attention_single_node_bf16():
    """N=1 with one self-loop: degenerate but in-domain; finite output."""
    d = 16
    attn = _attn(d_model=d, heads=4)
    ei = torch.tensor([[0], [0]], dtype=torch.long)
    torch.manual_seed(9)
    x = torch.randn(1, d).bfloat16()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = attn(x, ei)
    assert out.shape == (1, d) and torch.isfinite(out).all()


def test_attention_isolated_node_zero_row():
    """A node with no in-edges yields a zero output row (sparse.mm), never NaN."""
    N, d = 4, 16
    attn = _attn(d_model=d, heads=4)
    # node 3 has no incoming edge; nodes 0,1,2 form a self-looped path
    ei = torch.tensor([[0, 1, 2, 0, 1], [0, 1, 2, 1, 2]], dtype=torch.long)
    torch.manual_seed(10)
    x = torch.randn(N, d)
    out = attn(x, ei)
    assert torch.isfinite(out).all()
    # node 3 had no neighbor -> its aggregated row is exactly zero before W_O;
    # after the (bias-free) W_O it stays zero.
    assert torch.allclose(out[3], torch.zeros(d), atol=1e-6), out[3]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"{name}: PASS")
    print("done")
