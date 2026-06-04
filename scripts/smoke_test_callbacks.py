"""Smoke test for prism.eval.callbacks.GradientDebugCallback.

Exercises BOTH architecture branches end to end — install hooks, run a real
forward+backward, capture grad norms, fire on_log — and asserts the W&B payload
is well-formed and populated (grad norms > 0, signal norms finite, injection
count correct):

  A. GraphAugmentedLLM + R-PEARL-only encoder  (no GT blocks)
  B. GraphAugmentedLLM + GraphTransformer encoder  (GT sub-norms present)

Run:  PYTHONPATH=src python scripts/smoke_test_callbacks.py
Exits non-zero on any failed assertion.
"""

import sys
import types
from unittest.mock import MagicMock

# `callbacks.py` does `from prism.eval import evaluate`, and `evaluate` pulls a
# heavy import chain (spine, datasets↔huggingface_hub) that isn't importable in
# this env and is irrelevant to GradientDebugCallback (only EvalCallback uses it,
# at runtime). Register a stub submodule so the import binds without executing
# the real evaluate.py. PEP 562 __getattr__ returns a MagicMock for any symbol.
_ev = types.ModuleType("prism.eval.evaluate")
_ev.__getattr__ = lambda _attr: MagicMock()
sys.modules.setdefault("prism.eval.evaluate", _ev)

import torch
from torch import nn
from torch_geometric.data import Data
from transformers import LlamaConfig, LlamaForCausalLM

import prism.eval.callbacks as cbmod
from prism.eval.callbacks import GradientDebugCallback
from prism.models.gnn_llm import GraphAugmentedLLM
from prism.models.r_pearl import RandomGNNPositionalEncodings
from prism.models.gt import GraphTransformer

torch.manual_seed(0)

PASS, FAIL = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
_failures = []

HIDDEN, HEAD_DIM, NH, NKV, DMODEL = 64, 8, 8, 4, 16
N, SEQ = 5, 14
INJECTION = {0: [(2, 3)], 1: [(4, 5)], 2: [(6, 7)], 3: [(8, 9)], 4: [(10, 11)]}
N_INJ = sum(len(s) for spans in INJECTION.values() for s in [spans])  # 5 spans


def check(name, cond, detail=""):
    print(f"  [{PASS if cond else FAIL}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        _failures.append(name)


# --------------------------------------------------------------------------- #
# Shared fixtures
# --------------------------------------------------------------------------- #
class FakeWandb:
    """Stand-in for the wandb module: truthy .run + capturing .log()."""
    def __init__(self):
        self.run = object()
        self.logged = []

    def log(self, metrics, step=None):
        self.logged.append((step, dict(metrics)))


def fake_state():
    s = types.SimpleNamespace()
    s.global_step = 5
    s.epoch = 1.0
    s.log_history = [{"learning_rate": 1e-4}]
    return s


def tiny_llm():
    cfg = LlamaConfig(
        vocab_size=128, hidden_size=HIDDEN, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=NH, num_key_value_heads=NKV,
        head_dim=HEAD_DIM, max_position_embeddings=64, attn_implementation="eager",
    )
    return LlamaForCausalLM(cfg).eval()


_EI = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 0],
                    [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]], dtype=torch.long)


def real_graph():
    g = Data(x=torch.randn(N, 1), edge_index=_EI.clone())
    g.num_nodes = N
    return g


def rpearl_encoder():
    return RandomGNNPositionalEncodings(
        pe_hidden_channels=32, pe_num_layers=2, d_model=DMODEL,
        num_samples=6, dropout=0.0, k=2, eps=1e-6, use_layer_norm=True)


def gt_encoder():
    return GraphTransformer(
        num_layers=2, pe_hidden_channels=32, pe_num_layers=2, d_model=DMODEL,
        heads=4, num_samples=6, dropout=0.0, k_pe=2, k_gt=2, eps=1e-6,
        use_layer_norm=True)


def run_callback(model, graphs, *, with_grads=True):
    """Install hooks, do one forward+backward, capture grad norms, fire on_log.

    Returns the metrics dict the callback handed to wandb.log.
    """
    fake = FakeWandb()
    cbmod.wandb = fake  # redirect the module-level wandb reference
    cb = GradientDebugCallback()
    state = fake_state()

    cb.on_train_begin(None, state, None, model=model)

    ids = torch.randint(0, 128, (1, SEQ))
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                labels=ids.clone(), graphs=graphs, injection_maps=[INJECTION])
    if with_grads:
        out.loss.backward()
        cb._capture_grad_norms(model)

    cb.on_log(None, state, None, model=model)
    assert fake.logged, "on_log did not call wandb.log"
    return cb, fake.logged[-1][1]


# --------------------------------------------------------------------------- #
# A — GraphAugmentedLLM + R-PEARL (legacy, no GT blocks)
# --------------------------------------------------------------------------- #
def test_legacy_rpearl():
    print("Test A: GradientDebugCallback on GraphAugmentedLLM + R-PEARL")
    model = GraphAugmentedLLM(tiny_llm(), rpearl_encoder(), d_model=DMODEL, eps=1e-6).train()
    cb, m = run_callback(model, [real_graph()])

    for k in ("debug/grad_norm_gnn", "debug/grad_norm_pe_proj", "debug/grad_norm_pe_gain",
              "debug/grad_norm_lora", "debug/pe_output_norm", "debug/pe_has_nan",
              "debug/embedding_norm", "debug/num_injections", "debug/pe_gain",
              "debug/grad_norm_pe_gain", "debug/lr"):
        check(f"has {k}", k in m)
    check("num_injections correct", m.get("debug/num_injections") == N_INJ,
          f"{m.get('debug/num_injections')} vs {N_INJ}")
    check("grad_norm_gnn > 0", m.get("debug/grad_norm_gnn", 0) > 0, f"{m.get('debug/grad_norm_gnn'):.2e}")
    check("grad_norm_pe_proj > 0", m.get("debug/grad_norm_pe_proj", 0) > 0)
    check("grad_norm_pe_gain > 0", m.get("debug/grad_norm_pe_gain", 0) > 0)
    check("grad_norm_lora > 0", m.get("debug/grad_norm_lora", 0) > 0)
    check("pe_output_norm finite & set", 0 < m.get("debug/pe_output_norm", 0) < float("inf"),
          f"{m.get('debug/pe_output_norm'):.3e}")
    check("pe_has_nan == 0", m.get("debug/pe_has_nan") == 0)
    check("no GT sub-norms for R-PEARL-only", "debug/grad_norm_gt_blocks" not in m)


# --------------------------------------------------------------------------- #
# B — GraphAugmentedLLM + GraphTransformer (legacy, GT sub-norms present)
# --------------------------------------------------------------------------- #
def test_legacy_gt():
    print("Test B: GradientDebugCallback on GraphAugmentedLLM + GraphTransformer")
    model = GraphAugmentedLLM(tiny_llm(), gt_encoder(), d_model=DMODEL, eps=1e-6).train()
    cb, m = run_callback(model, [real_graph()])

    for k in ("debug/grad_norm_rpearl", "debug/grad_norm_gt_blocks",
              "debug/grad_norm_gt_output_norm"):
        check(f"has {k}", k in m)
    check("grad_norm_gt_blocks > 0", m.get("debug/grad_norm_gt_blocks", 0) > 0,
          f"{m.get('debug/grad_norm_gt_blocks'):.2e}")
    check("grad_norm_rpearl > 0", m.get("debug/grad_norm_rpearl", 0) > 0)
    check("pe_output_norm finite", 0 < m.get("debug/pe_output_norm", 0) < float("inf"))


def main():
    test_legacy_rpearl()
    test_legacy_gt()

    print()
    if _failures:
        print(f"{FAIL}: {len(_failures)} check(s) failed: {_failures}")
        sys.exit(1)
    print(f"{PASS}: all callback smoke checks passed.")


if __name__ == "__main__":
    main()
