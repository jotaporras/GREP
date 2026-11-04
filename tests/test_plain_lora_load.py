"""loaders.load_lora_state_dict: saved LoRA weights must actually land (2026-11-04).

Before the fix, the plain-LLM loader fed ``save_pretrained`` keys
(``...lora_A.weight``) to a PeftModel whose keys are ``...lora_A.default.weight``
with ``strict=False``: nothing loaded and every text-arm cross_eval ran the
untrained base model. These tests use PEFT's own save format on a tiny model.
"""

import pytest
import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file

from prism.models.loaders import load_lora_state_dict


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(8, 8)
        self.v_proj = torch.nn.Linear(8, 8)

    def forward(self, x):
        return self.v_proj(self.q_proj(x))


def _peft(seed):
    torch.manual_seed(seed)
    return get_peft_model(Tiny(), LoraConfig(r=2, target_modules=["q_proj", "v_proj"]))


def _lora(model):
    return {k: v.clone() for k, v in model.state_dict().items() if "lora_" in k}


def test_saved_adapter_loads_into_fresh_model(tmp_path):
    trained = _peft(0)
    with torch.no_grad():  # make B nonzero, as after training
        for n, p in trained.named_parameters():
            if "lora_B" in n:
                p.normal_()
    trained.save_pretrained(tmp_path)
    saved = load_file(tmp_path / "adapter_model.safetensors")
    assert not any(".default." in k for k in saved)  # the format that broke the old load

    fresh = _peft(1)
    n = load_lora_state_dict(fresh, saved, str(tmp_path))
    assert n == len(saved)
    want, got = _lora(trained), _lora(fresh)
    assert want.keys() == got.keys()
    for k in want:
        torch.testing.assert_close(got[k], want[k])


def test_mismatched_keys_fail_loud(tmp_path):
    trained = _peft(0)
    trained.save_pretrained(tmp_path)
    saved = load_file(tmp_path / "adapter_model.safetensors")
    wrong = {k.replace("q_proj", "k_proj"): v for k, v in saved.items()}
    with pytest.raises(RuntimeError, match="did not load"):
        load_lora_state_dict(_peft(1), wrong, str(tmp_path))
