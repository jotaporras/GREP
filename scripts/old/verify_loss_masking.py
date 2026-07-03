#!/usr/bin/env python
"""Verify the `loss_target` token-masking framework end to end.

Proves that the supervised-token spans built in
``prism.data.data.preprocess_dataset`` (`assistant_idx` for
``loss_target='responses'`, `edge_list_idx` for ``loss_target='edge_list'``) are
correct, survive the collator, and actually restrict the next-token loss to the
intended tokens.

Run locally with the cached small Gemma tokenizer + a local SPINE conversations
JSON (defaults below), or point it at the real e8 split on the cluster::

    # local (default)
    python scripts/verify_loss_masking.py

    # cluster, real e8 gemma split + the configured base tokenizer
    python scripts/verify_loss_masking.py \
        --tokenizer google/gemma-4-12B-it \
        --data data/revised/gen/nav100_n30_gemma_data/split/formatted_all_new__train.json

Sections
  A  unit tests (synthetic, pure logic): assistant_token_positions (multi- and
     single-turn), _mask_labels_to_positions (+ all-empty fallback), TrainConfig
     validation.
  B  real-data masking: decode the kept positions for `responses`, `edge_list`,
     `all` on real examples for BOTH architecture='llm' and 'rpearl_llm';
     prints the decoded spans for inspection.
  C  collator integration: index columns survive batching; masked non-(-100)
     labels == union of the per-example index lists.
  D  loss behaviour on a real tiny model (Qwen2.5-0.5B): masked loss differs
     from 'all'; perturbing a non-supervised label leaves the masked loss
     unchanged; HF loss == manual mean-CE over the kept span. Skip with
     --skip-model.

Exit code is nonzero if any check FAILs.
"""
import argparse
import json
import os
import sys
import warnings

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import datasets  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from prism.data import data  # noqa: E402
from prism.training import train_v2  # noqa: E402

datasets.disable_progress_bars()
datasets.utils.logging.set_verbosity_error()


# ---------------------------------------------------------------------------
# tiny PASS/FAIL harness
# ---------------------------------------------------------------------------
_RESULTS = []


def check(name, ok, detail=""):
    _RESULTS.append(bool(ok))
    tag = "PASS" if ok else "FAIL"
    line = f"  [{tag}] {name}"
    if detail:
        line += f"  --  {detail}"
    print(line)
    return ok


def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def special_token_strings(tok):
    """Rendered text of every special id (used to subtract template scaffolding)."""
    out = []
    for sid in getattr(tok, "all_special_ids", []) or []:
        try:
            out.append(tok.decode([sid]))
        except Exception:
            pass
    return [s for s in out if s]


def decode_positions(tok, input_ids, positions):
    return tok.decode([input_ids[p] for p in positions])


def gold_content_positions(messages, input_ids, tok, role):
    """Independent gold: token positions whose chars fall inside any `role` turn's
    *content*, via offset mapping over the rendered text (trusted only when the
    re-tokenized ids round-trip to input_ids exactly). Returns set or None if the
    offset path is untrustworthy for this tokenizer."""
    full_text = tok.apply_chat_template(messages, tokenize=False)
    enc = tok(full_text, add_special_tokens=False, return_offsets_mapping=True)
    if list(enc["input_ids"]) != list(input_ids):
        return None
    offsets = enc["offset_mapping"]
    gold = set()
    cursor = 0
    for m in messages:
        if m.get("role") != role:
            continue
        c = m["content"]
        cs = full_text.find(c, cursor)
        if cs == -1:
            return None
        ce = cs + len(c)
        cursor = ce
        for i, (a, b) in enumerate(offsets):
            if b > cs and a < ce:
                gold.add(i)
    return gold


# ---------------------------------------------------------------------------
# A. unit tests (synthetic)
# ---------------------------------------------------------------------------
def section_A(tok):
    section("A. UNIT TESTS (synthetic messages + real tokenizer)")

    # A1 multi-turn assistant_token_positions: distinctive content so a leak is obvious
    SYS = "SYSTEM_SECRET_PROMPT_DO_NOT_TRAIN"
    U1 = "USER_QUESTION_ALPHA_xyzzy"
    A1 = "ASSISTANT_ANSWER_ONE_qqqq first reply"
    U2 = "USER_QUESTION_BETA_plugh"
    A2 = "ASSISTANT_ANSWER_TWO_rrrr second reply"
    messages = [
        {"role": "system", "content": SYS},
        {"role": "user", "content": U1},
        {"role": "assistant", "content": A1},
        {"role": "user", "content": U2},
        {"role": "assistant", "content": A2},
    ]
    ids = tok.apply_chat_template(messages, tokenize=True, return_dict=False)
    pos = data.assistant_token_positions(messages, ids, tok)
    decoded = decode_positions(tok, ids, pos)
    check("A1 multi-turn: BOTH assistant answers reconstructed",
          (A1 in decoded) and (A2 in decoded),
          f"A1 in span={A1 in decoded}, A2 in span={A2 in decoded}")
    check("A1 multi-turn: system prompt NOT in supervised span", SYS not in decoded)
    check("A1 multi-turn: user turns NOT in supervised span",
          (U1 not in decoded) and (U2 not in decoded))

    # exact upper bound: residue after removing the two contents is only template scaffold
    residue = decoded.replace(A1, "").replace(A2, "")
    for s in special_token_strings(tok):
        residue = residue.replace(s, "")
    check("A1 multi-turn: span is EXACTLY contents + template terminators (no extra content)",
          residue.strip() == "", f"residue={residue.strip()!r}")

    # boundary / off-by-one: gold content positions ⊆ code positions ⊆ gold+specials
    gold = gold_content_positions(messages, ids, tok, "assistant")
    if gold is not None:
        codeset = set(pos)
        extra = codeset - gold
        extra_nonspecial = [p for p in extra if ids[p] not in set(tok.all_special_ids or [])
                            and tok.decode([ids[p]]).strip() != ""]
        check("A1 boundary: every assistant-content token is supervised (gold ⊆ code)",
              gold.issubset(codeset), f"missing={len(gold - codeset)}")
        check("A1 boundary: no non-terminator token leaks in (code\\gold ⊆ specials/whitespace)",
              not extra_nonspecial, f"stray={extra_nonspecial[:5]}")
    else:
        print("    (offset-mapping gold unavailable for this tokenizer; skipped strict boundary)")

    # A2 single-turn
    s_msgs = [{"role": "user", "content": U1}, {"role": "assistant", "content": A1}]
    s_ids = tok.apply_chat_template(s_msgs, tokenize=True, return_dict=False)
    s_pos = data.assistant_token_positions(s_msgs, s_ids, tok)
    s_dec = decode_positions(tok, s_ids, s_pos)
    check("A2 single-turn: answer reconstructed, user not in span",
          (A1 in s_dec) and (U1 not in s_dec))

    # A3 _mask_labels_to_positions masks everything outside the listed positions
    mixin = train_v2.LossTargetMixin()
    labels = torch.arange(1, 25).reshape(2, 12)  # all >=0 (no -100 to start)
    idx_lists = [[1, 3, 5], [0, 11]]
    inp = {"labels": labels.clone()}
    mixin._mask_labels_to_positions(inp, idx_lists, "responses")
    out = inp["labels"]
    kept = [(b, p.item()) for b in range(out.shape[0]) for p in (out[b] != -100).nonzero().flatten()]
    expected = {(0, 1), (0, 3), (0, 5), (1, 0), (1, 11)}
    check("A3 mask: non-(-100) labels == exactly the listed positions",
          set(kept) == expected, f"kept={sorted(kept)}")
    check("A3 mask: kept label VALUES are unchanged (original token ids)",
          all(out[b, p] == labels[b, p] for (b, p) in expected))

    # A4 all-empty fallback: labels unchanged + a warning
    inp2 = {"labels": labels.clone()}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mixin._mask_labels_to_positions(inp2, [[], []], "responses")
    check("A4 fallback: all-empty idx leaves labels UNCHANGED",
          torch.equal(inp2["labels"], labels))
    check("A4 fallback: a warning is emitted",
          any("no supervised tokens" in str(x.message) for x in w),
          f"{len(w)} warning(s)")

    # A5 TrainConfig validation
    def mk(**kw):
        base = dict(name="t", checkpoint_dir="/tmp/x", data="x.json")
        base.update(kw)
        return train_v2.TrainConfig(**base)
    ok_default = True
    try:
        mk()  # loss_target='all' default
    except Exception as e:
        ok_default = False
    check("A5 config: default loss_target='all' is valid", ok_default)
    try:
        mk(loss_target="bogus"); raised = False
    except ValueError:
        raised = True
    check("A5 config: invalid loss_target raises ValueError", raised)
    try:
        mk(loss_target="edge_list", text_edge_list="none"); raised = False
    except ValueError:
        raised = True
    check("A5 config: edge_list + text_edge_list='none' raises ValueError", raised)
    try:
        mk(loss_target="edge_list", text_edge_list="present"); ok = True
    except Exception:
        ok = False
    check("A5 config: edge_list + text_edge_list='present' is valid", ok)


# ---------------------------------------------------------------------------
# B. real-data masking verification
# ---------------------------------------------------------------------------
def load_records(path, n_wanted):
    with open(path) as f:
        raw = json.load(f)
    recs = raw if isinstance(raw, list) else [raw]
    recs = [r for r in recs if isinstance(r, dict) and "conversations" in r]
    # prefer multi-assistant-turn records first (more adversarial), then fill.
    def nass(r):
        return sum(1 for t in r["conversations"]
                   if (t.get("role") or t.get("from")) in ("assistant", "gpt", "model"))
    recs.sort(key=lambda r: -nass(r))
    return recs[:n_wanted]


def verify_example_responses(tok, ex, specials, label):
    ids = ex["input_ids"]
    pos = ex["assistant_idx"]
    contents = [m["content"] for m in ex["messages"] if m["role"] == "assistant"]
    decoded = decode_positions(tok, ids, pos)
    cover = all(c.strip() in decoded for c in contents)
    sys_leak = any("navigation planner" in decoded.lower() for _ in [0])
    user_leak = any(m["content"].strip() and m["content"].strip() in decoded
                    for m in ex["messages"] if m["role"] == "user")
    residue = decoded
    for c in contents:
        residue = residue.replace(c, "")
    for s in specials:
        residue = residue.replace(s, "")
    exact = residue.strip() == ""
    ok = cover and (not sys_leak) and (not user_leak) and exact
    check(f"B[{label}] responses: span == {len(contents)} assistant turn(s), no sys/user leak",
          ok, f"cover={cover} sysleak={sys_leak} userleak={user_leak} exact={exact}")
    return decoded


def verify_example_edges(tok, ex, specials, label):
    ids = ex["input_ids"]
    pos = ex["edge_list_idx"]
    if not pos:
        check(f"B[{label}] edge_list: located the bullet block", False, "edge_list_idx empty")
        return ""
    decoded = decode_positions(tok, ids, pos)
    full_text = tok.apply_chat_template(ex["messages"], tokenize=False)
    s = full_text.index("• Region Edges:")
    o = full_text.find("• Object Edges:", s)
    nl = full_text.find("\n", o)
    e = nl if nl != -1 else len(full_text)
    gold = full_text[s:e]
    # The edge block is the last line of the system turn, so the char span up to the
    # next newline includes the rendered end-of-turn special token (e.g. Gemma's
    # '<turn|>'). The implementation correctly EXCLUDES that special token, so strip
    # any trailing special-token renderings from the gold before the exact compare.
    changed = True
    while changed:
        changed = False
        gold = gold.rstrip()
        for sp in specials:
            if sp and gold.endswith(sp):
                gold = gold[:-len(sp)]
                changed = True
    no_special = all(ids[p] not in set(tok.all_special_ids or []) for p in pos)
    # boundary: token just AFTER the span must be the line terminator (special or newline)
    last = pos[-1]
    after_ok = True
    if last + 1 < len(ids):
        nxt = tok.decode([ids[last + 1]])
        after_ok = (ids[last + 1] in set(tok.all_special_ids or [])) or ("\n" in nxt)
    exact = decoded.strip() == gold.strip()
    starts_region = decoded.lstrip().startswith("• Region Edges:")
    has_object = "• Object Edges:" in decoded
    ok = exact and no_special and after_ok and starts_region and has_object
    check(f"B[{label}] edge_list: span == exactly the two edge bullets, no special tokens",
          ok, f"exact={exact} no_special={no_special} boundary_after={after_ok}")
    return decoded


def section_B(tok, data_path, n_examples, n_print):
    section(f"B. REAL-DATA MASKING  (data={data_path})")
    specials = special_token_strings(tok)
    records = load_records(data_path, n_examples)
    check("B load: found real `conversations` records", len(records) > 0,
          f"{len(records)} record(s)")
    if not records:
        return

    printed = 0
    for arch in ("llm", "rpearl_llm"):
        print(f"\n-- architecture='{arch}', text_edge_list='present' --")
        ds = datasets.Dataset.from_list([dict(r) for r in records])
        try:
            out = data.preprocess_dataset(ds, tok, architecture=arch,
                                          text_edge_list="present")
        except Exception as e:
            check(f"B[{arch}] preprocess_dataset ran", False, f"{type(e).__name__}: {e}")
            continue
        check(f"B[{arch}] preprocess_dataset produced index columns",
              {"assistant_idx", "edge_list_idx"}.issubset(out.column_names))

        for i in range(len(out)):
            ex = out[i]
            dec_r = verify_example_responses(tok, ex, specials, f"{arch}#{i}")
            dec_e = verify_example_edges(tok, ex, specials, f"{arch}#{i}")
            # loss_target='all' sanity: no masking column consulted -> full non-pad span
            n_nonpad = int(sum(ex["attention_mask"]))
            check(f"B[{arch}#{i}] all: 'all' uses no index column "
                  f"(full {n_nonpad}-token sequence supervised)",
                  train_v2._LOSS_TARGET_COLUMN.get("all") is None)
            if printed < n_print and arch == "llm":
                print(f"\n   ----- decoded spans, example #{i} (arch={arch}) -----")
                print("   [responses span]:")
                print("   " + repr(dec_r[:700]))
                print("   [edge_list span]:")
                print("   " + repr(dec_e[:400]))
                printed += 1
    return records


# ---------------------------------------------------------------------------
# C. collator integration
# ---------------------------------------------------------------------------
def section_C(tok, records):
    section("C. COLLATOR INTEGRATION (index columns survive batching)")
    ds = datasets.Dataset.from_list([dict(r) for r in records[:4]])
    out = data.preprocess_dataset(ds, tok, architecture="llm", text_edge_list="present")
    feats = [out[i] for i in range(len(out))]
    collator = data.TokenIndexCollator(tok, mlm=False)
    batch = collator(feats)

    check("C: assistant_idx survives as a per-example list",
          isinstance(batch.get("assistant_idx"), list) and len(batch["assistant_idx"]) == len(feats))
    check("C: edge_list_idx survives as a per-example list",
          isinstance(batch.get("edge_list_idx"), list) and len(batch["edge_list_idx"]) == len(feats))

    mixin = train_v2.LossTargetMixin()
    for target, col in (("responses", "assistant_idx"), ("edge_list", "edge_list_idx")):
        masked = {"labels": batch["labels"].clone()}
        mixin._mask_labels_to_positions(masked, batch[col], target)
        S = masked["labels"].shape[1]
        ok = True
        for b, idx in enumerate(batch[col]):
            kept = set((masked["labels"][b] != -100).nonzero().flatten().tolist())
            want = {p for p in idx if 0 <= p < S}
            if kept != want:
                ok = False
                break
        check(f"C: after masking ({target}), non-(-100) labels == union of per-example {col}", ok)


# ---------------------------------------------------------------------------
# D. real tiny-model loss behaviour
# ---------------------------------------------------------------------------
def manual_mean_ce(logits, labels):
    """Reference next-token mean-CE over labels != -100 (HF ForCausalLMLoss, reduction='mean')."""
    sl = logits[:, :-1, :].contiguous()
    tg = labels[:, 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(
        sl.view(-1, sl.size(-1)).float(), tg.view(-1), ignore_index=-100, reduction="mean")
    return loss


def section_D(model_id, records, max_seq):
    section(f"D. LOSS BEHAVIOUR on a real tiny model ({model_id})")
    from transformers import AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to(device).eval()

    ds = datasets.Dataset.from_list([dict(r) for r in records[:2]])
    out = data.preprocess_dataset(ds, tok, architecture="llm", text_edge_list="present")
    feats = [out[i] for i in range(len(out))]
    collator = data.TokenIndexCollator(tok, mlm=False)
    batch = collator(feats)
    input_ids = batch["input_ids"].to(device)
    attn = batch["attention_mask"].to(device)
    full_labels = batch["labels"].to(device)
    assistant_idx = batch["assistant_idx"]

    mixin = train_v2.LossTargetMixin()
    masked_in = {"labels": full_labels.clone()}
    mixin._mask_labels_to_positions(masked_in, assistant_idx, "responses")
    resp_labels = masked_in["labels"]

    n_all = int((full_labels != -100).sum())
    n_resp = int((resp_labels != -100).sum())
    check("D0: responses mask supervises a strict subset of 'all'",
          0 < n_resp < n_all, f"resp={n_resp} all={n_all}")

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attn).logits

    loss_all = manual_mean_ce(logits, full_labels).item()
    loss_resp = manual_mean_ce(logits, resp_labels).item()
    check("D1: masked (responses) loss != full ('all') loss  [masking changes the loss]",
          abs(loss_all - loss_resp) > 1e-5, f"all={loss_all:.5f} responses={loss_resp:.5f}")

    # D2: perturb a NON-supervised label -> 'all' loss changes, responses loss does NOT.
    nonsup = (resp_labels[0] == -100).nonzero().flatten()
    # pick a non-supervised position that is also a real (non-pad) token so 'all' grades it
    cand = [p.item() for p in nonsup if full_labels[0, p] != -100]
    pos0 = cand[len(cand) // 2]
    full_p = full_labels.clone()
    full_p[0, pos0] = (full_p[0, pos0] + 7) % logits.shape[-1]  # corrupt one non-assistant gold token
    resp_p_in = {"labels": full_p.clone()}
    mixin._mask_labels_to_positions(resp_p_in, assistant_idx, "responses")
    resp_p = resp_p_in["labels"]
    loss_all_p = manual_mean_ce(logits, full_p).item()
    loss_resp_p = manual_mean_ce(logits, resp_p).item()
    check("D2: perturbing a non-assistant label CHANGES the 'all' loss",
          abs(loss_all_p - loss_all) > 1e-6, f"all {loss_all:.6f}->{loss_all_p:.6f}")
    check("D2: the SAME perturbation leaves the responses-masked loss UNCHANGED",
          abs(loss_resp_p - loss_resp) < 1e-9, f"responses {loss_resp:.6f}->{loss_resp_p:.6f}")

    # D3: HF model loss with the masked labels == manual mean-CE over the kept span.
    with torch.no_grad():
        hf_loss = model(input_ids=input_ids, attention_mask=attn, labels=resp_labels).loss.item()
    check("D3: HF model(labels=masked).loss == manual mean-CE over kept tokens",
          abs(hf_loss - loss_resp) < 1e-3, f"hf={hf_loss:.5f} manual={loss_resp:.5f}")

    # D4: _set_loss_target disables the token-weighted loss path for masked targets.
    class _T(train_v2.LossTargetMixin):
        def __init__(self):
            self.model_accepts_loss_kwargs = True
    t_all, t_resp, t_edge = _T(), _T(), _T()
    t_all._set_loss_target("all")
    t_resp._set_loss_target("responses")
    t_edge._set_loss_target("edge_list")
    check("D4: _set_loss_target keeps loss-kwargs ON for 'all'",
          t_all.model_accepts_loss_kwargs is True)
    check("D4: _set_loss_target turns loss-kwargs OFF for 'responses' (reduction='mean' over span)",
          t_resp.model_accepts_loss_kwargs is False)
    check("D4: _set_loss_target turns loss-kwargs OFF for 'edge_list'",
          t_edge.model_accepts_loss_kwargs is False)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/gpt_gen_formatted.json",
                    help="local SPINE `conversations` JSON (point at the e8 split on cluster)")
    ap.add_argument("--tokenizer", default="google/gemma-4-E4B-it",
                    help="HF tokenizer id (same chat-template family as the base model)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                    help="small cached model for the real-forward loss check (section D)")
    ap.add_argument("--num-examples", type=int, default=4)
    ap.add_argument("--num-print", type=int, default=3)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--skip-model", action="store_true",
                    help="skip section D (no model forward)")
    args = ap.parse_args()

    print(f"tokenizer = {args.tokenizer}")
    print(f"data      = {args.data}")
    print(f"model (D) = {'(skipped)' if args.skip_model else args.model}")

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    # Match the training pipeline (train_v2.py): the precomputed index columns are
    # only valid in the padded batch under RIGHT padding (e.g. Gemma defaults to LEFT).
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    section_A(tok)
    records = section_B(tok, args.data, args.num_examples, args.num_print)
    if records:
        section_C(tok, records)
        if not args.skip_model:
            try:
                section_D(args.model, records, args.max_seq)
            except Exception as e:
                import traceback
                traceback.print_exc()
                check("D: tiny-model section ran", False, f"{type(e).__name__}: {e}")

    section("SUMMARY")
    n = len(_RESULTS)
    passed = sum(_RESULTS)
    print(f"  {passed}/{n} checks PASSED")
    if passed != n:
        print("  RESULT: FAIL")
        sys.exit(1)
    print("  RESULT: ALL PASS")


if __name__ == "__main__":
    main()
