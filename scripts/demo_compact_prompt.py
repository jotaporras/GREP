"""Demo: the compact plain-text prompt format fed to the GREP-PRISM models.

Shows exactly what training and eval prompts become once the verbose SPINE JSON
(long system prompt + few-shot ICL examples + full scene-graph JSON) is translated
to the compact format the LLM consumes:

  * the SPINE system prompt and ALL ICL examples are DROPPED (the format is taught
    by SFT, and dropping ICL keeps train and eval symmetric);
  * the scene graph becomes a compact ``Scene graph:`` block (bulleted node-name
    lists + robot location, NO edges — the GNN supplies connectivity) and is
    placed in a leading ``system`` message, ABOVE the first task;
  * tasks then stack as ``user``/``assistant`` pairs in the same conversation per
    graph; the assistant target wraps reasoning in
    ``<think>Relevant graph: …\\n\\nReasoning: …</think>`` followed by the bare plan.

The ``User:``/``Assistant:`` turn delimiters are produced by the tokenizer's chat
template (native role special tokens), NOT literal text — so with ``--tokenizer``
the demo renders the byte-for-byte prompt via ``apply_chat_template``; without it,
it prints the raw messages plus an illustrative ``Role:`` view.

Pure pre-processing; does NOT modify anything under data/.

Run:  PYTHONPATH=src python scripts/demo_compact_prompt.py
      PYTHONPATH=src python scripts/demo_compact_prompt.py --tokenizer meta-llama/Llama-3.2-3B-Instruct
"""

import argparse
import json
from pathlib import Path

from prism.data.compact_prompt import (
    assemble_training_conversation,
    format_eval_messages,
    format_training_messages,
    render,
    spine_to_compact_messages,
    strip_icl,
    try_load_json,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "gen" / "nav100_n30_gemma_data"
DEFAULT_PLAN = DATA_DIR / "generated_plans" / "sample_000_000.json"
DEFAULT_GRAPH = DATA_DIR / "populated_graphs" / "data_gen_000.json"
SECOND_GRAPH = DATA_DIR / "populated_graphs" / "data_gen_001.json"


def _rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _roles(messages) -> str:
    return " ".join(m["role"][0].upper() for m in messages)  # e.g. "S U A U A"


def _show(messages, tokenizer, add_generation_prompt: bool) -> None:
    print("\n-- messages (roles handled by apply_chat_template) --")
    print(json.dumps(messages, indent=2))
    label = "apply_chat_template" if tokenizer is not None else "illustrative Role: view"
    if add_generation_prompt:
        label += " (+generation prompt)"
    print(f"\n-- prompt the model receives [{label}] --")
    print(render(messages, tokenizer=tokenizer, add_generation_prompt=add_generation_prompt))


def _make_counter(tokenizer):
    if tokenizer is not None:
        return "tokens", lambda s: len(tokenizer.encode(s, add_special_tokens=False))
    return "chars", len


def _reduction_summary(before_msgs, after_msgs, tokenizer) -> None:
    label, count = _make_counter(tokenizer)
    b = count(render(before_msgs, tokenizer=tokenizer))
    a = count(render(after_msgs, tokenizer=tokenizer))
    saved = 100.0 * (b - a) / b if b else 0.0
    print(f"\n  context ({label}):  before={b:,}  after={a:,}  saved={saved:.1f}%")


def _verbose_baseline(graph_dict, task):
    """Illustrative 'before' baseline: the full graph dict restated per task."""
    return [{"role": "user", "content": f"task: {task}\nScene graph:{graph_dict}"}]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-sample", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--graph-sample", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--tokenizer", type=str, default="")
    args = parser.parse_args()

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    payload = try_load_json(args.graph_sample)
    graph_dict, tasks = payload["graph"], payload["tasks"]
    task = tasks[args.task_index]["task"]

    # --- 1. Training prompt: a real rollout -> compact (system + ICL dropped) -----
    conversation = try_load_json(args.plan_sample)
    training_messages = format_training_messages(conversation)
    _rule(f"TRAINING PROMPT  (from {args.plan_sample.name})")
    _show(training_messages, tokenizer, add_generation_prompt=False)
    _reduction_summary(strip_icl(conversation), training_messages, tokenizer)

    # --- 2. Eval prompt: graph + task -> compact (open assistant turn) ------------
    eval_messages = format_eval_messages(graph_dict, task)
    _rule(f"EVAL PROMPT  (from {args.graph_sample.name}, task {args.task_index})")
    _show(eval_messages, tokenizer, add_generation_prompt=True)  # mirrors query_llm

    # --- 3. MULTI-TURN translation: system + ICL dropped, planning turns kept -----
    # Synthesize a realistic eval-time SPINE conversation: system + one ICL example
    # (its own graph) + the real query + a receding-horizon planning continuation
    # (assistant plan -> updates -> assistant replan). spine_to_compact_messages is
    # exactly what the eval client feeds the LLM.
    icl_graph = try_load_json(SECOND_GRAPH)["graph"]
    answer_json = lambda r, p: json.dumps(
        {"primary_goal": "g", "relevant_graph": r, "reasoning": "...", "plan": f"[answer({p})]"})
    spine_convo = [
        {"role": "system", "content": "<5,992-char SPINE system prompt: API + JSON schema + advice>"},
        {"role": "user", "content": f"task: an ICL example task\nScene graph:{json.dumps(icl_graph)}"},
        {"role": "assistant", "content": answer_json("a_1, b_1", "a_1 -> b_1")},
        {"role": "user", "content": f"{task}\nAdvice: \n- ...\n\nScene graph:{json.dumps(graph_dict)}"},
        {"role": "assistant", "content": answer_json("hub_1, mess_hall_1", "hub_1 -> mess_hall_1")},
        {"role": "user", "content": "updates:[no_updates()]"},
        {"role": "assistant", "content": answer_json("hub_1, mess_hall_1", "hub_1 -> mess_hall_1")},
    ]
    compact = spine_to_compact_messages(spine_convo)
    _rule("MULTI-TURN SPINE -> COMPACT  (SPINE system + ICL dropped; graph hoisted to system)")
    print(f"  input  roles: [{_roles(spine_convo)}]  ({len(spine_convo)} turns: SPINE system + 1 ICL pair + query + replan loop)")
    print(f"  output roles: [{_roles(compact)}]  ({len(compact)} turns: system(graph) + query + planning continuation)")
    sys_msgs = [m for m in compact if m["role"] == "system"]
    assert len(sys_msgs) == 1 and sys_msgs[0]["content"].startswith("Scene graph:"), "graph not hoisted to one system msg"
    assert "<5,992-char SPINE system prompt" not in render(compact), "verbose SPINE system leaked"
    assert sum(c["content"].count("Scene graph:") for c in compact) == 1, "ICL graph leaked"
    print("  -> ONE system 'Scene graph:' block (the query graph); SPINE system + ICL gone.")
    _show(compact, tokenizer, add_generation_prompt=False)

    # --- 4. Multi-task over ONE graph: graph in system, tasks stacked -------------
    all_tasks = [t["task"] for t in tasks]
    multi_eval = format_eval_messages(graph_dict, all_tasks)
    roles_e = [m["role"] for m in multi_eval]
    _rule(f"MULTI-TASK EVAL  ({len(all_tasks)} tasks stacked under one shared graph)")
    print(f"  {len(multi_eval)} turns: 1 system(graph) + {roles_e.count('user')} stacked user tasks; "
          f"'Scene graph:' block appears {sum(c['content'].count('Scene graph:') for c in multi_eval)}x (once, in system).")
    _reduction_summary([m for t in all_tasks for m in _verbose_baseline(graph_dict, t)], multi_eval, tokenizer)

    plan_dir = args.plan_sample.parent
    graph_idx = args.plan_sample.stem.split("_")[1]
    rollout_files = sorted(plan_dir.glob(f"sample_{graph_idx}_*.json"))
    if len(rollout_files) > 1:
        rollouts = [try_load_json(p) for p in rollout_files]
        multi_train = assemble_training_conversation(rollouts)
        roles = [m["role"] for m in multi_train]
        _rule(f"MULTI-TASK TRAINING  ({len(rollout_files)} rollouts, graph {graph_idx}, tasks stacked under one graph)")
        print(f"  {len(multi_train)} turns ({roles.count('system')} system(graph) / {roles.count('user')} user / "
              f"{roles.count('assistant')} assistant); 'Scene graph:' block appears "
              f"{sum(m['content'].count('Scene graph:') for m in multi_train)}x (once, in system).")
        _reduction_summary([m for r in rollouts for m in strip_icl(r)], multi_train, tokenizer)

        # Rendered view: how multiple tasks per graph actually look stacked.
        few = assemble_training_conversation(rollouts[:3])
        n_tasks = sum(1 for m in few if m["role"] == "user")
        _rule(f"MULTIPLE TASKS PER GRAPH — rendered ({n_tasks} tasks stacked under one shared graph)")
        print(f"  one system(graph) message, then {n_tasks} (user task, assistant answer) pairs in the same conversation:")
        _show(few, tokenizer, add_generation_prompt=False)

    print()


if __name__ == "__main__":
    main()
