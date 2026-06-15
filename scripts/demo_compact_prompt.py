"""Demo: the compact plain-text prompt format fed to the GREP-PRISM models.

Loads a real nav100 sample and prints exactly what the training and eval prompts
look like once converted from the verbose SPINE JSON (long system prompt + full
scene-graph JSON) into the compact format: a chat ``messages`` list whose user
turn carries ``<task>`` + the ``Scene graph:`` block and whose assistant turn
(training only) carries the reasoning + ``Relevant Nodes:`` + ``Plan:``.

IMPORTANT: the ``User:`` / ``Assistant:`` turn delimiters are produced by the
tokenizer's chat template (native role special tokens), NOT literal text — so
this demo renders through ``apply_chat_template`` when a tokenizer is given,
which is the byte-for-byte prompt the model receives. Without ``--tokenizer`` it
prints the raw messages plus an illustrative ``Role:`` view.

This is pure pre-processing and does NOT modify anything under data/.

Run:  PYTHONPATH=src python scripts/demo_compact_prompt.py
      PYTHONPATH=src python scripts/demo_compact_prompt.py --tokenizer meta-llama/Llama-3.2-3B-Instruct
      PYTHONPATH=src python scripts/demo_compact_prompt.py \
          --plan-sample data/gen/nav100_n30_gemma_data/generated_plans/sample_001_001.json \
          --graph-sample data/gen/nav100_n30_gemma_data/populated_graphs/data_gen_000.json \
          --task-index 0
"""

import argparse
import json
from pathlib import Path

from prism.data.compact_prompt import (
    assemble_training_conversation,
    format_eval_messages,
    format_training_messages,
    render,
    strip_icl,
    try_load_json,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "gen" / "nav100_n30_gemma_data"
DEFAULT_PLAN = DATA_DIR / "generated_plans" / "sample_000_000.json"
DEFAULT_GRAPH = DATA_DIR / "populated_graphs" / "data_gen_000.json"


def _rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _show(messages, tokenizer, add_generation_prompt: bool) -> None:
    """Print the raw messages list and the rendered chat-template string."""
    print("\n-- messages (roles handled by apply_chat_template) --")
    print(json.dumps(messages, indent=2))
    label = "apply_chat_template" if tokenizer is not None else "illustrative Role: view"
    if add_generation_prompt:
        label += " (+generation prompt)"
    print(f"\n-- prompt the model receives [{label}] --")
    print(render(messages, tokenizer=tokenizer, add_generation_prompt=add_generation_prompt))


def _make_counter(tokenizer):
    """Return a (label, count_fn) pair for the context-reduction summary."""
    if tokenizer is not None:
        return "tokens", lambda s: len(tokenizer.encode(s, add_special_tokens=False))
    return "chars", len


def _reduction_summary(before_msgs, after_msgs, tokenizer) -> None:
    label, count = _make_counter(tokenizer)
    before = render(before_msgs, tokenizer=tokenizer)
    after = render(after_msgs, tokenizer=tokenizer)
    b, a = count(before), count(after)
    saved = 100.0 * (b - a) / b if b else 0.0
    print(f"\n  context ({label}):  before={b:,}  after={a:,}  saved={saved:.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-sample", type=Path, default=DEFAULT_PLAN,
                        help="generated_plans/*.json conversation (training prompt source)")
    parser.add_argument("--graph-sample", type=Path, default=DEFAULT_GRAPH,
                        help="populated_graphs/data_gen_*.json (eval prompt source)")
    parser.add_argument("--task-index", type=int, default=0,
                        help="which task in the graph sample to render as an eval prompt")
    parser.add_argument("--tokenizer", type=str, default="",
                        help="HF model/tokenizer id; renders the true chat-template prompt "
                             "and token counts. Default: messages + illustrative view, char counts.")
    args = parser.parse_args()

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # --- Training prompt: from a real generated-plan rollout --------------------
    conversation = try_load_json(args.plan_sample)
    training_messages = format_training_messages(conversation)
    _rule(f"TRAINING PROMPT  (from {args.plan_sample.name})")
    _show(training_messages, tokenizer, add_generation_prompt=False)
    # Baseline = the current verbose prompt (system + scene-graph JSON + answer),
    # ICL prefix stripped, in the same messages shape.
    _reduction_summary(strip_icl(conversation), training_messages, tokenizer)

    # --- Eval prompt: from a real populated graph -------------------------------
    payload = try_load_json(args.graph_sample)
    graph_dict = payload["graph"]
    task = payload["tasks"][args.task_index]["task"]
    eval_messages = format_eval_messages(graph_dict, task)
    _rule(f"EVAL PROMPT  (from {args.graph_sample.name}, task {args.task_index})")
    # add_generation_prompt=True mirrors GraphAugmentedInMemoryLLM.query_llm at eval.
    _show(eval_messages, tokenizer, add_generation_prompt=True)

    # --- Multi-task over ONE graph: graph node lists stated exactly once ---------
    # Eval side: all tasks of the graph in a single conversation (graph once).
    all_tasks = [t["task"] for t in payload["tasks"]]
    multi_eval = format_eval_messages(graph_dict, all_tasks)
    _rule(f"MULTI-TASK EVAL  ({len(all_tasks)} tasks, {args.graph_sample.name}, one shared graph)")
    print(f"  {len(multi_eval)} user turns; 'Scene graph:' block appears "
          f"{sum(c['content'].count('Scene graph:') for c in multi_eval)}x (once).")
    _reduction_summary(
        # Baseline: the verbose per-task prompt restated for every task.
        [m for t in all_tasks for m in format_eval_messages_verbose_baseline(graph_dict, t)],
        multi_eval, tokenizer,
    )

    # Training side: assemble the graph's per-task rollouts into one example.
    plan_dir = args.plan_sample.parent
    graph_idx = args.plan_sample.stem.split("_")[1]
    rollout_files = sorted(plan_dir.glob(f"sample_{graph_idx}_*.json"))
    if len(rollout_files) > 1:
        rollouts = [try_load_json(p) for p in rollout_files]
        multi_train = assemble_training_conversation(rollouts)
        _rule(f"MULTI-TASK TRAINING  ({len(rollout_files)} rollouts, graph {graph_idx}, one shared graph)")
        roles = [m["role"] for m in multi_train]
        print(f"  {len(multi_train)} turns ({roles.count('user')} user / "
              f"{roles.count('assistant')} assistant); 'Scene graph:' block appears "
              f"{sum(m['content'].count('Scene graph:') for m in multi_train)}x (once).")
        _reduction_summary(
            [m for r in rollouts for m in strip_icl(r)], multi_train, tokenizer)

    print()


def format_eval_messages_verbose_baseline(graph_dict, task):
    """Illustrative 'before' baseline: the full graph dict restated per task."""
    return [{"role": "user", "content": f"task: {task}\nScene graph:{graph_dict}"}]


if __name__ == "__main__":
    main()
