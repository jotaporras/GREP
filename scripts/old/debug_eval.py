"""Standalone debug script for the eval chain.

Runs each step of the EvalCallback → SPINE → LLM pipeline individually,
printing full intermediate state so you can see exactly where parsing fails.

Usage:
    python scripts/debug_eval.py \
        --checkpoint_dir outputs/e1_baseline/dev_e1_rpearl_llm_r16 \
        --eval_data data/eval/eval_1_multi_step.json \
        --sample_idx 0
"""

import argparse
import json
import re

import torch

from prism.models import loaders
from prism.models import inference
from prism.eval import evaluate
from prism.data import graph_sim
from spine.mapping import graph_util
from spine.prompts import prompts as spine_prompts
from spine import spine


def main():
    parser = argparse.ArgumentParser(description="Debug eval chain step-by-step")
    parser.add_argument("--checkpoint_dir", required=True, help="Path to GraphAugmentedLLM checkpoint")
    parser.add_argument("--eval_data", required=True, help="Path to eval JSON file")
    parser.add_argument("--sample_idx", type=int, default=0, help="Which eval sample to debug (0-indexed)")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens to generate")
    args = parser.parse_args()

    # ── Step 1: Load model ──────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Load model")
    print("=" * 60)
    model, tokenizer = loaders.graph_augmented_llm_from_pretrained(args.checkpoint_dir)
    model.eval()
    device = next(model.parameters()).device
    print(f"  Model type: {type(model).__name__}")
    print(f"  Device: {device}")
    print(f"  Tokenizer vocab size: {len(tokenizer)}")

    # ── Step 2: Load eval sample ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Load eval sample")
    print("=" * 60)
    with open(args.eval_data) as f:
        data = json.load(f)

    tasks = data["tasks"]
    graph_data = data["graph"]

    if args.sample_idx >= len(tasks):
        print(f"  ERROR: sample_idx={args.sample_idx} but only {len(tasks)} tasks available")
        return

    import os
    entry = tasks[args.sample_idx]
    sample = evaluate.EvalSample(
        task=entry["task"],
        answer=entry["answer"],
        graph=graph_data,
        init_node=entry["init_node"],
        graph_name=os.path.splitext(os.path.basename(args.eval_data))[0],
    )
    print(f"  Task: {sample.task}")
    print(f"  Answer regex: {sample.answer}")
    print(f"  Init node: {sample.init_node}")
    print(f"  Graph has {len(graph_data.get('regions', []))} regions, {len(graph_data.get('objects', []))} objects")

    # ── Step 3: Set up graph (same as eval_model) ───────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Set up graph + SPINE (same as EvalCallback)")
    print("=" * 60)
    graph_handler = graph_util.GraphHandler("")
    graph_sim_inst = graph_sim.GraphSim(graph_handler)
    client = inference.GraphAugmentedInMemoryLLM(model=model, tokenizer=tokenizer, include_edges=False)
    llm_planner = spine.SPINE(graph=graph_sim_inst.partial_graph, client=client)

    # Reset graph with sample data (same as eval_model loop)
    graph_sim_inst.reset(graph_as_dict=sample.graph, current_location=sample.init_node)
    llm_planner.graph = graph_sim_inst.partial_graph

    print(f"  Graph handler current_location: {graph_sim_inst.partial_graph.current_location}")
    print(f"  Graph node count: {len(graph_sim_inst.partial_graph.graph.nodes)}")

    # ── Step 4: Construct SPINE prompt ──────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Construct SPINE prompt")
    print("=" * 60)
    scene_graph_json = graph_sim_inst.partial_graph.as_json_str
    print(f"  Scene graph JSON (first 500 chars):")
    print(f"  {scene_graph_json[:500]}")

    # Check robot_location key
    scene_dict = json.loads(scene_graph_json)
    print(f"\n  robot_location in JSON: {'robot_location' in scene_dict}")
    print(f"  current_location in JSON: {'current_location' in scene_dict}")
    if "robot_location" in scene_dict:
        print(f"  robot_location value: {scene_dict['robot_location']}")
    if "current_location" in scene_dict:
        print(f"  current_location value: {scene_dict['current_location']}")

    request_str = f"task: {sample.task}"
    msg = spine_prompts.get_base_prompt_update_graph(
        request=request_str, scene_graph=scene_graph_json
    )

    print(f"\n  Message list has {len(msg)} messages")
    for i, m in enumerate(msg):
        role = m.get("role", "?")
        content_len = len(m.get("content", ""))
        preview = m.get("content", "")[:100].replace("\n", " ")
        print(f"    [{i}] role={role}, len={content_len}, preview: {preview}...")

    # Token count
    prompt_tokens = tokenizer.apply_chat_template(
        msg, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )["input_ids"]
    print(f"\n  Total prompt tokens: {prompt_tokens.shape[1]}")

    # ── Step 5: Check graph parsing (what inference.py sees) ────────────
    print("\n" + "=" * 60)
    print("STEP 5: Graph parsing (_parse_pyg_graph)")
    print("=" * 60)
    pyg_graph = client._parse_pyg_graph(msg)
    if pyg_graph is not None:
        print(f"  Graph found: True")
        print(f"  Node count: {pyg_graph.x.shape[0]}")
        print(f"  Node names: {pyg_graph.node_names}")
        print(f"  robot_location: {pyg_graph.robot_location}")
        print(f"  Edge index shape: {pyg_graph.edge_index.shape if hasattr(pyg_graph, 'edge_index') else 'N/A'}")
    else:
        print("  Graph found: False")
        print("  WARNING: _parse_pyg_graph returned None! The model will run WITHOUT graph augmentation.")
        # Debug: show what the regex is looking for
        for m in reversed(msg):
            if m.get("role") == "user":
                content = m.get("content", "")
                match = re.search(r"[Ss]cene graph: ?(.*})", content)
                if match:
                    print(f"  Regex matched, but parsing failed. Match: {match.group(1)[:200]}...")
                else:
                    print(f"  Regex did NOT match in user message (len={len(content)})")
                    print(f"  Looking for 'Scene graph:' in: ...{content[-300:]}")
                break

    # ── Step 6: Raw generation ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"STEP 6: Raw generation (max_new_tokens={args.max_new_tokens})")
    print("=" * 60)
    raw_output, success = client.query_llm(msg, max_new_tokens=args.max_new_tokens)
    print(f"\n  Generation success: {success}")
    print(f"  Output length: {len(raw_output)} chars")
    print(f"\n  --- FULL RAW OUTPUT ---")
    print(raw_output)
    print(f"  --- END RAW OUTPUT ---")

    # ── Step 7: SPINE parsing ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 7: SPINE parsing (try_parse + extract_plan)")
    print("=" * 60)

    parsed, is_valid_json = llm_planner.try_parse(raw_output)
    print(f"  try_parse success: {is_valid_json.success}")
    if not is_valid_json.success:
        print(f"  try_parse error: {is_valid_json.message}")
        print(f"  Cleaned output: {llm_planner.clean_llm_output(raw_output)[:500]}")
    else:
        print(f"  Parsed keys: {list(parsed.keys())}")
        print(f"  Parsed plan (raw): {parsed.get('plan', 'N/A')}")

        plan, is_valid_plan = llm_planner.extract_plan(parsed["plan"])
        print(f"\n  extract_plan success: {is_valid_plan.success}")
        if not is_valid_plan.success:
            print(f"  extract_plan error: {is_valid_plan.message}")
        else:
            print(f"  Validated plan: {plan}")

    # ── Step 8: Eval result ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 8: Eval result")
    print("=" * 60)

    # Simulate what run_planning returns on failure
    if is_valid_json.success and "plan" in parsed:
        planner_response = parsed
    else:
        planner_response = {"response": {}}

    result, formatted_answer = evaluate._construct_eval_result(planner_response, sample.answer)
    print(f"  Formatted: {result.formatted}")
    print(f"  Keyword match: {result.plan_keyword}")
    print(f"  Correct: {result.is_correct()}")
    print(f"  Answer key used: {sample.answer}")
    print(f"  Planner response passed to eval: {planner_response}")


if __name__ == "__main__":
    main()
