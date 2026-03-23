import json
import re
import traceback as traceback_mod
from collections import namedtuple
from dataclasses import asdict
from typing import Dict, List, Tuple

from spine.mapping.graph_util import GraphHandler
from spine.spine import SPINE
import spine.prompts.prompts as _spine_prompts
from spine.prompts.examples import EXAMPLE_1, EXAMPLE_2, EXAMPLE_3, EXAMPLE_4, EXAMPLE_5

# Fix operator-precedence bug in SPINE's get_base_prompt_update_graph.
_orig_get_base_prompt = _spine_prompts.get_base_prompt_update_graph

def _fixed_get_base_prompt(request, scene_graph, use_icl=True):
    if use_icl:
        return _orig_get_base_prompt(request, scene_graph, use_icl=True)
    header = [_spine_prompts.SYS_PROMPT] + EXAMPLE_1 + [
        {"role": "user",
         "content": f"{request}\nAdvice: \n- Recall the scene may be incomplete. \n- Carefully explain your reasoning in a step-by-step manner.\n- Reason over   connections, coordinates, and semantic relationships between objects and regions in the scene.\n\n"
                    f"Scene graph:{scene_graph}"}
    ]
    return header

_spine_prompts.get_base_prompt_update_graph = _fixed_get_base_prompt

from prism.data import graph_sim
from prism.data import planning_sim
from prism.models import gnn_llm
from prism.models import inference
from prism.models import loaders

# Modified to accept either a file path or a graph data dictionary
EvalSample = namedtuple("EvalSample", ["task", "answer", "graph", "init_node"])


class EvalResult:
    def __init__(self, formatted: bool, plan_keyword: bool):
        self.formatted = formatted
        self.plan_keyword = plan_keyword

    def is_correct(self):
        return self.formatted and self.plan_keyword


def correct_keys(answer: Dict[str, str]) -> bool:
    return (
        "primary_goal" in answer
        and "relevant_graph" in answer
        and "reasoning" in answer
        and "plan" in answer
    )


def eval_answer(parsed_answer: Dict[str, str], answer_key: str):
    """Check model output for correct JSON format and keyword match against expected answer.

    Validates that the parsed answer has the required keys (primary_goal, relevant_graph,
    reasoning, plan) and that the answer_key keyword appears in the plan field.
    """
    formatted = False
    keyphrase = False
    try:
        formatted = correct_keys(parsed_answer)

        keyphrase = bool(
            re.search(answer_key, str(parsed_answer["plan"]), re.IGNORECASE)
        )

        return EvalResult(formatted=formatted, plan_keyword=keyphrase), parsed_answer
    except Exception as e:
        print(f"[eval] eval_answer exception: {e}")
        return EvalResult(False, False), parsed_answer


def to_json(output: str) -> Tuple[str, bool]:
    try:
        s = json.loads(output)
        return s, True
    except Exception:
        return output, False


class Unsloth:
    def __init__(self, model_path: str, is_four_bit: bool):
        self.model, self.tokenizer = loaders.from_pretrained(
            path=model_path, inference=True, load_in_4bit=is_four_bit
        )

    def run(self, task: str, graph_handler: GraphHandler):
        messages = [
            {
                "role": "user",
                "content": f"task: {task}. scene graph {graph_handler.to_json_str()}",
            }
        ]

        print(f"\n====\n\ntask: {task}\n----\n")

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,  # Must add for generation
            return_tensors="pt",
        ).to(next(self.model.parameters()).device)

        outputs = self.model.generate(
            input_ids=inputs,
            max_new_tokens=4048,
            use_cache=True,
            temperature=0.01,
            min_p=0.1,
        )
        out = self.tokenizer.batch_decode(outputs)

        planner_response = out[0].split("end_header_id|>")[-1].split("<|eot_id|>")[0]
        return planner_response


def _is_graph_augmented(model) -> bool:
    """Check if model is a GraphAugmentedLLM, even under PEFT wrapping."""
    if isinstance(model, gnn_llm.GraphAugmentedLLM):
        return True
    # PEFT wrapping: PeftModel.base_model.model is the original module
    inner = getattr(getattr(model, 'base_model', None), 'model', None)
    return isinstance(inner, gnn_llm.GraphAugmentedLLM)


def eval_model(
    *,
    model_path: str = "",
    is_four_bit: bool = False,
    eval_samples: List[EvalSample],
    model=None,
    tokenizer=None,
    text_edge_list: str = "present",
) -> Tuple[float, List[Dict]]:
    """Run evaluation on a set of samples using the planning simulation loop.

    Accepts either a model_path (load from disk via HuggingFace) or an in-memory
    model+tokenizer pair (e.g. from an active training run). Sets up GraphSim and
    SPINE, runs PlanningSim per sample, and returns accuracy as fraction correct.
    """
    total_correct = 0
    sample_results = []

    multi_turn = True

    if multi_turn:
        graph_handler = GraphHandler("")
        graph_sim_inst = graph_sim.GraphSim(graph_handler)
        if model is not None and tokenizer is not None:
            # When model and tokenizer present, we're using an in-memory model (eg for eval during training.)
            strip_edges = text_edge_list == "none"
            is_gnn = _is_graph_augmented(model)
            if is_gnn:
                client = inference.GraphAugmentedInMemoryLLM(model=model, tokenizer=tokenizer, strip_edges=strip_edges)
            else:
                client = inference.InMemoryLLM(model=model, tokenizer=tokenizer, strip_edges=strip_edges)
            llm_planner = SPINE(graph=graph_sim_inst.partial_graph, client=client, use_icl=not is_gnn)
        else:
            llm_planner = SPINE(
                graph=graph_sim_inst.partial_graph,
                llm="huggingface",
                model_path=model_path,
            )

        model = planning_sim.PlanningSim(debug=False)
    else:
        model = Unsloth(model_path=model_path, is_four_bit=is_four_bit)

    for i, eval_sample in enumerate(eval_samples):
        graph_path = eval_sample.graph
        init_node = eval_sample.init_node
        task = eval_sample.task
        answer = eval_sample.answer

        planner_response = None
        planning_result = None
        try:
            if multi_turn:
                graph_sim_inst.reset(graph_as_dict=graph_path, current_location=init_node)
                llm_planner.graph = graph_sim_inst.partial_graph

                planning_result = model.run_planning(
                    llm_planner=llm_planner,
                    task=task,
                    graph_data_gen=graph_sim_inst,
                    max_iterations=10,
                )
                planner_response = planning_result.response

            else:
                planner_response = model(task=task, graph_handler=graph_handler)

                try:
                    planner_response = json.loads(planner_response)
                except:
                    planner_response = {"wrong": planner_response}

            result, formatted_answer = eval_answer(planner_response, answer)

            if result.formatted:
                print(formatted_answer)
            else:
                print(f"incorrect formatting\n{formatted_answer}")

            print(f"correct answer: {result.plan_keyword}")

            sample_results.append({
                "idx": i,
                "task": task,
                "answer_key": answer,
                "response": planner_response,
                "interaction_trace": [asdict(s) for s in planning_result.trace] if planning_result else [],
                "terminated_by": planning_result.terminated_by if planning_result else None,
                "formatted": result.formatted,
                "plan_keyword": result.plan_keyword,
                "correct": result.is_correct(),
                "error": None,
                "traceback": None,
            })

        except Exception as e:
            tb_str = traceback_mod.format_exc()
            print("!" * 80)
            print(f"[EVAL CRASH] Sample {i}/{len(eval_samples)} FAILED with unhandled exception!")
            print(f"[EVAL CRASH] Task: {task}")
            print(f"[EVAL CRASH] Error: {type(e).__name__}: {e}")
            print(tb_str)
            print("!" * 80)
            result = EvalResult(formatted=False, plan_keyword=False)

            sample_results.append({
                "idx": i,
                "task": task,
                "answer_key": answer,
                "response": None,
                "interaction_trace": [asdict(s) for s in planning_result.trace] if planning_result else [],
                "terminated_by": planning_result.terminated_by if planning_result else "exception",
                "formatted": False,
                "plan_keyword": False,
                "correct": False,
                "error": f"{type(e).__name__}: {e}",
                "traceback": tb_str,
            })

        print(f"\n=====\n")

        total_correct += result.plan_keyword

    return total_correct / len(eval_samples), sample_results
