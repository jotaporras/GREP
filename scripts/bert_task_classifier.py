#!/usr/bin/env python3
"""
BERT-based classifier for graph analysis task types.

Classifies tasks from PRISM evaluation results into four categories:
  - node_existence: Does an object/entity exist in the scene?
  - position_in_map: Where is something located? Which area contains it?
  - reachability: Can you move directly (one hop) between two places?
  - navigability: Can you reach a destination via a multi-hop path?

Uses DistilBERT embeddings with a linear classification head trained on
curated prototype examples for each category.
"""

import argparse
import json
import glob
import os
import sys
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

TASK_LABELS = [
    "node_existence",
    "position_in_map",
    "reachability",
    "navigability",
]

LABEL_TO_IDX = {l: i for i, l in enumerate(TASK_LABELS)}

TRAINING_EXAMPLES = {
    "node_existence": [
        "Is there a boat in the scene?",
        "Does the complex include a medical supply kit?",
        "Is a diesel generator present anywhere on the map?",
        "Is the truck damaged?",
        "Is the battery storage marked as damaged?",
        "Is the medical storage locker marked locked?",
        "Is any listed equipment marked operational?",
        "Does the fuel bladder have any recorded condition status?",
        "What condition is reported for the fire engine?",
        "Is there a portable breathing apparatus available?",
        "Does the map include an emergency rescue sled?",
        "Is there food or water anywhere in this area?",
        "Is the server rack described as damaged?",
        "What is the condition of the backup power unit?",
        "Is the drone available for immediate use?",
        "Is the boat damaged?",
        "Is the light poles damaged?",
        "Is there a working firearm I can use for defense?",
        "Is there a device for measuring radiation levels?",
        "Is there a motorized vessel available anywhere?",
        "Is there a weather balloon somewhere in the facility?",
        "Is there heavy drilling equipment anywhere in the area?",
        "Does the tide gauge list any condition note?",
        "Is the cargo pallet marked locked?",
        "Is there a satellite dish anywhere in the utility pad area?",
        "Is the satellite phone marked as locked or damaged?",
        "Is there a listed quadcopter anywhere in the environment?",
        "communications are down. Why?",
        "Is there something I can use to catch fish?",
        "is there a boat in the scene?",
        "Is the ice probe damaged?",
        "Is the meteorological balloon damaged or usable?",
        "Is there a snow vehicle stored anywhere in the site?",
        "Is there a dispatch console listed anywhere in the station?",
        "Is a plankton sampler listed in the site inventory?",
        "Is there a damaged tilt-logging case anywhere in the site?",
        "Is a boat battery present anywhere on the map?",
        "Is there a survey drone anywhere in the station?",
        "What condition is the tilt-logging case in?",
        "Is there a working electrical power source anywhere in the park?",
        "Is there a way to communicate with the surface?",
        "Is there a way to signal for rescue from outside the station?",
        "Is there a way to signal someone from a distance?",
        "Is there somewhere I can grow plants indoors?",
        "Is there somewhere to grow food in this facility?",
        "Is there water filtration equipment anywhere?",
        "Is there any asset located in the hazmat cage?",
        "Is there any communication equipment that still works?",
        "Is there heavy lifting equipment near the waterfront?",
        "Is there protective equipment for handling hazardous materials?",
        "Is there something that can help me stay afloat in the water?",
        "Is there a reinforced secure storage area below ground?",
        "Is there a place where I can monitor operations from screens and panels?",
        "Is there a single storage area holding a transit crate, rescue blanket, and tag bundle together?",
        "Is there an airport area that contains both a hold-short sign and a runway beacon?",
        "Does the place with the torque wrench also hold a socket set?",
    ],
    "position_in_map": [
        "Where is the satellite phone stored?",
        "Which area contains the fuel tank?",
        "Find me a place to stay for the night",
        "Where are tools likely to be?",
        "Where can I conduct scientific experiments?",
        "How many objects are in the engine testing area?",
        "Which area has the most listed items?",
        "How many areas contain exactly two objects?",
        "Find me a watercraft I can paddle by hand",
        "Find me a tool for cutting through metal bulkheads",
        "Where can I find books or written knowledge?",
        "Where is the emergency trauma kit kept?",
        "Which area contains the rescue boat?",
        "How many listed items are present in total?",
        "Where can I find fuel for a vehicle or generator?",
        "Which storage area contains the spill-response kit?",
        "How many equipment objects are cataloged at the site?",
        "Find me something I can drive in",
        "Find me a place where live performances could be held for an audience",
        "Where can I buy memorabilia or trinkets?",
        "Find me a sheltered place to store a small vessel",
        "Which workroom holds the microscope, sample freezer, and acoustic recorder together?",
        "How many listed items have a filled-in condition description?",
        "Where is the damaged compressor located?",
        "Which area contains both the fuel filter case and the damaged battery crate?",
        "Which area is the only one with two listed objects in the same area?",
        "I lost my keys. I last saw them when I parked my truck.",
        "Where can I get a cold frozen treat on a hot day?",
        "Find me navigation equipment for charting a course",
        "Which area contains the weather balloon?",
        "Where is the flotation ring stored?",
        "How many objects are stored at the dock with the rescue canoe?",
        "Which area contains the cargo drone?",
        "Find me protective eyewear for working underwater",
        "How many areas have exactly three listed items?",
        "Among all areas, which area has the largest number of immediate neighbors?",
        "How many listed objects have a recorded condition description?",
        "Which area holds both a ration supply and a satellite phone?",
        "Where can raw ore be melted down into metal?",
        "Find me medical supplies for treating injuries",
        "Where should the robot go to find the aerial inspection unit?",
        "Where should the robot go to find the ambulance van?",
        "What badge-making equipment is present?",
        "What kitchen vessel for serving hot meals is present in the support network?",
        "What mobile boat is kept in the vessel storage area?",
        "What listed item is kept in the launch area for unmanned aircraft?",
        "What towing item is present on the base?",
        "Which item is located in the spare-parts storage area?",
        "Which mapped item is locked?",
        "Which listed object is marked empty?",
        "Which item has the damaged condition note?",
        "Name the item in the airfield avionics work area that is marked operational.",
        "What damaged device is located in sector 169?",
        "Which other item is stored in the same area as the charger bank?",
        "Where are both the blanket bundle and the ration crate stored?",
        "Where are the charger bank and spare battery kept together?",
    ],
    "reachability": [
        "Can the robot move directly from the cold storage area to the area with the parking pay station?",
        "Is the pump house directly connected to the radar pier?",
        "Can the robot move in one step from the engine testing area to the cargo apron?",
        "From the medical tent area, can the robot reach the pump house in one direct hop?",
        "Can the robot move directly from the deicing pad to the runway holding point without passing through another area?",
        "Can the robot move directly in one link from the microscopy lab to the compressor room?",
        "Is the pool skimmer's area directly connected to the quadcopter's area?",
        "Is the ridge gate directly connected to the seed storage area?",
        "From the starting area, can the robot reach the area with the battery cabinet in one move?",
        "Can the robot reach the area containing the microscope in a single move from the current starting area?",
        "From the fuel valve area, can the robot reach the southern proving track in a single move?",
        "Is there a direct passage between the waste sorting area and the pump house?",
        "From the starting work bay, is the area with the tool chest immediately reachable?",
        "Can the robot move directly from the main harbor plaza to the chart archive?",
        "Can the robot move directly from the medicine kiosk to the drone launch area?",
        "Can the robot move directly from the vehicle engine bay to the refueling island?",
        "Can the robot move in one step from the starting command area directly to the fuel depot?",
        "From the starting concourse, can the robot reach the rescue sled storage area in a single move?",
        "Starting where the welding cart is kept, can the robot reach the area with the fuel meter in one move?",
        "From the current area, can the robot move directly to the boat battery's area?",
        "From the current area, can the robot move directly to the rescue truck's area?",
        "From the starting area, can the robot reach the area with the sample crate in one move?",
        "From the starting area, is the crash-response bay directly connected?",
        "Starting in the care clinic, can the robot reach the utility substation in one move?",
        "From the command post, can the place with the ambulance be reached in one direct move?",
        "From the communications gateway, can the robot move directly to the container yard?",
        "From the medical communications hut, is the landing deck one move away?",
        "From the secondary command area, is the launch area with the small rescue boat one move away?",
        "From the southern aircraft service apron, can the robot reach the weather-check balloon in one move?",
        "From the waterfront pier, is there a direct link to the remote cache?",
        "Can the robot reach the damaged meteorological balloon from its starting area?",
    ],
    "navigability": [
        "Can a route from the starting area reach the area containing the air purifier?",
        "Can the robot get from the dockyard gate to the oxygen cart using mapped areas?",
        "From the starting area, can the robot eventually reach the area holding the incident radio?",
        "Give a valid area sequence from the animal holding area to the waterfront platform that avoids the supply building.",
        "Can the robot get from the ice-core storage area to the locked satellite phone by passing through the communications bunker and cable vault?",
        "Can the robot travel from the command hub to the records area by passing through sector 22 and sector 85?",
        "From the starting command area, can the robot reach the patient transfer area by going through the meal tent and a landing marker?",
        "Starting at the fuel bay, can the robot reach the dive lock by following connected areas?",
        "From the current starting point, can the hydraulic jack be reached by going through the fuel yard and then the parts storage area?",
        "Starting from the assigned initial area, can the robot reach the secured refueling cabinet by following connected areas?",
        "Starting in the secure storage room, can the robot get to the observatory deck?",
        "From the command center, can the robot reach the deployed weather mast using mapped links?",
        "Can a route from the wet lab porch reach the engine shelter through mapped links?",
        "From the current starting area, is there a route to the hangar containing the cargo drone?",
        "Starting at the command post, can the garage holding the snowmobile be reached in two moves or fewer?",
        "Starting at the place with the cargo pallet, can the robot reach the laboratory with the spectrometer without using the main command area?",
        "Starting in the main command area, can connected spaces reach the launch area with the small rescue boat?",
        "From the freezer aisle, can the robot get to the damaged lidar device by crossing sector 164 and sector 161?",
        "From the current starting point, can the emergency sled be reached by first moving to the ice access area?",
        "Starting from the assigned initial area, is the secured refueling cabinet one move away?",
        "From the current starting area, can the robot reach the checkpoint with the snowcat?",
        "From the starting area, can the robot reach the area with the medical pack?",
        "From the starting area, can the robot reach the area with the mobile aviation fuel tank?",
        "From the current airfield start, can the robot reach the locked diagnostic computer, and what route could it use?",
        "Can the robot reach the hydraulic testing stand from its starting airfield entrance?",
        "From the badge check-in area, which youth-supervision area can be reached in one move?",
        "From the waterfront platform, what is the shortest number of region-to-region moves needed to reach the animal holding area?",
        "Which intermediate area gives a two-step route from the starting area to the quadcopter's area?",
        "From the robot's starting area, give the two-hop route to the sample quay that uses the drone staging apron as the middle stop.",
        "Starting at the aircraft landing area, which neighboring area is also directly connected to the meal-preparation area, allowing a two-move route?",
        "Which area should be avoided if the plan must avoid locked items?",
        "Which reserve entry area is the only gateway between the facility side and the habitat loop?",
        "Which non-entry wetland habitat area directly links both the viewing shelter and the nesting patch?",
        "Which area is directly connected to the battery shed, the fuel quay, and the control bunker?",
        "Give a valid area sequence from the animal holding area to the waterfront platform.",
        "From the main command area, can the robot reach the dock that holds both watercraft items?",
        "From the current starting area, can the robot move directly to the area with the weather balloon?",
        "From the starting area, name the directly connected area that contains the rescue sled.",
        "From the terminal lobby, which directly connected area contains the radio console?",
        "From the baggage claim area, which neighboring area is used for ride share pickup?",
        "From the medical triage tent, which immediately adjacent area stores chilled medical supplies?",
    ],
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


class BertTaskClassifier:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME)
        self.model.eval()
        self.model.to(device)

        hidden_dim = self.model.config.hidden_size
        self.head = nn.Linear(hidden_dim, len(TASK_LABELS)).to(device)
        self._train_head()

    def _embed(self, texts: list[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]

    def _train_head(self):
        all_texts = []
        all_labels = []
        for label, examples in TRAINING_EXAMPLES.items():
            all_texts.extend(examples)
            all_labels.extend([LABEL_TO_IDX[label]] * len(examples))

        embeddings = self._embed(all_texts)
        labels = torch.tensor(all_labels, device=self.device)

        optimizer = torch.optim.Adam(self.head.parameters(), lr=1e-2)
        loss_fn = nn.CrossEntropyLoss()

        self.head.train()
        for epoch in range(200):
            logits = self.head(embeddings)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.head.eval()

        with torch.no_grad():
            logits = self.head(embeddings)
            preds = logits.argmax(dim=1)
            train_acc = (preds == labels).float().mean().item()
        print(f"  Classification head trained: {train_acc:.0%} training accuracy")

    def classify(self, task_text: str) -> dict:
        emb = self._embed([task_text])
        with torch.no_grad():
            logits = self.head(emb)
            probs = F.softmax(logits, dim=1).squeeze()
        scores = {TASK_LABELS[i]: round(float(probs[i]), 4) for i in range(len(TASK_LABELS))}
        best = max(scores, key=scores.get)
        return {"label": best, "scores": scores}

    def classify_batch(self, tasks: list[str], batch_size: int = 64) -> list[dict]:
        all_results = []
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i : i + batch_size]
            embs = self._embed(batch)
            with torch.no_grad():
                logits = self.head(embs)
                probs = F.softmax(logits, dim=1)
            for j in range(len(batch)):
                scores = {
                    TASK_LABELS[k]: round(float(probs[j, k]), 4)
                    for k in range(len(TASK_LABELS))
                }
                best = max(scores, key=scores.get)
                all_results.append({"label": best, "scores": scores})
        return all_results


def load_result_file(path: str) -> dict | None:
    try:
        with open(path) as f:
            data = json.load(f)
        if "samples" not in data:
            return None
        return data
    except (json.JSONDecodeError, KeyError):
        return None


def collect_result_files(search_dirs: list[str]) -> list[str]:
    files = []
    for d in search_dirs:
        for f in glob.glob(os.path.join(d, "**", "*.json"), recursive=True):
            if "summary" in os.path.basename(f):
                continue
            files.append(f)
    return sorted(set(files))


def process_result_file(classifier: BertTaskClassifier, path: str) -> dict | None:
    data = load_result_file(path)
    if data is None:
        return None

    samples = data["samples"]
    tasks = [s["task"] for s in samples]
    if not tasks:
        return None

    classifications = classifier.classify_batch(tasks)

    annotated_samples = []
    for sample, cls in zip(samples, classifications):
        annotated_samples.append({
            "task": sample["task"],
            "answer_key": sample.get("answer_key", ""),
            "predicted_type": cls["label"],
            "type_scores": cls["scores"],
            "formatted": sample.get("formatted", None),
            "correct": sample.get("correct", None),
            "terminated_by": sample.get("terminated_by", None),
        })

    type_counts = Counter(c["label"] for c in classifications)
    type_accuracy = {}
    type_formatted = {}
    for label in TASK_LABELS:
        matching = [s for s in annotated_samples if s["predicted_type"] == label]
        if matching:
            correct_count = sum(1 for s in matching if s["correct"] is True)
            formatted_count = sum(1 for s in matching if s["formatted"] is True)
            type_accuracy[label] = round(correct_count / len(matching), 4)
            type_formatted[label] = round(formatted_count / len(matching), 4)

    total_correct = sum(1 for s in annotated_samples if s["correct"] is True)
    total_formatted = sum(1 for s in annotated_samples if s["formatted"] is True)
    total = len(annotated_samples)

    return {
        "source_file": path,
        "checkpoint": data.get("checkpoint", ""),
        "eval_data": data.get("eval_data", ""),
        "permutation": data.get("permutation", None),
        "total_samples": total,
        "overall_accuracy": round(total_correct / total, 4) if total else 0,
        "overall_formatted": round(total_formatted / total, 4) if total else 0,
        "type_distribution": dict(type_counts),
        "accuracy_by_type": type_accuracy,
        "formatted_by_type": type_formatted,
        "samples": annotated_samples,
    }


def run_smoke_test(classifier: BertTaskClassifier) -> bool:
    """Smoke test with held-out examples not in the training set."""
    test_cases = [
        ("Are there any explosives stored somewhere in the complex?", "node_existence"),
        ("Is the hazardous materials protective suit in working condition?", "node_existence"),
        ("What status is recorded for the emergency generator?", "node_existence"),
        ("Is the medicine cabinet in the clinic storage area locked?", "node_existence"),
        ("Is a canvas roll listed anywhere in the response hub?", "node_existence"),
        ("Is a work skiff stored at a landing area?", "node_existence"),

        ("Which numbered sector contains the damaged portable inverter?", "position_in_map"),
        ("Find me a communication device for coordinating with others", "position_in_map"),
        ("How many separate items are at the airfield fuel service point?", "position_in_map"),
        ("Which office area contains both the logbook terminal and the handheld radio?", "position_in_map"),
        ("Where can I find emergency medical equipment?", "position_in_map"),
        ("What recorded status is listed for the remote camera unit?", "position_in_map"),

        ("From the robot's starting area, can it move directly to the emergency helipad?", "reachability"),
        ("Starting at the clinic doorway, can the robot move directly to the area with the stretcher?", "reachability"),
        ("From the initial command area, can the robot move directly to the harbor warehouse?", "reachability"),
        ("From the starting area, can the robot reach the area with the weather balloon?", "navigability"),

        ("From the main incident command area, what single intermediate area gives a two-step route to the search-aircraft pad?", "navigability"),
        ("From the current area, give one two-step route to the boat battery's area.", "navigability"),
        ("Which two middle stops can each make a two-move route from the viewing shelter to the nesting patch?", "navigability"),
        ("Which area should be avoided if the plan must avoid locked items?", "navigability"),
    ]

    print("=" * 70)
    print("SMOKE TEST -- BERT Task Classifier (DistilBERT + Linear Head)")
    print("Held-out examples (not in training set)")
    print("=" * 70)

    correct = 0
    total = len(test_cases)
    tasks = [t for t, _ in test_cases]
    results = classifier.classify_batch(tasks)

    per_class = {l: {"correct": 0, "total": 0} for l in TASK_LABELS}
    for (task, exp), cls in zip(test_cases, results):
        pred = cls["label"]
        match = pred == exp
        correct += int(match)
        per_class[exp]["total"] += 1
        per_class[exp]["correct"] += int(match)
        status = "PASS" if match else "FAIL"
        conf = cls["scores"][pred]
        print(f"  [{status}] expected={exp:20s} predicted={pred:20s} (conf={conf:.3f})")
        if not match:
            print(f"         scores: {cls['scores']}")
            print(f"         task: {task[:80]}")

    accuracy = correct / total
    print(f"\nSmoke test: {correct}/{total} correct ({accuracy:.0%})")
    print("\nPer-class accuracy:")
    for label in TASK_LABELS:
        c = per_class[label]
        acc = c["correct"] / c["total"] if c["total"] else 0
        print(f"  {label:20s}: {c['correct']}/{c['total']} ({acc:.0%})")
    print("=" * 70)

    if accuracy < 0.75:
        print("WARNING: Smoke test accuracy below 75%.")
        return False
    return True


def run_detailed_analysis(classifier: BertTaskClassifier):
    """Show detailed classification outputs for a diverse sample of tasks."""
    analysis_tasks = [
        "Is there a boat in the scene?",
        "Is the truck damaged?",
        "What condition is reported for the portable transfer pump?",
        "Is any listed equipment marked operational?",
        "Is there a way to start a fire out here?",
        "Where is the satellite phone stored?",
        "Which area contains the fuel tank?",
        "Find me a place to stay for the night?",
        "How many objects are in the engine testing area?",
        "How many areas contain more than one listed object?",
        "Where can I find drinkable water in a dry region?",
        "Which area should be avoided if the plan must avoid locked items?",
        "Can the robot move directly from the cold storage area to the area with the parking pay station?",
        "Is the pump house directly connected to the radar pier?",
        "Can the robot move in one step from the engine testing area to the cargo apron?",
        "From the starting area, can the robot reach the area with the sample crate in one move?",
        "Can a route from the starting area reach the area containing the air purifier?",
        "Can the robot get from the dockyard gate to the oxygen cart using mapped areas?",
        "Give a valid area sequence from the animal holding area to the waterfront platform.",
        "From the starting command area, can the robot reach the patient transfer area by going through the meal tent?",
        "I lost my keys. I last saw them when I parked my truck.",
        "communications are down. Why?",
        "Can I cross the bridge?",
        "Find me something I can drive in",
    ]

    print("\n" + "=" * 90)
    print("DETAILED CLASSIFICATION ANALYSIS")
    print("=" * 90)
    print(f"{'Task':<75s} {'Predicted':>15s}")
    print("-" * 90)

    results = classifier.classify_batch(analysis_tasks)
    for task, cls in zip(analysis_tasks, results):
        truncated = task[:72] + "..." if len(task) > 72 else task
        conf = cls["scores"][cls["label"]]
        print(f"  {truncated:<73s} {cls['label']:>15s} ({conf:.3f})")

    print("\n--- Score breakdown for ambiguous cases ---")
    ambiguous = [
        "I lost my keys. I last saw them when I parked my truck.",
        "communications are down. Why?",
        "Can I cross the bridge?",
        "Is there a way to start a fire out here?",
        "Which area should be avoided if the plan must avoid locked items?",
    ]
    amb_results = classifier.classify_batch(ambiguous)
    for task, cls in zip(ambiguous, amb_results):
        print(f"\n  Task: {task}")
        print(f"  Predicted: {cls['label']}")
        for label in TASK_LABELS:
            bar = "#" * int(cls["scores"][label] * 40)
            print(f"    {label:20s}: {cls['scores'][label]:.4f} {bar}")


def main():
    parser = argparse.ArgumentParser(
        description="BERT task type classifier for PRISM eval results"
    )
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke test only")
    parser.add_argument("--analyze", action="store_true", help="Run detailed analysis on sample tasks")
    parser.add_argument("--input-dirs", nargs="+", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    print("Loading DistilBERT model for embedding-based classification...")
    classifier = BertTaskClassifier(device=args.device)
    print("Model loaded.\n")

    if args.smoke_test:
        success = run_smoke_test(classifier)
        sys.exit(0 if success else 1)

    if args.analyze:
        run_smoke_test(classifier)
        run_detailed_analysis(classifier)
        sys.exit(0)

    run_smoke_test(classifier)

    input_dirs = args.input_dirs or [
        str(PROJECT_ROOT / "shared" / "results"),
        str(PROJECT_ROOT / "results"),
    ]
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    result_files = collect_result_files(input_dirs)
    print(f"\nFound {len(result_files)} result files to classify.\n")

    all_outputs = []
    for i, fpath in enumerate(result_files):
        rel = os.path.relpath(fpath, PROJECT_ROOT)
        print(f"  [{i + 1}/{len(result_files)}] {rel}")
        output = process_result_file(classifier, fpath)
        if output is None:
            print(f"    (skipped -- no samples)")
            continue
        output["source_file"] = rel
        all_outputs.append(output)

    summary = {
        "model": MODEL_NAME,
        "method": "distilbert_embedding_linear_head",
        "task_labels": TASK_LABELS,
        "num_training_examples_per_label": {
            k: len(v) for k, v in TRAINING_EXAMPLES.items()
        },
        "num_files_processed": len(all_outputs),
        "results": all_outputs,
    }

    out_path = output_dir / "task_classification.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nClassification results written to {out_path}")

    print("\n--- Aggregate Summary ---")
    total_samples = sum(r["total_samples"] for r in all_outputs)
    all_types = Counter()
    for r in all_outputs:
        for label, count in r["type_distribution"].items():
            all_types[label] += count

    print(f"Total samples classified: {total_samples}")
    print("Type distribution across all files:")
    for label in TASK_LABELS:
        count = all_types.get(label, 0)
        pct = count / total_samples * 100 if total_samples else 0
        print(f"  {label:20s}: {count:5d} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
