"""One-off script to fix malformed JSON in training data.

The model's training data contains ~17 assistant responses per file where
commas are missing between JSON key-value pairs, e.g.:

    "primary_goal": "Find the robot."        "relevant_graph": ...

This script finds those cases, inserts the missing comma, validates the fix,
and writes the corrected file back.
"""

import json
import re
import sys
from pathlib import Path

# Pattern: closing " followed by whitespace then "key": with NO comma between
MISSING_COMMA_RE = re.compile(r'"(\s+)"(?=\w+"\s*:)')

# Pattern: "key": followed by a non-quote character (unquoted string value)
# e.g. "reasoning": Upon exploring...  ->  "reasoning": "Upon exploring...
UNQUOTED_VALUE_RE = re.compile(r'("(?:\w+)":\s*)(?=[A-Z])')

TARGET_FILES = [
    "data/gen/spine_exp1/formatted.json",
    "data/gpt_gen_formatted.json",
    "data/eval/gpt_gen_formatted.json",
]


def fix_missing_commas(content: str) -> str:
    """Insert commas where they're missing between JSON key-value pairs."""
    return MISSING_COMMA_RE.sub(r'",\1"', content)


def fix_unquoted_values(content: str) -> str:
    """Add opening quote to unquoted string values."""
    return UNQUOTED_VALUE_RE.sub(r'\1"', content)


def process_file(path: Path) -> dict:
    """Fix malformed assistant JSON in a single data file.

    Returns a summary dict with counts.
    """
    with open(path) as f:
        data = json.load(f)

    fixed = 0
    still_broken = 0

    for conversation_obj in data:
        for msg in conversation_obj["conversations"]:
            if msg["role"] != "assistant":
                continue

            content = msg["content"]

            # Check if it already parses
            try:
                json.loads(content)
                continue
            except json.JSONDecodeError:
                pass

            # Apply fixes
            new_content = fix_unquoted_values(content)
            new_content = fix_missing_commas(new_content)

            # Validate fix
            try:
                json.loads(new_content)
                msg["content"] = new_content
                fixed += 1
            except json.JSONDecodeError:
                still_broken += 1
                print(f"  WARNING: Could not fix entry in {path}:")
                print(f"    {content[:120]}...")

    # Write back
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    return {"fixed": fixed, "still_broken": still_broken}


def main():
    root = Path(__file__).resolve().parent.parent

    for rel_path in TARGET_FILES:
        path = root / rel_path
        if not path.exists():
            print(f"SKIP: {path} does not exist")
            continue

        print(f"Processing {rel_path} ...")
        result = process_file(path)
        print(f"  Fixed: {result['fixed']}, Still broken: {result['still_broken']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
