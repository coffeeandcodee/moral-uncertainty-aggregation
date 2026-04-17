"""Reformat response JSON files so responses are arrays of wrapped lines."""

import json
import textwrap
import sys

WIDTH = 100

def wrap_text(text, width=WIDTH):
    lines = text.split('\n')
    result = []
    for line in lines:
        if len(line) <= width:
            result.append(line)
        else:
            wrapped = textwrap.wrap(
                line, width=width,
                break_long_words=False, break_on_hyphens=False
            )
            result.extend(wrapped if wrapped else [''])
    return result


def format_file(filepath):
    with open(filepath) as f:
        data = json.load(f)

    for resp in data["responses"]:
        text = resp["response"]
        if isinstance(text, str):
            resp["response"] = wrap_text(text)

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Formatted {filepath}")


files = sys.argv[1:] or [
    "responses_dilemma_1.json",
    "responses_dilemma_2.json",
]

for f in files:
    format_file(f)
