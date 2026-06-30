"""Reformat response JSON files so responses are arrays of wrapped lines.

Usage:
  python3 format_responses.py              # all files in responses/
  python3 format_responses.py FILE [FILE...]
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

WIDTH = 100
ROOT = Path(__file__).resolve().parent
RESPONSES_DIR = ROOT / "responses"


def wrap_text(text: str, width: int = WIDTH) -> list[str]:
    result: list[str] = []
    for line in text.split("\n"):
        if len(line) <= width:
            result.append(line)
        else:
            wrapped = textwrap.wrap(
                line, width=width, break_long_words=False, break_on_hyphens=False
            )
            result.extend(wrapped if wrapped else [""])
    return result


def format_file(filepath: Path) -> None:
    with filepath.open() as f:
        data = json.load(f)
    for resp in data["responses"]:
        text = resp["response"]
        if isinstance(text, str):
            resp["response"] = wrap_text(text)
    with filepath.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Formatted {filepath.relative_to(ROOT)}")


def main() -> int:
    if len(sys.argv) > 1:
        files = [Path(arg) for arg in sys.argv[1:]]
    else:
        files = sorted(RESPONSES_DIR.glob("*.json"))
        if not files:
            print(f"No JSON files found in {RESPONSES_DIR.relative_to(ROOT)}/")
            return 1
    for f in files:
        format_file(f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
