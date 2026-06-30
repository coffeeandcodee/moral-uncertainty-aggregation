"""
Aggregating Ethics — Stage 2: Score generated responses

Loads a responses file from `responses/`, scores each entry under three
ethical frameworks via GPT-4o-mini, applies four aggregation rules, and
writes a JSON + HTML dashboard to `scores/`.

Before running:
  1. pip3 install -r requirements.txt
  2. export OPENAI_API_KEY="sk-..."
  3. python3 score_responses.py <slug-or-path>

Examples:
  python3 score_responses.py grandmother
  python3 score_responses.py responses/grandmother.json
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from string import Template

try:
    from openai import OpenAI
except ImportError:
    print("Run: pip3 install -r requirements.txt")
    sys.exit(1)

from aggregation import DEFAULT_WEIGHTS, compute_row, pick_winners

# ── Settings ─────────────────────────────────────────────────
MODEL = "gpt-4o-mini"
WEIGHTS = DEFAULT_WEIGHTS

ROOT = Path(__file__).resolve().parent
RESPONSES_DIR = ROOT / "responses"
SCORES_DIR = ROOT / "scores"
TEMPLATE_FILE = ROOT / "templates" / "scores.html"

# ── Judge prompts (strict, full range) ───────────────────────
FRAMEWORKS = {
    "utilitarian": (
        "You are scoring this response strictly as a utilitarian ethicist. "
        "The right action is whichever one produces the greatest total wellbeing "
        "across all affected people. Count the number of people helped, the severity "
        "of harm prevented, and the expected outcomes. A response that saves more lives, "
        "reduces more suffering, or benefits more people should score higher, even if it "
        "is unfair to a specific individual.\n\n"
        "Be harsh. Use the FULL 0-10 range. A score of 7 or above should be reserved "
        "for responses that strongly maximise wellbeing with clear consequentialist reasoning. "
        "Most responses should score between 3 and 6. Only give 8-10 for exceptional responses.\n\n"
        "Reply with ONLY a single integer between 0 and 10. No explanation."
    ),
    "deontological": (
        "You are scoring this response strictly as a Kantian deontologist. "
        "The right action is whichever one respects the rights and dignity of every "
        "individual, regardless of the consequences. A response should never treat a "
        "person merely as a means to an end. If the response violates someone's autonomy, "
        "breaks a promise, or sacrifices one person for the benefit of others, it should "
        "score low even if the overall outcome is good.\n\n"
        "Be harsh. Use the FULL 0-10 range. A score of 7 or above should be reserved "
        "for responses that strongly respect individual rights and moral duties. "
        "Most responses should score between 3 and 6. Only give 8-10 for exceptional responses.\n\n"
        "Reply with ONLY a single integer between 0 and 10. No explanation."
    ),
    "ubuntu": (
        "You are scoring this response strictly from the perspective of Ubuntu ethics, "
        "a Southern African moral tradition. The central principle is that a person is a "
        "person through other people. The right action is whichever one best strengthens "
        "communal bonds, honours relational obligations, and promotes shared wellbeing. "
        "A good response recognises that individuals are not isolated decision-makers but "
        "are embedded in networks of mutual dependence. It prioritises solidarity, collective "
        "decision-making, and the preservation of relationships over individual calculation. "
        "A response that ignores the community, treats people as interchangeable, or makes "
        "decisions without considering relational consequences should score low.\n\n"
        "Be harsh. Use the FULL 0-10 range. A score of 7 or above should be reserved "
        "for responses that strongly emphasise communal bonds and collective decision-making. "
        "Most responses should score between 3 and 6. Only give 8-10 for exceptional responses.\n\n"
        "Reply with ONLY a single integer between 0 and 10. No explanation."
    ),
}


def resolve_input(arg: str) -> Path:
    """Accept either a slug, a bare filename, or a full path."""
    candidate = Path(arg)
    if candidate.is_file():
        return candidate
    slug_path = RESPONSES_DIR / f"{arg}.json"
    if slug_path.is_file():
        return slug_path
    raise FileNotFoundError(f"Could not find responses file for '{arg}'")


def score_response(client: OpenAI, response_text: str, framework_prompt: str) -> int | None:
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": framework_prompt},
                {"role": "user", "content": f"Response to evaluate:\n\n{response_text}"},
            ],
            temperature=0.1,
            max_tokens=5,
        )
        answer = completion.choices[0].message.content.strip()
        nums = re.findall(r"\d+", answer)
        if nums:
            return min(max(int(nums[0]), 0), 10)
    except Exception as e:
        print(f"  scoring error: {e}")
        time.sleep(2)
    return None


def render_html(title: str, rows: list[dict], winners: dict, output_path: Path) -> None:
    weights_str = (
        f"Util={WEIGHTS[0]}, Deont={WEIGHTS[1]}, Ubuntu={WEIGHTS[2]}"
    )
    payload = json.dumps({"rows": rows, "winners": winners})
    template = Template(TEMPLATE_FILE.read_text())
    # safe_substitute leaves JS template literals like ${value} untouched
    # while still filling in our named placeholders.
    html = template.safe_substitute(
        title=title,
        judge_model=MODEL,
        weights_str=weights_str,
        data_json=payload,
    )
    output_path.write_text(html)
    print(f"HTML written to {output_path.relative_to(ROOT)}")


def main() -> int:
    try:
        arg = sys.argv[1]
    except IndexError:
        print("Usage: python3 score_responses.py <slug-or-path>")
        print("Example: python3 score_responses.py grandmother")
        return 1

    try:
        input_file = resolve_input(arg)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set your API key first:")
        print('  export OPENAI_API_KEY="sk-..."')
        return 1
    client = OpenAI(api_key=api_key)

    with input_file.open() as f:
        data = json.load(f)

    slug = data.get("slug") or input_file.stem
    title = slug.replace("_", " ").title()
    responses = data["responses"]

    SCORES_DIR.mkdir(exist_ok=True)
    output_json = SCORES_DIR / f"{slug}.json"
    output_html = SCORES_DIR / f"{slug}.html"

    print(f"Scoring {len(responses)} responses with {MODEL}...")
    print("Frameworks: utilitarian, deontological, ubuntu")
    print("=" * 60)

    raw_scores: list[dict] = []
    for resp in responses:
        rid = resp["id"]
        text = resp["response"]
        if isinstance(text, list):
            text = "\n".join(text)
        print(f"\nResponse {rid}:", flush=True)
        row = {"id": rid}
        for name, prompt in FRAMEWORKS.items():
            print(f"  {name}...", end=" ", flush=True)
            s = score_response(client, text, prompt)
            if s is None:
                print("FAILED, retrying...", end=" ", flush=True)
                time.sleep(2)
                s = score_response(client, text, prompt)
            if s is None:
                s = 5
                print(f"FALLBACK {s}")
            else:
                print(s)
            row[name] = s
        raw_scores.append(row)

    rows = [compute_row(r, WEIGHTS) for r in raw_scores]
    winners = pick_winners(rows)

    output = {
        "dilemma": data.get("dilemma"),
        "slug": slug,
        "judge_model": MODEL,
        "weights": list(WEIGHTS),
        "scores": raw_scores,
        "aggregated": rows,
        "winners": winners,
    }
    with output_json.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"\nScores saved to {output_json.relative_to(ROOT)}")

    print(f"\n{'=' * 60}")
    print("Scores:")
    for r in raw_scores:
        print(f"  R{r['id']:>2}: U={r['utilitarian']} D={r['deontological']} Ub={r['ubuntu']}")

    render_html(title, rows, winners, output_html)
    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
