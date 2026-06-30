"""
Filter `data/daily_dilemmas.json` for dilemmas where the three ethical
frameworks (utilitarian, deontological, ubuntu) would disagree, and
reformat them as open-ended prompts compatible with `generate_responses.py`.

How it works
------------
Each row in the source corpus describes one side of a binary dilemma
(`action_type` is either "to_do" or "not_to_do") and carries a list of
values that implicitly back that action. Two rows with the same
`dilemma_idx` form a complete dilemma.

Values are mapped to the three frameworks via keyword sets below. For
each dilemma we count framework-aligned values on each side (these are
keyword match counts, NOT the 0-10 ethical scores produced later by
score_responses.py). Each framework "prefers" the side with more aligned
values (or is neutral on a tie). A dilemma is kept iff the frameworks
split — at least one framework prefers `to_do`, at least one prefers
`not_to_do`.

Caveats
-------
The mapping is heuristic. It catches dilemmas that are likely to
generate framework disagreement, not ones that are guaranteed to.

Usage
-----
  python3 filter_dilemmas.py
  python3 filter_dilemmas.py --limit 50          # keep the top 50 by strength
  python3 filter_dilemmas.py --output data/filtered_dilemmas.json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_FILE = ROOT / "data" / "daily_dilemmas.json"
DEFAULT_OUTPUT = ROOT / "data" / "filtered_dilemmas.json"

# ── Value → framework mapping ────────────────────────────────
# Lower-case keywords. Matching is substring-based on the cleaned value
# string, so e.g. "respect for privacy" matches both the full phrase and
# the bare "privacy" keyword.

UTILITARIAN_KEYWORDS = {
    "wellbeing", "well-being", "welfare", "happiness", "satisfaction",
    "survival", "health", "safety", "comfort", "relief", "peace",
    "peace of mind", "efficiency", "productivity", "profit",
    "profitability", "financial stability", "financial security",
    "job security", "stability", "protection", "prosperity",
    "outcome", "harm reduction", "resilience",
}

DEONTOLOGICAL_KEYWORDS = {
    "honesty", "integrity", "fairness", "justice", "lawfulness",
    "duty", "accountability", "responsibility", "transparency",
    "respect for rules", "professionalism", "professional integrity",
    "professional responsibility", "trust", "trustworthiness",
    "respect", "dignity", "autonomy", "right to life",
    "right to privacy", "right to information", "right to autonomy",
    "respect for privacy", "respect for autonomy", "respect for life",
    "freedom", "freedom of expression", "consent", "truth",
    "open communication", "discipline",
}

UBUNTU_KEYWORDS = {
    "compassion", "empathy", "solidarity", "unity", "teamwork",
    "cooperation", "support", "care", "concern", "generosity",
    "love", "loyalty", "kinship", "community", "togetherness",
    "selflessness", "sacrifice", "understanding", "tolerance",
    "patience", "guidance", "consideration", "respect for diversity",
    "respect for others", "friendship", "mutual aid",
}

FRAMEWORK_KEYWORDS = {
    "utilitarian": UTILITARIAN_KEYWORDS,
    "deontological": DEONTOLOGICAL_KEYWORDS,
    "ubuntu": UBUNTU_KEYWORDS,
}


def load_rows(path: Path) -> list[dict]:
    """Load a JSONL file (one JSON object per line)."""
    rows = []
    with path.open() as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  warn: skipping malformed line {line_num}: {e}", file=sys.stderr)
    return rows


def parse_values(raw: str) -> list[str]:
    """`values_aggregated` is a stringified Python list. Parse defensively."""
    if not isinstance(raw, str):
        return []
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, (list, tuple)):
            return [str(v).strip().lower() for v in parsed if v]
    except (ValueError, SyntaxError):
        pass
    return []


def count_value_matches(values: list[str]) -> dict[str, int]:
    """Count how many of a side's values map to each framework's keywords."""
    matches = {f: 0 for f in FRAMEWORK_KEYWORDS}
    for value in values:
        for framework, keywords in FRAMEWORK_KEYWORDS.items():
            if any(kw in value for kw in keywords):
                matches[framework] += 1
                break  # one value, one framework — avoid double counting
    return matches


def preference(to_do_matches: int, not_to_do_matches: int) -> str:
    if to_do_matches > not_to_do_matches:
        return "to_do"
    if not_to_do_matches > to_do_matches:
        return "not_to_do"
    return "neutral"


# ── Prompt reformatting ──────────────────────────────────────

# Yes/no question starters that mark the trailing prompt we want to strip.
QUESTION_STARTERS = (
    "should ", "would ", "do ", "did ", "is it ", "would it ",
    "are you ", "can ", "could ", "must ", "will ", "shall ",
)


def strip_trailing_yes_no_question(text: str) -> str:
    """Drop the final yes/no question, keeping the scenario setup."""
    text = text.strip()
    # Find last sentence boundary, then take what comes after as the candidate question.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if not sentences:
        return text
    last = sentences[-1].strip()
    if last.endswith("?") and last.lower().startswith(QUESTION_STARTERS):
        return " ".join(sentences[:-1]).strip()
    return text


def detect_actor(text: str) -> str:
    """Best-effort guess at the actor for the open-ended question."""
    lower = text.lower()
    if re.search(r"\b(you|your|you're|you've|yourself)\b", lower):
        return "you"
    if re.search(r"\b(i|i'm|i've|my|me|myself)\b", lower):
        return "you"
    return "they"


def reformat_as_open_ended(dilemma_situation: str) -> str | None:
    setup = strip_trailing_yes_no_question(dilemma_situation)
    if not setup:
        return None
    actor = detect_actor(setup)
    return (
        f"{setup}\n\nWhat should {actor} do? "
        "Give a clear recommendation and explain your reasoning."
    )


# ── Filtering ────────────────────────────────────────────────


def filter_dilemmas(rows: list[dict]) -> list[dict]:
    grouped: dict[int, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        action_type = row.get("action_type")
        if action_type in ("to_do", "not_to_do"):
            grouped[row["dilemma_idx"]][action_type] = row

    kept: list[dict] = []
    for dilemma_idx, sides in grouped.items():
        if "to_do" not in sides or "not_to_do" not in sides:
            continue

        to_do = sides["to_do"]
        not_to_do = sides["not_to_do"]

        to_do_matches = count_value_matches(parse_values(to_do.get("values_aggregated", "")))
        not_to_do_matches = count_value_matches(parse_values(not_to_do.get("values_aggregated", "")))

        prefs = {
            f: preference(to_do_matches[f], not_to_do_matches[f])
            for f in FRAMEWORK_KEYWORDS
        }
        unique_prefs = {p for p in prefs.values() if p != "neutral"}
        if len(unique_prefs) < 2:
            continue  # all frameworks agree (or only one expressed a preference)

        prompt = reformat_as_open_ended(to_do.get("dilemma_situation", ""))
        if not prompt:
            continue

        strength = sum(
            abs(to_do_matches[f] - not_to_do_matches[f]) for f in FRAMEWORK_KEYWORDS
        )

        kept.append({
            "source_dilemma_idx": dilemma_idx,
            "basic_situation": to_do.get("basic_situation", "").strip(),
            "topic_group": to_do.get("topic_group"),
            "prompt": prompt,
            "frameworks": {
                f: {
                    "prefers": prefs[f],
                    "to_do_value_matches": to_do_matches[f],
                    "not_to_do_value_matches": not_to_do_matches[f],
                }
                for f in FRAMEWORK_KEYWORDS
            },
            "to_do_action": to_do.get("action"),
            "not_to_do_action": not_to_do.get("action"),
            "disagreement_strength": strength,
        })

    kept.sort(key=lambda d: d["disagreement_strength"], reverse=True)
    return kept


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text[:60] or "dilemma"


def to_output_schema(kept: list[dict]) -> dict:
    """Match the schema of `data/dilemmas.json` so other scripts can consume it."""
    output: dict[str, dict] = {}
    used_slugs: set[str] = set()
    for i, item in enumerate(kept, start=1):
        base_slug = slugify(item["basic_situation"]) or f"dilemma_{item['source_dilemma_idx']}"
        slug = base_slug
        n = 2
        while slug in used_slugs:
            slug = f"{base_slug}_{n}"
            n += 1
        used_slugs.add(slug)
        output[str(i)] = {
            "slug": slug,
            "title": item["basic_situation"].rstrip(".").capitalize() or slug,
            "prompt": item["prompt"],
            "source_dilemma_idx": item["source_dilemma_idx"],
            "topic_group": item["topic_group"],
            "frameworks": item["frameworks"],
            "to_do_action": item["to_do_action"],
            "not_to_do_action": item["not_to_do_action"],
            "disagreement_strength": item["disagreement_strength"],
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=SOURCE_FILE,
                        help=f"Source JSONL file (default: {SOURCE_FILE.relative_to(ROOT)})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Output JSON file (default: {DEFAULT_OUTPUT.relative_to(ROOT)})")
    parser.add_argument("--limit", type=int, default=None,
                        help="Keep only the top N dilemmas by disagreement strength")
    args = parser.parse_args()

    if not args.source.is_file():
        print(f"ERROR: source file not found: {args.source}")
        return 1

    print(f"Loading {args.source.relative_to(ROOT)}...")
    rows = load_rows(args.source)
    print(f"  {len(rows)} rows loaded")

    kept = filter_dilemmas(rows)
    print(f"  {len(kept)} dilemmas show framework disagreement")

    if args.limit is not None:
        kept = kept[: args.limit]
        print(f"  trimmed to top {len(kept)} by disagreement strength")

    output = to_output_schema(kept)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(output)} dilemmas to {args.output.relative_to(ROOT)}")

    if kept:
        print("\nTop 5 by disagreement strength:")
        for item in kept[:5]:
            prefs_str = " | ".join(
                f"{f[:3]}:{item['frameworks'][f]['prefers']}" for f in FRAMEWORK_KEYWORDS
            )
            print(f"  [{item['disagreement_strength']}] {item['basic_situation'][:50]:<50} {prefs_str}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
