"""
Aggregating Ethics — Stage 1: Generate candidate responses

Sends a moral dilemma to Llama 3.1 8B via Ollama and collects diverse
responses using temperature sampling and varied system prompts.

Before running:
  1. Make sure Ollama is running (icon in your menu bar).
  2. pip3 install -r requirements.txt
  3. python3 generate_responses.py <dilemma_number>
"""

from __future__ import annotations

import json
import sys
import textwrap
import time
from pathlib import Path

import requests

# ── Settings ─────────────────────────────────────────────────
MODEL = "llama3.1:8b"
NUM_RESPONSES = 16
TEMPERATURE = 1.0
TOP_P = 0.95
WRAP_WIDTH = 100
OLLAMA_URL = "http://localhost:11434/api/generate"

ROOT = Path(__file__).resolve().parent
DILEMMAS_FILE = ROOT / "data" / "dilemmas.json"
RESPONSES_DIR = ROOT / "responses"

# ── Varied system prompts for diverse responses ──────────────
SYSTEM_PROMPTS = [
    None,
    "You believe that important decisions should be made collectively by the community, not by individuals acting alone.",
    "You believe that every person has the right to make their own decisions, and that right must be respected above all else.",
    "You believe the right action is whatever produces the best outcome for the most people. Focus on measurable consequences.",
    "You believe that cultural and spiritual traditions are the foundation of community life and should not be overridden by outsiders.",
    "You believe that a doctor's duty is to save lives, period. Professional obligations come before personal or cultural considerations.",
    "You believe that the most important thing in any decision is how it affects the relationships between people involved.",
    "You believe that when people disagree deeply, the right approach is to find a middle ground that everyone can accept.",
    "You believe that the welfare of children should be the primary consideration in any decision that affects them.",
    "You believe that elders hold wisdom that younger generations lack, and their guidance should be followed in times of crisis.",
    "You are a practical person who focuses on what is actually achievable given the constraints of the situation.",
    "You think about the long-term consequences for the entire community, not just the immediate situation.",
    "You believe that outsiders should not impose their values on communities with established traditions, even when they disagree.",
    "You believe that in life-or-death emergencies, there is no time for debate. Action must be taken immediately.",
    "You believe that the process of making a decision is just as important as the decision itself. Everyone affected should have a voice.",
    "You believe that spiritual wellbeing is just as important as physical wellbeing, and that medical decisions must account for both.",
]


def load_dilemma(number: int) -> dict:
    with DILEMMAS_FILE.open() as f:
        dilemmas = json.load(f)
    key = str(number)
    if key not in dilemmas:
        available = sorted(int(k) for k in dilemmas)
        raise KeyError(f"Dilemma {number} not found. Available: {available}")
    return dilemmas[key]


def wrap_text(text: str, width: int = WRAP_WIDTH) -> list[str]:
    """Wrap long lines while preserving short ones and blank lines."""
    result: list[str] = []
    for line in text.strip().split("\n"):
        if len(line) <= width:
            result.append(line)
        else:
            result.extend(
                textwrap.wrap(line, width=width, break_long_words=False, break_on_hyphens=False)
                or [""]
            )
    return result


def generate_one_response(prompt: str, system_prompt: str | None = None) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": TEMPERATURE, "top_p": TOP_P},
    }
    if system_prompt:
        payload["system"] = system_prompt
    response = requests.post(OLLAMA_URL, json=payload)
    return response.json()["response"]


def main() -> int:
    try:
        dilemma_number = int(sys.argv[1])
    except (IndexError, ValueError):
        print("Usage: python3 generate_responses.py <dilemma_number>")
        return 1

    try:
        dilemma = load_dilemma(dilemma_number)
    except (FileNotFoundError, KeyError) as e:
        print(f"ERROR: {e}")
        return 1

    slug = dilemma["slug"]
    prompt = dilemma["prompt"]
    output_file = RESPONSES_DIR / f"{slug}.json"
    RESPONSES_DIR.mkdir(exist_ok=True)

    print(f"Generating {NUM_RESPONSES} diverse responses for: {dilemma.get('title', slug)}")
    print(f"Model: {MODEL} | Temperature: {TEMPERATURE}")
    print("=" * 60)

    responses = []
    for i in range(NUM_RESPONSES):
        sys_prompt = SYSTEM_PROMPTS[i]
        label = "baseline (no system prompt)" if sys_prompt is None else sys_prompt[:60] + "..."
        print(f"\nResponse {i+1}/{NUM_RESPONSES} [{label}]")
        print("  Generating...", end=" ", flush=True)
        start = time.time()

        try:
            text = generate_one_response(prompt, sys_prompt)
            elapsed = time.time() - start
            print(f"done ({elapsed:.1f}s)")
            print(f"  >> {text.strip()[:120]}...")
            responses.append(
                {"id": i + 1, "system_prompt": sys_prompt, "response": wrap_text(text)}
            )
        except Exception as e:
            print(f"ERROR: {e}")
            print("  Is Ollama running? Check your menu bar.")

    output = {
        "dilemma": prompt.strip(),
        "slug": slug,
        "model": MODEL,
        "temperature": TEMPERATURE,
        "num_responses": len(responses),
        "responses": responses,
    }
    with output_file.open("w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Done! {len(responses)} responses saved to {output_file.relative_to(ROOT)}")
    print("\nNext: score these with the three ethical judges.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
