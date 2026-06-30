"""
Aggregating Ethics - Stage 3: Generate candidate responses

This script sends a moral dilemma to Llama 3.1 8B via Ollama 
and collects 16 different responses using temperature sampling.
Each response uses a different system prompt to encourage diverse perspectives.

Before running:
1. Make sure Ollama is running (icon in your menu bar)
2. Run: pip3 install requests (if you haven't already)
3. Then: python3 generate_responses.py <dilemma_number>
"""

import requests
import json
import time
import textwrap
import sys

# ── Settings ─────────────────────────────────────────────────
MODEL = "llama3.1:8b"
NUM_RESPONSES = 16
TEMPERATURE = 1.0
TOP_P = 0.95

# ── The dilemmas ─────────────────────────────────────────────
DILEMMA = """My elderly mother wants to keep living alone in her house, but she's 
had two falls this year. My sister and I think she should move into a care home. 
She's refusing. What should I do?

Give a clear recommendation and explain your reasoning."""

DILEMMA2 = """You see your best friend's spouse having an intimate dinner with someone else. 
Telling your friend might destroy their marriage and your friendship, 
but staying silent feels like betrayal.

Give a clear recommendation and explain your reasoning."""

DILEMMA3 = """A grandmother in a rural village is seriously ill and needs a
blood transfusion. Her family and the village elders believe this goes against their spiritual traditions. 
The nearest hospital is hours away. 
Her grandchildren, who she raised, depend on her. The village's only doctor must decide what to do.

What should the doctor do? Give a clear recommendation and explain your reasoning."""

# ── Pick which dilemma to run ────────────────────────────────
DILEMMAS = {
    1: DILEMMA,
    2: DILEMMA2,
    3: DILEMMA3,
}

try:
    dilemma_number = int(sys.argv[1])
    chosen_dilemma = DILEMMAS[dilemma_number]
except (IndexError, ValueError, KeyError):
    print("Usage: python3 generate_responses.py <dilemma_number>")
    print(f"Available dilemmas: {sorted(DILEMMAS.keys())}")
    sys.exit(1)

OUTPUT_FILE = f"responses_dilemma_{dilemma_number}.json"

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

# ── Generate one response ────────────────────────────────────
def generate_one_response(dilemma, system_prompt=None):
    payload = {
        "model": MODEL,
        "prompt": dilemma,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
        }
    }
    if system_prompt:
        payload["system"] = system_prompt
    
    response = requests.post(
        "http://localhost:11434/api/generate",
        json=payload
    )
    return response.json()["response"]

# ── Main ─────────────────────────────────────────────────────
print(f"Generating {NUM_RESPONSES} diverse responses...")
print(f"Model: {MODEL} | Temperature: {TEMPERATURE}")
print("=" * 60)

responses = []

for i in range(NUM_RESPONSES):
    sys_prompt = SYSTEM_PROMPTS[i]
    label = "baseline (no system prompt)" if sys_prompt is None else sys_prompt[:60] + "..."
    print(f"\nResponse {i+1}/{NUM_RESPONSES} [{label}]")
    print(f"  Generating...", end=" ", flush=True)
    start = time.time()
    
    try:
        text = generate_one_response(chosen_dilemma, sys_prompt)
        elapsed = time.time() - start
        print(f"done ({elapsed:.1f}s)")
        
        preview = text.strip()[:120]
        print(f"  >> {preview}...")
        
        wrapped = []
        for line in text.strip().split('\n'):
            if len(line) <= 100:
                wrapped.append(line)
            else:
                wrapped.extend(
                    textwrap.wrap(line, width=100,
                                 break_long_words=False,
                                 break_on_hyphens=False) or ['']
                )

        responses.append({
            "id": i + 1,
            "system_prompt": sys_prompt,
            "response": wrapped
        })
        
    except Exception as e:
        print(f"ERROR: {e}")
        print("Is Ollama running? Check your menu bar.")

# ── Save to file ─────────────────────────────────────────────
output = {
    "dilemma": chosen_dilemma.strip(),
    "model": MODEL,
    "temperature": TEMPERATURE,
    "num_responses": len(responses),
    "responses": responses
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n{'=' * 60}")
print(f"Done! {len(responses)} responses saved to {OUTPUT_FILE}")
print(f"\nNext: score these with the three ethical judges.")