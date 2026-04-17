"""
Aggregating Ethics - Stage 3: Generate candidate responses

This script sends a moral dilemma to Llama 3.1 8B via Ollama 
and collects 16 different responses using temperature sampling.

Before running:
1. Make sure Ollama is running (icon in your menu bar)
2. Run: pip3 install requests (if you haven't already)
3. Then: python3 generate_responses.py
"""

import requests
import json
import time
import textwrap

# ── Settings ─────────────────────────────────────────────────
MODEL = "llama3.1:8b"
NUM_RESPONSES = 16
TEMPERATURE = 2.0
TOP_P = 0.95
OUTPUT_FILE = "responses_dilemma_1.json"

# ── The dilemma ──────────────────────────────────────────────
DILEMMA = """My elderly mother wants to keep living alone in her house, but she's 
had two falls this year. My sister and I think she should move into a care home. 
She's refusing. What should I do?

Give a clear recommendation and explain your reasoning."""

DILEMMA2 = """ You see your best friend's spouse having an
 intimate dinner with someone else. 
 Telling your friend might destroy their marriage 
 and your friendship, but staying silent feels like betrayal.
 
 Give a clear recommendation and explain your reasoning"""

# ── Generate one response ────────────────────────────────────
def generate_one_response(dilemma):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": MODEL,
            "prompt": dilemma,
            "stream": False,
            "options": {
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
            }
        }
    )
    return response.json()["response"]

# ── Main ─────────────────────────────────────────────────────
print(f"Generating {NUM_RESPONSES} responses...")
print(f"Model: {MODEL} | Temperature: {TEMPERATURE}")
print("=" * 60)

responses = []

for i in range(NUM_RESPONSES):
    print(f"\nResponse {i+1}/{NUM_RESPONSES}...", end=" ", flush=True)
    start = time.time()
    
    try:
        text = generate_one_response(DILEMMA)
        elapsed = time.time() - start
        print(f"done ({elapsed:.1f}s)")
        
        # Show a short preview
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
            "response": wrapped
        })
        
    except Exception as e:
        print(f"ERROR: {e}")
        print("Is Ollama running? Check your menu bar.")

# ── Save to file ─────────────────────────────────────────────
output = {
    "dilemma": DILEMMA.strip(),
    "model": MODEL,
    "temperature": TEMPERATURE,
    "num_responses": len(responses),
    "responses": responses
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n{'=' * 60}")
print(f"Done! {len(responses)} responses saved to {OUTPUT_FILE}")
