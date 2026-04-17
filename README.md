# Aggregating Ethics Experimentation

This repository is a small, local experiment for comparing how different moral aggregation methods select "best" AI advice.

The workflow is:

1. Generate multiple candidate answers to a moral dilemma.
2. Score each answer under three ethical frameworks.
3. Aggregate those framework scores with multiple decision rules.
4. Visualize which response each rule picks.

The project uses Ollama locally with `llama3.1:8b` for both generation and scoring.

## What is in this repo

- `generate_responses.py`  
  Produces 16 sampled responses for a dilemma and saves them as JSON.
- `format_responses.py`  
  Normalizes response JSON so each response is stored as a wrapped list of lines.
- `score_responses.py`  
  Scores each response under utilitarian/deontological/Ubuntu lenses, computes aggregation winners, and writes an HTML table dashboard.
- `responses_dilemma_1.json`  
  Generated responses for the elder-care scenario.
- `responses_dilemma_2.json`  
  Generated responses for the friendship/possible-infidelity scenario.
- `elder_care_dilemma_scores.html` and `friendship_dilemma_scores.html`  
  Visualization outputs showing framework scores plus aggregation winners.

## Core idea

Each candidate response gets 3 scores (0-10):

- Utilitarian (overall outcomes/well-being)
- Deontological (duties/honesty/principles)
- Ubuntu (community/relational harmony)

Then four aggregation methods are compared:

- **Expected choiceworthiness (EC)**: weighted sum (`0.4, 0.35, 0.25`)
- **Maximin**: maximize the minimum framework score
- **Nash parliament**: maximize product of scores
- **Baseline**: utilitarian score only

The experiment asks: do different aggregation rules select different responses?

## Requirements

- Python 3
- Ollama running locally
- Model pulled: `llama3.1:8b`
- Python package: `requests`

Install dependency:

```bash
pip3 install requests
```

Prepare Ollama (if needed):

```bash
ollama pull llama3.1:8b
ollama serve
```

## How to run

From the project root:

1. Generate candidate responses:

```bash
python3 generate_responses.py
```

2. (Optional) Reformat JSON response line wrapping:

```bash
python3 format_responses.py
```

3. Score and produce visualization:

```bash
python3 score_responses.py
```

Open the generated HTML file in your browser to inspect winners by method.

## Important current behavior

- `generate_responses.py` currently writes to `responses_dilemma_1.json` and currently calls `DILEMMA` (elder-care prompt) in the main loop.
- `score_responses.py` currently reads `responses_dilemma_2.json` and writes `friendship_dilemma_scores.html`.

If you want both scripts to operate on the same scenario in one run, update those constants first.

## Output format notes

The response JSON schema is:

- top-level metadata: `dilemma`, `model`, `temperature`, `num_responses`
- `responses`: array of objects with:
  - `id` (1..N)
  - `response` (list of wrapped text lines)

`score_responses.py` tolerates either string or list-form responses.

## Caveats

- This is an experimental setup, not a normative ethics engine.
- Scores are produced by the same model family, so judge-model bias is likely.
- Randomness (`temperature`) means reruns can produce different response pools and different winners.

