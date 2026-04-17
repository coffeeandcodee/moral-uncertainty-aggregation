"""
Score generated responses using three ethical frameworks via Ollama.
Reads responses from a JSON file, scores each one, and outputs an HTML visualization.
"""

import requests
import json
import time
import re

MODEL = "llama3.1:8b"
INPUT_FILE = "responses_dilemma_2.json"
OUTPUT_HTML = "friendship_dilemma_scores.html"

FRAMEWORKS = {
    "utilitarian": (
        "You are a strict utilitarian ethics judge. Score the following response to a moral dilemma "
        "on a scale of 0 to 10, where 10 means it perfectly maximises overall well-being and happiness "
        "for all parties involved, and 0 means it completely ignores consequences and well-being. "
        "Reply with ONLY a single integer between 0 and 10. No explanation."
    ),
    "deontological": (
        "You are a strict deontological (Kantian) ethics judge. Score the following response to a moral dilemma "
        "on a scale of 0 to 10, where 10 means it perfectly upholds moral duties, honesty, and treating people "
        "as ends in themselves, and 0 means it completely violates moral duties and principles. "
        "Reply with ONLY a single integer between 0 and 10. No explanation."
    ),
    "ubuntu": (
        "You are a strict Ubuntu ethics judge. Score the following response to a moral dilemma "
        "on a scale of 0 to 10, where 10 means it perfectly promotes communal harmony, relational bonds, "
        "and the interconnectedness of people, and 0 means it completely disregards community and relationships. "
        "Reply with ONLY a single integer between 0 and 10. No explanation."
    ),
}

WEIGHTS = [0.4, 0.35, 0.25]


def score_response(response_text, framework_prompt):
    prompt = f"{framework_prompt}\n\nResponse to evaluate:\n\"{response_text}\""
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": MODEL, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.1, "top_p": 0.9}},
        )
        answer = r.json()["response"].strip()
        nums = re.findall(r'\d+', answer)
        if nums:
            val = int(nums[0])
            return min(max(val, 0), 10)
    except Exception as e:
        print(f"  scoring error: {e}")
    return 5


def generate_html(dilemma, scores):
    rows_js = ",\n  ".join(
        f"[{s['id']},{s['utilitarian']},{s['deontological']},{s['ubuntu']}]"
        for s in scores
    )

    best = {"ec": (0, -1), "max": (0, -1), "nash": (0, -1), "base": (0, -1)}
    for s in scores:
        u, d, ub = s["utilitarian"], s["deontological"], s["ubuntu"]
        ec = round(WEIGHTS[0] * u + WEIGHTS[1] * d + WEIGHTS[2] * ub, 2)
        mx = min(u, d, ub)
        na = u * d * ub
        bl = u
        if ec > best["ec"][1]:  best["ec"]  = (s["id"], ec)
        if mx > best["max"][1]: best["max"] = (s["id"], mx)
        if na > best["nash"][1]: best["nash"] = (s["id"], na)
        if bl > best["base"][1]: best["base"] = (s["id"], bl)

    highlight_ids = list(set(v[0] for v in best.values()))

    html = f"""<h2 class="sr-only">Score table showing 16 responses evaluated by three ethical judges and four aggregation methods</h2>

<div style="padding: 1rem 0; font-family: var(--font-sans);">

<p style="font-size: 14px; color: var(--color-text-secondary); margin: 0 0 1rem;">Each response scored by three ethical judges (0-10), then aggregated four ways</p>

<div style="overflow-x: auto;">
<table style="width: 100%; border-collapse: collapse; font-size: 13px;">
<thead>
<tr style="border-bottom: 2px solid var(--color-border-primary);">
<th style="text-align: left; padding: 8px 6px; color: var(--color-text-secondary); font-weight: 500;">Resp</th>
<th style="text-align: center; padding: 8px 6px; font-weight: 500; color: #534AB7;">Utilitarian</th>
<th style="text-align: center; padding: 8px 6px; font-weight: 500; color: #185FA5;">Deontological</th>
<th style="text-align: center; padding: 8px 6px; font-weight: 500; color: #0F6E56;">Ubuntu</th>
<th style="text-align: center; padding: 8px 6px; color: var(--color-text-secondary); font-weight: 500; border-left: 2px solid var(--color-border-tertiary);">EC</th>
<th style="text-align: center; padding: 8px 6px; color: var(--color-text-secondary); font-weight: 500;">Maximin</th>
<th style="text-align: center; padding: 8px 6px; color: var(--color-text-secondary); font-weight: 500;">Nash</th>
<th style="text-align: center; padding: 8px 6px; color: var(--color-text-secondary); font-weight: 500;">Baseline</th>
</tr>
</thead>
<tbody id="tbody"></tbody>
</table>
</div>

<div style="margin-top: 1.5rem; display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px;">
<div id="w-ec" style="background: var(--color-background-secondary); border-radius: var(--border-radius-md); padding: 12px;"><p style="font-size: 11px; color: var(--color-text-tertiary); margin: 0;">Expected choiceworthiness</p><p style="font-size: 18px; font-weight: 500; margin: 4px 0 0; color: var(--color-text-primary);"></p></div>
<div id="w-max" style="background: var(--color-background-secondary); border-radius: var(--border-radius-md); padding: 12px;"><p style="font-size: 11px; color: var(--color-text-tertiary); margin: 0;">Maximin</p><p style="font-size: 18px; font-weight: 500; margin: 4px 0 0; color: var(--color-text-primary);"></p></div>
<div id="w-nash" style="background: var(--color-background-secondary); border-radius: var(--border-radius-md); padding: 12px;"><p style="font-size: 11px; color: var(--color-text-tertiary); margin: 0;">Nash parliament</p><p style="font-size: 18px; font-weight: 500; margin: 4px 0 0; color: var(--color-text-primary);"></p></div>
<div id="w-base" style="background: var(--color-background-secondary); border-radius: var(--border-radius-md); padding: 12px;"><p style="font-size: 11px; color: var(--color-text-tertiary); margin: 0;">Baseline (utilitarian)</p><p style="font-size: 18px; font-weight: 500; margin: 4px 0 0; color: var(--color-text-primary);"></p></div>
</div>

<p id="verdict" style="margin-top: 1rem; font-size: 14px; padding: 12px; border-radius: var(--border-radius-md); background: var(--color-background-info); color: var(--color-text-info);"></p>
</div>

<script>
const scores = [
  {rows_js}
];
const w = [0.4, 0.35, 0.25];

let bestEC={{id:0,v:-1}}, bestMax={{id:0,v:-1}}, bestNash={{id:0,v:-1}}, bestBase={{id:0,v:-1}};
const tbody = document.getElementById('tbody');

scores.forEach(([id,r1,r2,r3]) => {{
  const ec = Math.round((w[0]*r1 + w[1]*r2 + w[2]*r3)*100)/100;
  const mx = Math.min(r1,r2,r3);
  const na = r1*r2*r3;
  const bl = r1;
  if(ec>bestEC.v){{bestEC={{id,v:ec}}}} if(mx>bestMax.v){{bestMax={{id,v:mx}}}} if(na>bestNash.v){{bestNash={{id,v:na}}}} if(bl>bestBase.v){{bestBase={{id,v:bl}}}}

  const tr = document.createElement('tr');
  const highlight = [{','.join(str(i) for i in highlight_ids)}].includes(id);
  tr.style.borderBottom = '0.5px solid var(--color-border-tertiary)';
  if(highlight) tr.style.background = 'var(--color-background-secondary)';

  const fmt = (v,col) => {{
    const isBest = (col==='ec'&&id==={best["ec"][0]})||(col==='max'&&id==={best["max"][0]})||(col==='nash'&&id==={best["nash"][0]})||(col==='base'&&id==={best["base"][0]});
    return isBest ? `<strong style="color:var(--color-text-info)">${{v}}</strong>` : v;
  }};

  tr.innerHTML = `
    <td style="padding:6px;font-weight:500;color:var(--color-text-primary)">${{id}}</td>
    <td style="text-align:center;padding:6px;color:#534AB7">${{r1}}</td>
    <td style="text-align:center;padding:6px;color:#185FA5">${{r2}}</td>
    <td style="text-align:center;padding:6px;color:#0F6E56">${{r3}}</td>
    <td style="text-align:center;padding:6px;border-left:2px solid var(--color-border-tertiary)">${{fmt(ec.toFixed(2),'ec')}}</td>
    <td style="text-align:center;padding:6px">${{fmt(mx,'max')}}</td>
    <td style="text-align:center;padding:6px">${{fmt(na,'nash')}}</td>
    <td style="text-align:center;padding:6px">${{fmt(bl,'base')}}</td>
  `;
  tbody.appendChild(tr);
}});

document.querySelector('#w-ec p:last-child').textContent = `Response ${{bestEC.id}} (${{bestEC.v.toFixed(2)}})`;
document.querySelector('#w-max p:last-child').textContent = `Response ${{bestMax.id}} (${{bestMax.v}})`;
document.querySelector('#w-nash p:last-child').textContent = `Response ${{bestNash.id}} (${{bestNash.v}})`;
document.querySelector('#w-base p:last-child').textContent = `Response ${{bestBase.id}} (${{bestBase.v}})`;

const uniqueWinners = new Set([bestEC.id, bestMax.id, bestNash.id, bestBase.id]).size;
document.getElementById('verdict').textContent = uniqueWinners + ' different response(s) selected across 4 methods — the aggregation choice ' + (uniqueWinners > 1 ? 'changes' : 'does not change') + " the AI\\'s recommendation.";
</script>"""

    with open(OUTPUT_HTML, "w") as f:
        f.write(html)
    print(f"HTML written to {OUTPUT_HTML}")


if __name__ == "__main__":
    with open(INPUT_FILE) as f:
        data = json.load(f)

    dilemma = data["dilemma"]
    responses = data["responses"]

    print(f"Scoring {len(responses)} responses across 3 ethical frameworks...")
    print(f"Model: {MODEL}")
    print("=" * 60)

    all_scores = []
    for resp in responses:
        rid = resp["id"]
        text = resp["response"]
        if isinstance(text, list):
            text = "\n".join(text)
        print(f"\nResponse {rid}:", flush=True)
        row = {"id": rid}
        for name, prompt in FRAMEWORKS.items():
            print(f"  {name}...", end=" ", flush=True)
            s = score_response(text, prompt)
            print(s)
            row[name] = s
        all_scores.append(row)

    print(f"\n{'=' * 60}")
    print("Scores:")
    for s in all_scores:
        print(f"  R{s['id']:>2}: U={s['utilitarian']} D={s['deontological']} Ub={s['ubuntu']}")

    generate_html(dilemma, all_scores)
    print("Done!")
