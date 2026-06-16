"""Turn a local SHAP explanation into a natural-language English explanation.

Uses an LLM when OPENAI_API_KEY is set; otherwise falls back to a deterministic
template so the endpoint always returns a clear, human-readable result.
"""
from __future__ import annotations

import os


SYSTEM_PROMPT = (
    "You are an urban-planning analyst who explains machine-learning results to "
    "non-technical policymakers in clear, natural English. "
    "Use ONLY the evidence you are given. Do not invent facts or numbers. "
    "The probability and the feature contributions are the model's final output — "
    "do not dispute or change them. Your job is to turn the technical evidence into "
    "a short, fluent explanation that a busy decision-maker can read at a glance. "
    "Write in flowing prose, never as a bulleted list, and never use jargon such as "
    "'SHAP', 'NDBI', 'NDVI', or raw column names."
)


def _join(items: list[str]) -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _evidence_lines(expl: dict) -> str:
    lines = []
    for f in expl["top_features"]:
        toward_slum = str(f["pushes_toward"]).lower().startswith("slum")
        direction = "raises" if toward_slum else "lowers"
        lines.append(
            f"- {f['description']} (value={f['value']}): {direction} the likelihood "
            f"of being a slum (contribution={f['shap_value']:+.3f})"
        )
    return "\n".join(lines)


def build_user_prompt(expl: dict, unit_id=None, city=None) -> str:
    header = " | ".join(p for p in [f"Unit: {unit_id}" if unit_id else "",
                                    f"City: {city}" if city else ""] if p)
    pct = expl["slum_probability"] * 100
    thr = expl["threshold"] * 100
    return (
        f"{header}\n"
        f"Model prediction: {expl['label']}\n"
        f"Slum probability: {pct:.1f}% (decision threshold {thr:.1f}%)\n\n"
        f"Most influential factors for this prediction:\n{_evidence_lines(expl)}\n\n"
        "Write a clear 2-4 sentence explanation in natural English for a policymaker:\n"
        "1. Open with the conclusion — whether this area is predicted to be a slum or "
        "not — and how confident the model is, by comparing the probability to the "
        "threshold (e.g. comfortably below, just under, well above).\n"
        "2. Explain the 2-3 main reasons in plain language, describing what the "
        "neighbourhood physically looks like rather than naming the raw indicators.\n"
        "3. If the probability is close to the threshold, note that the result is "
        "borderline.\n"
        "Keep it to flowing prose, not a list."
    )


def _confidence_phrase(prob: float, thr: float) -> str:
    margin = abs(prob - thr)
    if margin < 0.05:
        return "this is a borderline call, sitting right around the decision threshold"
    if margin >= 0.15:
        return "the model is confident in this result"
    return "the model leans this way, but not by a wide margin"


def _fallback(expl: dict, unit_id=None, city=None) -> str:
    prob = expl["slum_probability"]
    pct = prob * 100
    thr_frac = expl["threshold"]
    thr = thr_frac * 100
    is_slum = str(expl["label"]).lower().startswith("slum")

    drivers = [f for f in expl["top_features"]
               if str(f["pushes_toward"]).lower().startswith("slum") == is_slum]
    reasons = _join([d["description"] for d in drivers[:3]]) \
        or "a combination of its building pattern and satellite imagery"

    loc = unit_id or "This area"
    if city:
        loc = f"{loc} ({city})"
    verdict = "a slum area" if is_slum else "not a slum area"
    confidence = _confidence_phrase(prob, thr_frac)

    return (
        f"{loc} is predicted to be {verdict}. Its estimated slum probability is "
        f"{pct:.1f}%, against a {thr:.1f}% decision threshold, so {confidence}. "
        f"The reading is driven mainly by {reasons}."
    )


def explain_with_llm(expl: dict, unit_id=None, city=None) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"explanation": _fallback(expl, unit_id, city), "source": "fallback"}

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL") or None)
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(expl, unit_id, city)},
            ],
            temperature=0.3,
            max_tokens=400,
        )
        return {"explanation": resp.choices[0].message.content.strip(),
                "source": f"llm:{model}"}
    except Exception as e:  # noqa: BLE001 - degrade gracefully, never 500 on LLM issues
        return {"explanation": _fallback(expl, unit_id, city),
                "source": "fallback", "llm_error": str(e)}
