"""
rewrite_utils.py
Three-call agentic rewrite pipeline.

Flow:
  1. Drafter  – rewrites the text targeting the chosen style / clarity goal.
  2. Critic   – evaluates the draft and scores it; suggests a micro-fix.
  3. (loop)   – if critic score < CRITIC_THRESHOLD, Drafter tries again
                with the feedback attached (max MAX_DRAFT_LOOPS total drafts).
  4. Refiner  – synthesises the original + best draft + critique into the
                final polished version.

Public API:
  run_rewrite_pipeline(client, text, style, target_score) -> dict
"""

import re
from groq import Groq

# ── Model assignments ──────────────────────────────────────────────────────────
# The Drafter and Refiner use the large model for quality output.
# The Critic uses the fast 8B model — catching obvious issues doesn't need 70B.
_DRAFTER_MODEL = "llama-3.3-70b-versatile"
_CRITIC_MODEL  = "llama-3.1-8b-instant"
_REFINER_MODEL = "llama-3.3-70b-versatile"

_CRITIC_THRESHOLD = 7   # scores below this trigger a second drafter pass
_MAX_DRAFT_LOOPS  = 2   # hard cap on drafter iterations


# ── Internal: Call 1 — Drafter ─────────────────────────────────────────────────

def _call_drafter(
    client: Groq,
    text: str,
    style: str,
    target_score: int,
    critic_feedback: str = "",
) -> dict:
    """
    Ask the Drafter to rewrite `text`.
    If `critic_feedback` is provided (second pass), the prompt includes the
    critic's notes so the model can address them directly.
    """
    system = (
        f"You are an expert {style} writing coach. "
        f"Rewrite the text the user provides so it reaches a clarity score of "
        f"{target_score}/10. Keep the original meaning intact; only improve "
        "structure, word choice, and clarity. "
        "Respond in EXACTLY this format — no extra commentary:\n\n"
        "REWRITE: <the improved text>\n"
        "CHANGES_MADE: <one sentence describing the main changes you made>"
    )

    user_msg = f"Original text:\n---\n{text}\n---"
    if critic_feedback:
        user_msg += (
            f"\n\nA critic reviewed your previous draft and noted:\n"
            f"{critic_feedback}\n"
            "Please address these points in your new rewrite."
        )

    response = client.chat.completions.create(
        model=_DRAFTER_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.7,
        max_tokens=1024,
    )
    return _parse_drafter(response.choices[0].message.content.strip())


def _parse_drafter(raw: str) -> dict:
    result = {"rewrite": "", "changes_made": "", "raw": raw}
    # DOTALL so the rewrite body can span multiple lines
    m_rewrite  = re.search(r"(?i)REWRITE:\s*([\s\S]*?)(?=CHANGES_MADE:|$)", raw)
    m_changes  = re.search(r"(?i)CHANGES_MADE:\s*([\s\S]*?)$", raw)
    if m_rewrite:
        result["rewrite"] = m_rewrite.group(1).strip()
    if m_changes:
        result["changes_made"] = m_changes.group(1).strip()
    if not result["rewrite"]:
        result["rewrite"] = raw   # graceful fallback if the model ignores the format
    return result


# ── Internal: Call 2 — Critic ──────────────────────────────────────────────────

def _call_critic(
    client: Groq,
    original: str,
    draft: str,
    style: str,
    target_score: int,
) -> dict:
    """
    Ask the Critic (small, fast model) to evaluate the draft.
    Returns a score and actionable feedback.
    """
    system = (
        f"You are a critical writing editor evaluating a rewrite for {style} "
        f"style targeting clarity {target_score}/10. Be concise and specific. "
        "Respond in EXACTLY this format — no extra text:\n\n"
        "CRITIC_SCORE: <integer 1-10>\n"
        "IMPROVED: <one sentence on what got better>\n"
        "STILL_WEAK: <one sentence on the biggest remaining weakness>\n"
        "MICRO_FIX: <one concrete edit that would most improve the draft>"
    )

    response = client.chat.completions.create(
        model=_CRITIC_MODEL,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"Original:\n---\n{original}\n---\n\n"
                    f"Draft rewrite:\n---\n{draft}\n---"
                ),
            },
        ],
        temperature=0.1,   # low temperature → consistent, deterministic critique
        max_tokens=256,
    )
    return _parse_critic(response.choices[0].message.content.strip())


def _parse_critic(raw: str) -> dict:
    result = {
        "critic_score": 5,
        "improved":     "",
        "still_weak":   "",
        "micro_fix":    "",
        "raw":          raw,
    }
    for line in raw.splitlines():
        upper = line.upper()
        if upper.startswith("CRITIC_SCORE:"):
            nums = re.findall(r"\d+", line.split(":", 1)[-1])
            if nums:
                result["critic_score"] = max(1, min(10, int(nums[0])))
        elif upper.startswith("IMPROVED:"):
            result["improved"] = line.split(":", 1)[-1].strip()
        elif upper.startswith("STILL_WEAK:"):
            result["still_weak"] = line.split(":", 1)[-1].strip()
        elif upper.startswith("MICRO_FIX:"):
            result["micro_fix"] = line.split(":", 1)[-1].strip()
    return result


# ── Internal: Call 3 — Refiner ─────────────────────────────────────────────────

def _call_refiner(
    client: Groq,
    original: str,
    best_draft: str,
    critique: str,
    style: str,
    target_score: int,
) -> dict:
    """
    Ask the Refiner to synthesise the original, the best draft, and the
    editorial critique into a single final polished version.
    """
    system = (
        f"You are a master editor producing the final polished version of a "
        f"{style} text targeting clarity {target_score}/10. "
        "You are given the original text, a draft rewrite, and editorial "
        "feedback. Synthesise these into the best possible final version. "
        "Respond in EXACTLY this format — no extra text:\n\n"
        "FINAL_TEXT: <the final polished version>\n"
        "CONFIDENCE: <integer 1-10 confidence this meets the goal>"
    )

    response = client.chat.completions.create(
        model=_REFINER_MODEL,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"Original:\n---\n{original}\n---\n\n"
                    f"Draft:\n---\n{best_draft}\n---\n\n"
                    f"Editorial feedback:\n---\n{critique}\n---"
                ),
            },
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    return _parse_refiner(response.choices[0].message.content.strip())


def _parse_refiner(raw: str) -> dict:
    result = {"final_text": "", "confidence": 7, "raw": raw}
    m_final = re.search(r"(?i)FINAL_TEXT:\s*([\s\S]*?)(?=CONFIDENCE:|$)", raw)
    m_conf  = re.search(r"(?i)CONFIDENCE:\s*(\d+)", raw)
    if m_final:
        result["final_text"] = m_final.group(1).strip()
    if m_conf:
        result["confidence"] = max(1, min(10, int(m_conf.group(1))))
    if not result["final_text"]:
        result["final_text"] = raw
    return result


# ── Internal: Call 4 — Lessons ─────────────────────────────────────────────────

def _call_lessons(
    client: Groq,
    original: str,
    final: str,
    style: str,
) -> list[str]:
    """
    Call 4 (lightweight): compare the original and final texts and extract
    3 transferable writing rules the author can apply in future work.

    Uses the small model — this is a pattern-extraction task, not creative work.
    The goal is to make the pipeline a *teacher*, not just a ghostwriter.
    """
    system = (
        f"You are a writing teacher. A student wrote the original text and an expert "
        f"rewrote it for {style} style. "
        "Extract exactly 3 specific, transferable writing rules the student can apply "
        "to their next piece — rules, not descriptions of what changed. "
        "Each lesson should start with an action verb (e.g. 'Use...', 'Avoid...', 'Lead...'). "
        "Respond in EXACTLY this format — no extra text:\n\n"
        "LESSON_1: <rule>\n"
        "LESSON_2: <rule>\n"
        "LESSON_3: <rule>"
    )

    response = client.chat.completions.create(
        model=_CRITIC_MODEL,   # small model is sufficient for extraction
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"Original:\n---\n{original}\n---\n\n"
                    f"Rewritten:\n---\n{final}\n---"
                ),
            },
        ],
        temperature=0.3,
        max_tokens=256,
    )
    raw = response.choices[0].message.content.strip()
    lessons: list[str] = []
    for line in raw.splitlines():
        upper = line.upper()
        if upper.startswith("LESSON_"):
            lessons.append(line.split(":", 1)[-1].strip())
    # Graceful fallback if the model ignores the format
    return lessons if lessons else [
        "Compare the two versions and note the patterns in every change made."
    ]


# ── Public API ─────────────────────────────────────────────────────────────────

def run_rewrite_pipeline(
    client: Groq,
    text: str,
    style: str = "General",
    target_score: int = 7,
) -> dict:
    """
    Orchestrates the full three-call rewrite pipeline.

    Args:
        client:       Initialised Groq client (from ai_utils.get_groq_client()).
        text:         The text to rewrite.
        style:        Intended writing style (e.g. 'Academic', 'Business').
        target_score: Desired clarity score 1-10.

    Returns a dict with:
        draft         (str)  – Drafter's best attempt.
        changes_made  (str)  – What the Drafter changed.
        critic_score  (int)  – How the Critic scored the draft (final pass).
        improved      (str)  – What improved per the Critic.
        still_weak    (str)  – Remaining weaknesses per the Critic.
        micro_fix     (str)  – The Critic's one-line suggestion.
        final_text    (str)  – Refiner's final polished version.
        confidence    (int)  – Refiner's self-assessed confidence (1-10).
        iterations    (int)  – How many Drafter passes ran (1 or 2).
        steps         (list) – Human-readable log of each pipeline step.
        lessons       (list) – 3 transferable writing rules extracted from the diff.
    """
    steps: list[str] = []

    # ── Step 1: first draft ───────────────────────────────────────────────────
    steps.append("Drafter (pass 1): generating initial rewrite…")
    drafter = _call_drafter(client, text, style, target_score)
    best_draft   = drafter["rewrite"]
    changes_made = drafter["changes_made"]
    iterations   = 1

    # ── Step 2: critic evaluates ──────────────────────────────────────────────
    steps.append("Critic: evaluating draft quality…")
    critic = _call_critic(client, text, best_draft, style, target_score)

    # ── Step 3 (conditional): loop if the critic isn't satisfied ─────────────
    if critic["critic_score"] < _CRITIC_THRESHOLD and iterations < _MAX_DRAFT_LOOPS:
        feedback = (
            f"Still weak: {critic['still_weak']} "
            f"Suggested fix: {critic['micro_fix']}"
        )
        steps.append(
            f"Critic score {critic['critic_score']}/10 < {_CRITIC_THRESHOLD} "
            "— Drafter (pass 2): revising with feedback…"
        )
        drafter2     = _call_drafter(client, text, style, target_score,
                                     critic_feedback=feedback)
        best_draft   = drafter2["rewrite"]
        changes_made = drafter2["changes_made"]
        iterations   = 2

        steps.append("Critic: re-evaluating revised draft…")
        critic = _call_critic(client, text, best_draft, style, target_score)
    else:
        steps.append(
            f"Critic score {critic['critic_score']}/10 "
            f"(≥ {_CRITIC_THRESHOLD}) — no additional pass needed."
        )

    # ── Step 4: refiner produces the final version ────────────────────────────
    critique_summary = (
        f"Improved: {critic['improved']} "
        f"Still weak: {critic['still_weak']} "
        f"Micro-fix: {critic['micro_fix']}"
    )
    steps.append("Refiner: producing final polished version…")
    refiner = _call_refiner(client, text, best_draft, critique_summary,
                            style, target_score)

    # ── Step 5: lessons — what the author can learn from the diff ─────────────
    steps.append("Lessons: extracting transferable writing rules…")
    lessons = _call_lessons(client, text, refiner["final_text"], style)

    return {
        "draft":        best_draft,
        "changes_made": changes_made,
        "critic_score": critic["critic_score"],
        "improved":     critic["improved"],
        "still_weak":   critic["still_weak"],
        "micro_fix":    critic["micro_fix"],
        "final_text":   refiner["final_text"],
        "confidence":   refiner["confidence"],
        "iterations":   iterations,
        "steps":        steps,
        "lessons":      lessons,
    }
