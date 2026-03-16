"""
ai_utils.py
Handles all AI prediction logic using the Groq API.
"""

from groq import Groq
import re

# Module-level client kept as None until configure_groq() is called
_client: Groq | None = None


def configure_groq(api_key: str):
    """
    Creates the Groq client with the given API key
    Call this once at startup before calling analyze_document().
    """
    global _client
    _client = Groq(api_key=api_key)   # instantiate the SDK client


def analyze_document(text: str, style: str = "General", target_score: int = 7) -> dict:
    """
    Sends the document to Groq (llama-3.3-70b-versatile) and returns a structured analysis
    The prompt forces the model to respond in a fixed key:value format so we can
    parse it reliably without needing JSON mode

    Args:
        text:         The document text to analyze
        style:        The intended writing style/audience (e.g. 'Academic', 'Business')
        target_score: The user's target clarity score (1-10); used for contextual feedback

    Returns:
        A dict with keys: clarity_score (int), tone (str), suggestions (list), raw (str)

    Raises:
        RuntimeError: If configure_groq() hasn't been called yet
    """
    if _client is None:
        raise RuntimeError(
            "Groq client is not initialised. Call configure_groq(api_key) first."
        )

    # The system prompt does two things: sets the model's role as a style-specific coach
    # and locks the response into a rigid format so _parse_analysis_response can read it
    system_prompt = (
        f"You are an expert writing coach specialising in {style} writing. " # sets the model's role as style-specific coach
        f"The author's target clarity score is {target_score}/10. " # sets the target clarity score
        "Analyze the user's document and return your analysis in EXACTLY this format " # sets the output format
        "— no extra commentary, no markdown:\n\n"
        "CLARITY_SCORE: <integer 1-10>\n"
        "TONE: <one or two sentence description of the writing tone>\n"
        "SUGGESTION_1: <first specific suggestion tailored to the chosen style>\n"
        "SUGGESTION_2: <second specific suggestion tailored to the chosen style>\n"
        "SUGGESTION_3: <third specific suggestion tailored to the chosen style>"
    )

    response = _client.chat.completions.create(
        model="llama-3.3-70b-versatile",   # Groq's fastest large model at time of writing
        messages=[
            {"role": "system", "content": system_prompt},
            # wrap the document in --- delimiters so the model knows where it starts and ends
            {"role": "user", "content": f"Document:\n---\n{text}\n---"},
        ],
        temperature=0.4,    # low temperature keeps the output deterministic and structured
        max_tokens=512,     # analysis is concise, 512 tokens is plenty for 3 suggestions
    )

    raw_text = response.choices[0].message.content.strip()
    result = _parse_analysis_response(raw_text)
    result["raw"] = raw_text   # keep the raw string so the debug expander can show it
    return result


def get_groq_client() -> Groq:
    """
    Returns the initialised Groq client.
    Used by other modules (e.g. rewrite_utils) that need the shared client
    without re-initialising it.

    Raises:
        RuntimeError: If configure_groq() hasn't been called yet.
    """
    if _client is None:
        raise RuntimeError(
            "Groq client is not initialised. Call configure_groq(api_key) first."
        )
    return _client


def chat_with_document(
    messages: list,
    document_context: str,
    style: str = "General",
) -> str:
    """
    Multi-turn chat with the document injected as system context.

    Args:
        messages:         Full conversation history in OpenAI format
                          [{"role": "user"|"assistant", "content": "..."}].
                          The latest user message must already be appended.
        document_context: The current document text (truncated to 3 000 chars
                          to stay within the context budget).
        style:            The writing style target, forwarded from the sidebar.

    Returns:
        The assistant's reply as a plain string.
    """
    if _client is None:
        raise RuntimeError(
            "Groq client is not initialised. Call configure_groq(api_key) first."
        )

    system = (
        f"You are a helpful writing assistant. The user is working on a {style} document. "
        "Help them understand, improve, and refine their writing. "
        "You can answer questions about their document, suggest rewrites of specific "
        "passages, explain stylistic choices, or discuss readability improvements. "
        "Be concise but thorough.\n\n"
        f"Document context:\n---\n{document_context[:3000]}\n---"
    )

    response = _client.chat.completions.create(
        model="llama-3.1-8b-instant",   # fast model — latency matters in chat
        messages=[{"role": "system", "content": system}] + messages,
        temperature=0.6,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


def _parse_analysis_response(raw_text: str) -> dict:
    """
    Parses the model's response into a plain Python dict
    If anything is missing the defaults below act as a safe fallback
    """
    # start with safe defaults in case the model skips a field
    result: dict = {
        "clarity_score": None,
        "tone": "Unable to determine tone",
        "suggestions": [],
        "raw": "",
    }

    for line in raw_text.splitlines():
        line = line.strip()
        upper = line.upper()   # normalise to uppercase so matching is case-insensitive

        if upper.startswith("CLARITY_SCORE:"):
            # extract the first integer on the line and clamp it to the 1-10 range
            numbers = re.findall(r"\d+", line.split(":", 1)[-1])
            if numbers:
                result["clarity_score"] = max(1, min(10, int(numbers[0])))

        elif upper.startswith("TONE:"):
            # everything after the colon is the tone description
            result["tone"] = line.split(":", 1)[-1].strip()

        elif upper.startswith("SUGGESTION_1:"):
            result["suggestions"].append(line.split(":", 1)[-1].strip())

        elif upper.startswith("SUGGESTION_2:"):
            result["suggestions"].append(line.split(":", 1)[-1].strip())

        elif upper.startswith("SUGGESTION_3:"):
            result["suggestions"].append(line.split(":", 1)[-1].strip())

    # if the model didn't return a score for some reason default to the middle of the scale
    if result["clarity_score"] is None:
        result["clarity_score"] = 5

    return result
