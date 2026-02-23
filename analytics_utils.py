"""
analytics_utils.py
Local analytics using textstat, NLTK, wordcloud, and matplotlib
No API calls needed here
"""

import io
import re
import string
from collections import Counter
import matplotlib
matplotlib.use("Agg")  # Streamlit runs in a server process: we need a non-interactive backend
import matplotlib.pyplot as plt
import nltk
import textstat
from wordcloud import WordCloud, STOPWORDS


# This runs quietly so it doesn't clutter the app output on startup
def _ensure_nltk():
    for resource, path in [
        ("punkt_tab", "tokenizers/punkt_tab"),   # sentence tokenizer model
        ("stopwords", "corpora/stopwords"),        # common filler word list
    ]:
        try:
            nltk.data.find(path)   # check if the resource is already present locally
        except LookupError:
            nltk.download(resource, quiet=True)   # download silently if it's missing


def get_readability_stats(text: str) -> dict:
    """
    Runs a handful of readability formulas on the given text via textstat
    Returns a dict so the caller can pick whatever metrics it wants to display

    Keys:
        fk_grade        (float): Flesch-Kincaid Grade Level
        flesch_ease     (float): Flesch Reading Ease (0-100, higher = easier)
        smog            (float): SMOG Index
        avg_sentence    (float): Average words per sentence
        avg_syllables   (float): Average syllables per word
        word_count      (int):   Total word count
    """
    if not text.strip():
        return {}   # return early if there's nothing to analyse

    return {
        "fk_grade":       textstat.flesch_kincaid_grade(text),      # grade level required to understand the text
        "flesch_ease":    textstat.flesch_reading_ease(text),        # 0-100 score, higher means easier to read
        "smog":           textstat.smog_index(text),                 # similar to FK grade, better for short texts
        "avg_sentence":   textstat.avg_sentence_length(text),        # average number of words per sentence
        "avg_syllables":  textstat.avg_syllables_per_word(text),     # higher values suggest more complex vocabulary
        "word_count":     textstat.lexicon_count(text, removepunct=True),  # total words, ignoring punctuation
    }


def ease_label(score: float) -> str:
    """Converts a raw Flesch Reading Ease score into something a human can actually read."""
    if score >= 90:
        return "Very Easy"    # think children's books
    elif score >= 70:
        return "Easy"
    elif score >= 60:
        return "Standard"     # most newspaper writing falls around here
    elif score >= 50:
        return "Fairly Difficult"
    elif score >= 30:
        return "Difficult"    # academic papers typically land in this range
    return "Very Confusing"   # legal documents, dense technical specs


def generate_wordcloud_bytes(text: str) -> bytes:
    """
    Renders a word cloud from the document text and returns it as PNG bytes.
    The transparent background works well on both light and dark themes.

    Returns raw bytes so the caller can pass them straight to st.image().
    """
    wc = WordCloud(
        width=800,
        height=400,
        background_color=None,   # transparent background so it blends with any theme
        mode="RGBA",             # RGBA needed for transparency support
        colormap="cool",         # blue-purple palette that fits the dark UI
        stopwords=STOPWORDS,     # filter out words like "the", "and", "is"
        max_words=80,            # cap word count so the cloud doesn't get too noisy
        prefer_horizontal=0.85,  # most words are horizontal — easier to read
        font_path=None,          # None lets wordcloud pick a system font
    ).generate(text)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_alpha(0)              # make the figure background transparent
    ax.imshow(wc, interpolation="bilinear")   # bilinear smooths edges between words
    ax.axis("off")                      # hide axes, we only want the image
    plt.tight_layout(pad=0)             # remove whitespace padding around the image

    buf = io.BytesIO()   # write to an in-memory buffer instead of a file on disk
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor="none", transparent=True)
    plt.close(fig)  # release the figure from memory so it doesn't accumulate across renders
    buf.seek(0)     # rewind the buffer to the start before reading
    return buf.read()


def get_sentence_lengths(text: str) -> list[int]:
    """
    Splits the text into sentences and returns a list of word counts.
    Useful for spotting whether the writing is overly uniform or choppy.
    """
    _ensure_nltk()
    sentences = nltk.sent_tokenize(text)   # NLTK's tokeniser handles abbreviations like "Dr." correctly
    # filter out empty strings that can appear at the start or end of a block
    lengths = [len(s.split()) for s in sentences if s.strip()]
    return lengths


# Extend the standard stopword list with a few more filler words that
# tend to show up as "frequent" without actually meaning much
_STOPWORDS = set(STOPWORDS) | {
    "said", "also", "would", "could", "should", "may", "might",
    "one", "two", "three", "us", "like", "get", "got", "use",
}


def get_overused_words(text: str, top_n: int = 8) -> list[tuple[str, int]]:
    """
    Finds the most repeated meaningful words after filtering out stopwords.
    Only looks at words with 3+ characters so single-letter tokens don't clutter results.

    Returns a list of (word, count) tuples, sorted most-frequent first.
    """
    # extract only alphabetic tokens that are at least 3 characters long
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    # drop stopwords — we only care about content words
    filtered = [t for t in tokens if t not in _STOPWORDS]
    return Counter(filtered).most_common(top_n)


def build_annotated_tokens(text: str, highlight_words: set[str]) -> list:
    """
    Breaks the text into a mix of plain strings and annotated tuples
    that the streamlit-annotated-text component understands.

    Each word in highlight_words gets a consistent colour so the same word
    always looks the same across the document preview.
    """
    # eight colours, one assigned per unique highlighted word
    palette = ["#f59e0b", "#34d399", "#60a5fa", "#f472b6",
               "#a78bfa", "#fb923c", "#38bdf8", "#4ade80"]
    # sort the words so the colour assignment is stable across re-renders
    word_color: dict[str, str] = {
        w: palette[i % len(palette)]
        for i, w in enumerate(sorted(highlight_words))
    }

    # split on whitespace but keep the whitespace tokens so the spacing in the
    # original text is preserved when the annotated component reassembles it
    parts = re.split(r"(\s+)", text)
    tokens: list = []
    for part in parts:
        # strip punctuation and quotes before checking if the word is in our highlight set
        clean = part.strip(string.punctuation + "\"'""''").lower()
        if clean in word_color:
            # annotated-text expects a 3-tuple: (display text, label, background colour)
            tokens.append((part, "overused", word_color[clean]))
        else:
            tokens.append(part)   # plain string — no highlight needed
    return tokens
