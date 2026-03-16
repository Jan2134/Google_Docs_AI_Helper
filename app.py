"""
app.py
AI Writing Optimizer for Google Docs — Main Streamlit Application.
Assignment 2 additions:
  - Rewrite tab: 3-call agentic pipeline (Drafter → Critic → Refiner)
  - Chat tab:    multi-turn document Q&A assistant
"""

import streamlit as st
import os
import difflib
import pandas as pd

from google_docs_utils import fetch_document_text, update_document_text
from ai_utils import configure_groq, analyze_document, get_groq_client, chat_with_document
from rewrite_utils import run_rewrite_pipeline
from analytics_utils import (
    get_readability_stats,
    ease_label,
    generate_wordcloud_bytes,
    get_sentence_lengths,
    get_overused_words,
    build_annotated_tokens,
)
from annotated_text import annotated_text

# must be the first Streamlit call in the script
st.set_page_config(
    page_title="AI Writing Optimizer",
    page_icon="✍️",
    layout="wide",
)


def get_groq_api_key() -> str:
    """
    Looks for the Groq API key first in st.secrets, then falls back to an
    environment variable. If neither is set the app stops with a helpful message.
    """
    try:
        return st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        key = os.environ.get("GROQ_API_KEY", "")
        if not key:
            st.error(
                "Groq API Key not found. "
                "Add `GROQ_API_KEY` to `.streamlit/secrets.toml` or set it as an environment variable."
            )
            st.stop()
        return key


# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        }
        .main-title {
            font-size: 2.4rem;
            font-weight: 800;
            background: linear-gradient(90deg, #a78bfa, #60a5fa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            color: #94a3b8;
            font-size: 1rem;
            margin-bottom: 1.5rem;
        }
        section[data-testid="stSidebar"] {
            background: rgba(15, 12, 41, 0.85);
            border-right: 1px solid rgba(167, 139, 250, 0.2);
        }
        div[data-testid="metric-container"] {
            background: rgba(96, 165, 250, 0.08);
            border: 1px solid rgba(96, 165, 250, 0.25);
            border-radius: 12px;
            padding: 16px;
        }
        .stAlert { border-radius: 10px; }
        div[data-testid="stButton"] > button {
            background: linear-gradient(135deg, #7c3aed, #2563eb);
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            padding: 0.6rem 1.4rem;
            width: 100%;
            transition: opacity 0.2s ease;
        }
        div[data-testid="stButton"] > button:hover { opacity: 0.88; }
        textarea {
            background: rgba(255,255,255,0.04) !important;
            border: 1px solid rgba(167,139,250,0.25) !important;
            border-radius: 10px !important;
            color: #e2e8f0 !important;
            font-family: 'Courier New', monospace !important;
            font-size: 0.88rem !important;
        }
        hr { border-color: rgba(167,139,250,0.2); }
        .raw-output {
            background: rgba(0,0,0,0.3);
            border: 1px solid rgba(167,139,250,0.2);
            border-radius: 8px;
            padding: 1rem;
            font-family: 'Courier New', monospace;
            font-size: 0.82rem;
            color: #94a3b8;
            white-space: pre-wrap;
        }
        /* Pipeline step badge */
        .pipeline-step {
            background: rgba(167,139,250,0.1);
            border-left: 3px solid #a78bfa;
            border-radius: 4px;
            padding: 0.4rem 0.8rem;
            margin: 0.3rem 0;
            font-size: 0.84rem;
            color: #c4b5fd;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state ──────────────────────────────────────────────────────────────
if "doc_text" not in st.session_state:
    st.session_state.doc_text = ""
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "score_history" not in st.session_state:
    st.session_state.score_history = []
# Rewrite tab state
if "rewrite_result" not in st.session_state:
    st.session_state.rewrite_result = None
if "rewrite_input_text" not in st.session_state:
    st.session_state.rewrite_input_text = ""
# Chat tab state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✍️ AI Writing Optimizer")
    st.markdown("---")

    st.markdown("### 📄 Google Doc")
    st.caption(
        "Before fetching, share your doc with **Editor** access:\n\n"
        "`agent-249@docs-optimizer.iam.gserviceaccount.com`"
    )
    doc_id = st.text_input(
        "Document ID",
        placeholder="Paste your Doc ID here…",
        help="Found in the Google Docs URL: `docs.google.com/document/d/<ID>/edit`",
    )
    fetch_btn = st.button("Fetch Document", use_container_width=True)

    st.markdown("---")

    st.markdown("### Analysis Settings")
    writing_style = st.selectbox(
        "Writing Style Target",
        options=["General", "Academic", "Business", "Creative", "Technical", "Casual"],
        index=0,
        help="The AI will tailor tone feedback and suggestions to this style.",
    )

    target_clarity = st.slider(
        "Target Clarity Score",
        min_value=1,
        max_value=10,
        value=7,
        step=1,
        help="Your desired clarity level.",
    )

    st.markdown("---")
    st.markdown(
        """
        <small style='color:#64748b;'>
        Auth uses a Google Service Account — no browser login required.
        </small>
        """,
        unsafe_allow_html=True,
    )

# ── Fetch document handler ─────────────────────────────────────────────────────
if fetch_btn:
    if not doc_id.strip():
        st.sidebar.warning("Please enter a Google Document ID first.")
    else:
        with st.spinner("Connecting to Google Docs…"):
            try:
                text = fetch_document_text(doc_id.strip())
                if not text:
                    st.sidebar.warning("The document appears to be empty.")
                else:
                    st.session_state.doc_text = text
                    st.session_state.doc_text_area = text
                    st.session_state.analysis = None
                    st.toast("Document fetched successfully!", icon="📄")
            except FileNotFoundError as e:
                st.sidebar.error(f"{e}")
            except Exception as e:
                st.sidebar.error(f"Error fetching document: {e}")

# ── Header + Save button (above tabs, always visible) ─────────────────────────
hdr_title, hdr_btn = st.columns([4, 1])

with hdr_title:
    st.markdown('<p class="main-title">AI Writing Optimizer for Google Docs</p>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="subtitle">Style: <b>{writing_style}</b> &nbsp;·&nbsp; '
        f'Target clarity: <b>{target_clarity}/10</b> &nbsp;·&nbsp; '
        'Powered by Groq · Llama 3.3 70B</p>',
        unsafe_allow_html=True,
    )

with hdr_btn:
    st.markdown("<div style='margin-top:1.6rem;'></div>", unsafe_allow_html=True)
    can_save = bool(doc_id.strip() and st.session_state.doc_text.strip())
    save_btn = st.button(
        "💾 Save to Google Docs",
        use_container_width=True,
        disabled=not can_save,
        help=(
            "Writes the current text back to your Google Doc."
            if can_save
            else "Fetch a document first, then edit the text before saving."
        ),
    )

if save_btn and can_save:
    with st.status("Saving to Google Docs…", expanded=True) as save_status:
        st.write(f"Document ID: `{doc_id.strip()}`")
        st.write(f"Writing {len(st.session_state.doc_text.split())} words…")
        try:
            update_document_text(doc_id.strip(), st.session_state.doc_text)
            save_status.update(label="Saved successfully!", state="complete", expanded=False)
            st.toast("Document saved to Google Docs!", icon="✅")
        except Exception as e:
            save_status.update(label="Save failed", state="error", expanded=True)
            st.error(f"Could not save: {e}")

st.markdown("---")

# ── Word-level diff renderer ───────────────────────────────────────────────────

def render_word_diff(original: str, revised: str) -> str:
    """
    Returns an HTML string with word-level differences highlighted.
    Deleted words: red background + strikethrough.
    Inserted words: green background.
    Unchanged words: plain.
    """
    orig_words = original.split()
    rev_words  = revised.split()
    matcher    = difflib.SequenceMatcher(None, orig_words, rev_words, autojunk=False)
    parts: list[str] = []

    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            parts.append(" ".join(orig_words[i1:i2]))
        elif op == "replace":
            deleted = " ".join(orig_words[i1:i2])
            added   = " ".join(rev_words[j1:j2])
            parts.append(
                f'<span style="background:rgba(239,68,68,0.25); color:#fca5a5; '
                f'text-decoration:line-through; border-radius:3px; padding:1px 3px; '
                f'margin:0 1px;">{deleted}</span>'
            )
            parts.append(
                f'<span style="background:rgba(34,197,94,0.25); color:#86efac; '
                f'border-radius:3px; padding:1px 3px; margin:0 1px;">{added}</span>'
            )
        elif op == "delete":
            deleted = " ".join(orig_words[i1:i2])
            parts.append(
                f'<span style="background:rgba(239,68,68,0.25); color:#fca5a5; '
                f'text-decoration:line-through; border-radius:3px; padding:1px 3px; '
                f'margin:0 1px;">{deleted}</span>'
            )
        elif op == "insert":
            added = " ".join(rev_words[j1:j2])
            parts.append(
                f'<span style="background:rgba(34,197,94,0.25); color:#86efac; '
                f'border-radius:3px; padding:1px 3px; margin:0 1px;">{added}</span>'
            )

    return " ".join(parts)


# ── Main tabs ──────────────────────────────────────────────────────────────────
tab_analyze, tab_rewrite, tab_chat = st.tabs([
    "📊 Analyze",
    "✏️ Rewrite",
    "💬 Chat",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ANALYZE (original Assignment 1 content, unchanged)
# ══════════════════════════════════════════════════════════════════════════════
with tab_analyze:
    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.markdown("### 📝 Document Content")
        with st.expander("Show / Hide Document Text", expanded=True):
            doc_text_area = st.text_area(
                label="Document Text",
                value=st.session_state.doc_text,
                height=420,
                placeholder=(
                    "Your Google Doc content will appear here after fetching…\n\n"
                    "You can also paste text directly for a quick analysis."
                ),
                label_visibility="collapsed",
                key="doc_text_area",
            )
            if doc_text_area != st.session_state.doc_text:
                st.session_state.doc_text = doc_text_area

        word_count = len(st.session_state.doc_text.split()) if st.session_state.doc_text.strip() else 0
        char_count = len(st.session_state.doc_text)
        st.caption(f"{word_count} words · {char_count} characters")

    with right_col:
        st.markdown("### AI Analysis")
        st.caption("Results appear below after analysis.")

        analyze_btn = st.button("🔍 Analyze Writing", use_container_width=True)

        if analyze_btn:
            text_to_analyze = st.session_state.doc_text.strip()
            if not text_to_analyze:
                st.warning("No text to analyze. Fetch a document or paste text on the left.")
            else:
                api_key = get_groq_api_key()
                configure_groq(api_key)

                with st.status("Analyzing your writing…", expanded=True) as status:
                    st.write(f"Style target: **{writing_style}**")
                    st.write(f"Clarity goal: **{target_clarity}/10**")
                    st.write("Sending to Llama 3.3 70B via Groq…")
                    try:
                        analysis = analyze_document(
                            text_to_analyze,
                            style=writing_style,
                            target_score=target_clarity,
                        )
                        st.session_state.analysis = analysis

                        rs = get_readability_stats(text_to_analyze)
                        label = (
                            doc_id.strip()[:20] + "…"
                            if doc_id.strip()
                            else f"Doc {len(st.session_state.score_history)+1}"
                        )
                        st.session_state.score_history.append({
                            "Document": label,
                            "Clarity Score": analysis.get("clarity_score", 0),
                            "FK Grade Level": rs.get("fk_grade", 0.0),
                        })
                        st.write("Analysis complete!")
                        status.update(label="Analysis ready!", state="complete", expanded=False)
                        st.toast("Analysis complete!", icon="✍️")
                    except Exception as e:
                        status.update(label="Analysis failed", state="error", expanded=True)
                        st.error(f"Analysis failed: {e}")

        if not st.session_state.analysis:
            st.markdown(
                """
                <div style='
                    background: rgba(167,139,250,0.05);
                    border: 1px dashed rgba(167,139,250,0.3);
                    border-radius: 12px;
                    padding: 1.5rem 1rem;
                    text-align: center;
                    margin-top: 1rem;
                    color: #64748b;
                '>
                    <div style='font-size:2rem;'>✍️</div>
                    <p style='margin:0.4rem 0 0; font-size:0.86rem;'>
                        Click <b>Analyze Writing</b><br>to get AI insights.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Analysis results (full-width, below the two columns)
    if st.session_state.analysis:
        analysis = st.session_state.analysis
        score    = analysis.get("clarity_score", 0)
        delta    = score - target_clarity
        delta_label = f"{delta:+d} vs your target" if delta != 0 else "On target!"
        suggestions = analysis.get("suggestions", [])

        st.markdown("---")
        st.markdown("### 📋 Analysis Results")

        res_score, res_suggestions = st.columns([1, 2])

        with res_score:
            st.metric(
                label="Clarity Score",
                value=f"{score} / 10",
                delta=delta_label,
                delta_color="normal",
            )
            st.markdown("**Tone**")
            st.info(analysis.get("tone", "—"), icon="💬")

        with res_suggestions:
            st.markdown("#### 💡 Suggestions")
            icons = ["🔸", "🔹", "🔺"]
            for i, suggestion in enumerate(suggestions):
                st.warning(f"{icons[i]} **Tip {i+1}:** {suggestion}", icon=None)

    # Local analytics dashboard
    st.markdown("---")
    st.markdown("### 📊 Local Analytics Dashboard")
    st.caption("All metrics computed locally — no API calls required.")

    has_text = bool(st.session_state.doc_text.strip())

    tab_read, tab_wc, tab_sent, tab_annot = st.tabs([
        "📈 Readability",
        "Word Cloud",
        "Sentence Lengths",
        "Annotated Text",
    ])

    with tab_read:
        if not has_text:
            st.info("Fetch or paste a document to see readability metrics.")
        else:
            stats = get_readability_stats(st.session_state.doc_text)
            c1, c2, c3 = st.columns(3)
            c1.metric("FK Grade Level", f"{stats['fk_grade']:.1f}",
                      help="Flesch-Kincaid Grade Level: US school grade needed to understand the text.")
            c2.metric("Reading Ease", f"{stats['flesch_ease']:.1f} / 100",
                      delta=ease_label(stats["flesch_ease"]), delta_color="off",
                      help="Flesch Reading Ease: 0 = very hard, 100 = very easy.")
            c3.metric("SMOG Index", f"{stats['smog']:.1f}",
                      help="SMOG Index: years of education needed.")
            c4, c5, c6 = st.columns(3)
            c4.metric("Words", stats["word_count"])
            c5.metric("Avg Sentence Length", f"{stats['avg_sentence']:.1f} words")
            c6.metric("Avg Syllables/Word", f"{stats['avg_syllables']:.2f}")

            with st.expander("How to interpret these scores"):
                st.markdown("""\
| Metric | Ideal range |
|---|---|
| FK Grade Level | 6–10 for general audiences; <6 for mass-market |
| Flesch Reading Ease | ≥60 is accessible; <30 is academic/legal |
| SMOG Index | Rough synonym for FK Grade; use for medical text |
| Avg Sentence Length | 15–20 words is a sweet spot |
| Avg Syllables/Word | <1.5 keeps text accessible |
""")

    with tab_wc:
        if not has_text:
            st.info("Fetch or paste a document to generate a word cloud.")
        elif len(st.session_state.doc_text.split()) < 10:
            st.warning("Need at least 10 words to generate a word cloud.")
        else:
            with st.spinner("Generating word cloud…"):
                try:
                    wc_bytes = generate_wordcloud_bytes(st.session_state.doc_text)
                    st.image(wc_bytes, use_container_width=True,
                             caption="Most frequent terms (stopwords excluded)")
                except Exception as e:
                    st.error(f"Could not generate word cloud: {e}")

    with tab_sent:
        if not has_text:
            st.info("Fetch or paste a document to see sentence length distribution.")
        else:
            lengths = get_sentence_lengths(st.session_state.doc_text)
            if not lengths:
                st.warning("Could not detect any sentences.")
            else:
                df_sent = pd.DataFrame({"Words per Sentence": lengths})
                st.bar_chart(df_sent["Words per Sentence"].value_counts().sort_index(),
                             x_label="Words", y_label="Sentence count",
                             use_container_width=True, color="#a78bfa")
                avg = sum(lengths) / len(lengths)
                mn, mx = min(lengths), max(lengths)
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Avg Length", f"{avg:.1f} words")
                col_b.metric("Shortest", f"{mn} words")
                col_c.metric("Longest", f"{mx} words")
                if avg < 10:
                    st.info("Short sentences detected — the writing may feel choppy.")
                elif avg > 25:
                    st.warning("Long sentences detected — the writing may feel dense.")
                else:
                    st.success("Sentence length looks good (10–25 words on average).")

    with tab_annot:
        if not has_text:
            st.info("Fetch or paste a document to see overused word highlighting.")
        else:
            overused = get_overused_words(st.session_state.doc_text, top_n=8)
            if not overused:
                st.info("Not enough content to identify overused words.")
            else:
                overused_words = {w for w, _ in overused}
                st.markdown("**Top overused words** (excluding stopwords):")
                freq_cols = st.columns(min(len(overused), 4))
                for idx, (word, count) in enumerate(overused):
                    freq_cols[idx % 4].metric(f"`{word}`", f"{count}×")
                st.markdown("---")
                st.markdown("**Document with overused words highlighted:**")
                preview_text = st.session_state.doc_text[:1500]
                if len(st.session_state.doc_text) > 1500:
                    preview_text += " …"
                tokens = build_annotated_tokens(preview_text, overused_words)
                with st.expander("Show annotated text", expanded=True):
                    annotated_text(*tokens)

    # Session progress tracker
    st.markdown("---")
    st.markdown("### 📈 Session Progress Tracker")
    st.caption("Tracks Clarity Score and Readability across every document analyzed this session.")

    if not st.session_state.score_history:
        st.markdown(
            """
            <div style='
                background: rgba(167,139,250,0.05);
                border: 1px dashed rgba(167,139,250,0.3);
                border-radius: 10px;
                padding: 1.2rem;
                text-align: center;
                color: #64748b;
            '>
                Analyze at least one document to start tracking progress.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        df_history = pd.DataFrame(st.session_state.score_history)
        df_history = df_history.set_index("Document")

        h_left, h_right = st.columns(2)
        with h_left:
            st.markdown("**Clarity Score over documents**")
            st.line_chart(df_history[["Clarity Score"]], color="#a78bfa", use_container_width=True)
        with h_right:
            st.markdown("**Flesch-Kincaid Grade Level over documents**")
            st.line_chart(df_history[["FK Grade Level"]], color="#60a5fa", use_container_width=True)

        with st.expander("Full history table"):
            st.dataframe(df_history.reset_index(), use_container_width=True)

        if st.button("Clear history", use_container_width=False):
            st.session_state.score_history = []
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — REWRITE (3-call agentic pipeline)
# Wrapped in @st.fragment so the button triggers a fragment-only rerun,
# keeping the active tab in place instead of resetting to the first tab.
# ══════════════════════════════════════════════════════════════════════════════

@st.fragment
def render_rewrite_tab(writing_style: str, target_clarity: int) -> None:
    st.markdown("### ✏️ Rewrite Assistant")
    st.caption(
        "Three-call agentic pipeline: **Drafter** rewrites your text → "
        "**Critic** scores and annotates it → "
        "*(if score < 7)* **Drafter** revises with feedback → "
        "**Refiner** synthesises the final version."
    )

    st.markdown("---")

    # Settings info
    st.markdown(
        f"<p style='text-align:center; color:#94a3b8; font-size:0.9rem;'>"
        f"Style: <b>{writing_style}</b> &nbsp;·&nbsp; "
        f"Target clarity: <b>{target_clarity}/10</b> &nbsp;·&nbsp; "
        f"Drafter &amp; Refiner: <code>llama-3.3-70b</code> &nbsp;·&nbsp; "
        f"Critic: <code>llama-3.1-8b</code>"
        f"</p>",
        unsafe_allow_html=True,
    )

    has_doc = bool(st.session_state.doc_text.strip())
    if not has_doc:
        st.info("Fetch or paste a document in the **Analyze** tab first.")

    _, btn_col, _ = st.columns([2, 1, 2])
    with btn_col:
        run_pipeline_btn = st.button(
            "🚀 Run Rewrite Pipeline",
            use_container_width=True,
            disabled=not has_doc,
        )

    if run_pipeline_btn:
        api_key = get_groq_api_key()
        configure_groq(api_key)
        client = get_groq_client()

        # snapshot the text at run-time so the diff stays stable even if doc_text changes later
        st.session_state.rewrite_input_text = st.session_state.doc_text.strip()
        st.session_state.rewrite_result = None

        with st.status("Running rewrite pipeline…", expanded=True) as pipe_status:
            try:
                result = run_rewrite_pipeline(
                    client=client,
                    text=st.session_state.rewrite_input_text,
                    style=writing_style,
                    target_score=target_clarity,
                )
                for step in result["steps"]:
                    st.write(step)
                st.session_state.rewrite_result = result
                pipe_status.update(
                    label="Pipeline complete!", state="complete", expanded=False
                )
                st.toast("Rewrite pipeline finished!", icon="✏️")
            except Exception as e:
                pipe_status.update(label="Pipeline failed", state="error", expanded=True)
                st.error(f"Pipeline error: {e}")

    # ── Pipeline results ──────────────────────────────────────────────────────
    if st.session_state.rewrite_result:
        r        = st.session_state.rewrite_result
        original = st.session_state.rewrite_input_text

        st.markdown("---")
        st.markdown("### Results")

        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Critic Score", f"{r['critic_score']}/10",
                  help="Below 7 triggers a second Drafter pass.")
        m2.metric("Drafter Passes", r["iterations"],
                  help="1 = draft was good enough; 2 = Drafter revised with feedback.")
        m3.metric("Refiner Confidence", f"{r['confidence']}/10")
        m4.metric("LLM Calls", 4 + (1 if r["iterations"] == 2 else 0),
                  help="Drafter + Critic (+ optional 2nd Drafter + Critic) + Refiner + Lessons.")

        # ── Section 1: Tracked-changes diff ───────────────────────────────────
        st.markdown("---")
        st.markdown(
            "#### What changed — and why "
            "<span style='font-size:0.8rem; font-weight:400; color:#94a3b8;'>"
            "🟥 removed &nbsp; 🟩 added</span>",
            unsafe_allow_html=True,
        )
        st.caption(
            "This is a *reference*, not a replacement. "
            "Read what changed, understand why, then rewrite in your own words."
        )
        diff_html = render_word_diff(original, r["final_text"])
        st.markdown(
            f'<div style="background:rgba(255,255,255,0.03); border:1px solid rgba(167,139,250,0.2); '
            f'border-radius:10px; padding:1.2rem 1.4rem; line-height:1.8; font-size:0.92rem; '
            f'color:#e2e8f0;">{diff_html}</div>',
            unsafe_allow_html=True,
        )

        # ── Section 2: Critic's notes ──────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Critic's Notes")
        crit_a, crit_b, crit_c = st.columns(3)
        with crit_a:
            st.success(f"**What improved:** {r['improved'] or '—'}")
        with crit_b:
            st.warning(f"**Still weak:** {r['still_weak'] or '—'}")
        with crit_c:
            st.info(f"**Micro-fix applied:** {r['micro_fix'] or '—'}")

        # ── Section 3: Lessons learned ────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Lessons for your next draft")
        st.caption(
            "These rules were extracted from the changes above. "
            "Apply them yourself — don't copy the rewrite."
        )
        for lesson in r.get("lessons", []):
            st.markdown(
                f'<div style="background:rgba(167,139,250,0.08); border-left:3px solid #a78bfa; '
                f'border-radius:6px; padding:0.6rem 1rem; margin:0.4rem 0; color:#e2e8f0; '
                f'font-size:0.9rem;">📌 {lesson}</div>',
                unsafe_allow_html=True,
            )

        # ── Section 4: Full rewritten version (tucked away) ───────────────────
        st.markdown("---")
        with st.expander(
            f"Full rewritten version (reference only — confidence: {r['confidence']}/10)"
        ):
            st.caption(
                "Use this as a benchmark to compare against your own revision, "
                "not as text to copy."
            )
            st.text_area(
                "final_ref", value=r["final_text"],
                height=260, disabled=True, label_visibility="collapsed",
                key="final_display",
            )
            ref_col, _ = st.columns([1, 3])
            with ref_col:
                if st.button("Use as reference → update document text", use_container_width=True):
                    st.session_state.doc_text = r["final_text"]
                    st.toast("Document updated. Use Save to Google Docs when ready.", icon="📋")

        # Pipeline step log
        with st.expander("Pipeline step log"):
            for step in r["steps"]:
                st.markdown(
                    f'<div class="pipeline-step">{step}</div>',
                    unsafe_allow_html=True,
                )

    elif not run_pipeline_btn:
        # Placeholder shown before the first pipeline run
        st.markdown(
            """
            <div style='
                background: rgba(167,139,250,0.05);
                border: 1px dashed rgba(167,139,250,0.3);
                border-radius: 12px;
                padding: 2rem 1rem;
                text-align: center;
                margin-top: 1.5rem;
                color: #64748b;
            '>
                <div style='font-size:2rem;'>✏️</div>
                <p style='margin:0.4rem 0 0; font-size:0.86rem;'>
                    Fetch a document in the <b>Analyze</b> tab, then click
                    <b>Run Rewrite Pipeline</b>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


with tab_rewrite:
    render_rewrite_tab(writing_style, target_clarity)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CHAT (multi-turn document Q&A)
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown("### 💬 Writing Assistant Chat")
    st.caption(
        "Ask anything about your document — the full text is injected as context. "
        "Multi-turn: the assistant remembers earlier messages in this session."
    )

    if not st.session_state.doc_text.strip():
        st.info(
            "Fetch or paste a document first (Analyze tab). "
            "The chat uses your document text as context."
        )
    else:
        word_count_ctx = len(st.session_state.doc_text.split())
        st.caption(
            f"Context: {word_count_ctx} words loaded "
            f"(first 3 000 chars sent per turn) · Style: **{writing_style}**"
        )

        # Render existing conversation
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # New user message
        if prompt := st.chat_input("Ask anything about your document…"):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        api_key = get_groq_api_key()
                        configure_groq(api_key)
                        reply = chat_with_document(
                            messages=st.session_state.chat_history,
                            document_context=st.session_state.doc_text,
                            style=writing_style,
                        )
                        st.markdown(reply)
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": reply}
                        )
                    except Exception as e:
                        err_msg = f"Chat error: {e}"
                        st.error(err_msg)

        # Clear chat button (only shown once there's history)
        if st.session_state.chat_history:
            if st.button("🗑️ Clear chat history", use_container_width=False):
                st.session_state.chat_history = []
                st.rerun()

        # Suggested starter prompts shown when history is empty
        if not st.session_state.chat_history:
            st.markdown("---")
            st.markdown("**Suggested questions:**")
            suggestions_chat = [
                "What is the overall tone of this document?",
                "Which sentences are hardest to understand?",
                "Rewrite the opening paragraph to be more engaging.",
                "What are the three most important ideas in this text?",
                "How can I make this more suitable for a business audience?",
            ]
            for s in suggestions_chat:
                st.markdown(f"- *{s}*")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#475569; font-size:0.8rem;'>"
    "Powered by <b>Groq · Llama 3.3 70B / 3.1 8B</b> · Google Docs API · Built with Streamlit"
    "</p>",
    unsafe_allow_html=True,
)
