"""
app.py
AI Writing Optimizer for Google Docs — Main Streamlit Application.
"""

import streamlit as st
import os
import datetime
import pandas as pd

from google_docs_utils import fetch_document_text, update_document_text
from ai_utils import configure_groq, analyze_document
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
    layout="wide",   # wide layout gives us more room for the side-by-side columns
)


def get_groq_api_key() -> str:
    """
    Looks for the Groq API key first in st.secrets, then falls back to an
    environment variable. If neither is set the app stops with a helpful message.
    """
    try:
        # st.secrets reads from .streamlit/secrets.toml — best for local dev
        return st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        # fall back to a regular environment variable (useful for deployed environments)
        key = os.environ.get("GROQ_API_KEY", "")
        if not key:
            st.error(
                "🔑 **Groq API Key not found.** "
                "Add `GROQ_API_KEY` to `.streamlit/secrets.toml` or set it as an environment variable. "
                "Get a free key (no credit card) at https://console.groq.com"
            )
            st.stop()   # halt the script so nothing else renders without a valid key
        return key


# Custom CSS injected directly into the page — keeps all visual styling in one place
# rather than scattering it across individual component arguments
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
        /* used by the raw model output expander */
        .raw-output {
            background: rgba(0,0,0,0.3);
            border: 1px solid rgba(167,139,250,0.2);
            border-radius: 8px;
            padding: 1rem;
            font-family: 'Courier New', monospace;
            font-size: 0.82rem;
            color: #94a3b8;
            white-space: pre-wrap;   /* preserve line breaks from the model response */
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit re-runs the entire script on every user interaction,
# so session_state is how we keep values alive between those re-runs
if "doc_text" not in st.session_state:
    st.session_state.doc_text = ""         # raw text from the fetched or typed document
if "analysis" not in st.session_state:
    st.session_state.analysis = None       # last AI analysis result, or None if not run yet
if "score_history" not in st.session_state:
    # each entry stores a document label, clarity score, and FK grade for the progress chart
    st.session_state.score_history = []

with st.sidebar:
    st.markdown("## ✍️ AI Writing Optimizer")
    st.markdown("---")

    st.markdown("### 📄 Google Doc")
    doc_id = st.text_input(
        "Document ID",
        placeholder="Paste your Doc ID here…",
        # the ID is the long alphanumeric string between /d/ and /edit in the URL
        help="Found in the Google Docs URL: `docs.google.com/document/d/<ID>/edit`",
    )
    fetch_btn = st.button("📥 Fetch Document", use_container_width=True)

    st.markdown("---")

    st.markdown("### Analysis Settings")
    writing_style = st.selectbox(
        "Writing Style Target",
        options=["General", "Academic", "Business", "Creative", "Technical", "Casual"],
        index=0,   # default to "General"
        help="The AI will tailor tone feedback and suggestions to this style.",
    )

    target_clarity = st.slider(
        "Target Clarity Score",
        min_value=1,
        max_value=10,
        value=7,    # 7/10 is a reasonable default — clear but not oversimplified
        step=1,
        help="Your desired clarity level. The metric delta shows how far you are from this goal.",
    )

    st.markdown("---")
    # small-print authentication reminder shown below the controls
    st.markdown(
        """
        <small style='color:#64748b;'>
        Auth uses <code>credentials.json</code> OAuth.<br><br>
        <b>First run?</b> A browser window opens for Google sign-in.
        Token cached in <code>token.json</code>.
        </small>
        """,
        unsafe_allow_html=True,
    )

# handle the Fetch Document button — this block only runs when the button is clicked
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
                    st.session_state.analysis = None   # clear any previous analysis result
                    st.toast("Document fetched successfully!", icon="📄")
            except FileNotFoundError as e:
                # most likely means credentials.json is missing
                st.sidebar.error(f"{e}")
            except Exception as e:
                st.sidebar.error(f"Error fetching document: {e}")

# header row — wide title on left, save button on right in a 4:1 ratio
hdr_title, hdr_btn = st.columns([4, 1])

with hdr_title:
    st.markdown('<p class="main-title">AI Writing Optimizer for Google Docs</p>', unsafe_allow_html=True)
    # show the current settings inline so the user can see what mode they're in at a glance
    st.markdown(
        f'<p class="subtitle">Style: <b>{writing_style}</b> &nbsp;·&nbsp; '
        f'Target clarity: <b>{target_clarity}/10</b> &nbsp;·&nbsp; '
        'Powered by Groq · Llama 3.3 70B</p>',
        unsafe_allow_html=True,
    )

with hdr_btn:
    # small spacer so the button sits roughly level with the bottom of the title
    st.markdown("<div style='margin-top:1.6rem;'></div>", unsafe_allow_html=True)
    # save is only enabled when we have both a doc_id and some text to write back
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

# two-column layout: the document editor takes up 2/3, the analysis panel takes 1/3
left_col, right_col = st.columns([2, 1])

with left_col:
    st.markdown("### 📝 Document Content")

    # the expander lets the user collapse the text area once they're done editing
    with st.expander("Show / Hide Document Text", expanded=True):
        doc_text_area = st.text_area(
            label="Document Text",
            value=st.session_state.doc_text,
            height=420,
            placeholder=(
                "Your Google Doc content will appear here after fetching…\n\n"
                "You can also paste text directly for a quick analysis."
            ),
            label_visibility="collapsed",   # hide the label — the expander title is enough
            key="doc_text_area",
        )
        # sync any manual edits back into session state so the rest of the page sees them
        if doc_text_area != st.session_state.doc_text:
            st.session_state.doc_text = doc_text_area

    # quick stats shown below the text area as a subtle caption
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
            configure_groq(api_key)   # initialise the Groq client with the retrieved key

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

                    # compute readability locally (no API cost) and append to history
                    rs = get_readability_stats(text_to_analyze)
                    # use the first 20 chars of the doc ID as the row label, or a numbered fallback
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

    # show a placeholder card while no analysis has been run yet
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

# analysis results section — rendered full-width below the two-column editor area
if st.session_state.analysis:
    analysis = st.session_state.analysis
    score     = analysis.get("clarity_score", 0)
    delta     = score - target_clarity   # positive means better than target, negative means below
    delta_label = f"{delta:+d} vs your target" if delta != 0 else "On target!"
    suggestions = analysis.get("suggestions", [])

    st.markdown("---")
    st.markdown("### 📋 Analysis Results")

    # three columns: score + download export | tone + raw output | suggestions list
    res_score, res_tone, res_suggestions = st.columns([1, 1, 2])

    with res_score:
        st.metric(
            label="Clarity Score",
            value=f"{score} / 10",
            delta=delta_label,
            delta_color="normal",   # green if positive delta, red if negative
        )
        st.markdown("")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        # build a plain-text export of the analysis for the download button
        export_lines = [
            f"AI Writing Analysis — {timestamp}",
            f"Style: {writing_style}  |  Target Clarity: {target_clarity}/10",
            "=" * 50,
            f"Clarity Score : {score}/10  (delta: {delta:+d} vs target)",
            f"Tone          : {analysis.get('tone', '—')}",
            "",
            "Suggestions:",
        ]
        for i, s in enumerate(suggestions, 1):
            export_lines.append(f"  {i}. {s}")
        st.download_button(
            label="⬇️ Download Analysis",
            data="\n".join(export_lines),
            file_name=f"writing_analysis_{timestamp}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with res_tone:
        st.markdown("#### Tone")
        st.info(analysis.get("tone", "—"), icon="💬")
        # raw output expander is useful for debugging or checking the exact model response
        with st.expander("Raw model output"):
            st.markdown(
                f'<div class="raw-output">{analysis.get("raw", "No raw output available.")}</div>',
                unsafe_allow_html=True,
            )

    with res_suggestions:
        st.markdown("#### 💡 Suggestions")
        icons = ["🔸", "🔹", "🔺"]   # three distinct icons to visually separate each tip
        for i, suggestion in enumerate(suggestions):
            st.warning(f"{icons[i]} **Tip {i+1}:** {suggestion}", icon=None)

# local analytics section — all calculations run client-side, no API calls needed
st.markdown("---")
st.markdown("### 📊 Local Analytics Dashboard")
st.caption("All metrics computed locally — no API calls required.")

# only show meaningful content when there's actually text to analyse
has_text = bool(st.session_state.doc_text.strip())

# four tabs group the different analytics views without cluttering the page
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
        c1.metric(
            "FK Grade Level",
            f"{stats['fk_grade']:.1f}",
            help="Flesch-Kincaid Grade Level: US school grade needed to understand the text.",
        )
        c2.metric(
            "Reading Ease",
            f"{stats['flesch_ease']:.1f} / 100",
            delta=ease_label(stats["flesch_ease"]),   # show a text label instead of a number
            delta_color="off",   # "off" means the label is grey regardless of the value
            help="Flesch Reading Ease: 0 = very hard, 100 = very easy.",
        )
        c3.metric(
            "SMOG Index",
            f"{stats['smog']:.1f}",
            help="SMOG Index: years of education needed. Best for text with 30+ sentences.",
        )
        # second row of metrics for the more granular stats
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
        # the wordcloud library raises an error if there aren't enough words
        st.warning("Need at least 10 words to generate a word cloud.")
    else:
        with st.spinner("Generating word cloud…"):
            try:
                wc_bytes = generate_wordcloud_bytes(st.session_state.doc_text)
                # pass raw bytes directly — Streamlit handles PNG decoding internally
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
            # value_counts groups sentences by length, sort_index orders them left-to-right
            st.bar_chart(df_sent["Words per Sentence"].value_counts().sort_index(),
                         x_label="Words", y_label="Sentence count",
                         use_container_width=True, color="#a78bfa")

            avg = sum(lengths) / len(lengths)
            mn, mx = min(lengths), max(lengths)
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Avg Length", f"{avg:.1f} words")
            col_b.metric("Shortest", f"{mn} words")
            col_c.metric("Longest", f"{mx} words")

            # give the user a quick verdict based on the average sentence length
            if avg < 10:
                st.info("Short sentences detected — the writing may feel choppy. Consider combining a few.")
            elif avg > 25:
                st.warning("Long sentences detected — the writing may feel dense. Consider breaking some up.")
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
            # extract just the word strings into a set for fast membership lookups
            overused_words = {w for w, _ in overused}

            st.markdown("**Top overused words** (excluding stopwords):")
            # cap at 4 columns so the metrics don't get too narrow on small screens
            freq_cols = st.columns(min(len(overused), 4))
            for idx, (word, count) in enumerate(overused):
                freq_cols[idx % 4].metric(f"`{word}`", f"{count}×")

            st.markdown("---")
            st.markdown("**Document with overused words highlighted:**")

            # only preview the first 1500 characters — long docs can make this component sluggish
            preview_text = st.session_state.doc_text[:1500]
            if len(st.session_state.doc_text) > 1500:
                preview_text += " …"   # let the user know the preview is truncated

            tokens = build_annotated_tokens(preview_text, overused_words)
            with st.expander("Show annotated text", expanded=True):
                # unpack the token list — annotated_text expects individual positional args
                annotated_text(*tokens)

# session progress tracker — persists across analyses as long as the tab stays open
st.markdown("---")
st.markdown("### 📈 Session Progress Tracker")
st.caption("Tracks Clarity Score and Readability across every document analyzed this session.")

if not st.session_state.score_history:
    # placeholder shown before the first analysis is run
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
    df_history = df_history.set_index("Document")   # use the doc label as the x-axis

    h_left, h_right = st.columns(2)
    with h_left:
        st.markdown("**Clarity Score over documents**")
        # double brackets keep the result as a DataFrame so line_chart gets column labels
        st.line_chart(df_history[["Clarity Score"]], color="#a78bfa", use_container_width=True)
    with h_right:
        st.markdown("**Flesch-Kincaid Grade Level over documents**")
        st.line_chart(df_history[["FK Grade Level"]], color="#60a5fa", use_container_width=True)

    with st.expander("Full history table"):
        # reset_index brings "Document" back as a regular column so it shows in the table
        st.dataframe(df_history.reset_index(), use_container_width=True)

    if st.button("Clear history", use_container_width=False):
        st.session_state.score_history = []
        st.rerun()   # force a re-run so the empty state placeholder appears immediately

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#475569; font-size:0.8rem;'>"
    "Powered by <b>Groq · Llama 3.3 70B</b> · Google Docs API · Built with Streamlit"
    "</p>",
    unsafe_allow_html=True,
)
