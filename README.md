# AI Writing Optimizer for Google Docs

A Streamlit app that connects to your Google Docs, runs your text through a multi-stage AI pipeline for writing feedback, and layers on a suite of local analytics — all without leaving your browser.

The tool is designed as a **writing coach, not a ghostwriter**. The AI shows you what changed and why, extracts transferable lessons, and keeps your original text visible throughout — so you improve as a writer, not just the document.

---

## Features

### Analyze tab
- **AI Analysis** — sends your document to Llama 3.3 70B (via Groq) and returns a clarity score, tone description, and three targeted improvement suggestions
- **Writing Style Targeting** — choose from General, Academic, Business, Creative, Technical, or Casual to tailor the feedback
- **Readability Metrics** — Flesch-Kincaid Grade Level, Flesch Reading Ease, SMOG Index, average sentence length, and average syllables per word, all computed locally
- **Word Cloud** — visual overview of the most frequent content words in your document
- **Sentence Length Distribution** — bar chart of word counts per sentence so you can spot choppy or dense writing at a glance
- **Overused Word Highlighting** — finds the most repeated meaningful words and highlights them inline using colour-coded annotations
- **Session Progress Tracker** — line charts that compare clarity score and readability grade across every document you analyse in one session
- **Save Back to Google Docs** — write an edited version of the text directly back to the original document

### Rewrite tab *(Assignment 2)*
A four-call agentic pipeline that acts as a writing coach:

1. **Drafter** (`llama-3.3-70b`, temp 0.7) — rewrites your text targeting the chosen style and clarity goal
2. **Critic** (`llama-3.1-8b`, temp 0.1) — scores the draft (1–10) and identifies what improved, what's still weak, and a micro-fix
3. **Drafter pass 2** *(conditional)* — if the critic scores below 7, the Drafter revises again with the feedback attached
4. **Refiner** (`llama-3.3-70b`, temp 0.3) — synthesises the original, best draft, and critique into a final polished version
5. **Lessons** (`llama-3.1-8b`) — extracts 3 transferable writing rules from the diff so you can apply them yourself next time

Results are shown as a **word-level tracked-changes diff** (red strikethrough for deletions, green for insertions) — the same style as Google Docs "suggest edits". The full rewritten version is tucked behind an expander labelled *reference only*, reinforcing that the goal is learning, not copying.

### Chat tab *(Assignment 2)*
A multi-turn conversation assistant with your document injected as context. Ask questions about your writing, request explanations of specific suggestions, or explore style alternatives. The full conversation history is maintained within the session.

---

## Workflow

```
1. Analyze tab  →  fetch document, get baseline clarity score + readability metrics
2. Rewrite tab  →  run pipeline, read the diff + lessons, revise your own text
3. Analyze tab  →  re-analyze your revision to measure improvement
4. (optional)   →  save the updated text back to Google Docs
```

---

## Project Structure

```
.
├── app.py                  # main Streamlit application (3 tabs)
├── ai_utils.py             # Groq client, single-call analysis, multi-turn chat
├── rewrite_utils.py        # 4-call agentic rewrite pipeline
├── analytics_utils.py      # local analytics (readability, word cloud, annotations)
├── google_docs_utils.py    # Google Docs OAuth and read/write helpers
├── requirements.txt
├── .gitignore
└── .streamlit/
    └── secrets.toml        # (not committed) — stores GROQ_API_KEY
```

---

## Setup

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd <repo-folder>
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Get a Groq API key

Sign up at [console.groq.com](https://console.groq.com) — it's free and requires no credit card.

Create `.streamlit/secrets.toml` and add your key:

```toml
GROQ_API_KEY = "gsk_..."
```

Alternatively, export it as an environment variable:

```bash
export GROQ_API_KEY="gsk_..."
```

### 5. Set up Google Docs access

1. Go to the [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project and enable the **Google Docs API**
3. Create an **OAuth 2.0 Client ID** (Desktop application)
4. Download the credentials file and save it as `credentials.json` in the project root

On the first run a browser window will open for Google sign-in. After you approve, a `token.json` file is created and subsequent runs are silent.

> **Note:** `credentials.json` and `token.json` are listed in `.gitignore` and should never be committed.

---

## Running the App

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` by default.

---

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `groq` | Groq API client (LLM inference) |
| `google-api-python-client` | Google Docs API |
| `google-auth`, `google-auth-oauthlib` | OAuth 2.0 authentication |
| `textstat` | Readability formula calculations |
| `wordcloud` | Word cloud image generation |
| `matplotlib` | Rendering the word cloud to PNG |
| `nltk` | Sentence tokenisation |
| `st-annotated-text` | Inline word highlighting component |
| `difflib` | Word-level diff for tracked-changes view (stdlib) |

---

## Models used

| Role | Model | Why |
|---|---|---|
| Analyzer | `llama-3.3-70b-versatile` | High-quality structured analysis |
| Drafter | `llama-3.3-70b-versatile` | Creative rewriting requires the larger model |
| Critic | `llama-3.1-8b-instant` | Fast, cheap — catching obvious issues doesn't need 70B |
| Refiner | `llama-3.3-70b-versatile` | Final synthesis requires quality |
| Lessons | `llama-3.1-8b-instant` | Pattern extraction — small model is sufficient |
| Chat | `llama-3.1-8b-instant` | Low latency matters in conversational turns |

---

## Security Notes

The following files are excluded from version control via `.gitignore`:

- `credentials.json` — OAuth client secret
- `token.json` — cached user access token
- `.streamlit/secrets.toml` — Groq API key
- `.claude/` — local editor settings

Never commit any of these files to a public repository.
