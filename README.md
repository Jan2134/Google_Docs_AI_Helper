# AI Writing Optimizer for Google Docs

A Streamlit app that connects to your Google Docs, runs your text through a large language model for writing feedback, and layers on a suite of local analytics — all without leaving your browser.

---

## Features

- **AI Analysis** — sends your document to Llama 3.3 70B (via Groq) and returns a clarity score, tone description, and three targeted improvement suggestions
- **Writing Style Targeting** — choose from General, Academic, Business, Creative, Technical, or Casual to tailor the feedback
- **Readability Metrics** — Flesch-Kincaid Grade Level, Flesch Reading Ease, SMOG Index, average sentence length, and average syllables per word, all computed locally
- **Word Cloud** — visual overview of the most frequent content words in your document
- **Sentence Length Distribution** — bar chart of word counts per sentence so you can spot choppy or dense writing at a glance
- **Overused Word Highlighting** — finds the most repeated meaningful words and highlights them inline using colour-coded annotations
- **Session Progress Tracker** — line charts that compare clarity score and readability grade across every document you analyse in one session
- **Save Back to Google Docs** — write an edited version of the text directly back to the original document

---

## Project Structure

```
.
├── app.py                  # main Streamlit application
├── ai_utils.py             # Groq API client and response parser
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

## Usage

1. Paste your **Google Document ID** into the sidebar (the long string in the URL between `/d/` and `/edit`)
2. Click **Fetch Document** to load the text
3. Select a **Writing Style** and set your **Target Clarity Score**
4. Click **Analyze Writing** to get AI feedback
5. Browse the **Local Analytics Dashboard** tabs for readability stats, word cloud, sentence distribution, and overused word highlighting
6. Edit the text in the document editor and click **Save to Google Docs** to write changes back

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

---

## Security Notes

The following files are excluded from version control via `.gitignore`:

- `credentials.json` — OAuth client secret
- `token.json` — cached user access token
- `.streamlit/secrets.toml` — Groq API key

Never commit any of these files to a public repository.
