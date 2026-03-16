"""
google_docs_utils.py
Authenticates via a Google Service Account and reads/writes Google Docs content.

Setup:
  1. Share each Google Doc with the service account email:
       agent-249@docs-optimizer.iam.gserviceaccount.com  (Editor access)
  2. Place the service account JSON key file in the project root, or store its
     contents in .streamlit/secrets.toml under [gcp_service_account].

Local development:
  The file docs-optimizer-2ee59fbaa2a8.json is loaded automatically when running
  locally. It is excluded from version control via .gitignore.

Streamlit Cloud:
  Add each field from the JSON key file to secrets.toml:
      [gcp_service_account]
      type = "service_account"
      project_id = "..."
      private_key_id = "..."
      private_key = "..."
      client_email = "..."
      token_uri = "..."
      # ... (all remaining fields from the downloaded JSON)
"""

import os
import json
import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/documents"]

# Local service account key file (gitignored — never commit this)
_SERVICE_ACCOUNT_FILE = "docs-optimizer-2ee59fbaa2a8.json"


def _get_credentials():
    """
    Returns service account credentials.
    Prefers Streamlit secrets (for cloud deployment); falls back to the local
    JSON key file (for local development).
    """
    # Streamlit Cloud: credentials stored in secrets.toml under [gcp_service_account]
    if "gcp_service_account" in st.secrets:
        info = dict(st.secrets["gcp_service_account"])
        return service_account.Credentials.from_service_account_info(info, scopes=SCOPES)

    # Local development: load from the JSON key file
    if os.path.exists(_SERVICE_ACCOUNT_FILE):
        return service_account.Credentials.from_service_account_file(
            _SERVICE_ACCOUNT_FILE, scopes=SCOPES
        )

    raise FileNotFoundError(
        f"No service account credentials found. "
        f"Add [gcp_service_account] to .streamlit/secrets.toml, or place "
        f"'{_SERVICE_ACCOUNT_FILE}' in the project root."
    )


def get_google_docs_service():
    """Returns a Google Docs API service object authenticated via service account."""
    creds = _get_credentials()
    return build("docs", "v1", credentials=creds)


def fetch_document_text(doc_id: str) -> str:
    """
    Pulls the plain-text content out of a Google Doc.

    The Docs API returns a nested structure of paragraphs and text runs,
    so we walk through that tree and join all the text pieces together.

    Args:
        doc_id: The Google Document ID (the long string in the URL between /d/ and /edit).

    Returns:
        The full document text as a single string.

    Note:
        The document must be shared with agent-249@docs-optimizer.iam.gserviceaccount.com
        (Editor access) before it can be fetched.
    """
    service = get_google_docs_service()
    document = service.documents().get(documentId=doc_id).execute()

    content = document.get("body", {}).get("content", [])
    text_parts = []

    for element in content:
        if "paragraph" in element:
            for para_element in element["paragraph"].get("elements", []):
                text_content = para_element.get("textRun", {}).get("content", "")
                text_parts.append(text_content)

    return "".join(text_parts).strip()


def update_document_text(doc_id: str, new_text: str) -> None:
    """
    Replaces the entire body of a Google Doc with new_text.

    The approach is: delete everything first, then insert the new content.
    This avoids partial-update headaches with the Docs batchUpdate API.

    Args:
        doc_id:   The Google Document ID.
        new_text: The full replacement text to write.

    Note:
        The document must be shared with agent-249@docs-optimizer.iam.gserviceaccount.com
        (Editor access) before it can be written.
    """
    service = get_google_docs_service()

    document = service.documents().get(documentId=doc_id).execute()
    body_content = document.get("body", {}).get("content", [])
    end_index = body_content[-1].get("endIndex", 1) - 1

    requests = []

    if end_index > 1:
        requests.append({
            "deleteContentRange": {
                "range": {
                    "startIndex": 1,
                    "endIndex": end_index,
                }
            }
        })

    if new_text:
        requests.append({
            "insertText": {
                "location": {"index": 1},
                "text": new_text,
            }
        })

    if requests:
        service.documents().batchUpdate(
            documentId=doc_id,
            body={"requests": requests},
        ).execute()
