"""
google_docs_utils.py
Handles OAuth authentication and reading/writing Google Docs content via the Google API.
"""

import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# Full read+write scope — upgraded from .readonly so we can save changes back to the doc.
# Note: if you previously authenticated with the readonly scope, delete token.json first
# so a fresh token with write permissions gets created on the next run.
SCOPES = ["https://www.googleapis.com/auth/documents"]
TOKEN_FILE = "token.json"           # cached credentials; created automatically after first sign-in
CREDENTIALS_FILE = "credentials.json"   # OAuth client config downloaded from Google Cloud Console


def get_google_docs_service():
    """
    Authenticates via OAuth 2.0 and returns a Google Docs API service object.

    On the first run this opens a browser for Google sign-in. After that the
    token is cached in token.json so subsequent runs are silent.
    """
    creds = None

    # try to load an existing token from disk to avoid prompting the user every run
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            # the token has expired but we have a refresh token, so renew it silently
            creds.refresh(Request())
        else:
            # no token at all — kick off the interactive OAuth flow in the browser
            if not os.path.exists(CREDENTIALS_FILE):
                raise FileNotFoundError(
                    f"'{CREDENTIALS_FILE}' not found. Please download it from the "
                    "Google Cloud Console and place it in the project root directory."
                )
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)   # port=0 lets the OS pick a free port

        # persist the new token so the next run doesn't need user interaction
        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())

    # build() returns a Resource object that provides methods for each Docs API endpoint
    service = build("docs", "v1", credentials=creds)
    return service


def fetch_document_text(doc_id: str) -> str:
    """
    Pulls the plain-text content out of a Google Doc.

    The Docs API returns a nested structure of paragraphs and text runs,
    so we walk through that tree and join all the text pieces together.

    Args:
        doc_id: The Google Document ID (the long string in the URL between /d/ and /edit).

    Returns:
        The full document text as a single string.
    """
    service = get_google_docs_service()
    # execute() sends the HTTP request; the result is the full document JSON
    document = service.documents().get(documentId=doc_id).execute()

    # the document body is a list of structural elements (paragraphs, tables, etc.)
    content = document.get("body", {}).get("content", [])
    text_parts = []

    for element in content:
        if "paragraph" in element:
            # each paragraph can contain multiple text runs (e.g. bold/italic segments)
            for para_element in element["paragraph"].get("elements", []):
                text_content = para_element.get("textRun", {}).get("content", "")
                text_parts.append(text_content)

    # join everything into one string and strip leading/trailing whitespace
    return "".join(text_parts).strip()


def update_document_text(doc_id: str, new_text: str) -> None:
    """
    Replaces the entire body of a Google Doc with new_text.

    The approach is: delete everything first, then insert the new content.
    This avoids partial-update headaches with the Docs batchUpdate API.

    Args:
        doc_id:   The Google Document ID.
        new_text: The full replacement text to write.
    """
    service = get_google_docs_service()

    # find out how long the document currently is so we know what range to delete
    document = service.documents().get(documentId=doc_id).execute()
    body_content = document.get("body", {}).get("content", [])
    # subtract 1 to exclude the mandatory trailing newline that Docs always keeps
    end_index = body_content[-1].get("endIndex", 1) - 1

    requests = []

    # only send a delete request if there's actually content to delete
    # (a brand-new empty doc has end_index == 1)
    if end_index > 1:
        requests.append({
            "deleteContentRange": {
                "range": {
                    "startIndex": 1,       # index 1 is the very start of the body text
                    "endIndex": end_index,
                }
            }
        })

    # insert the replacement text at position 1, right after the document's structural start
    if new_text:
        requests.append({
            "insertText": {
                "location": {"index": 1},
                "text": new_text,
            }
        })

    # batchUpdate sends all operations in a single request, reducing round-trips
    if requests:
        service.documents().batchUpdate(
            documentId=doc_id,
            body={"requests": requests},
        ).execute()
