#!/usr/bin/env python3
"""Synchronize AskDUNE feedback from SQLite to Google Sheets."""

import os
import sqlite3
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build


PROJECT_DIR = Path(__file__).resolve().parent.parent

DB_PATH = Path(
    "/exp/dune/data/users/dune/askdune/web_feedback/web_feedback.sqlite3"
)

CREDENTIALS_PATH = PROJECT_DIR / "credentials/google-service-account.json"

SPREADSHEET_ID = "1u_xVMaYqKLNZmsdLVIxkd_hDG-HRwLB9ZZIRj2VcC_c"

SHEET_NAME = "Sheet1"


def sync_feedback():
    credentials = service_account.Credentials.from_service_account_file(
        str(CREDENTIALS_PATH),
        scopes=["https://www.googleapis.com/auth/spreadsheets"],
    )

    service = build("sheets", "v4", credentials=credentials)

    # Retrieve IDs already present in Google Sheets.
    result = service.spreadsheets().values().get(
        spreadsheetId=SPREADSHEET_ID,
        range=f"'{SHEET_NAME}'!A2:A",
    ).execute()

    existing_ids = {
        str(row[0])
        for row in result.get("values", [])
        if row
    }

    # Read feedback without modifying SQLite.
    db_uri = f"file:{DB_PATH}?mode=ro"

    with sqlite3.connect(db_uri, uri=True) as conn:
        rows = conn.execute(
            """
            SELECT id, response_id, timestamp, rating, reasons,
                   comment, question, answer, references_json, model
            FROM feedback
            ORDER BY id
            """
        ).fetchall()

    new_rows = [
        ["" if value is None else str(value) for value in row]
        for row in rows
        if str(row[0]) not in existing_ids
    ]

    if not new_rows:
        print("No new feedback records to synchronize.")
        print(f"Total SQLite records: {len(rows)}")
        return

    # Upload missing records to Google Sheets.
    service.spreadsheets().values().append(
        spreadsheetId=SPREADSHEET_ID,
        range=f"'{SHEET_NAME}'!A:J",
        valueInputOption="RAW",
        insertDataOption="INSERT_ROWS",
        body={"values": new_rows},
    ).execute()

    print(f"Successfully synchronized {len(new_rows)} new records.")
    print(f"Total SQLite records: {len(rows)}")


if __name__ == "__main__":
    sync_feedback()
