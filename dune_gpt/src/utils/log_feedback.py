from datetime import datetime
from logging import Logger

import gspread

from config import (
    SPREADSHEET_ID,
    CREDENTIALS_PATH,
)


def log_to_google_sheet(
    user_id: str,
    feedback_type: str,
    logger: Logger,
    reason: str = None,
    user_email: str = None,
    user_query: str = None,
    ai_response: str = None,
    additional_details: str = None,
) -> None:
    try:
        gc = gspread.service_account(filename=CREDENTIALS_PATH)
        sh = gc.open_by_key(SPREADSHEET_ID)
        worksheet = sh.sheet1

        current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        row_to_append = [
            current_date,
            user_id,
            user_email,
            feedback_type,
            reason,
            additional_details,
            user_query,
            ai_response,
        ]

        worksheet.append_row(row_to_append)
        logger.debug(f"Successfully logged {feedback_type} feedback for user {user_id}")
    except Exception as e:
        logger.error(f"Failed to log feedback to Google Sheets: {e}")
