from slack_bolt import App

from .views import (
    handle_negative_feedback_submission,
)


def register(app: App) -> None:
    app.view("negative_feedback_submitted")(handle_negative_feedback_submission)
