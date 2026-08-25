from slack_bolt import App

from .actions import (
    handle_feedback,
    handle_reason_selection,
)


def register(app: App) -> None:
    app.action("feedback")(handle_feedback)
    app.action("reason_selected")(handle_reason_selection)
