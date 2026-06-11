from logging import Logger

from slack_bolt import Say
from slack_sdk import WebClient


def assistant_thread_started(
    say: Say,
    payload: dict,
    client: WebClient,
    logger: Logger,
) -> None:
    """Handle the assistant thread start event by greeting the user.

    Args:
        say: Function to send messages to the thread from the app.
        payload: Event payload with thread details (channel, user, etc.).
        client: Slack WebClient for making API calls.
        logger: Logger instance for error tracking.
    """
    try:
        assistant_thread = payload["assistant_thread"]
        channel_id = assistant_thread["channel_id"]
        user_id = assistant_thread["user_id"]
        thread_ts = assistant_thread["thread_ts"]

        say(text="Hello, how can I assist you today?")

        client.chat_postEphemeral(
            channel=channel_id,
            user=user_id,
            thread_ts=thread_ts,
            text=(
                ":bulb: *Tip:* To search Indico & DocDB, type `!docs` before your question; "
                "or for general inquiries not requiring a citation, you can just ask me directly."
            ),
        )
    except Exception as e:
        logger.exception(f"Failed to handle an assistant_thread_started event: {e}")
        say(f":warning: Something went wrong! ({e})")
