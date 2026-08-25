from logging import Logger

from slack_bolt import Ack
from slack_sdk import WebClient
from slack_sdk.models.blocks import InputBlock, PlainTextInputElement
from slack_sdk.models.views import View

from src.slack.listeners.views.negative_feedback_modal import (
    create_negative_feedback_modal,
)
from src.utils.log_feedback import log_to_google_sheet


def handle_feedback(
    ack: Ack,
    body: dict,
    client: WebClient,
    logger: Logger,
) -> None:
    """Handles user feedback on AI-gnerated responses via thumbs up/down buttons.

    Args:
        ack: Function to acknowledge the action request.
        body: Action payload containing feedback details (message, channel, user, action value).
        client: Slack WebClient for making API calls.
        logger: Logger instance for debugging and error tracking.
    """
    try:
        ack()
        message_ts = body["message"]["ts"]
        channel_id = body["channel"]["id"]
        user_id = body["user"]["id"]
        trigger_id = body["trigger_id"]
        feedback_type = body["actions"][0]["value"]
        is_positive = feedback_type == "good-feedback"

        if is_positive:
            log_to_google_sheet(
                user_id=user_id, feedback_type="positive", logger=logger
            )

            client.chat_postEphemeral(
                channel=channel_id,
                user=user_id,
                thread_ts=message_ts,
                text="We're glad you found this useful.",
            )
        else:
            client.views_open(
                trigger_id=trigger_id,
                view=create_negative_feedback_modal(channel_id, message_ts),
            )

        logger.debug(f"Handled feedback: type={feedback_type}, message_ts={message_ts}")
    except Exception as e:
        logger.error(f"Failed to handle a feedback event: {e}")


def handle_reason_selection(
    ack: Ack,
    body: dict,
    client: WebClient,
    logger: Logger,
):
    try:
        ack()
        view = body["view"]
        view_id = view["id"]
        hash = view["hash"]

        action = body["actions"][0]
        action_id = action["action_id"]

        selected_options = action.get("selected_options", [])

        print(selected_options)
        blocks = []

        for block in view["blocks"]:
            if block.get("block_id") == "additional_details":
                continue

            if "element" in block and block["element"].get("action_id") == action_id:
                if selected_options:
                    block["element"]["initial_options"] = selected_options
                else:
                    block["element"].pop("initial_options", None)

            blocks.append(block)

        if "other" in [opt["value"] for opt in selected_options]:
            blocks.append(
                InputBlock(
                    block_id="additional_details",
                    label="Provide additonal feedback",
                    optional=True,
                    element=PlainTextInputElement(
                        action_id="text",
                        multiline=True,
                    ),
                )
            )

        client.views_update(
            view_id=view_id,
            hash=hash,
            view=View(
                type="modal",
                callback_id=view["callback_id"],
                title=view["title"],
                submit=view["submit"],
                close=view["close"],
                blocks=blocks,
            ),
        )
    except Exception as e:
        logger.error(f"Failed to handle reason selection event: {e}")
