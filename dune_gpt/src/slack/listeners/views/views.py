import json
from logging import Logger

from slack_bolt import Ack
from slack_sdk import WebClient
from slack_sdk.models.views import View
from slack_sdk.models.blocks import AlertBlock

from src.utils.log_feedback import log_to_google_sheet


def handle_negative_feedback_submission(
    ack: Ack,
    body: dict,
    view: dict,
    client: WebClient,
    logger: Logger,
):
    try:
        ack(
            response_action="update",
            view=View(
                type="modal",
                title="Success!",
                close="Close",
                blocks=[
                    AlertBlock(
                        text="Thank you! Feedback successfully submitted!",
                        level="success",
                    )
                ],
            ),
        )

        user_id = body["user"]["id"]
        name = body["user"]["name"]

        input_values = view["state"]["values"]
        selected_options = input_values["reason"]["reason_selected"]["selected_options"]
        reason = ", ".join([option["value"] for option in selected_options])

        additional_details = (
            input_values.get("additional_details", {}).get("text", {}).get("value")
        )

        user_info = client.users_info(user=user_id)
        user_email = user_info["user"]["profile"].get("email")

        metadata_str = view.get("private_metadata", "{}")
        metadata = json.loads(metadata_str) if metadata_str else {}

        channel_id = metadata.get("channel_id")
        message_ts = metadata.get("message_ts")

        user_query = None
        ai_response = None

        if channel_id and message_ts:
            replies = client.conversations_replies(
                channel=channel_id,
                ts=message_ts,
                latest=message_ts,
                inclusive=True,
                limit=2,
            )
            messages = replies.get("messages", [])

        log_to_google_sheet(
            user_id=name,
            user_email=user_email,
            feedback_type="negative",
            reason=reason,
            user_query=user_query,
            ai_response=ai_response,
            additional_details=additional_details,
            logger=logger,
        )

    except Exception as e:
        logger.error(f"Failed to handle negative feedback submission event: {e}")
