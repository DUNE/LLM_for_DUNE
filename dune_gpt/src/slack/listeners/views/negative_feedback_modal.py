import json
from slack_sdk.models.views import View
from slack_sdk.models.blocks import (
    SectionBlock,
    InputBlock,
    CheckboxesElement,
    Option,
)


def create_negative_feedback_modal(channel_id: str, thread_ts: str, message_ts: str) -> View:
    metadata = {
        "channel_id": channel_id,
        "thread_ts": thread_ts,
        "message_ts": message_ts,
    }

    return View(
        type="modal",
        callback_id="negative_feedback_submitted",
        private_metadata=json.dumps(metadata),
        submit="Submit",
        close="Cancel",
        title="What went wrong?",
        blocks=[
            SectionBlock(text="Your feedback helps make AskDUNE better for everyone."),
            InputBlock(
                block_id="reason",
                label="Please select a reason or reasons:",
                element=CheckboxesElement(
                    action_id="reason_selected",
                    options=[
                        Option(text="Inaccurate", value="inaccurate"),
                        Option(text="Other", value="other"),
                    ],
                ),
                dispatch_action=True,
            ),
        ],
    )
