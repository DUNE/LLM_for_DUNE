import re
from typing import Dict, List
from logging import Logger

from slack_bolt import BoltContext, Say, SetStatus
from slack_sdk import WebClient
from langchain_core.runnables import Runnable

from src.slack.listeners.views.feedback_block import create_feedback_block
from config import RAG_COMMAND, HISTORY_LIMIT

RAG_PATTERN = re.compile(rf"^{re.escape(RAG_COMMAND)}\s+(?=\S)", re.IGNORECASE)


def message(
    client: WebClient,
    context: BoltContext,
    logger: Logger,
    payload: dict,
    say: Say,
    set_status: SetStatus,
    qa_chain: Runnable,
) -> None:
    """Handles when users send messages and generates AI responses.

    Args:
        client: Slack WebClient for making API calls.
        context: Bolt context containing channel and thread information.
        logger: Logger instance for error tracking.
        payload: Event payload with message details (channel, user, text etc.)
        say: Function to send messages to the thread.
        set_status: Function to update assistant's status.
        qa_chain:
    """
    try:
        channel_id = payload["channel"]
        team_id = context.team_id
        thread_ts = payload["thread_ts"]
        ts = payload["ts"]
        user_id = context.user_id
        user_message = payload["text"]

        if match := RAG_PATTERN.match(user_message):
            is_rag_enabled = True
            question = user_message[match.end() :]
            set_status(
                status="searching docs...",
                loading_messages=[
                    "Sifting through DUNE DocDB...",
                    "Scanning Indico meeting notes...",
                    "Assembling relevant documentation...",
                ],
            )
        else:
            is_rag_enabled = False
            question = user_message
            set_status(status="thinking...")

        thread = client.conversations_replies(
            channel=channel_id,
            ts=thread_ts,
            latest=ts,
            limit=HISTORY_LIMIT,
        )

        thread_history: List[Dict[str, str]] = []
        for thread_message in thread["messages"]:
            role = "user" if thread_message.get("bot_id") is None else "assistant"
            thread_history.append({"role": role, "content": thread_message["text"]})

        streamer = client.chat_stream(
            channel=channel_id,
            recipient_team_id=team_id,
            recipient_user_id=user_id,
            thread_ts=thread_ts,
            task_display_mode="timeline",
        )

        for chunk in qa_chain.stream(
            {
                "question": question,
                "chat_history": thread_history,
                "is_rag_enabled": is_rag_enabled,
            }
        ):
            streamer.append(markdown_text=chunk)

        feedback_block = create_feedback_block()
        streamer.stop(
            blocks=feedback_block,
        )

    except Exception as e:
        logger.exception(f"Failed to handle a user message event: {e}")
        say(f":warning: Something went wrong! ({e})")
