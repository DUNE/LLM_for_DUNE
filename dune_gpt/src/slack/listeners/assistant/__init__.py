from slack_bolt import App, Assistant
from langchain_core.runnables import Runnable
from functools import partial

from .assistant_thread_started import assistant_thread_started
from .message import message


def register(app: App, qa_chain: Runnable) -> None:
    assistant = Assistant()

    assistant.thread_started(assistant_thread_started)
    assistant.user_message(partial(message, qa_chain=qa_chain))

    app.assistant(assistant)
