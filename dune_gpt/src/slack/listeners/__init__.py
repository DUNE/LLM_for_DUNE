from slack_bolt import App
from langchain_core.runnables import Runnable

from src.slack.listeners import actions, assistant


def register_listeners(app: App, qa_chain: Runnable) -> None:
    actions.register(app)
    assistant.register(app, qa_chain)
