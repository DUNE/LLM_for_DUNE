import os

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient

from src.slack.listeners import register_listeners
from src.core.rag import init_llm, create_qa_chain, get_qa_prompt
from src.indexing.chroma_manager import ChromaManager
from src.utils.logger import get_logger
from config import (
    LLM_MODEL,
    FERMILAB_BASE_URL,
    FERMILAB_API_KEY,
    CHROMA_PATH,
    K_DOCS,
    DEFAULT_TOP_K,
)

logger = get_logger(__name__)

app = App(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    client=WebClient(
        base_url=os.environ.get("SLACK_API_URL", "https://slack.com/api"),
        token=os.environ.get("SLACK_BOT_TOKEN"),
    ),
    logger=logger,
)

llm = init_llm(LLM_MODEL, FERMILAB_API_KEY, FERMILAB_BASE_URL)
retriever = ChromaManager(CHROMA_PATH).as_retriever(k_docs=K_DOCS, top_k=DEFAULT_TOP_K)
prompt = get_qa_prompt()
qa_chain = create_qa_chain(retriever, llm, prompt)

register_listeners(app, qa_chain)

if __name__ == "__main__":
    SocketModeHandler(app, os.environ.get("SLACK_APP_TOKEN")).start()
