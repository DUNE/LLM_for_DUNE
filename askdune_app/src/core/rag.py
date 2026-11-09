from langchain_openai import ChatOpenAI
from langchain_core.runnables import chain, Runnable
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser


def init_llm(
    model: str,
    api_key: str,
    base_url: str,
) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
    )


def get_qa_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "{instruction}",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "human",
                "<context>\n{context}\n</context>\n\n<question>\n{question}\n</question>",
            ),
        ]
    )


def format_docs(docs: list[str]) -> str:
    return "\n\n".join(docs)


def create_qa_chain(
    retriever: Runnable,
    llm: ChatOpenAI,
    prompt: ChatPromptTemplate,
) -> Runnable:

    llm_chain = prompt | llm | StrOutputParser()

    @chain
    def qa_chain(input: dict):
        question = input["question"]
        chat_history = input["chat_history"]
        is_rag_enabled = input["is_rag_enabled"]

        links = []

        if is_rag_enabled:
            content, links = retriever.invoke(question)
            context = format_docs(content)
            instruction = (
                "You are an expert on DUNE documentation. Provide answers strictly based on the "
                "provided context. If the context does not contain the answer, precede your "
                "response with 'This answer does not reference Indico nor DUNE DocDB.'"
            )
        else:
            context = "No documentation provided."
            instruction = (
                "You are a general scientific assitant. Provide concise, helpful answers based on "
                "your general knowledege."
            )

        for chunk in llm_chain.stream(
            {
                "instruction": instruction,
                "chat_history": chat_history,
                "context": context,
                "question": question,
            }
        ):
            yield chunk

        if links:
            sources = "\n\n**Sources:**\n" + "\n".join(
                f"{i}. <{link}>" for i, link in enumerate(links, 1)
            )
            yield sources

    return qa_chain
