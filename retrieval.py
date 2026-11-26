import os
from pathlib import Path

from chromadb.errors import NotFoundError
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

PROJECT_DIR = Path(__file__).resolve().parent
INDEX_DIR = PROJECT_DIR / "chroma_db"
CHUNK_LIMIT = 20
EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_CHAT_MODEL = "gpt-6-luna"

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
你是一名机器学习知识库助手。
请根据本轮提供的资料回答问题。
回答时请标明相关知识的来源页码，例如“来源：南瓜书 PDF 第 2 页”。
只能引用本轮资料中明确标注的页码，不得编造页码。
如果资料不足，请说明资料不足，不要使用未经验证的外部知识补充答案。

资料：
{context}
"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
请根据聊天记录完善用户最新的问题，使其在没有聊天记录的情况下也能被理解。
如果问题已经完整，则直接返回原问题。只返回问题，不要回答问题。
"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


class SetupError(Exception):
    pass


def check_openai_key():
    load_dotenv(PROJECT_DIR / ".env")
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise SetupError("Set OPENAI_API_KEY in .env before using OpenAI.")


def open_index(embeddings=None, *, create=False):
    if not create and not (INDEX_DIR / "chroma.sqlite3").is_file():
        raise NotFoundError("Local Chroma database is missing.")
    return Chroma(
        collection_name="langchain",
        persist_directory=str(INDEX_DIR),
        embedding_function=embeddings,
        create_collection_if_not_exists=create,
    )


def combine_docs(docs):
    contents = []
    for doc in docs:
        page = doc.metadata.get("page")
        if isinstance(page, int) and "total_pages" in doc.metadata:
            # The original PyMuPDFLoader index uses zero-based pages; the standalone loader uses one-based pages.
            page += 1
        source = f"PDF 第 {page} 页" if isinstance(page, int) and page > 0 else "页码未知"
        contents.append(f"[{source}]\n{doc.page_content}")
    return "\n\n".join(contents)


def build_qa_chain(llm, retriever):
    rewrite = REWRITE_PROMPT | llm | StrOutputParser()
    search_query = RunnableBranch(
        (lambda request: not request["chat_history"], lambda request: request["input"]),
        rewrite,
    )
    with_context = RunnablePassthrough.assign(
        context=search_query | retriever | combine_docs,
    )
    return with_context.assign(answer=QA_PROMPT | llm | StrOutputParser())


def load_qa_chain():
    try:
        index = open_index()
    except NotFoundError:
        raise SetupError("Build the local index with python document_loader.py.") from None

    if len(index.get(include=[])["ids"]) != CHUNK_LIMIT:
        raise SetupError("The local index is incomplete. Run python document_loader.py.")

    check_openai_key()
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    llm = ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL") or DEFAULT_CHAT_MODEL)
    retriever = open_index(embeddings).as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )
    return build_qa_chain(llm, retriever)
