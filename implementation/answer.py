"""
answer.py, call by the outside world
Two key functions:
- fetch_context(question): Retrieve relevant context documents for a question. 
- answer_question(question, history): Answer the given question with RAG; return the answer
"""
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document

from dotenv import load_dotenv

# Vercel自带sqlite3版本较低
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
import os


load_dotenv(override=True)

MODEL = "gpt-4.1-nano"
DB_NAME = str(Path(__file__).parent.parent / "vector_db")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
RETRIEVAL_K = 10


SYSTEM_PROMPT = """
角色设定： 你是 Otonoma 公司的资深分布式系统架构师，负责 Paranet 协议与 Paraflow 语言的技术支持，本学期你与uw ece 有capstone 合作项目。

背景： 你正在协助一名具备后端开发背景（正在学习微服务、gRPC 和分布式一致性）的 UW ECE 研究生。他正在使用 Paranet 开发 Clipper Race 的实时预测工具。

核心原则：

基于文档： 优先根据 {context} 提供的信息回答。

技术映射： 如果涉及 Paranet 的抽象概念，请尝试将其与传统的分布式系统概念（如 Kafka、gRPC、Stateful Actors）进行对比并讲解异同，以便用户快速理解。

处理缺失信息： 如果文档未提及某个功能，不要编造。但请基于分布式系统的一般原理，分析为什么该功能在当前的 Paranet 架构下可能存在限制或尚未实现。

代码要求： 提供 Paraflow 代码示例时，必须确保符合文档中的语法模式，并在代码中添加详细注释。

任务： 请分析用户输入的关于 Clipper Race 预测或 Paranet 接入的问题，并给出具备技术深度的回答。

"""
# 本地配置
# vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)

# 云端配置
remote_client = chromadb.HttpClient(
    host="https://api.trychroma.com",
    headers={"x-chroma-token": os.environ.get("CHROMA_CLOUD_API_KEY")},
    tenant=os.environ.get("CHROMA_TENANT"),
    database=os.environ.get("CHROMA_DATABASE")
)

vectorstore = Chroma(
    client=remote_client,
    collection_name="paradoc_collection",
    embedding_function=embeddings
)


retriever = vectorstore.as_retriever()
llm = ChatOpenAI(temperature=0, model_name=MODEL)


def fetch_context(question: str) -> list[Document]:
    """
    Retrieve relevant context documents for a question.
    """
    return retriever.invoke(question, k=RETRIEVAL_K)


def combined_question(question: str, history: list[dict] = []) -> str:
    """
    Combine all the user's messages into a single string.
    """
    prior = "\n".join(m["content"] for m in history if m["role"] == "user")
    return prior + "\n" + question


def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    """
    Answer the given question with RAG; return the answer and the context documents.
    """
    combined = combined_question(question, history)
    docs = fetch_context(combined)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT.format(context=context)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    return response.content, docs
