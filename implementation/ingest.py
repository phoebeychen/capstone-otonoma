"""
ingest.py 
- Read in Knowledge Base
- Turn documents into chunks
- Vectorize chunks
- Store in Chroma
"""


import os
import glob
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings


from dotenv import load_dotenv

MODEL = "gpt-4.1-nano"

DB_NAME = str(Path(__file__).parent.parent / "vector_db")
KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "knowledge-base")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

load_dotenv(override=True)

# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


def fetch_documents():
    folders = glob.glob(str(Path(KNOWLEDGE_BASE) / "*"))
    documents = []
    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(
            folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}
        )
        folder_docs = loader.load()
        for doc in folder_docs:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)
    return documents


def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks


def create_embeddings(chunks):

    # 修改前：本地模式
    # if os.path.exists(DB_NAME):
    #     Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()
    # vectorstore = Chroma.from_documents(
    #    documents=chunks, embedding=embeddings, persist_directory=DB_NAME
    # )


    # 修改后：云端模式
    import chromadb
    from chromadb.config import Settings

    api_key = os.environ.get("CHROMA_CLOUD_API_KEY")
    tenant = os.environ.get("CHROMA_TENANT")
    database = os.environ.get("CHROMA_DATABASE")

    remote_client = chromadb.HttpClient(
        host="https://api.trychroma.com", # 或是您的专属域名
        headers={"x-chroma-token": api_key},
        tenant=tenant,
        database=database,
        ssl=True
    )

    try:
        remote_client.delete_collection("paradoc_collection")
        print("Deleted existing collection.")
    except Exception:
        print("Collection does not exist, creating new one.")

    # 向量化 + 上传数据
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="paradoc_collection",
        client=remote_client
    )

    collection = vectorstore._collection
    count = collection.count()

    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")

    return vectorstore


if __name__ == "__main__":
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion complete")
