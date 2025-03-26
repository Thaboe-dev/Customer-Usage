from typing import Any
from langchain_community.document_loaders.text import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv

docs: list[str] = [
    "data_ETL\CABS.txt",
    "data_ETL\CBZ.txt",
    "data_ETL\ecobank.txt",
    "data_ETL\POSB.txt"
]
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
load_dotenv()


def ingest_docs(embeddings: Any):
    loaded_docs: list[Document] = []
    for item in docs:
        loader = TextLoader(item)
        raw_docs = loader.load()
        loaded_docs += raw_docs
        print(f"loaded {len(raw_docs)} documents")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=6000, 
        chunk_overlap=600,
    )
    documents = text_splitter.split_documents(loaded_docs)

    print(f"Going to index {len(documents)} to Pinecone")
    PineconeVectorStore.from_documents(
        documents, 
        embeddings, 
        index_name="pricing-engine"
    )
    print("-----Loading to VectorStore Done------")


if __name__ == "__main__":
    ingest_docs(embeddings=embeddings)
    # loader = PyPDFDirectoryLoader(docs_path)
    # print("--------------Ingesting----------")
    # raw_docs = loader.load()
    # print("---------------Done---------------")
    # response: dict = raw_docs[0].metadata
    # src: str = response["source"]
