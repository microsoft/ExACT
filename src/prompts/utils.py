import os
import logging
from langchain.schema.document import Document
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_openai import OpenAIEmbeddings
from src.logging import time_it


logger = logging.getLogger("retriever")


def display_multimodal_openai_messages(messages):
    all_texts = ""
    for i, message in enumerate(messages):
        role = message["role"]
        all_texts += f"Turn {i + 1} with role {role}\n"
        content = message["content"]
        if isinstance(content, str):
            all_texts += f"{content}\n"
        else:
            for c in content:
                data_type = c["type"]
                if data_type == "text":
                    all_texts += f"{c['text']}\n"
                elif data_type == "image_url":
                    truncated_image_url = c["image_url"]["url"][:50]
                    all_texts += f"Image URL: {truncated_image_url}...\n"
        all_texts += "\n\n"
    return all_texts.strip()


class FaissRetriever:
    def __init__(
        self,
        index_save_path: str,
        docs: list[Document],
        embeddings: OpenAIEmbeddings = None,
        **kwargs
    ):
        fs = LocalFileStore(index_save_path)

        self.docs = docs
        self.embeddings = CacheBackedEmbeddings.from_bytes_store(
            embeddings, fs, namespace="customized"
        )
        self.retriever = self._init_retriever(**kwargs)
        return
    
    def _init_retriever(self, **kwargs):
        docsearch = FAISS.from_documents(
            self.docs, self.embeddings,
            distance_strategy=DistanceStrategy.COSINE
        )
        # retriever = docsearch.as_retriever(**kwargs)
        return docsearch
    
    @time_it
    def retrieve(self, query: str, min_score: float = 0.2, k: int = 3) -> list[Document]:
        self.retriever: FAISS
        docs_w_score = self.retriever._similarity_search_with_relevance_scores(query, k=k)
        logger.debug(f"retrieved {len(docs_w_score)} docs with scores {[score for _, score in docs_w_score]}")
        docs = []
        for doc, score in docs_w_score:
            if score >= min_score:
                docs.append(doc)
        return docs