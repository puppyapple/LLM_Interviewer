import requests
import jieba
import os
from loguru import logger
from pathlib import Path
from typing import Optional, Generator, Iterable
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core import SimpleDirectoryReader
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.storage import StorageContext
from llama_index.core import load_index_from_storage
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from typing import List
from llama_index.core import Settings
from llama_index.postprocessor.flag_embedding_reranker import (
    FlagEmbeddingReranker,
)
from .text_chunker import split_text

from .private_consts import JINA_BASE_URL, JINA_API_KEY
from .custom_modules import ChineseBM25Retriever, SimilarityPostprocessorWithSigmoid
from .llm_client import get_llama_index_client


def chinese_tokenizer(text: str) -> Generator[str, None, None]:
    # Use jieba to segment Chinese text
    # return list(jieba.cut(text))
    return jieba.cut_for_search(text)


def custom_splitter(text: str) -> List[str]:

    url = JINA_BASE_URL
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}",
    }
    data = {
        "content": text,
        "return_tokens": False,
        "return_chunks": True,
        "max_chunk_length": 200,
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()["chunks"]


class KnowledgeBase:
    def __init__(
        self,
        api_type: str = "qwen",
        embedding_model: str = "local:BAAI/bge-large-zh-v1.5",
        index_path: Optional[str | Path] = None,
        save_dir: Optional[str | Path] = None,
        file_dir: Optional[str | Path] = None,
        stop_words: Optional[List[str]] = None,
        top_k: int = 20,
        # similarity_cutoff: float = 0.7,
    ):
        self.llm_model_name, self.llm = get_llama_index_client(api_type)
        Settings.llm = self.llm
        self.embedding_model = embedding_model
        Settings.embed_model = self.embedding_model
        self.rerank = FlagEmbeddingReranker(
            model="BAAI/bge-reranker-v2-m3", top_n=top_k
        )

        if index_path and os.path.exists(index_path):
            self.load_index(index_path)
        elif file_dir:
            self.build_index(file_dir)
            if save_dir:
                self.save_index(save_dir)
        else:
            raise ValueError("file_dir 和 index_path 不能同时为空")
        self.setup_query_engine(stop_words=stop_words, top_k=top_k)

    def load_index(self, path):
        self.index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=path)
        )
        logger.info(f"从 {path} 加载索引")

    def build_index(self, doc_dir: str | Path):
        if isinstance(doc_dir, str):
            doc_dir = Path(doc_dir)

        documents = []
        docs_loader = SimpleDirectoryReader(
            input_dir=str(doc_dir),
            recursive=True,
        )
        docs = docs_loader.load_data(num_workers=os.cpu_count())
        documents.extend(docs)
        # pdf_loader = PDFReader()
        # for pdf_file in doc_dir.glob("*.pdf"):
        #     docs = pdf_loader.load_data(file=pdf_file)
        #     doc_text = "\n".join([doc.text for doc in docs])
        #     documents.append(Document(text=doc_text))

        logger.info(f"已加载 {len(documents)} 个文件")

        node_parser = SentenceWindowNodeParser.from_defaults(
            sentence_splitter=split_text,
            window_size=1,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )
        sentence_nodes = node_parser.get_nodes_from_documents(documents)
        logger.info(f"已构建 {len(sentence_nodes)} 个节点")

        self.index = VectorStoreIndex(
            sentence_nodes, show_progress=True, embed_model=self.embedding_model
        )
        logger.info("索引构建完成")

    def setup_query_engine(
        self,
        stop_words: Optional[Iterable[str]] = None,
        top_k: int = 20,
        similarity_cutoff: float = 0.7,
    ):
        if stop_words is None:
            stop_words = []
        vector_retriever = self.index.as_retriever(similarity_top_k=top_k)

        bm25_retriever = ChineseBM25Retriever.from_defaults(
            docstore=self.index.docstore,
            similarity_top_k=top_k,
            stop_words=stop_words,
            tokenizer=chinese_tokenizer,
        )
        retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            llm=None,
            similarity_top_k=top_k,
            num_queries=1,  # set this to 1 to disable query generation
            retriever_weights=[0.6, 0.4],
            mode=FUSION_MODES.DIST_BASED_SCORE,
            use_async=True,
            verbose=True,
        )
        self.similarity_postprocessor = SimilarityPostprocessorWithSigmoid(
            similarity_cutoff=similarity_cutoff,
        )
        self.query_engine = RetrieverQueryEngine.from_args(
            retriever,
            response_mode=ResponseMode.NO_TEXT,
            node_postprocessors=[self.rerank, self.similarity_postprocessor],
        )
        logger.info("查询引擎设置完成")

    def query(self, question, similarity_cutoff=0.7):
        if not self.query_engine:
            raise ValueError("查询引擎尚未设置,请先调用setup_query_engine()")
        self.similarity_postprocessor.similarity_cutoff = similarity_cutoff
        response = self.query_engine.query(question)
        return response

    def save_index(self, path):
        if not self.index:
            raise ValueError("索引尚未建,请先调用build_index()")
        self.index.storage_context.persist(persist_dir=path)
        logger.info(f"索引已保存到 {path}")
