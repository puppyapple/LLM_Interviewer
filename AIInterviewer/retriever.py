import torch
from haystack import Pipeline
from haystack.components.retrievers.in_memory import (
    InMemoryBM25Retriever,
    InMemoryEmbeddingRetriever,
)
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.joiners import DocumentJoiner
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.utils import ComponentDevice


class HybridSearchRetriever:
    def __init__(self, document_store):
        self.document_store = document_store

        # 初始化文本嵌入器
        self.text_embedder = SentenceTransformersTextEmbedder(
            model="jinaai/jina-embeddings-v2-base-zh",
            device=ComponentDevice.from_str(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            ),
        )

        # 初始化BM25检索器和嵌入检索器
        self.bm25_retriever = InMemoryBM25Retriever(document_store=self.document_store)
        self.embedding_retriever = InMemoryEmbeddingRetriever(
            document_store=self.document_store
        )

        # 初始化文档合并器
        self.document_joiner = DocumentJoiner()

        # 初始化排序器
        self.ranker = TransformersSimilarityRanker(
            model="jinaai/jina-reranker-v2-base-multilingual",
        )

        # 创建混合检索管道
        self.pipeline = self._create_pipeline()

    def _create_pipeline(self):
        pipeline = Pipeline()
        pipeline.add_component("text_embedder", self.text_embedder)
        pipeline.add_component("embedding_retriever", self.embedding_retriever)
        pipeline.add_component("bm25_retriever", self.bm25_retriever)
        pipeline.add_component("document_joiner", self.document_joiner)
        pipeline.add_component("ranker", self.ranker)

        # 连接组件
        pipeline.connect("text_embedder", "embedding_retriever")
        pipeline.connect("bm25_retriever", "document_joiner")
        pipeline.connect("embedding_retriever", "document_joiner")
        pipeline.connect("document_joiner", "ranker")

        return pipeline

    def retrieve(self, query, top_k=5):
        result = self.pipeline.run(
            {
                "text_embedder": {"text": query},
                "bm25_retriever": {"query": query, "top_k": top_k},
                "embedding_retriever": {"query": query, "top_k": top_k},
                "weight_merger": {
                    "retrievers": ["bm25_retriever", "embedding_retriever"],
                    "weights": [0.3, 0.7],
                    "top_k": top_k,
                },
                "ranker": {"query": query, "top_k": top_k},
            }
        )
        return [doc.content for doc in result["ranker"]["documents"]]
