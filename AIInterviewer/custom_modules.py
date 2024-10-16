import jieba
import math
from typing import List, Optional, cast, Callable, Generator, Iterable
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.storage.docstore import BaseDocumentStore
from llama_index.core import VectorStoreIndex
from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core.schema import BaseNode, IndexNode, NodeWithScore, QueryBundle
from llama_index.core.vector_stores.utils import (
    node_to_metadata_dict,
    metadata_dict_to_node,
)
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.retrievers.bm25 import BM25Retriever


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def chinese_tokenizer(text: str) -> Generator[str, None, None]:
    # Use jieba to segment Chinese text
    # return list(jieba.cut(text))
    return jieba.cut_for_search(text)


class ChineseBM25Retriever(BM25Retriever):
    """A BM25 retriever that uses the BM25 algorithm to retrieve nodes.

    Args:
        nodes (List[BaseNode], optional):
            The nodes to index. If not provided, an existing BM25 object must be passed.
        similarity_top_k (int, optional):
            The number of results to return. Defaults to DEFAULT_SIMILARITY_TOP_K.
        callback_manager (CallbackManager, optional):
            The callback manager to use. Defaults to None.
        objects (List[IndexNode], optional):
            The objects to retrieve. Defaults to None.
        object_map (dict, optional):
            A map of object IDs to nodes. Defaults to None.
        verbose (bool, optional):
            Whether to show progress. Defaults to False.
    """

    def __init__(
        self,
        nodes: Optional[List[BaseNode]] = None,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        callback_manager: Optional[CallbackManager] = None,
        objects: Optional[List[IndexNode]] = None,
        object_map: Optional[dict] = None,
        verbose: bool = False,
        stop_words: Optional[Iterable[str]] = set(),
        tokenizer: Optional[
            Callable[[str], Generator[str, None, None] | List[str]]
        ] = chinese_tokenizer,
    ) -> None:

        super().__init__(
            nodes=nodes,
            similarity_top_k=similarity_top_k,
            callback_manager=callback_manager,
            objects=objects,
            object_map=object_map,
            verbose=verbose,
        )

        # change the stop words for Chinese
        self.stop_words = set(stop_words)

        corpus_tokens = [
            [
                word
                for word in tokenizer(node.get_content())
                if word not in self.stop_words and word.strip("\n")
            ]
            for node in nodes
        ]
        corpus = [node_to_metadata_dict(node) for node in nodes]
        self.bm25.corpus = corpus
        self.bm25.index(corpus_tokens, show_progress=True)

    @classmethod
    def from_defaults(
        cls,
        verbose: bool = False,
        index: Optional[VectorStoreIndex] = None,
        nodes: Optional[List[BaseNode]] = None,
        docstore: Optional[BaseDocumentStore] = None,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        tokenizer: Optional[
            Callable[[str], Generator[str, None, None] | List[str]]
        ] = chinese_tokenizer,
        stop_words: Optional[Iterable[str]] = None,
    ) -> "BM25Retriever":

        # ensure only one of index, nodes, or docstore is passed
        if sum(bool(val) for val in [index, nodes, docstore]) != 1:
            raise ValueError("Please pass exactly one of index, nodes, or docstore.")

        if index is not None:
            docstore = index.docstore

        if docstore is not None:
            nodes = cast(List[BaseNode], list(docstore.docs.values()))

        assert (
            nodes is not None
        ), "Please pass exactly one of index, nodes, or docstore."

        return cls(
            nodes=nodes,
            similarity_top_k=similarity_top_k,
            verbose=verbose,
            stop_words=stop_words,
            tokenizer=tokenizer,
        )

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query = query_bundle.query_str

        tokenized_query = [
            [
                word
                for word in jieba.cut_for_search(query)
                if word not in self.stop_words
            ]
        ]

        indexes, scores = self.bm25.retrieve(
            tokenized_query, k=self.similarity_top_k, show_progress=self._verbose
        )

        # batched, but only one query
        indexes = indexes[0]
        scores = scores[0]

        nodes: List[NodeWithScore] = []
        for idx, score in zip(indexes, scores):
            # idx can be an int or a dict of the node
            if isinstance(idx, dict):
                node = metadata_dict_to_node(idx)
            else:
                node_dict = self.corpus[int(idx)]
                node = metadata_dict_to_node(node_dict)
            nodes.append(NodeWithScore(node=node, score=float(score)))

        return nodes


class SimilarityPostprocessorWithSigmoid(SimilarityPostprocessor):
    """Similarity-based Node processor. Return always one result if result is empty"""

    @classmethod
    def class_name(cls) -> str:
        return "SimilarityPostprocessorWithAtLeastOneResult"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        # Call parent class's _postprocess_nodes method first
        for node in nodes:
            node.score = sigmoid(node.score)
        new_nodes = super()._postprocess_nodes(nodes, query_bundle)

        return new_nodes
