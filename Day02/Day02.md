让大家就等啦，之所以这次更新等了这么久，主要是对上一期（[你说让我用LangChain，和忽悠我节后开盘就满仓有什么区别？｜从零手搓AI面试官｜Day01](https://mp.weixin.qq.com/s/IG2JbNyQZS0IvHIsGUhm3w)）里的几个框架的的`API`做了一些测试和对比。

测试主要是针对`RAG`构建这方面的。

在我的设想里，为了避免幻觉（尤其是使用小尺寸的本地大模型的时候），`AI`面试官最好是基于**本地知识库**来生成问题和评估回答，因此`RAG`的构建是首当其冲的部分。

经过反复对比，最终还是决定使用`Llamaindex`来构建知识库，主要原因还是索引构建方面的`API`比较丰富。（前面的文章也提过了，`Llamaindex`的优势就是构建各种各样的索引，天然适合做`RAG`。）

实现过程中也遇到了一些坑，**文章最后**会列举几个比较重要的点，希望能帮到大家。

## 文件加载和向量索引构建

主流的文档格式一般就是`txt`、`pdf`、`md`、`docx`这几种，`Llamaindex`里内置了这几种格式的`Reader`。

并且直接使用`SimpleDirectoryReader`就可以直接加载指定目录下的所有文件。

除此之外甚至还能支持**PPT、图片、音频**之类的更复杂的文件格式。

![](https://erxuanyi-1257355350.cos.ap-beijing.myqcloud.com/simple_reader-2024-10-14-14-41-52.png)

这里得到的`docs`就是`Document`对象的列表，每个`Document`对象就代表一个文件。

`LlamaIndex`里把每个索引文件称为`Node`，里内置了多种`NodeParser`，可以用于将`Document`对象转换为`Node`对象。
  
这里我选择了`SentenceWindowNodeParser`，它将文档中的每个句子作为一个节点，然后通过`window_size`参数指定每个节点的前后`window_size`个句子，召回的时候将整个窗口的句子一起返回，整体传递给`LLM`。这样可以一定程度上使得返回的上下文**语义更加连贯**。
  
当然，除此之外`LlamaIndex`里还内置了很多文本检索块的策略，有兴趣的大家可以自己去看一下[Node Parser Modules - LlamaIndex](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules/)。

整体的索引构建代码如下：
```python
def build_index(self, doc_dir: str | Path):
    if isinstance(doc_dir, str):
        doc_dir = Path(doc_dir)

    documents = []

    # 处理普通文本文件
    docs_loader = SimpleDirectoryReader(input_dir=str(doc_dir), file_extractor={})
    docs = docs_loader.load_data(num_workers=os.cpu_count())
    documents.extend(docs)

    logger.info(f"已加载 {len(documents)} 个文件")

    node_parser = SentenceWindowNodeParser.from_defaults(
        sentence_splitter=custom_splitter,
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
```

## 混合检索

`RAG`技术里，混合检索（`Hybrid Search`）可以算是**基操**了。

最常见的策略就是**向量检索**和**关键词检索（BM25等）**的结合。

然后`LlamaIndex`里提供了`QueryFusionRetriever`这类检索策略类，可以用于将多个检索策略的结果进行融合，从而提高召回效果。

不过这里有个小问题：`LlamaIndex`里的`BM25Retriever`对中文的支持不太友好，思索再三决定自己继承实现一个`ChineseBM25Retriever`。
    
这部分并没有什么难度，主要就是重写`_retrieve`方法，使其支持中文分词，代码如下：

```python
import jieba
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
from llama_index.retrievers.bm25 import BM25Retriever

def chinese_tokenizer(text: str) -> Generator[str, None, None]:
    # Use jieba to segment Chinese text
    # return list(jieba.cut(text))
    return jieba.cut_for_search(text)


class ChineseBM25Retriever(BM25Retriever):
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
```

## 检索引擎

有了向量索引（`VectorStoreIndex`）和关键词检索（`ChineseBM25Retriever`）之后，就可以构建检索引擎了。
    
```python
def setup_query_engine(self, stop_words: Optional[Iterable[str]] = None):
    if stop_words is None:
        stop_words = []
    vector_retriever = self.index.as_retriever(similarity_top_k=5)

    bm25_retriever = ChineseBM25Retriever.from_defaults(
        docstore=self.index.docstore,
        similarity_top_k=5,
        stop_words=stop_words,
        tokenizer=chinese_tokenizer,
    )
    retriever = QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        llm=None,
        similarity_top_k=5,
        num_queries=1,  # set this to 1 to disable query generation
        retriever_weights=[0.6, 0.4],
        mode=FUSION_MODES.DIST_BASED_SCORE,
        use_async=True,
        verbose=True,
    )
    self.query_engine = RetrieverQueryEngine.from_args(
        retriever, response_mode=ResponseMode.NO_TEXT
    )
    logger.info("查询引擎设置完成")

```
这里列举几个我感觉重要的点：

1. `QueryFusionRetriever`的`retriever_weights`参数，用于指定每个检索策略的权重，这里我取的是`0.6`和`0.4`，表示向量检索的权重为`0.6`，关键词检索的权重为`0.4`。
2. `QueryFusionRetriever`的`mode`参数，用于指定融合模式，这里我取的是`FUSION_MODES.DIST_BASED_SCORE`，表示基于距离的融合模式，即根据每个检索策略的距离得分进行加权融合（会将距离得分归一化到0-1之间）。
3. `num_queries`参数**需要格外注意下**，如果大于1，会启用查询生成策略，即根据当前的查询语句生成新的查询语句，然后分别在各个检索策略中进行检索（如果不希望有额外的大模型开销，这里取1即可）。
4. `RetrieverQueryEngine`的`response_mode`参数，用于指定返回的响应模式，这里我取的是`ResponseMode.NO_TEXT`，表示只返回检索到的节点，不对结果进行任何处理（否则其他的`mode`都会调度大模型进行`summary`这类的操作）。

## 小结

有了上面的各个模块之后，一个最简单的`RAG`的索引构建和查询引擎就实现完成了。

后续需要测试解锁的效果和性能，得先收集一些真实的**知识库**数据。

我打算从开源的`github`项目中收集一些`md`格式的文档，具体的效果等我测试完再和大家分享吧！


