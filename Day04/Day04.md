这期歪个楼，之前[微软的GraphRAG](https://github.com/microsoft/graphrag)爆火，后续也出来了很多相关的工作，这不最近港大发布了一个[轻量且快速版的LightRAG](https://github.com/HKUDS/LightRAG)，两周迅速揽🌟`5.6k`。

![lightrag](https://erxuanyi-1257355350.cos.ap-beijing.myqcloud.com/lightrag-2024-10-24-11-32-34.png)

于是我也忍不住跟风尝试一下，看看效果如何，这里记录一下自己尝试的过程，希望对大家有所帮助。

首先安装一下LightRAG库，官方建议从源码安装。


```python
git clone https://github.com/HKUDS/LightRAG.git
cd LightRAG
pip install -e .
```

由于所有的`GraphRAG`都依赖大模型进行文本的**图谱构建**，是相当耗费资源的，所以这里我选择使用`Ollama`部署本地的开源大模型。

模型选用的是`qwen2.5`的`14b`，这里有个小操作：

由于`Ollama`默认了文本的`num_ctx`是`2048`，但是`qwen2.5`的`14b`可以支持更大，所以可以通过修改`Modelfile`的配置来新建一个模型。

具体操作如下：


```python
ollama pull qwen2.5:14b
ollama show --modelfile qwen2.5:14b > Modelfile
# 修改Modelfile中的num_ctx为32768，并且创建模型，命名为qwen2.5:14b_long
ollama create -f Modelfile qwen2.5:14b_long
curl http://0.0.0.0:11434/api/generate -d '{"model": "qwen2.5:14b_long", "keep_alive": -1}'
```

> 这里用`curl`命令启动模型，可以让模型在后台运行，方便后续的调用。
> 具体更多的操作大家可以参考[ollama的api文档](https://github.com/ollama/ollama/blob/main/docs/api.md)

接着我们来配置`LightRAG`。

这里我没有使用大模型的`Embedding`，而是选择了`BAAI/bge-large-zh-v1.5`，效果会更好。

但这里就需要自定义`LightRAG`提供的`EmbeddingFunc`实例，主要是设置`embedding_dim`和`max_token_size`，以及提供具体的`embedding`获取函数（得是`batch`粒度的接口）。


```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from lightrag.utils import EmbeddingFunc

embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-zh-v1.5")
embedding_func = EmbeddingFunc(
    embedding_dim=1024,
    max_token_size=8192,
    func=lambda texts: embedding_model.aget_text_embedding_batch(texts),
)
```

然后就可以初始化`LightRAG`了。

> ⚠️ 注意：如果是在`Jupyter Notebook`中运行，需要使用`nest_asyncio`的`apply()`方法来避免`asyncio`的阻塞问题。


```python
import nest_asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm import ollama_model_complete
from glob import glob

nest_asyncio.apply()
rag = LightRAG(
    working_dir="./interview_lightrag",
    llm_model_func=ollama_model_complete,
    llm_model_name="qwen2.5:14b_long",
    embedding_func=embedding_func,
)
```

接下来只需要向`LightRAG`中插入数据。

这里我用的是我自己本地搜集的一些关于`LLM`的`markdown`文档。

由于文档数量比较多（90个），然后用的是本地的`Ollama`，没有什么并发，相比使用`OpenAI`之类的商业`API`速度肯定是要慢很多的。

但是没法子，谁叫咱穷呢😭；我依稀清楚地记得当初用`OpenAI`的`gpt-4o-mini`跑`GraphRAG`的`demo`时候，就耗费了我`6+`刀，不要太肉疼。

处理完我的全部文件，耗时正好`1h`，如果是并发的`API`，速度应该要快很多。


```python
for file in glob("../AIInterviewer/docs/**/**.md", recursive=True):
    with open(file, "r", encoding="utf-8") as f:
        rag.insert(f.read())
```

提个问试试看吧（我的文档里是有`MQA`的介绍的）。


```python
print(rag.query("介绍下MHA & MQA & MGA？", param=QueryParam(mode="global")))
```


```markdown
看起来您提到的是三个不同的概念，它们分别是：

1. **MHA（MySQL Master HA）**：
   - **定义**：MHA是一种用于增强MySQL数据库高可用性的工具。它通过自动或手动方式实现主从复制模式下的故障转移。
   - **特点**：
     - 自动检测到主机发生故障并进行切换，减少服务中断时间；
     - 可以在主机崩溃前主动将数据复制给备机（基于日志），确保备机上具备最新的数据库状态；

2. **MQA（MySQL Query Analyzer）**：
   - **定义**：这不是一个官方的、广泛认可的技术术语。不过，根据上下文推测，可能指的是用于分析和优化MySQL查询性能的相关工具或技术。
   - **用途**：这种类型的工具有助于找出数据库中慢速执行的SQL语句，并提供改善性能建议。

3. **MGA（Microsoft Graph API）**：
   - 这与前两者无关。它属于微软的一个API服务，用以访问Office 365和Windows Azure Active Directory中的资源。
   - **定义**：Microsoft Graph是一个统一端点，通过它可以访问来自不同服务的数据，并实现应用之间的集成。

综上所述，MHA专注于MySQL高可用性管理，MQA（假设为Query Analyzer）则可能涉及查询性能分析工具，而MGA指的是微软提供的一系列API用于数据交互和应用程序整合。这三个缩写分别代表完全不同的技术领域和技术功能。
```


很遗憾，得到的答复是胡说八道的。

我试了好几个`mode`，包括`naive`、`local`、`global`、`hybrid`，结果都不尽人意。

我分析了一下原因：

1. 首先，`LightRAG`的`prompt`模板是全英文的，所以对于中文的数据和大模型，效果肯定是有影响的。
2. 其次，使用的是开源的`Qwen2.5`的`14b`模型，抽取和图谱构建效果肯定不如`gpt`级别的模型（这一点通过我下面绘制的图谱可以看出来）。
3. 最后，我的文档数据都是专业性很强的大模型新技术，所以`embedding`的效果和信息抽取的难度肯定也更大。

简单看看构建图谱的可视化效果（这段在`Jupyter Notebook`中运行可能不会成功，需要放到脚本中运行得到一个`html`文件然后用浏览器打开）。


```python
import networkx as nx
from pyvis.network import Network

# Load the GraphML file
G = nx.read_graphml("./interview_lightrag/graph_chunk_entity_relation.graphml")

# Create a Pyvis network
net = Network(notebook=True)

# Convert NetworkX graph to Pyvis network
net.from_nx(G)

# Save and display the network
net.show("knowledge_graph.html")
```

图谱大概长这样：

![](https://erxuanyi-1257355350.cos.ap-beijing.myqcloud.com/graph-2024-10-24-15-56-41.png)

可以看到图谱非常稀疏，抽取到的三元组质量也差强人意，所以`LightRAG`的回答效果不好也在情理之中。

官方`issue`里后续也有中文`prompt`的优化计划，等后续有机会我会争取用更大的`LLM`再尝试一下。

不论如何也算是跟风玩了一下`LightRAG`，希望对大家有所帮助。