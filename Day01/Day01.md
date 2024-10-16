
上一期预告过了要给自己新开一个坑：基于大模型来「从零构建AI面试官」。

由于涉及到`Agent`的构建，那么选择一个合适的`Agent`框架就非常重要。

什么？你说`LangChain`？我和你无冤无仇，你为什么要害我？

哈哈玩笑归玩笑，但是`LangChain`这个爆火的库最近确实遭到了越来越多的质疑。甚至很多开发者有一种**被套牢**的感觉。

这里我就先给大家细数一下`LangChain`的**原罪**。

然后介绍几个我调研之后觉得还不错的**替代品**以及它们各自的特点。

> 声明一下，吐槽框架本身的问题对事不对人，没有任何贬低`LangChain`开发人员和开源社区贡献者的意思。

## LangChain的那些原罪

### 抽象！抽象！抽象！

首先不得不承认，`LangChain`作为在大模型时代早期就异军突起的框架，有着丰富的组件和工具集，这和开发人员和开源社区的贡献者的努力是分不开的。

但`AI`领域尤其是大模型领域，发展速度极其迅猛，导致框架需要在不断的变化中进行迭代，以适应新的需求。

而设计能经得起时间考验的抽象，是一件极其困难的事情，就像`LangChain`遇到的情况一样。

下面是一个使用`OpenAI`的`API`来实现翻译功能的代码示例：

```python
from openai import OpenAI

client = OpenAI(api_key="<your_api_key>")
text = "你好！"
language = "English"

messages = [
    {"role": "system", "content": "你是一个翻译专家"},
    {"role": "user", "content": f"将以下内容从中文翻译成{language}"},
    {"role": "user", "content": f"{text}"},
]

response = client.chat.completions.create(model="gpt-4o", messages=messages)
result = response.choices[0].message.content
```

这个例子里除了`OpenAI`的`API`调用之外，都是标准的`Python`。

如果使用`LangChain`，那么代码可能就变成了这样：

```python
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

os.environ["OPENAI_API_KEY"] = "<your_api_key>"
text = "你好！"
language = "English"


prompt_template = ChatPromptTemplate.from_messages(
    [("system", "你是一个翻译专家"),
     ("user", "将以下内容从中文翻译成{language}"),
     ("user", "{text}")]
)

parser = StrOutputParser()
chain = prompt_template | model | parser
result = chain.invoke({"language": language, "text": text})
```

为了实现相同的功能，这里不得不引入`LangChain`的`ChatPromptTemplate`、`StrOutputParser`，以及隐含的`Chains`（`|`运算符）—— **足足3个新的抽象**，除了增加了代码的复杂度和理解的难度，并没有带来任何实质性的好处。

不光如此，里面看起来丰富且诱人的**辅助函数**，进入源码一看竟然大多是对标准`Python`库的简单封装。

更可怕的是，`LangChain`还习惯在抽象之上再抽象。

抽象本身没有错，但是为了抽象而抽象，甚至引入不必要的抽象，那就是画蛇添足了。

这些无疑都增加了理解和学习的成本。

在快速开发原型的时候，也许还感觉不到问题，但是一旦需要进行产品应用基本的开发，我们就不得不花费大量的时间去理解封装这些抽象的框架代码。

## 文档！文档！文档！

`LangChain`的文档也是出了名的糟糕。

糟糕到什么程度呢？当你要搜索某个`feature`或者`API`的用法的时候，你会发现相关的信息散落在文档网页的**各个角落**，混杂着参考和解释，甚至还有大量的**过时信息**。

网上大家对此已经诟病已久。

![](https://erxuanyi-1257355350.cos.ap-beijing.myqcloud.com/langchain_github-2024-10-10-14-36-35.png)

![](https://erxuanyi-1257355350.cos.ap-beijing.myqcloud.com/langchain_reddit-2024-10-10-14-36-51.png)


## 隐藏！隐藏！隐藏！

`LangChain`的源码中**隐藏了很多不通用的细节**，导致开发人员的应用在不知情的情况下出现了不符合预期的行为。

例如，很多模块中会`hard code`一些`prompt`，这些`prompt`往往没有文档说明，也没有可以修改的接口。

再例如，网上有开发者观察到`ConversationRetrievalChain`会对输入问题的重新表述，这种重新表述有时可能过于泛化，以至于会破坏对话原本的自然流程并使其脱离上下文。

![](https://erxuanyi-1257355350.cos.ap-beijing.myqcloud.com/langchain_medium-2024-10-10-14-45-18.png)


## 我们有其他选择么？

**有！**
### [LlamaIndex](https://www.llamaindex.ai/)
![](https://erxuanyi-1257355350.cos.ap-beijing.myqcloud.com/llamaindex-2024-10-10-16-02-39.png)

从名字就能看出来，这个框架的侧重点在`Index`，也就是数据检索。

它可以处理多达`160`种不同格式的数据源，对于`RAG`的构建来说，可以说是非常的方便。

而且`LlamaIndex`在索引技术方面提供了很强的控制能力和广泛的功能覆盖。

不过功能的强大也伴随着学习成本的增加，`LlamaIndex`的学习曲线相对较陡峭，需要花费一定的时间去理解和掌握。

### [Haystack](https://docs.haystack.deepset.ai/docs/intro) 
![](https://erxuanyi-1257355350.cos.ap-beijing.myqcloud.com/haystack-2024-10-10-16-02-47.png)

这个框架其实比`LangChain`和`LlamaIndex`都要早，但是国内社区相对不活跃，所以知名度不如它们。

`haystack`的目标是用于构建生产基本的的大模型应用，几乎可以与所有主流模型和数据库集成。

同时，框架内提供了非常丰富的组件（数据处理、`embedding`、`rerank`等），并且能够独立使用。

新发布的`2.0`版本具备了更加开放和灵活的架构，尤其擅长构建**语义搜索**和**问答系统**。

### **啥也不用！**

是的你没看错，我们也可以选择啥框架也不用！

其实大多情况下，`LLM`的使用就是简单而直接的`API`调用。我们需要做的就是编写串联流程的代码，以及不断地优化迭代`prompts`。

绝大多数的任务都可以通过简单的代码和相对少的外部工具辅助来实现。

> 这里不包括`Multi-Agent`的场景，因为那需要一个更复杂的框架来调度。

## 小结一下

无论怎么吐槽，选择工具的最重要原则就是**适合**。

每个框架都有其独特的设计哲学和侧重点，我们只需要根据自己的需求选择合适的工具即可。

在调研的时候，了解框架优点的同时一定要同时关注缺点，评估这些缺点是否可以接受，以及在不在自己的可控范围内。

最后，还是那句老话，**没有银弹（No Silver Bullet）**。

