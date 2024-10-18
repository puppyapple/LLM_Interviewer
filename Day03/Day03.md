这一期内容很短但是妥妥的纯干货。

[上一期](https://mp.weixin.qq.com/s/meuW5qrf-dcu_BasP7modg)完整介绍了使用`LlamaIndex`构建一个**基于知识库构建RAG**基本方案和代码。

其中其实没有深入一个细节，那就是对**文本的切分**。

稍微接触过`RAG`的就知道，由于文本结构差异很大，切分时候的坑比较多：

标题、列表、区块引用、代码块、表格、HTML标签、LaTeX表达，等等这些复杂的内容的集合，都给文本切分带来了很大的挑战。

现在很多方案会引入语言模型来进行语义切分，但大大增加了复杂度和开销。

`Jina AI`的`CEO`最近对此发表了自己的看法，并且开源分享了一段50多行的正则为核心的代码，能够高效处理各种复杂的文本切分场景。

![](https://erxuanyi-1257355350.cos.ap-beijing.myqcloud.com/jina_ai-2024-10-18-11-11-56.png)

我在前一期的代码里尝试了很多开源库自带的`splitter`，发现效果都不太好，于是打算试试这个。

这里有个问题在于`Jina AI`的`CEO`的[原始代码](https://gist.github.com/hanxiao/3f60354cf6dc5ac698bc9154163b4e6a)是基于`Typescript`的，里面的正则直接拿到`Python`里又没法直接使用，所以需要一些改造。

下面直接给出改造后的代码，大家有需要的直接拿走不谢：


```python
import regex as re

MAX_HEADING_LENGTH = 10
MAX_HEADING_CONTENT_LENGTH = 200
MAX_HEADING_UNDERLINE_LENGTH = 200
MAX_HTML_HEADING_ATTRIBUTES_LENGTH = 100
MAX_LIST_ITEM_LENGTH = 200
MAX_NESTED_LIST_ITEMS = 6
MAX_LIST_INDENT_SPACES = 7
MAX_BLOCKQUOTE_LINE_LENGTH = 200
MAX_BLOCKQUOTE_LINES = 15
MAX_CODE_BLOCK_LENGTH = 1500
MAX_CODE_LANGUAGE_LENGTH = 20
MAX_INDENTED_CODE_LINES = 20
MAX_TABLE_CELL_LENGTH = 200
MAX_TABLE_ROWS = 20
MAX_HTML_TABLE_LENGTH = 2000
MIN_HORIZONTAL_RULE_LENGTH = 3
MAX_SENTENCE_LENGTH = 400
MAX_QUOTED_TEXT_LENGTH = 300
MAX_PARENTHETICAL_CONTENT_LENGTH = 200
MAX_NESTED_PARENTHESES = 5
MAX_MATH_INLINE_LENGTH = 100
MAX_MATH_BLOCK_LENGTH = 500
MAX_PARAGRAPH_LENGTH = 1000
MAX_STANDALONE_LINE_LENGTH = 800
MAX_HTML_TAG_ATTRIBUTES_LENGTH = 100
MAX_HTML_TAG_CONTENT_LENGTH = 1000
LOOKAHEAD_RANGE = 100

chunk_regex = re.compile(
    r"(" +
    # 1. Headings (Setext-style, Markdown, and HTML-style)
    rf"(?:^(?:[#*=-]{{1,{MAX_HEADING_LENGTH}}}|\w[^\r\n]{{0,{MAX_HEADING_CONTENT_LENGTH}}}\r?\n[-=]{{2,{MAX_HEADING_UNDERLINE_LENGTH}}}|<h[1-6][^>]{{0,{MAX_HTML_HEADING_ATTRIBUTES_LENGTH}}}>)[^\r\n]{{1,{MAX_HEADING_CONTENT_LENGTH}}}(?:</h[1-6]>)?(?:\r?\n|$))"
    + "|"
    +
    # 2. Citations
    rf"(?:\[[0-9]+\][^\r\n]{{1,{MAX_STANDALONE_LINE_LENGTH}}})" + "|" +
    # 3. List items (Adjusted to handle indentation correctly)
    rf"(?:(?:^|\r?\n)[ \t]{{0,3}}(?:[-*+•]|\d{{1,3}}\.\w\.|\[[ xX]\])[ \t]+(?:[^\r\n]{{1,{MAX_LIST_ITEM_LENGTH}}})(?:\r?\n[ \t]{{2,}}(?:[^\r\n]{{1,{MAX_LIST_ITEM_LENGTH}}}))*)"
    + "|"
    +
    # 4. Block quotes (Handles nested quotes without chunking)
    rf"(?:(?:^>(?:>|\\s{{2,}}){{0,2}}(?:[^\r\n]{{0,{MAX_BLOCKQUOTE_LINE_LENGTH}}})(?:\r?\n[ \t]+[^\r\n]{{0,{MAX_BLOCKQUOTE_LINE_LENGTH}}})*?\r?\n?))"
    + "|"
    +
    # 5. Code blocks
    rf"(?:(?:^|\r?\n)(?:```|~~~)(?:\w{{0,{MAX_CODE_LANGUAGE_LENGTH}}})?\r?\n[\s\S]{{0,{MAX_CODE_BLOCK_LENGTH}}}?(?:```|~~~)\r?\n?)"
    + rf"|(?:(?:^|\r?\n)(?: {{4}}|\t)[^\r\n]{{0,{MAX_LIST_ITEM_LENGTH}}}(?:\r?\n(?: {{4}}|\t)[^\r\n]{{0,{MAX_LIST_ITEM_LENGTH}}}){{0,{MAX_INDENTED_CODE_LINES}}}\r?\n?)"
    + rf"|(?:<pre>(?:<code>)[\s\S]{{0,{MAX_CODE_BLOCK_LENGTH}}}?(?:</code>)?</pre>)"
    + "|"
    +
    # 6. Tables
    rf"(?:(?:^|\r?\n)\|[^\r\n]{{0,{MAX_TABLE_CELL_LENGTH}}}\|(?:\r?\n\|[-:]{{1,{MAX_TABLE_CELL_LENGTH}}}\|)?(?:\r?\n\|[^\r\n]{{0,{MAX_TABLE_CELL_LENGTH}}}\|){{0,{MAX_TABLE_ROWS}}})"
    + rf"|<table>[\s\S]{{0,{MAX_HTML_TABLE_LENGTH}}}?</table>"
    + "|"
    +
    # 7. Horizontal rules
    rf"(?:^(?:[-*_]){{{MIN_HORIZONTAL_RULE_LENGTH},}}\s*$|<hr\s*/?>)" + "|" +
    # 8. Standalone lines or phrases (Prevent chunking by treating indented lines as part of the same block)
    rf"(?:^(?:<[a-zA-Z][^>]{{0,{MAX_HTML_TAG_ATTRIBUTES_LENGTH}}}>[^\r\n]{{1,{MAX_STANDALONE_LINE_LENGTH}}}(?:[.!?…]|\.\.\.|[\u2026\u2047-\u2049]|\p{{Emoji_Presentation}}\p{{Extended_Pictographic}})?(?:</[a-zA-Z]+>)?(?:\r?\n|$))"
    + rf"(?:\r?\n[ \t]+[^\r\n]*)*)"
    + "|"
    +
    # 9. Sentences (Allow sentences to include multiple lines if they are indented)
    rf"(?:[^\r\n]{{1,{MAX_SENTENCE_LENGTH}}}(?:[.!?…]|\.\.\.|[\u2026\u2047-\u2049]|\p{{Emoji_Presentation}}\p{{Extended_Pictographic}})?(?=\s|$)(?:\r?\n[ \t]+[^\r\n]*)*)"
    + "|"
    +
    # 10. Quoted text, parentheticals, or bracketed content
    rf"(?<!\w)\"\"\"[^\"]{{0,{MAX_QUOTED_TEXT_LENGTH}}}\"\"\"(?!\w)"
    + rf"|(?<!\w)(?:['\"\`])[^\r\n]{{0,{MAX_QUOTED_TEXT_LENGTH}}}\g<1>(?!\w)"
    + rf"|\([^\r\n()]{0,{MAX_PARENTHETICAL_CONTENT_LENGTH}}(?:\([^\r\n()]{0,{MAX_PARENTHETICAL_CONTENT_LENGTH}}\)[^\r\n()]{0,{MAX_PARENTHETICAL_CONTENT_LENGTH}}){{0,{MAX_NESTED_PARENTHESES}}}\)"
    + rf"|\[[^\r\n\[\]]{{0,{MAX_PARENTHETICAL_CONTENT_LENGTH}}}(?:\[[^\r\n\[\]]{{0,{MAX_PARENTHETICAL_CONTENT_LENGTH}}}\][^\r\n\[\]]{{0,{MAX_PARENTHETICAL_CONTENT_LENGTH}}}){{0,{MAX_NESTED_PARENTHESES}}}\]"
    + rf"|\$[^\r\n$]{{0,{MAX_MATH_INLINE_LENGTH}}}\$"
    + rf"|`[^\r\n`]{{0,{MAX_MATH_INLINE_LENGTH}}}`"
    + "|"
    +
    # 11. Paragraphs (Treats indented lines as part of the same paragraph)
    rf"(?:(?:^|\r?\n\r?\n)(?:<p>)?(?:(?:[^\r\n]{{1,{MAX_PARAGRAPH_LENGTH}}}(?:[.!?…]|\.\.\.|[\u2026\u2047-\u2049]|\p{{Emoji_Presentation}}\p{{Extended_Pictographic}})?(?=\s|$))|(?:[^\r\n]{{1,{MAX_PARAGRAPH_LENGTH}}}(?=[\r\n]|$))|(?:[^\r\n]{{1,{MAX_PARAGRAPH_LENGTH}}}(?=[.!?…]|\.\.\.|[\u2026\u2047-\u2049]|\p{{Emoji_Presentation}}\p{{Extended_Pictographic}}])(?:.{{1,{LOOKAHEAD_RANGE}}}(?:[.!?…]|\.\.\.|[\u2026\u2047-\u2049]|\p{{Emoji_Presentation}}\p{{Extended_Pictographic}}])(?=\s|$))?))(?:</p>)?(?:\r?\n[ \t]+[^\r\n]*)*)"
    + "|"
    +
    # 12. HTML-like tags and their content
    rf"(?:<[a-zA-Z][^>]{{0,{MAX_HTML_TAG_ATTRIBUTES_LENGTH}}}(?:>[\s\S]{{0,{MAX_HTML_TAG_CONTENT_LENGTH}}}</[a-zA-Z]+>|\s*/>))"
    + "|"
    +
    # 13. LaTeX-style math expressions
    rf"(?:(?:\$\$[\s\S]{{0,{MAX_MATH_BLOCK_LENGTH}}}?\$\$)|(?:\$[^\$\r\n]{{0,{MAX_MATH_INLINE_LENGTH}}}\$))"
    + "|"
    +
    # 14. Fallback for any remaining content (Keep content together if it's indented)
    rf"(?:(?:[^\r\n]{{1,{MAX_STANDALONE_LINE_LENGTH}}}(?:[.!?…]|\.\.\.|[\u2026\u2047-\u2049]|\p{{Emoji_Presentation}}\p{{Extended_Pictographic}})?(?=\s|$))|(?:[^\r\n]{{1,{MAX_STANDALONE_LINE_LENGTH}}}(?=[\r\n]|$))|(?:[^\r\n]{{1,{MAX_STANDALONE_LINE_LENGTH}}}(?=[.!?…]|\.\.\.|[\u2026\u2047-\u2049]|\p{{Emoji_Presentation}}\p{{Extended_Pictographic}}])(?:.{{1,{LOOKAHEAD_RANGE}}}(?:[.!?…]|\.\.\.|[\u2026\u2047-\u2049]|\p{{Emoji_Presentation}}\p{{Extended_Pictographic}}])(?=\s|$))(?:\r?\n[ \t]+[^\r\n]*)?))"
    + r")",
    re.MULTILINE | re.UNICODE,
)


def split_text(text):
    matches = chunk_regex.findall(text)
    # print("Raw matches:")
    # for i, match in enumerate(matches):
    #     print(f"{i}: {match}")

    # 提取非空匹配结果，并过滤掉空白片段
    result = [match for group in matches for match in group if match]
    filtered_result = [
        item.strip() for item in result if item and len(item.strip()) > 0
    ]

    # print("\nFiltered result:")
    # for i, item in enumerate(filtered_result):
    #     print(f"{i}: {item}")

    return filtered_result
```

我用上一篇文章的`Markdown`文档测试了一下，耗时只要`621 μs`。


```python
doc = open("../Day02/Day02.md", "r", encoding="utf-8").read()
%time chunks = split_text(doc)
```
```text
CPU times: user 893 μs, sys: 170 μs, total: 1.06 ms
Wall time: 698 μs
```


以**代码块**的情况为例子，看看切分效果：


```python
print(chunks[15])
```

```text
    ```python
    def build_index(self, doc_dir: str | Path):
        if isinstance(doc_dir, str):
            doc_dir = Path(doc_dir)
    
        documents = []
    
        # 处理普通文本文件
        docs_loader = SimpleDirectoryReader(input_dir=str(doc_dir), file_extractor={}) docs = docs_loader.load_data(num_workers=os.cpu_count())
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
```

完美地切出了代码块，比很多开源库自带的`splitter`效果都要好。

尽管没有任何语义理解的介入，但在很多应用场景里，这个程度的文档切分已经很足够了，加上一些`chunk`检索的策略组合，就能实现非常不错的效果。


