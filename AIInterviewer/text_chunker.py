import regex

MAX_HEADING_LENGTH = 7
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


"""
    // 1. Headings (Setext-style, Markdown, and HTML-style, with length constraints)
    `(?:^(?:[#*=-]{1,${MAX_HEADING_LENGTH}}|\\w[^\\r\\n]{0,${MAX_HEADING_CONTENT_LENGTH}}\\r?\\n[-=]{2,${MAX_HEADING_UNDERLINE_LENGTH}}|<h[1-6][^>]{0,${MAX_HTML_HEADING_ATTRIBUTES_LENGTH}}>)[^\\r\\n]{1,${MAX_HEADING_CONTENT_LENGTH}}(?:</h[1-6]>)?(?:\\r?\\n|$))` +
    "|" +
    // New pattern for citations
    `(?:\\[[0-9]+\\][^\\r\\n]{1,${MAX_STANDALONE_LINE_LENGTH}})` +
"""

heading_regex = regex.compile(
    # 1. Headings (Setext-style, Markdown, and HTML-style, with length constraints)
    rf"(?:^"  # 开始匹配行的开头
    rf"(?:^(?:[#*=-]{{1,{MAX_HEADING_LENGTH}}}"  # Markdown 风格的标题
    rf"|\w[^\r\n]{{0,{MAX_HEADING_CONTENT_LENGTH}}}\r?\n[-=]{{2,{MAX_HEADING_UNDERLINE_LENGTH}}}"  # Setext 风格的标题
    rf"|<h[1-6][^>]{{0,{MAX_HTML_HEADING_ATTRIBUTES_LENGTH}}}>[^\r\n]{{1,{MAX_HEADING_CONTENT_LENGTH}}}"  # HTML 风格的标题
    rf")"
    rf"[^\r\n]{{1,{MAX_HEADING_CONTENT_LENGTH}}})"  # 匹配标题内容
    rf"(?:</h[1-6]>)?"  # 可选的闭合标签
    rf"(?:\r?\n|$)"  # 行尾匹配
    rf")"
    rf"|"
    rf"(?:\[[0-9]+\][^\r\n]{{1,{MAX_STANDALONE_LINE_LENGTH}}})",
    regex.MULTILINE | regex.UNICODE,
)


"""
    // 2. List items (bulleted, numbered, lettered, or task lists, including nested, up to three levels, with length constraints)
    `(?:(?:^|\\r?\\n)[ \\t]{0,3}(?:[-*+•]|\\d{1,3}\\.\\w\\.|\\[[ xX]\\])[ \\t]+
    (?:(?:\\b[^\\r\\n]{1,${MAX_LIST_ITEM_LENGTH}}\\b(?:[.!?…]|\\.{3}|[\\u2026\\u2047-\\u2049]|
    [\\p{Emoji_Presentation}\\p{Extended_Pictographic}])(?=\\s|$))|
    (?:\\b[^\\r\\n]{1,${MAX_LIST_ITEM_LENGTH}}\\b(?=[\\r\\n]|$))|
    (?:\\b[^\\r\\n]{1,${MAX_LIST_ITEM_LENGTH}}\\b(?=[.!?…]|\\.{3}|[\\u2026\\u2047-\\u2049]|[\\p{Emoji_Presentation}\\p{Extended_Pictographic}])
    (?:.{1,${LOOKAHEAD_RANGE}}(?:[.!?…]|\\.{3}|[\\u2026\\u2047-\\u2049]|[\\p{Emoji_Presentation}\\p{Extended_Pictographic}])(?=\\s|$))?))` +


    `(?:(?:\\r?\\n[ \\t]{2,5}(?:[-*+•]|\\d{1,3}\\.\\w\\.|\\[[ xX]\\])[ \\t]+
    (?:(?:\\b[^\\r\\n]{1,${MAX_LIST_ITEM_LENGTH}}\\b
    (?:[.!?…]|\\.{3}|[\\u2026\\u2047-\\u2049]|[\\p{Emoji_Presentation}\\p{Extended_Pictographic}])(?=\\s|$))
    |(?:\\b[^\\r\\n]{1,${MAX_LIST_ITEM_LENGTH}}\\b(?=[\\r\\n]|$))
    |(?:\\b[^\\r\\n]{1,${MAX_LIST_ITEM_LENGTH}}\\b(?=[.!?…]|\\.{3}|[\\u2026\\u2047-\\u2049]|[\\p{Emoji_Presentation}\\p{Extended_Pictographic}])
    (?:.{1,${LOOKAHEAD_RANGE}}(?:[.!?…]|\\.{3}|[\\u2026\\u2047-\\u2049]|[\\p{Emoji_Presentation}\\p{Extended_Pictographic}])(?=\\s|$))?)))` +


    `{0,${MAX_NESTED_LIST_ITEMS}}(?:\\r?\\n[ \\t]{4,${MAX_LIST_INDENT_SPACES}}(?:[-*+•]|\\d{1,3}\\.\\w\\.|\\[[ xX]\\])[ \\t]+
    (?:(?:\\b[^\\r\\n]{1,${MAX_LIST_ITEM_LENGTH}}\\b
    (?:[.!?…]|\\.{3}|[\\u2026\\u2047-\\u2049]|[\\p{Emoji_Presentation}\\p{Extended_Pictographic}])(?=\\s|$))
    |(?:\\b[^\\r\\n]{1,${MAX_LIST_ITEM_LENGTH}}\\b(?=[\\r\\n]|$))
    |(?:\\b[^\\r\\n]{1,${MAX_LIST_ITEM_LENGTH}}\\b(?=[.!?…]|\\.{3}|[\\u2026\\u2047-\\u2049]|[\\p{Emoji_Presentation}\\p{Extended_Pictographic}])
    (?:.{1,${LOOKAHEAD_RANGE}}(?:[.!?…]|\\.{3}|[\\u2026\\u2047-\\u2049]|[\\p{Emoji_Presentation}\\p{Extended_Pictographic}])(?=\\s|$))?)))` +
    `{0,${MAX_NESTED_LIST_ITEMS}})?)` +
"""
list_item_full_regex = regex.compile(
    # 2. List items (bulleted, numbered, lettered, or task lists, including nested, up to three levels, with length constraints)
    rf"(?:(?:^|\r?\n)[ \t]{{0,3}}(?:[-*+•]|\d{{1,3}}\.\w\.|\[[ xX]\])[ \t]+"
    rf"(?:(?:\b[^\r\n]{{1,{MAX_LIST_ITEM_LENGTH}}}\b(?:[.!?…]|\.\.\.|[\u2026\u2047-\u2049]|[\p{{Emoji_Presentation}}\p{{Extended_Pictographic}}])(?=\s|$))"
    rf"|(?:\b[^\r\n]{{1,{MAX_LIST_ITEM_LENGTH}}}\b(?=[\r\n]|$))"
    rf"|(?:\b[^\r\n]{{1,{MAX_LIST_ITEM_LENGTH}}}\b(?=[.!?…]|\.\.\.|[\u2026\u2047-\u2049]|[\p{{Emoji_Presentation}}\p{{Extended_Pictographic}}])"
    rf"(?:.{{1,{LOOKAHEAD_RANGE}}}(?:[.!?…]|\.\.\.|[\u2026\u2047-\u2049]|[\p{{Emoji_Presentation}}\p{{Extended_Pictographic}}])(?=\s|$))?))"
    rf"(?:(?:\r?\n[ \t]{{2,5}}(?:[-*+•]|\d{{1,3}}\.\w\.|\[[ xX]\])[ \t]+"
    rf"(?:(?:\b[^\r\n]{{1,{MAX_LIST_ITEM_LENGTH}}}\b"
    rf"(?:[.!?…]|\.\.\.|[\u2026\u2047-\u2049]|[\p{{Emoji_Presentation}}\p{{Extended_Pictographic}}])(?=\s|$))"
    rf"|(?:\b[^\r\n]{{1,{MAX_LIST_ITEM_LENGTH}}}\b(?=[\r\n]|$))"
    rf"|(?:\b[^\r\n]{{1,{MAX_LIST_ITEM_LENGTH}}}\b(?=[.!?…]|\.\.\.|[\u2026\u2047-\u2049]|[\p{{Emoji_Presentation}}\p{{Extended_Pictographic}}])"
    rf"(?:.{{1,{LOOKAHEAD_RANGE}}}(?:[.!?…]|\.\.\.|[\u2026\u2047-\u2049]|[\p{{Emoji_Presentation}}\p{{Extended_Pictographic}}])(?=\s|$))?)))"
    rf"{{0,{MAX_NESTED_LIST_ITEMS}}}"
    rf"(?:\r?\n[ \t]{{4,{MAX_LIST_INDENT_SPACES}}}(?:[-*+•]|\d{{1,3}}\.\w\.|\[[ xX]\])[ \t]+"
    rf"(?:(?:\b[^\r\n]{{1,{MAX_LIST_ITEM_LENGTH}}}\b"
    rf"(?:[.!?…]|\.\.\.|[\u2026\u2047-\u2049]|[\p{{Emoji_Presentation}}\p{{Extended_Pictographic}}])(?=\s|$))"
    rf"|(?:\b[^\r\n]{{1,{MAX_LIST_ITEM_LENGTH}}}\b(?=[\r\n]|$))"
    rf"|(?:\b[^\r\n]{{1,{MAX_LIST_ITEM_LENGTH}}}\b(?=[.!?…]|\.\.\.|[\u2026\u2047-\u2049]|[\p{{Emoji_Presentation}}\p{{Extended_Pictographic}}])"
    rf"(?:.{{1,{LOOKAHEAD_RANGE}}}(?:[.!?…]|\.\.\.|[\u2026\u2047-\u2049]|[\p{{Emoji_Presentation}}\p{{Extended_Pictographic}}])(?=\s|$))?)))"
    rf"{{0,{MAX_NESTED_LIST_ITEMS}}})?)",
    regex.MULTILINE | regex.UNICODE,
)


"""
    // 3. Block quotes (including nested quotes and citations, up to three levels, with length constraints)
    `(?:(?:^>(?:>|\\s{2,}){0,2}(?:
    (?:\\b[^\\r\\n]{0,${MAX_BLOCKQUOTE_LINE_LENGTH}}\\b(?:[.!?…]|\\.{3}|[\\u2026\\u2047-\\u2049]|[\\p{Emoji_Presentation}\\p{Extended_Pictographic}])(?=\\s|$))
    |(?:\\b[^\\r\\n]{0,${MAX_BLOCKQUOTE_LINE_LENGTH}}\\b(?=[\\r\\n]|$))
    |(?:\\b[^\\r\\n]{0,${MAX_BLOCKQUOTE_LINE_LENGTH}}\\b(?=[.!?…]|\\.{3}|[\\u2026\\u2047-\\u2049]|[\\p{Emoji_Presentation}\\p{Extended_Pictographic}])
    (?:.{1,${LOOKAHEAD_RANGE}}(?:[.!?…]|\\.{3}|[\\u2026\\u2047-\\u2049]|[\\p{Emoji_Presentation}\\p{Extended_Pictographic}])
    (?=\\s|$))?))\\r?\\n?){1,${MAX_BLOCKQUOTE_LINES}})` +
"""


block_quotes_regex = regex.compile(
    # 3. Block quotes (including nested quotes and citations, up to three levels, with length constraints)
    rf"(?:(?:^>(?:>|\s{{2,}}){{0,2}}(?:"
    rf"(?:\b[^\r\n]{{0,{MAX_BLOCKQUOTE_LINE_LENGTH}}}\b(?:[.!?…]|\.{3}|[\u2026\u2047-\u2049]|[\p{{Emoji_Presentation}}\p{{Extended_Pictographic}}])(?=\s|$))"
    rf"|(?:\b[^\r\n]{{0,{MAX_BLOCKQUOTE_LINE_LENGTH}}}\b(?=[\r\n]|$))"
    rf"|(?:\b[^\r\n]{{0,{MAX_BLOCKQUOTE_LINE_LENGTH}}}\b(?=[.!?…]|\.{3}|[\u2026\u2047-\u2049]|[\p{{Emoji_Presentation}}\p{{Extended_Pictographic}}])"
    rf"(?:.{{1,{LOOKAHEAD_RANGE}}}(?:[.!?…]|\.{3}|[\u2026\u2047-\u2049]|[\p{{Emoji_Presentation}}\p{{Extended_Pictographic}}])"
    rf"(?=\s|$))?))\r?\n?){{1,{MAX_BLOCKQUOTE_LINES}}})",
    regex.MULTILINE | regex.UNICODE,
)


# """
#     // 4. Code blocks (fenced, indented, or HTML pre/code tags, with length constraints)
#     `(?:(?:^|\\r?\\n)(?:\`\`\`|~~~)(?:\\w{0,${MAX_CODE_LANGUAGE_LENGTH}})?\\r?\\n[\\s\\S]{0,${MAX_CODE_BLOCK_LENGTH}}?(?:\`\`\`|~~~)\\r?\\n?` +
#     `|(?:(?:^|\\r?\\n)(?: {4}|\\t)[^\\r\\n]{0,${MAX_LIST_ITEM_LENGTH}}(?:\\r?\\n(?: {4}|\\t)[^\\r\\n]{0,${MAX_LIST_ITEM_LENGTH}}){0,${MAX_INDENTED_CODE_LINES}}\\r?\\n?)` +
#     `|(?:<pre>(?:<code>)?[\\s\\S]{0,${MAX_CODE_BLOCK_LENGTH}}?(?:</code>)?</pre>))` +
# """


code_blocks_regex = regex.compile(
    # 4. Code blocks (fenced, indented, or HTML pre/code tags, with length constraints)
    rf"(?:(?:^|\r?\n)(?:\`\`\`|~~~)(?:\w{{0,{MAX_CODE_LANGUAGE_LENGTH}}})?\r?\n[\s\S]{{0,{MAX_CODE_BLOCK_LENGTH}}}?(?:\`\`\`|~~~)\r?\n?"
    rf"|(?:(?:^|\r?\n)(?: {4}|\t)[^\r\n]{{0,{MAX_LIST_ITEM_LENGTH}}}(?:\r?\n(?: {4}|\t)[^\r\n]{{0,{MAX_LIST_ITEM_LENGTH}}}){{0,{MAX_INDENTED_CODE_LINES}}}\r?\n?)"
    rf"|(?:<pre>(?:<code>)?[\s\S]{{0,{MAX_CODE_BLOCK_LENGTH}}}?(?:</code>)?</pre>))",
    regex.MULTILINE | regex.UNICODE,
)


"""
    // 5. Tables (Markdown, grid tables, and HTML tables, with length constraints)
    `(?:(?:^|\\r?\\n)(?:\\|[^\\r\\n]{0,${MAX_TABLE_CELL_LENGTH}}\\|(?:\\r?\\n\\|[-:]{1,${MAX_TABLE_CELL_LENGTH}}\\|){0,1}
    (?:\\r?\\n\\|[^\\r\\n]{0,${MAX_TABLE_CELL_LENGTH}}\\|){0,${MAX_TABLE_ROWS}}` +
    `|<table>[\\s\\S]{0,${MAX_HTML_TABLE_LENGTH}}?</table>))` +
"""


tables_regex = regex.compile(
    # 5. Tables (Markdown, grid tables, and HTML tables, with length constraints)
    rf"(?:(?:^|\r?\n)(?:\|[^\r\n]{{0,{MAX_TABLE_CELL_LENGTH}}}\|(?:\r?\n\|[-:]{{1,{MAX_TABLE_CELL_LENGTH}}}\|)?"
    rf"(?:\r?\n\|[^\r\n]{{0,{MAX_TABLE_CELL_LENGTH}}}\|){{0,{MAX_TABLE_ROWS}}}"
    rf"|<table>[\s\S]{{0,{MAX_HTML_TABLE_LENGTH}}}?</table>))",
    regex.MULTILINE | regex.UNICODE,
)


"""
    // 6. Horizontal rules (Markdown and HTML hr tag)
    `(?:^(?:[-*_]){${MIN_HORIZONTAL_RULE_LENGTH},}\\s*$|<hr\\s*/?>)` +
"""


horizontal_rules_regex = regex.compile(
    # 6. Horizontal rules (Markdown and HTML hr tag)
    rf"(?:^(?:[-*_]){{{MIN_HORIZONTAL_RULE_LENGTH},}}\s*$|<hr\s*/?>)",
    regex.MULTILINE | regex.UNICODE,
)


"""
    // 10. Standalone lines or phrases (including single-line blocks and HTML elements, with length constraints)
    `(?:^(?:<[a-zA-Z][^>]{0,${MAX_HTML_TAG_ATTRIBUTES_LENGTH}}>)?(?:(?:[^\\r\\n]{1,${MAX_STANDALONE_LINE_LENGTH}}(?:[.!?…]|\\.{3}|[\\u2026\\u2047-\\u2049]|[\\p{Emoji_Presentation}\\p{Extended_Pictographic}])(?=\\s|$))|(?:[^\\r\\n]{1,${MAX_STANDALONE_LINE_LENGTH}}(?=[\\r\\n]|$))|(?:[^\\r\\n]{1,${MAX_STANDALONE_LINE_LENGTH}}(?=[.!?…]|\\.{3}|[\\u2026\\u2047-\\u2049]|[\\p{Emoji_Presentation}\\p{Extended_Pictographic}])(?:.{1,${LOOKAHEAD_RANGE}}(?:[.!?…]|\\.{3}|[\\u2026\\u2047-\\u2049]|[\\p{Emoji_Presentation}\\p{Extended_Pictographic}])(?=\\s|$))?))(?:</[a-zA-Z]+>)?(?:\\r?\\n|$))` +
"""


standalone_lines_regex = regex.compile(
    # 10. Standalone lines or phrases (including single-line blocks and HTML elements, with length constraints)
    rf"(?:^(?:<[a-zA-Z][^>]{{0,{MAX_HTML_TAG_ATTRIBUTES_LENGTH}}}>)?(?:(?:[^\r\n]{{1,{MAX_STANDALONE_LINE_LENGTH}}}(?:[.!?…]|\.\.\.|[\u2026\u2047-\u2049]|[\p{{Emoji_Presentation}}\p{{Extended_Pictographic}}])(?=\s|$))|(?:[^\r\n]{{1,{MAX_STANDALONE_LINE_LENGTH}}}(?=[\r\n]|$))|(?:[^\r\n]{{1,{MAX_STANDALONE_LINE_LENGTH}}}(?=[.!?…]|\.\.\.|[\u2026\u2047-\u2049]|[\p{{Emoji_Presentation}}\p{{Extended_Pictographic}}])(?:.{{1,{LOOKAHEAD_RANGE}}}(?:[.!?…]|\.\.\.|[\u2026\u2047-\u2049]|[\p{{Emoji_Presentation}}\p{{Extended_Pictographic}}])(?=\s|$))?))(?:</[a-zA-Z]+>)?(?:\r?\n|$))",
    regex.MULTILINE | regex.UNICODE,
)


"""
    // 7. Sentences or phrases ending with punctuation (including ellipsis and Unicode punctuation)
    `(?:(?:[^\\r\\n]{1,${MAX_SENTENCE_LENGTH}}(?:[.!?…]|\\.{3}|[\\u2026\\u2047-\\u2049]|[\\p{Emoji_Presentation}\\p{Extended_Pictographic}])(?=\\s|$))|(?:[^\\r\\n]{1,${MAX_SENTENCE_LENGTH}}(?=[\\r\\n]|$))|(?:[^\\r\\n]{1,${MAX_SENTENCE_LENGTH}}(?=[.!?…]|\\.{3}|[\\u2026\\u2047-\\u2049]|[\\p{Emoji_Presentation}\\p{Extended_Pictographic}])(?:.{1,${LOOKAHEAD_RANGE}}(?:[.!?…]|\\.{3}|[\\u2026\\u2047-\\u2049]|[\\p{Emoji_Presentation}\\p{Extended_Pictographic}])(?=\\s|$))?))` +
"""


sentences_regex = regex.compile(
    # 7. Sentences or phrases ending with punctuation (including ellipsis and Unicode punctuation)
    rf"(?:(?:[^\r\n]{{1,{MAX_SENTENCE_LENGTH}}}(?:[.!?]|\.\.\.|[\u2026\u2047-\u2049]|[\p{{Emoji_Presentation}}\p{{Extended_Pictographic}}])(?=\s|$))|(?:[^\r\n]{{1,{MAX_SENTENCE_LENGTH}}}(?=[\r\n]|$))|(?:[^\r\n]{{1,{MAX_SENTENCE_LENGTH}}}(?=[.!?…]|\.\.\.|[\u2026\u2047-\u2049]|[\p{{Emoji_Presentation}}\p{{Extended_Pictographic}}])(?:.{{1,{LOOKAHEAD_RANGE}}}(?:[.!?…]|\.\.\.|[\u2026\u2047-\u2049]|[\p{{Emoji_Presentation}}\p{{Extended_Pictographic}}])(?=\s|$))?))",
    regex.MULTILINE | regex.UNICODE,
)


# // 8. Quoted text, parenthetical phrases, or bracketed content (with length constraints)
# "(?:" +
# `(?<!\\w)\"\"\"[^\"]{0,${MAX_QUOTED_TEXT_LENGTH}}\"\"\"(?!\\w)` +
# `|(?<!\\w)(?:['\"\`'"])[^\\r\\n]{0,${MAX_QUOTED_TEXT_LENGTH}}\\1(?!\\w)` +
# `|\\([^\\r\\n()]{0,${MAX_PARENTHETICAL_CONTENT_LENGTH}}(?:\\([^\\r\\n()]{0,${MAX_PARENTHETICAL_CONTENT_LENGTH}}\\)[^\\r\\n()]{0,${MAX_PARENTHETICAL_CONTENT_LENGTH}}){0,${MAX_NESTED_PARENTHESES}}\\)` +
# `|\\[[^\\r\\n\\[\\]]{0,${MAX_PARENTHETICAL_CONTENT_LENGTH}}(?:\\[[^\\r\\n\\[\\]]{0,${MAX_PARENTHETICAL_CONTENT_LENGTH}}\\][^\\r\\n\\[\\]]{0,${MAX_PARENTHETICAL_CONTENT_LENGTH}}){0,${MAX_NESTED_PARENTHESES}}\\]` +
# `|\\$[^\\r\\n$]{0,${MAX_MATH_INLINE_LENGTH}}\\$` +
# `|\`[^\`\\r\\n]{0,${MAX_MATH_INLINE_LENGTH}}\`` +
# ")" +


quoted_regex = regex.compile(
    # 8. Quoted text, parenthetical phrases, or bracketed content (with length constraints)
    rf"(?:"
    rf"(?<!\w)\"\"\"[^\"]{{0,{MAX_QUOTED_TEXT_LENGTH}}}\"\"\"(?!\w)"
    rf"|(?<!\w)(?:['\"\`\'])[^\r\n]{{0,{MAX_QUOTED_TEXT_LENGTH}}}\\1(?!\w)"
    rf"|\([^\r\n()]{{0,{MAX_PARENTHETICAL_CONTENT_LENGTH}}}(?:\([^\r\n()]{{0,{MAX_PARENTHETICAL_CONTENT_LENGTH}}}\)[^\r\n()]{{0,{MAX_PARENTHETICAL_CONTENT_LENGTH}}}){{0,{MAX_NESTED_PARENTHESES}}}\)"
    rf"|\[[^\r\n\[\]]{{0,{MAX_PARENTHETICAL_CONTENT_LENGTH}}}(?:\[[^\r\n\[\]]{{0,{MAX_PARENTHETICAL_CONTENT_LENGTH}}}\][^\r\n\[\]]{{0,{MAX_PARENTHETICAL_CONTENT_LENGTH}}}){{0,{MAX_NESTED_PARENTHESES}}}\]"
    rf"|\$[^\r\n$]{{0,{MAX_MATH_INLINE_LENGTH}}}\$"
    rf"|\"[^\"\r\n]{{0,{MAX_MATH_INLINE_LENGTH}}}\""
    rf")",
    regex.MULTILINE | regex.UNICODE,
)


"""
    // 9. Paragraphs (with length constraints)
    `(?:(?:^|\\r?\\n\\r?\\n)(?:<p>)?(?:(?:[^\\r\\n]{1,${MAX_PARAGRAPH_LENGTH}}(?:[.!?…]|\\.{3}|[\\u2026\\u2047-\\u2049]|[\\p{Emoji_Presentation}\\p{Extended_Pictographic}])(?=\\s|$))|(?:[^\\r\\n]{1,${MAX_PARAGRAPH_LENGTH}}(?=[\\r\\n]|$))|(?:[^\\r\\n]{1,${MAX_PARAGRAPH_LENGTH}}(?=[.!?…]|\\.{3}|[\\u2026\\u2047-\\u2049]|[\\p{Emoji_Presentation}\\p{Extended_Pictographic}])(?:.{1,${LOOKAHEAD_RANGE}}(?:[.!?…]|\\.{3}|[\\u2026\\u2047-\\u2049]|[\\p{Emoji_Presentation}\\p{Extended_Pictographic}])(?=\\s|$))?))(?:</p>)?(?=\\r?\\n\\r?\\n|$))` +
"""


paragraphs_regex = regex.compile(
    # 9. Paragraphs (with length constraints)
    rf"(?:(?:^|\r?\n\r?\n)(?:<p>)?(?:(?:[^\r\n]{{1,{MAX_PARAGRAPH_LENGTH}}}(?:[.!?…]|\.{3}|[\u2026\u2047-\u2049]|[\p{{Emoji_Presentation}}\p{{Extended_Pictographic}}])(?=\s|$))|(?:[^\r\n]{{1,{MAX_PARAGRAPH_LENGTH}}}(?=[\r\n]|$))|(?:[^\r\n]{{1,{MAX_PARAGRAPH_LENGTH}}}(?=[.!?…]|\.{3}|[\u2026\u2047-\u2049]|[\p{{Emoji_Presentation}}\p{{Extended_Pictographic}}])(?:.{{1,{LOOKAHEAD_RANGE}}}(?:[.!?…]|\.{3}|[\u2026\u2047-\u2049]|[\p{{Emoji_Presentation}}\p{{Extended_Pictographic}}])(?=\s|$))?))(?:</p>)?(?=\r?\n\r?\n|$))",
    regex.MULTILINE | regex.UNICODE,
)


"""
    // 11. HTML-like tags and their content (including self-closing tags and attributes, with length constraints)
    `(?:<[a-zA-Z][^>]{0,${MAX_HTML_TAG_ATTRIBUTES_LENGTH}}(?:>[\\s\\S]{0,${MAX_HTML_TAG_CONTENT_LENGTH}}?</[a-zA-Z]+>|\\s*/>))` +
"""


HTML_like_tags_regex = regex.compile(
    # 11. HTML-like tags and their content (including self-closing tags and attributes, with length constraints)
    rf"(?:<[a-zA-Z][^>]{{0,{MAX_HTML_TAG_ATTRIBUTES_LENGTH}}}(?:>[\s\S]{{0,{MAX_HTML_TAG_CONTENT_LENGTH}}}?</[a-zA-Z]+>|\s*/>))",
    regex.MULTILINE | regex.UNICODE,
)


"""
    // 12. LaTeX-style math expressions (inline and block, with length constraints)
    `(?:(?:\\$\\$[\\s\\S]{0,${MAX_MATH_BLOCK_LENGTH}}?\\$\\$)|(?:\\$[^\\$\\r\\n]{0,${MAX_MATH_INLINE_LENGTH}}\\$))` +
"""


LaTeX_style_math_regex = regex.compile(
    # 12. LaTeX-style math expressions (inline and block, with length constraints)
    rf"(?:(?:\$\$[\s\S]{{0,{MAX_MATH_BLOCK_LENGTH}}}?\$\$)|(?:\$[^\$\r\n]{{0,{MAX_MATH_INLINE_LENGTH}}}\$))",
    regex.MULTILINE | regex.UNICODE,
)


"""
    // 14. Fallback for any remaining content (with length constraints)
    `(?:(?:[^\\r\\n]{1,${MAX_STANDALONE_LINE_LENGTH}}(?:[.!?…]|\\.{3}|[\\u2026\\u2047-\\u2049]|[\\p{Emoji_Presentation}\\p{Extended_Pictographic}])(?=\\s|$))|(?:[^\\r\\n]{1,${MAX_STANDALONE_LINE_LENGTH}}(?=[\\r\\n]|$))|(?:[^\\r\\n]{1,${MAX_STANDALONE_LINE_LENGTH}}(?=[.!?…]|\\.{3}|[\\u2026\\u2047-\\u2049]|[\\p{Emoji_Presentation}\\p{Extended_Pictographic}])(?:.{1,${LOOKAHEAD_RANGE}}(?:[.!?…]|\\.{3}|[\\u2026\\u2047-\\u2049]|[\\p{Emoji_Presentation}\\p{Extended_Pictographic}])(?=\\s|$))?))` +
"""


fallback_regex = regex.compile(
    # 14. Fallback for any remaining content (with length constraints)
    rf"(?:(?:[^\r\n]{{1,{MAX_STANDALONE_LINE_LENGTH}}}(?:[.!?…]|\.{3}|[\u2026\u2047-\u2049]|[\p{{Emoji_Presentation}}\p{{Extended_Pictographic}}])(?=\s|$))|(?:[^\r\n]{{1,{MAX_STANDALONE_LINE_LENGTH}}}(?=[\r\n]|$))|(?:[^\r\n]{{1,{MAX_STANDALONE_LINE_LENGTH}}}(?=[.!?…]|\.{3}|[\u2026\u2047-\u2049]|[\p{{Emoji_Presentation}}\p{{Extended_Pictographic}}])(?:.{{1,{LOOKAHEAD_RANGE}}}(?:[.!?…]|\.{3}|[\u2026\u2047-\u2049]|[\p{{Emoji_Presentation}}\p{{Extended_Pictographic}}])(?=\s|$))?))",
    regex.MULTILINE | regex.UNICODE,
)


def split_text(text):
    """
    使用所有定义的正则表达式对文本进行切分,并过滤掉空白片段。

    参数:
    text (str): 要切分的输入文本

    返回:
    list: 切分后的非空文本片段列表
    """
    # 将所有正则表达式组合成一个列表
    all_regexes = [
        heading_regex,
        list_item_full_regex,
        block_quotes_regex,
        code_blocks_regex,
        tables_regex,
        horizontal_rules_regex,
        standalone_lines_regex,
        sentences_regex,
        quoted_regex,
        paragraphs_regex,
        HTML_like_tags_regex,
        LaTeX_style_math_regex,
        fallback_regex,
    ]

    # 使用 '|' 连接所有正则表达式，创建一个大的组合正则表达式
    combined_regex = regex.compile(
        "|".join(f"({r.pattern})" for r in all_regexes), regex.MULTILINE | regex.UNICODE
    )

    # 使用组合正则表达式对文本进行切分
    matches = combined_regex.findall(text)

    # print("Raw matches:")
    # for i, match in enumerate(matches):
    #     print(f"{i}: {match}")

    # 提取非空匹配结果,并过滤掉空白片段
    result = [match for group in matches for match in group if match]
    filtered_result = [item.strip() for item in result if item and len(item.strip())]

    # print("\nFiltered result:")
    # for i, item in enumerate(filtered_result):
    #     print(f"{i}: {item}")

    return filtered_result


# 示例使用
if __name__ == "__main__":
    sample_text = """
    # 标题1
    
    这是一个段落。它包含了一些文本。
    
    - 列表项1
    - 列表项2
    
    > 这是一个引用。
        ```python
    def hello_world():
        print("Hello, World!")    ```
    
    这是另一个段落，带有一些**粗体**和*斜体*文本。
    """

    result = split_text(sample_text)
    for i, item in enumerate(result, 1):
        print(f"{i}. {item}")
