> 大家**国庆快乐**呀！前几天没什么时间码字，快收假了抽空梳理梳理下一阶段的**主线**更新计划，也希望大家一如既往地多多监督！

之前的[「从零手搓中文大模型」](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkyMzczMjkxMA%3D%3D&action=getalbum&album_id=3599032183991779337&scenenote=https://mp.weixin.qq.com/s?__biz%3DMzkyMzczMjkxMA%3D%3D%26mid%3D2247484081%26idx%3D1%26sn%3Da740d8346704d27dc215950d4ce1d99b%26chksm%3Dc00d71f4e1214c2fe35302fe98ec6f076f29eea2dcd67568a3861863ba6c7e6f4832e89bd955%26scene%3D126%26sessionid%3D1727572824%26subscene%3D91%26clicktime%3D1727572828%26enterid%3D1727572828%26ascene%3D3%26devicetype%3DiOS18.0%26version%3D18003424%26nettype%3D3G+%26abtest_cookie%3DAAACAA%253D%253D%26lang%3Dzh_CN%26countrycode%3DCN%26fontScale%3D100%26exportkey%3Dn_ChQIAhIQW8oJys10cUzrj5zuSU%252BgVxLZAQIE97dBBAEAAAAAAHzMJ7V4CqgAAAAOpnltbLcz9gKNyK89dVj0sfQwZoaQr6vXojy9g0gYCI6hqbyXfvmfGWyzXj89VcxBNbuR8UWmidc%252BZmUF7swRYb8m%252B2xvF3w5fnuadG8%252BJGgBjcdjxf7HCNRfjL1PgQtGWh2VnTyvx%252FXzrqVXYUBpVTQtVBuyiX1nHVUMHr794yaLscaxaDJW497sreHaivMobvVpeRHMVxsd2nDJ%252FQ1SZhYwoN6ZvQUwB9iQsbEyBN7Idn2CKc7%252F8I2Sx%252BXt7%252FTf8cI%253D%26pass_ticket%3DuYco99quUGGX4QRAQ9HXe0zE2X3NtF%252FtqemWE%252BE9lnPOBmDjyzH5rX2KPimEbPQK%26wx_header%3D3&nolastread=1&devicetype=iMac+MacBookAir10,1+OSX+OSX+14.5+build(23F79)&version=13080810&lang=zh_CN&nettype=WIFI&ascene=0&fontScale=100&uin=&key=)系列文章的**主线**算是告一段落了，当然后续还会展开更新一些大模型相关知识点，大家敬请期待。

接下来的主要计划是打算倒腾一个应用落地层面的学习项目，思考良久之后想到了一个自认为还挺有趣的赛道——**AI面试**。

本以为自己这个想法非常之新颖，结果`Google`一波之后发现自己还是太天真：国内外都已经有相关的产品化应用了。

不过值得一提的是，我看到的这些应用都稍微有些**不正经**，因为他们的核心功能切入点是**用AI辅助用户进行视频面试**，说白了就是识别音频之后将面试里的提问转成文本发给大模型来生成答案进行「**作弊**」。

无论是作为求职者还是面试官，我经历过的大大小小面试也不算少数了，个人对这种**「黑科技」**还是有些不太认同的，毕竟这违背了基本的诚信原则，因此这里也不展开分享这些应用了。

我自己的思路其实是相反的：构造一个`LLM based agent`来充当**面试官**，让大家可以通过和它交互来进行**模拟面试**的练习，提升自己面试的技巧，个人认为这是一个更**健康**的应用方式。

从我检索到的一些新闻里看到，某些求职类`APP`似乎已经上线了**AI面试**的功能，部分求职者在平台上已经体验过来自`AI`面试官的面试了，但这些商业化产品无法看到其内部的实现方式，因此抱着**学习的目的**，我决定做一个简版的实现尝试，主要是为了让自己熟悉和实践大模型落地相关的一些技术（例如`RAG`、`Agent`等等）。

我理想中的**AI面试官**应该具备如下的核心能力：

- 有丰富的**知识储备**（底层有不同领域的知识库做支撑）
- 针对**简历**里的实际项目经验和上面的知识库来进行**个性化**的提问生成
- 能够对**面试者**的回答进行相对准确的评估，这又包含以下几点：
  1. 回答的整体<u>正确性</u>（是否有明显的错误），能够触发反问确认
  2. 答案的<u>精细程度</u>（是否有重要细节的遗漏），能够触发细节追问
  3. 识别出回答中<u>新出现的知识点</u>，能够进行展开提问
- 能够在回答不够理想的时候，适当地对受试者进行**提示和引导**
- 最后能对面试进行全面的**面试评价**，指出受试者的优点和不足

这些其实也是根据我自己诸多面试体验总结而来，可能有一定的主观性和个人偏好在其中，仅供大家参考哈。

基于上面的设想，我进而简单地预判了一下其中可能涉及到的技术点：

- **文档解析**，对上传的**简历**（文本或文件）能进行正确的识别和解析
- `RAG(Retrieval-Augmented Generation)`，支持知识库的**自定义和切换**
- `LLM Agent`，基于大模型的对话`Agent`构建，这显然是一个很大的点（内部的具体拆分这里我一时还无法给出，需要后面实践过程中逐步摸索）
- **多轮对话**，能够记录并结合上下文，进行**一致性强**的问答
- **文本总结**，这其实可以算作`LLM Agent`功能的一部分，但由于它直接影响到**面试评价**这一关键环节，所以单拎出来以示重视
- **语音交互**，`ASR`和`TTS`（前期可能非必需，所以优先级可以放后一些）

开源的项目我也进行了一番**搜刮**，能参考的项目不多，大概列举几个在下面：

- [AowerDmax 面试音频处理](https://github.com/AowerDmax/Simple-Interview-Audio-Processing)
- [JasonJarvan/interview-helper: 开源的AI面试助手，使用OpenAI Whipser模型进行STT（Speak to Text 语音转文字）转录，然后将问题交给ChatGPT回答。](https://github.com/JasonJarvan/interview-helper)
- [IliaLarchenko/Interviewer: AI Mock Interviewer](https://github.com/IliaLarchenko/Interviewer)

还没有来得及挨个细看，但是整体上感觉这几个项目里要么侧重的是对音频的识别和生成，要么在模拟内容上偏向于常规的知识库问答，和我想象的效果有一定的差距，但前人的工作一定是有值得学习借鉴的地方的，待我后面慢慢挖掘。

相比之前的大模型构建系列，这个系列我隐约感觉**执行难度和风险要大很多**。

一是参考资料偏少，二是涉及到的技术栈很多，再者就是我预感它的各个功能的效果评估会比较难做。

前期我会做很多实验，得到有价值的结果之后再和大家分享，所以更新的间隔有可能会比之前要久一些。

这中间也许会穿插着分享一些前面做构建工程化时候一带而过的**大模型的知识点细节**，来填补主线进度的稀疏。

那这个`Flag`帖就到这里啦，欢迎大家后面监督催更，再次祝大家**节日快乐**！
