# 严格冷启动场景下的典型推荐案例分析

为避免仅展示单个随机种子的偶然结果，本文将 seed 0、1、2 的 Top-10 推荐列表通过倒数排名投票进行聚合，并从测试集中选取具有代表性的案例。案例结果显示，文本语义分支能够补充新 Mashup 缺失的结构信息，而图结构分支有助于保持总体推荐准确率；与此同时，热门偏置和新 API 冷启动仍是当前模型的主要局限。

## 案例一：图文融合显著改善正确 API 的排序

**Mashup：Anaphonr。** 该应用用于寻找 anaphone（发音重排词），并允许用户通过短信发送查询。真实调用服务为 **Twilio SMS**。Popularity 将 Twilio SMS 排在第 10 位，Graph-only 未将其排入 Top-10，BGE-only 也未命中；Graph+BGE 则在三个随机种子中均将其稳定排在第 1 位。

该案例表明，纯图结构分支容易受到全局流行度支配，而纯文本分支虽然检索到了语音、押韵和自然语言相关服务，却没有识别出短信通信这一关键功能。标准化融合将 Mashup 的文本需求与 Twilio 的通信结构先验结合，使真实 API 从边缘位置提升至首位，体现了两类信号的互补性。

## 案例二：文本语义帮助命中 Middle API

**Mashup：30 Boxes EvokeTV。** 其功能是将 EvokeTV 节目列表自动添加到 30 Boxes 日历中，真实 API 为 **30 Boxes**，训练频次仅为 3，属于 Middle 组。Popularity 与 Graph-only 均未命中，BGE-only 和 Graph+BGE 均将其排在第 1 位，且三个随机种子结果一致。

由于该 API 的交互频次很低，图结构难以从协同关系中学习到稳定表示；但 Mashup 名称和描述中直接包含“30 Boxes”与“calendar”等强语义线索，因此文本分支可以准确识别目标服务。融合模型保留了这一语义证据，说明 BGE 分支是模型获得低频服务检索能力的重要来源。

## 案例三：Tail API 能被文本分支发现，但融合后排名受到热门偏置影响

**Mashup：PDXster。** 该应用用于查看和评论美国波特兰市议会议程，真实 API 为 **PDXCouncilConnect**，训练频次仅为 1，属于 Tail 组。Popularity 与 Graph-only 均未命中；BGE-only 在三个随机种子中均将其排在第 1 位；Graph+BGE 的共识排名为第 5 位，并只在两个随机种子的 Top-10 中出现。

该案例说明，文本相似度能够直接匹配“Portland city council agenda”等高度专门化的语义，因此 BGE-only 对极低频 API 具有较强检索能力。但加入图分数后，Technorati、Sunlight Labs Congress、Google Maps 等更热门的服务获得了更高综合得分，从而压低了真实 Tail API 的排名。这与多样性分析中约 99% 推荐槽位仍属于 Head API 的现象一致，说明当前融合方法只是部分缓解热门偏置。

## 案例四：文本分支能够在热门服务内部进行语义区分

**Mashup：DocuSign for Outlook。** 该应用在 Outlook 中提供电子签名功能，真实 API 为 **DocuSign Enterprise**。Popularity 与 Graph-only 均未命中；BGE-only 和 Graph+BGE 均在三个随机种子中将真实 API 排在第 1 位。

DocuSign Enterprise 本身属于 Head API，但并非全局最热门服务，因此单纯依赖流行度仍会漏检。文本分支通过“Outlook、electronic signature、document signing”等语义信息准确区分了 DocuSign 与其他常见热门 API。该案例表明，BGE 的作用不仅是扩大长尾覆盖，还能提高同一流行度层级内部的语义排序质量。

## 案例五：包含未见服务的复杂多 API 失败案例

**Mashup：KissAPI。** 该应用从 Facebook 图片生成拼贴，并可将图片制作成杯子、枕头、项链等实体商品。真实 API 共 6 个，包括 **HP Cloud Object Storage、Facebook Graph、HP Labs Multimedia Analytic Platform、Amazon S3、Zazzle 和 Amazon EC2**。其中前述 HP Cloud Object Storage 与 HP Labs Multimedia Analytic Platform 在训练集中没有交互，属于 Unseen API；其余服务属于 Head 组。四种方法均未命中任何真实 API。

这一失败不能仅归因于 Unseen API，因为模型也未命中 Facebook Graph、Amazon S3、Zazzle 和 Amazon EC2 等已见服务。该 Mashup 同时涉及社交图片获取、图像分析、云存储、云计算和定制商品生成，需求链条较长，而当前模型只使用单一 Top-K 排序分数，难以恢复多阶段服务组合。该案例揭示了两个后续方向：一是针对未见 API 构建不依赖交互边的语义冷启动分支；二是显式建模 API 之间的功能互补和组合顺序。

## 综合结论

案例分析与总体实验结论一致：Graph-only 的推荐列表高度接近 Popularity，缺乏针对新 Mashup 的个性化能力；BGE-only 能够发现语义相关的 Middle/Tail API，但总体准确率不足；Graph+BGE 在多数案例中保留了有效文本证据并结合图结构先验，从而获得最佳总体排序。然而，图分支仍会把部分 Tail API 压到较低位置，对完全未见 API 和复杂多服务组合的建模能力也仍然有限。