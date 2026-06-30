# SCF 最终案例分析

## 选择原则

- 使用 seed 0/1/2 的倒数排名投票生成共识 Top-10，避免只展示单个随机种子的偶然结果。
- Middle、Tail 和训练交互未见 API 案例优先从严格清洗子集中选择。
- 成功案例会额外排除 Mashup 名称或描述中直接出现真实 API 名称的样本，包括 Moo 这类三字符短名称。
- 优先选择 SCF 至少在 2/3 个 seed 中命中、而 Inductive LightGCN 未命中的样本。
- “Unseen”仅表示该 API 在训练交互中的频次为 0，不表示预训练文本编码器从未接触相关概念。

## 案例 1：Middle API 语义协同补救案例

**Mashup：** Rentals By Owner Canada（ID=6361）

**是否属于严格清洗子集：** 是

**自动选择规则：** 严格清洗子集；SCF 至少 2/3 seed 命中；LightGCN 与 Graph+BGE 均未命中

**Mashup 描述：** rentals-by-owner-canada provides landlord free advertising rental property mapping

### 真实 API

| 真实 API | 分组 | 训练频次 | 与 Mashup 共享关键词 |
| --- | --- | --- | --- |
| Google Maps | Head | 1898 | mapping |
| Geocoder.ca | Middle | 4 | mapping, canada |

### 各方法共识 Top-10 表现

| 方法 | 命中数 | 共识 Recall@10 | 首个命中排名 | 命中的真实 API |
| --- | --- | --- | --- | --- |
| BGE-only | 1 | 0.5000 | 9 | Geocoder.ca |
| Graph+BGE | 1 | 0.5000 | 1 | Google Maps |
| Inductive LightGCN | 1 | 0.5000 | 1 | Google Maps |
| SCF-LightGCN+BGE | 2 | 1.0000 | 1 | Google Maps; Geocoder.ca |

### 共识推荐列表

| 方法 | Top-10（✓表示真实 API） |
| --- | --- |
| BGE-only | 1.Zoopla[Tail]；2.Rentometer[Unseen]；3.Microsoft Mappoint[Middle]；4.Roomorama[Unseen]；5.1map[Tail]；6.Tourcms Marketplace[Middle]；7.Hotelston[Unseen]；8.Facebook Business Mapping[Tail]；9.Geocoder.ca[Middle]✓；10.Globexplorer[Middle] |
| Graph+BGE | 1.Google Maps[Head]✓；2.Commission Junction[Head]；3.Amazon Product Advertising[Head]；4.Microsoft Bing Maps[Head]；5.Facebook[Head]；6.Google Base[Head]；7.Youtube[Head]；8.Zillow[Head]；9.Ebay[Head]；10.Panoramio[Head] |
| Inductive LightGCN | 1.Google Maps[Head]✓；2.Oodle[Head]；3.Yahoo Geocoding[Head]；4.Bigtribe[Middle]；5.Geocoder[Head]；6.Google Fusion Tables[Head]；7.Sletoh.com[Middle]；8.Google Adsense[Head]；9.Google Maps Data[Head]；10.Google Earth[Head] |
| SCF-LightGCN+BGE | 1.Google Maps[Head]✓；2.Tourcms Marketplace[Middle]；3.Geocoder.ca[Middle]✓；4.Globexplorer[Middle]；5.Microsoft Mappoint[Middle]；6.Openstreetmap[Head]；7.Oodle[Head]；8.Geocoder[Head]；9.Mapstraction[Tail]；10.Facebook Business Mapping[Tail] |

### 分析

- 目标真实 API 为 **Geocoder.ca**，属于 **Middle**，训练交互频次为 **4**。
- SCF 在 3/3 个 seed 中将其放入 Top-10，共识排名为 **3**；Inductive LightGCN 为 **未进入共识Top-10**。
- BGE-only 共识排名为 **9**。
- Mashup 与该 API 文本的可见共享关键词包括：**mapping, canada**。这些词只是解释线索，不等同于完整的 BGE 语义机制。
- 该案例表明，直接 Mashup–API 语义匹配可以补充历史 Mashup 协同迁移对中频 API 的遗漏。

## 案例 2：Tail API 语义协同补救案例

**Mashup：** Fix It Repair Guides Android（ID=5242）

**是否属于严格清洗子集：** 是

**自动选择规则：** 严格清洗子集；SCF 至少 2/3 seed 命中；LightGCN 与 Graph+BGE 均未命中

**Mashup 描述：** fix-it-repair-guides-android android app thousand repair guide tutorial game console mobile phone computer household device car mobile

### 真实 API

| 真实 API | 分组 | 训练频次 | 与 Mashup 共享关键词 |
| --- | --- | --- | --- |
| Ifixit | Tail | 1 | device, repair, guide |

### 各方法共识 Top-10 表现

| 方法 | 命中数 | 共识 Recall@10 | 首个命中排名 | 命中的真实 API |
| --- | --- | --- | --- | --- |
| BGE-only | 1 | 1.0000 | 1 | Ifixit |
| Graph+BGE | 0 | 0.0000 | - |  |
| Inductive LightGCN | 0 | 0.0000 | - |  |
| SCF-LightGCN+BGE | 1 | 1.0000 | 2 | Ifixit |

### 共识推荐列表

| 方法 | Top-10（✓表示真实 API） |
| --- | --- |
| BGE-only | 1.Ifixit[Tail]✓；2.Handset Detection[Tail]；3.Producteev[Tail]；4.Windshieldrepair Tech Auto Glass[Tail]；5.Mobile Phone Megastore[Tail]；6.Hacker News Mobile[Tail]；7.Allogarage[Tail]；8.Pricerunner[Head]；9.Moves[Tail]；10.Evernote[Head] |
| Graph+BGE | 1.Google Base[Head]；2.Youtube[Head]；3.Google Homepage[Head]；4.Ebay[Head]；5.Twitter[Head]；6.Google Maps[Head]；7.411sync[Head]；8.Twilio[Head]；9.Amazon S3[Head]；10.Pricerunner[Head] |
| Inductive LightGCN | 1.Wikia[Tail]；2.Windshieldrepair Tech Auto Glass[Tail]；3.Mediatemple[Middle]；4.Recognize.im[Middle]；5.Angellist[Head]；6.Gatekrash[Tail]；7.Tesco[Tail]；8.Food[Middle]；9.Coindesk[Head]；10.Gamesradar[Middle] |
| SCF-LightGCN+BGE | 1.Windshieldrepair Tech Auto Glass[Tail]；2.Ifixit[Tail]✓；3.Producteev[Tail]；4.Handset Detection[Tail]；5.Mobile Phone Megastore[Tail]；6.Allogarage[Tail]；7.Bike Index[Tail]；8.Evoca[Tail]；9.Moves[Tail]；10.Evernote[Head] |

### 分析

- 目标真实 API 为 **Ifixit**，属于 **Tail**，训练交互频次为 **1**。
- SCF 在 3/3 个 seed 中将其放入 Top-10，共识排名为 **2**；Inductive LightGCN 为 **未进入共识Top-10**。
- BGE-only 共识排名为 **1**。
- Mashup 与该 API 文本的可见共享关键词包括：**device, repair, guide**。这些词只是解释线索，不等同于完整的 BGE 语义机制。
- 该案例表明，SCF 能在不完全依赖热门协同信号的情况下恢复低频 Tail API，提高长尾服务的可发现性。

## 案例 3：训练交互未见 API 零样本补救案例

**Mashup：** Get Digitalhealth Patient Generated Health Data Pilots（ID=5544）

**是否属于严格清洗子集：** 是

**自动选择规则：** 严格清洗子集；SCF 至少 2/3 seed 命中；LightGCN 与 Graph+BGE 均未命中

**Mashup 描述：** get-digitalhealth-patient-generated-health-data-pilots get digitalhealth mashup enables clinical trial sponsor clinical research organization academic researcherinvestigators set study select wearablesdevices start receiving data study participant participant click invite link sign consent provide authorization device data accessed get digitalhealth platform provides study team ability export data spreadsheet visualize directly push data clinical trial management system medidata openclinical ma…

### 真实 API

| 真实 API | 分组 | 训练频次 | 与 Mashup 共享关键词 |
| --- | --- | --- | --- |
| Gethealth | Unseen | 0 | applications, healthcare, wearable, device, health |

### 各方法共识 Top-10 表现

| 方法 | 命中数 | 共识 Recall@10 | 首个命中排名 | 命中的真实 API |
| --- | --- | --- | --- | --- |
| BGE-only | 1 | 1.0000 | 1 | Gethealth |
| Graph+BGE | 0 | 0.0000 | - |  |
| Inductive LightGCN | 0 | 0.0000 | - |  |
| SCF-LightGCN+BGE | 1 | 1.0000 | 1 | Gethealth |

### 共识推荐列表

| 方法 | Top-10（✓表示真实 API） |
| --- | --- |
| BGE-only | 1.Gethealth[Unseen]✓；2.Microsoft Healthvault[Tail]；3.Ihealth[Tail]；4.Datarella[Tail]；5.Health 2.0[Unseen]；6.Information Machine[Tail]；7.Cnet[Head]；8.Strava[Middle]；9.Bioid Web Services[Middle]；10.Misfit[Unseen] |
| Graph+BGE | 1.Youtube[Head]；2.Twitter[Head]；3.Fitbit[Head]；4.Cnet[Head]；5.Facebook[Head]；6.Twilio[Head]；7.Twilio Sms[Head]；8.Amazon Product Advertising[Head]；9.Google Base[Head]；10.Last.fm[Head] |
| Inductive LightGCN | 1.Fitbit[Head]；2.Withings[Middle]；3.Yolink[Middle]；4.Ihealth[Tail]；5.Datarella[Tail]；6.Runkeeper Health Graph[Middle]；7.Mynetdiary Food Search[Tail]；8.Jawbone Up[Middle]；9.Donorschoose[Head]；10.Microsoft Healthvault[Tail] |
| SCF-LightGCN+BGE | 1.Gethealth[Unseen]✓；2.Microsoft Healthvault[Tail]；3.Fitbit[Head]；4.Ihealth[Tail]；5.Datarella[Tail]；6.Withings[Middle]；7.Runkeeper Health Graph[Middle]；8.Information Machine[Tail]；9.Health 2.0[Unseen]；10.Donorschoose[Head] |

### 分析

- 目标真实 API 为 **Gethealth**，属于 **Unseen**，训练交互频次为 **0**。
- SCF 在 3/3 个 seed 中将其放入 Top-10，共识排名为 **1**；Inductive LightGCN 为 **未进入共识Top-10**。
- BGE-only 共识排名为 **1**。
- Mashup 与该 API 文本的可见共享关键词包括：**applications, healthcare, wearable, device, health**。这些词只是解释线索，不等同于完整的 BGE 语义机制。
- LightGCN 无法从训练交互中学习该 API 的有效协同表示，SCF 主要依靠直接 BGE 文本分支完成零样本检索。

## 案例 4：复杂多 API 失败案例

**Mashup：** Esync Dashboard（ID=4799）

**是否属于严格清洗子集：** 是

**自动选择规则：** 严格清洗子集；四种方法三个 seed 均未命中

**Mashup 描述：** esync-dashboard kosmos central esync dashboard used manage esync partner app marketplace connecting apis together esync api also white labeled used software company also looking add additional connection api driven software integration mobile rest api-strategy api ipaas cloud enterprise api-management ecommerce

### 真实 API

| 真实 API | 分组 | 训练频次 | 与 Mashup 共享关键词 |
| --- | --- | --- | --- |
| Shopify Admin | Head | 7 | ecommerce, apis |
| Magento Soap | Head | 6 | ecommerce |
| Woocommerce | Middle | 2 | ecommerce |
| Bigcommerce | Middle | 5 | ecommerce, used |
| Revel Systems | Unseen | 0 | management, ecommerce, mobile |
| Lightspeed Retail | Unseen | 0 | ecommerce, company, manage |

### 各方法共识 Top-10 表现

| 方法 | 命中数 | 共识 Recall@10 | 首个命中排名 | 命中的真实 API |
| --- | --- | --- | --- | --- |
| BGE-only | 0 | 0.0000 | - |  |
| Graph+BGE | 0 | 0.0000 | - |  |
| Inductive LightGCN | 0 | 0.0000 | - |  |
| SCF-LightGCN+BGE | 0 | 0.0000 | - |  |

### 共识推荐列表

| 方法 | Top-10（✓表示真实 API） |
| --- | --- |
| BGE-only | 1.Sap Anywhere[Unseen]；2.Bookingmarkets[Tail]；3.Redbooth[Unseen]；4.Cross.io[Tail]；5.Elance[Unseen]；6.Constantcontact[Unseen]；7.Sugarsync[Middle]；8.Envato[Tail]；9.Marketo[Tail]；10.Arcweb[Unseen] |
| Graph+BGE | 1.Google Maps[Head]；2.Twitter[Head]；3.Youtube[Head]；4.Facebook[Head]；5.Ebay[Head]；6.Amazon Product Advertising[Head]；7.Yahoo Maps[Head]；8.Flickr[Head]；9.Twilio[Head]；10.Last.fm[Head] |
| Inductive LightGCN | 1.Amazon S3[Head]；2.Box[Head]；3.Sendgrid[Head]；4.Twilio[Head]；5.Twilio Sms[Head]；6.Dropbox[Head]；7.Microsoft Bing Maps[Head]；8.Heroku[Head]；9.Google Drive[Head]；10.Google Storage[Middle] |
| SCF-LightGCN+BGE | 1.Amazon S3[Head]；2.Sugarsync[Middle]；3.Microsoft Bing Maps[Head]；4.Sendgrid[Head]；5.Twilio[Head]；6.Box[Head]；7.Marketo[Tail]；8.Twilio Sms[Head]；9.Cloud Elements Platform[Tail]；10.Sugarcrm[Middle] |

### 分析

- 该 Mashup 共有 **6** 个真实 API，分组构成为：Head, Unseen, Head, Middle, Unseen, Middle。
- SCF、Inductive LightGCN、Graph+BGE 与 BGE-only 在三个随机种子下均未命中任何真实 API，说明当前方法仍难以从宽泛的平台集成描述中恢复由多个互补服务构成的复杂 API 组合。
- 该失败案例应作为局限性展示，而不是通过继续查看测试集后调整0.35/0.65 融合权重来修复。

## 案例分析结论

- Middle 案例中，LightGCN 能稳定召回热门的 Google Maps，但遗漏了训练频次仅为 4 的 Geocoder.ca；SCF 将两类信号结合后同时命中两个真实 API。
- Tail 案例中，直接语义分支将训练频次仅为 1 的 iFixit 提升到前列，说明语义匹配能够补充协同信号对低频 API 的覆盖不足。
- 训练交互未见案例中，SCF 能推荐训练频次为 0 的 Gethealth，表明文本分支提供了一定的零样本新 API 推荐能力。
- 失败案例显示，对于描述宽泛、同时涉及多个互补服务的 Mashup，固定权重的后期融合仍可能无法恢复真实 API 组合。
- 严格清洗排除了精确 API 名称直出和高相似近重复样本，但并不排除“Fix It–iFixit”或“Digitalhealth–Gethealth”这类较强词汇与语义关联，因此案例解释应使用“语义与词汇线索共同作用”，不应将提升完全归因于深层语义推理。

## 人工复核清单

1. 对照原始数据确认自动恢复的 Mashup/API 名称是否正确。
2. 检查案例描述中是否直接出现真实 API 名称；脚本已增加短名称过滤，但仍需人工复核。
3. 对 Unseen API 使用“训练交互未见”措辞，不使用“模型从未见过”。
4. 论文正文建议放 3 个成功案例和 1 个失败案例；完整 Top-10 可放附录。
5. 不根据案例结果重新调整模型权重或重新选择测试样本阈值。
