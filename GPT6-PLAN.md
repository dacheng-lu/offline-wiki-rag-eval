# Offline Wikipedia domain RAG — GPT-6 plan

建议锁定 **200 道英文测试题＋80 道独立开发题**，评估“分层抽样语料上的证据问答”，不宣称代表整个 Wikipedia。全程不修改现有文件；以下是执行计划。已读本机 `run_eval.py`，未访问 Deck 的 ZIM，因此不提供未经核验的文章名。

| 设计项 | 固定方案 |
|---|---|
| 八领域 | 数学、物理与天文、化学与材料、生命与医学、地理与环境、历史、社会科学、艺术与人文 |
| 每领域测试 | 25题：20可答＋5不可答；可答来自10篇文章，每篇2题 |
| 每领域开发 | 10题：8可答＋2不可答；另选文章族 |
| 索引文章 | 测试每领域40篇：10篇出题＋30篇干扰；开发每领域10篇：4篇出题＋6篇干扰；共400篇 |
| 不可答来源 | 测试40篇、开发16篇独立来源，**不进入任何索引**；保留答案与证据供核验 |
| 文档族 | 重定向、别名、列表与主体、同一人物/事件系列、近重复文本合并；整族划分 dev/test，跨领域指定唯一主领域 |
| 题型 | 每领域可答题：实体8、日期/数值6、定义/关系6；仅短答案、单证据跨度，不出开放论述题 |

先冻结候选池和文章抽样，再写题。不可答指“完整冻结语料不支持”，不是世界上没有答案；须对全部索引文本做答案别名检索和人工主题检查，发现支持证据就换题。所有系统提示统一要求“仅据所给证据回答，否则弃答”；base 另报闭卷知识准确率。

每领域预注册6个宽泛英文检索词，每词取前100条，并从命中文章抽取一跳内部链接。统一去重后按固定种子抽样，搜索结果与链接扩展各占一半；每领域候选不少于150篇，预留替补顺序。记录排名分段，避免只取首屏。该方法仍是搜索覆盖样本，不是 Wikipedia 等概率抽样。

每题由一人写、另一人盲验：问题自然改写；接受答案、单位、日期格式预先列全；证据必须足以独立推出答案。不得调用被测模型辅助选题。

**以下命令在 Deck 执行；脚本名称是待实现接口，不是假定已有工具。**

```bash
ROOT=/home/dav-sd/deck-rag/pilot3
BOOK=wikipedia_en_all_nopic_2026-06
BASE=http://127.0.0.1:8080

# QUERY 来自冻结的领域检索词表；分页起点为 0、20、40、60、80
curl --fail --get "$BASE/search" \
  --data-urlencode "content=$BOOK" \
  --data-urlencode "pattern=$QUERY" \
  --data-urlencode "pageStart=0" \
  --data-urlencode "pageLength=20"

# ARTICLE_HREF 必须取自解析后的真实搜索/正文链接，禁止猜标题
curl --fail --location "$BASE$ARTICLE_HREF"
```

HTML 用 DOM 解析，不用正则抽正文：限定同源及该书 `/content/` 链接；保存请求 URL、最终 URL、标题和原始 HTML。先检查实际 DOM，再冻结正文选择器；去导航、脚本、编辑链接和参考文献，保留标题、段落、列表及有表头的表格行。清洗后逐段编号，跨度采用 Unicode 字符半开区间，并断言 `text[start:end] == quote`。随机抽查每领域5篇及全部出题证据。HTTP 即 `zimdump` 缺失时的正式替代方案，无须安装它。

```text
articles.jsonl:
{article_id,domain,family_id,split,url,html_sha256,text_sha256,text}

gold/test.jsonl:                  # runner 不读取 gold
{id,domain,family_id,question,label,accepted_answers,
 source:{article_id,start,end,quote},evidence_chunk_ids,absence_audit}

runner/test.jsonl:
{id,domain,question}

chunks.jsonl:
{chunk_id,article_id,start,end,text}

manifest.json:
{zim_sha256,selection_seed,files_sha256,models,prompts,
 extraction_version,chunking,retrieval,scoring,seeds,run_order}
```

| 索引与资源 | 计划 |
|---|---|
| 隔离 | dev/test 分别建80/320篇索引；测试索引包含其可答证据是正常 RAG，不得包含题目、答案标注或不可答来源 |
| 分块 | 按生成器 tokenizer：384 tokens，重叠64；保留标题路径和原文偏移；长篇完整保留 |
| 数量 | 预计4,000–12,000块；超过15,000块先暂停，冻结前按预注册长度分层重新抽样 |
| 模型 | `qwen3-embedding:0.6b`，冻结实际模型 digest、维数、查询模板和归一化；nomic 仅独立开发实验 |
| 存储 | float32 连续矩阵，精确余弦 top-k；若实测维数1024，向量约16–49 MB；避免 Python 浮点 JSON |
| 构建 | 先用开发集200块测吞吐；按总 tokens 外推。暂预算30–120分钟，非实测保证 |
| 运行 | 单请求、batch 8起；嵌入建库时卸载生成模型；峰值总内存<9 GiB、VRAM<3.5 GiB；在线共存失败则嵌入放CPU并重测 |

开发只比较 `k=3/5/8`，其他设置固定；冻结后不得挑参数。

| Arm | 行为与指标 |
|---|---|
| `base` | LFM3无上下文；报闭卷 EM/F1、弃答及证据协议分数，解释其证据限制 |
| `retrieval_only` | 新增：只嵌入＋检索；可答题报告文章 hit@1/3/5/8、**完整证据跨度** recall@k、MRR；无生成质量分 |
| `embed` | 原有 LFM3＋embedding；主结果 |
| `embed_decision` | 可选 d1 重排消融；同候选、同上下文预算，单独报告成本 |

生成固定 JSON：`{"answer":"","abstain":true,"citations":[]}`；引用使用 `chunk_id`。

| 评分 | 确切定义 |
|---|---|
| 可答 `answer_ok` | 规范化答案精确匹配预注册别名，且至少一条有效引用覆盖金标证据；引用必须实际提供给模型，所有引用都须有效 |
| 不可答 `answer_ok` | `abstain=true`、答案为空、引用为空 |
| EM/F1 | 只算 `answer` 字段，取最佳金标；数字与单位规则预注册；另报可答题弃答率、不可答弃答率及错误作答率 |
| 引用 | 分别报告合法率、证据覆盖率；不能把文章 ID 字符串出现当作支持 |
| 延迟 | 查询嵌入、检索、d1、生成、端到端分别计时；报冷启动与热态 p50/p95，包含实际模型切换 |
| 汇总 | 每领域及领域等权宏平均；三次重复先按题聚合，保留波动 |
| 95% CI | 领域内按文档族重采样，整族携带题目与重复；10,000次配对 bootstrap 比较 arms。另报“领域→族”两层区间，并注明仅8领域不稳定 |

适配现有 harness：扩展 `ARMS/run_arm`；`fmt_context` 输出块 ID；替换 `score` 的子串规则；`evaluate` 将提前批量查询嵌入改为逐题计时；`validate_dev/seal_dev` 泛化为 split 检查；`summarize` 增加分层统计。保留 allowlist、seed、原始请求响应及重复聚合。另保留旧 `keyword_any AND citation_ok` 为 `legacy_answer_ok`，不作主分。

| 阶段 | 产物、预算与停止条件 |
|---|---|
| 1. 预检与抽样 | `selection.json`、候选与替补日志；2–4小时。HTTP错误、候选不足、跨 split 同族即停止 |
| 2. 抽取与建库 | `raw/ text/ index/`；2–4小时含人工检查。跨度错位、截断、资源超限即停止 |
| 3. 写题与复核 | `gold/{dev,test}.jsonl`；约2–4人日。歧义、不可答无法核验则用预定替补 |
| 4. 开发与封存 | 调参2–4小时；冻结语料、问题、gold、评分器、代码、模型、提示、预算、运行顺序及种子11/22/33 |
| 5. 锁定运行 | 200题×2生成 arms×3次≈80–100分钟纯生成；预留2–4小时；d1另加约1–2小时 |
| 6. 审计 | 1–2小时；超时/格式错误计失败；基础设施错误保留日志后按固定规则重跑，不删除难题 |

```bash
# 完成上述适配后；config 路径由新适配层支持
python /home/dav-sd/deck-rag/pilot2/dev/run_eval.py preflight \
  --config "$ROOT/config.locked.json" --scope test

python /home/dav-sd/deck-rag/pilot2/dev/run_eval.py evaluate \
  --config "$ROOT/config.locked.json" \
  --cases "$ROOT/runner/test.jsonl" \
  --arms base,retrieval_only,embed --output "$ROOT/runs/locked"
# 三次重复、种子及轮换顺序由 locked config 驱动；禁止 --force
```

| 可委派提示 | 输入→输出；禁令、验收、停止 |
|---|---|
| Data builder | 领域配额＋Kiwix→`selection/ articles/ draft/`；禁止猜标题和调用被测模型；配额、来源、跨度齐全；候选不足停止 |
| Locked-set verifier | 草稿＋完整索引→`gold/ verification.json`；禁止查看模型输出；逐题双验、隔离成立才签封；歧义停止 |
| Runner | manifest＋无答案题库→`runs/`；禁止读取 gold、改参数或挑重复；哈希一致且全部请求有记录；哈希/服务异常停止 |
| Auditor | 封存文件＋全部运行→`report/`；禁止事后改答案或删失败；独立复算分数和族级CI；数据泄漏即判整轮无效 |

主要薄弱点是人工核验成本、搜索抽样偏差和每领域仅25题的宽区间；Deck 算力不是首要障碍。最小修正是限定结论到这套抽样语料、保留全部领域，并优先减少可选 d1 实验。发现测试集缺陷应发布新版本重跑，不能看输出后修金标。
