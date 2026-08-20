# 系统架构设计文档

> 本文档描述电商领域智能体的完整架构：分层设计、请求生命周期、核心模块职责、Agent 实现细节、数据层与安全设计。
> 面向对象：维护者、贡献者，以及希望深入理解系统设计的读者（面试深挖前建议通读）。
> 快速上手与 API 文档见根目录 [README](../README.md)。

---

## 1. 系统总览

电商领域智能体是一个**多智能体 + ReAct 工具循环**的对话系统，核心目标：

1. 用一个统一对话入口承接电商场景的多种诉求：搜索、推荐、订单、售后、数据分析、闲聊
2. 让 LLM **自主决策**调用哪个工具、按什么顺序调用，而不是写死路由
3. 在功能完整的前提下控制**成本与延迟**（模型分级、多级缓存、Prompt 压缩）
4. 保证**可观测、可降级、可评估**，任何单点故障都不让系统整体不可用

系统由四个层次组成：

| 层次 | 模块 | 职责 |
|------|------|------|
| API 层 | `api/app.py` | HTTP 入口、SSE 流式、认证、限流 |
| 编排层 | `orchestration/*` | 意图路由、ReAct 循环、记忆、成本优化、可观测性、Agent 注册 |
| Agent 层 | `agents/*` | 8 个业务 Agent，每个封装一类领域能力 |
| 数据层 | Neo4j / MySQL / FAISS | 知识图谱、业务数据、向量索引 |

---

## 2. 请求生命周期

以一次流式对话请求为例，完整链路如下：

```text
Streamlit 前端
   │ POST /api/chat/stream (SSE)
   ▼
FastAPI (api/app.py)
   │ API Key 校验 → IP 限流 → CORS
   ▼
ReactOrchestrator.ainvoke_stream (orchestration/react_orchestrator.py)
   │
   ├─ 1. RouterAgent.route() 关键词快速路由 → 意图 (intent)
   ├─ 2. cost_optimizer.cache_get() 三层缓存查询 (L1/L2/L3)
   │     └─ 命中 → 直接返回，仅记录记忆
   ├─ 3. MemoryManager.build_context() 构建历史上下文
   │     ├─ 长期记忆 (向量检索 Top-K)
   │     └─ 短期记忆 (摘要 + 最近 10 轮窗口)
   ├─ 4. AgentExecutor.invoke() ReAct 工具循环
   │     ├─ LLM 决策 → 调用 1~N 个工具 (search/kg_qa/classify/recommend/...)
   │     ├─ RepeatDetectionCallback 防死循环
   │     └─ 失败 → 降级到 ECommerceOrchestrator (LangGraph 固定路由)
   ├─ 5. LLM 原生 astream() 逐 token 生成最终回答
   └─ 6. 流式结束后由调用方写入记忆 + 非闲聊结果写入缓存
   ▼
SSE: start → token* → done
```

同步入口 `/api/chat` 走 `ainvoke()`，流程相同但最后一次性返回完整回答。

---

## 3. 核心模块详解

### 3.1 API 层 (`api/app.py`)

FastAPI 应用，生命周期内完成系统启动装配：

1. 初始化 Neo4j 驱动（失败则置空，图相关功能降级）
2. 创建统一 LLM 实例（DeepSeek）
3. `register_all_agents()` 注册 8 个 Agent
4. 预加载客服 Agent 的向量库；加载共享 BGE 嵌入模型
5. 创建 RouterAgent、MemoryManager，按 `REACT_ENABLED` 选择主编排器
6. 无论主编排器是什么，都额外创建一个 LangGraph 固定路由编排器作为降级

中间件（按执行顺序）：

| 中间件 | 实现 | 说明 |
|--------|------|------|
| CORS | 白名单配置 | 默认允许 Streamlit 前端来源 |
| API Key | `X-API-Key` 请求头 | 配置了 `API_ACCESS_KEY` 才启用，`/api/health` 豁免 |
| 限流 | 内存滑动窗口 | 默认 60 次/分钟/IP，超限返回 429 |

接口一览：

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查 |
| GET | `/api/agents` | 已注册 Agent 列表 |
| POST | `/api/chat` | 同步对话（ReAct + 记忆） |
| POST | `/api/chat/stream` | SSE 流式对话 |
| POST | `/api/classify` | 独立分类 |
| POST | `/api/search` | 独立混合搜索 |
| GET | `/api/stats` | Token / 缓存 / 记忆统计 |
| GET | `/api/trace` | 最近请求追踪树 |
| DELETE | `/api/session/{id}` | 清除会话记忆 |

**流式输出机制**：真实 token 级流式——先在线程池中执行同步的 ReAct 工具循环（工具调用本身通常 <2s），拿到工具结果后用 `llm.astream()` 逐 token 生成最终回答，通过 SSE 推送。`STREAMING_ENABLED=false` 或 ReAct 不可用时降级为"整段回答按 5 字符分块"的模拟流式。

### 3.2 意图路由 (`orchestration/router.py`)

两级路由策略：

- **Level 1 关键词路由**（零延迟、确定性）：按固定优先级匹配，顺序为 `order → customer_service → analytics → kg_qa → classify → recommend → search → chitchat`
  - 关键设计：`kg_qa` 的"属于什么/什么分类"等问句式规则**必须**在 `classify` 的"分类"之前匹配，否则问句会被误判为分类请求
  - 同一意图内按规则数组顺序匹配
- **Level 2 LLM 语义路由**（Level 1 未命中时）：prompt 内给出 8 个意图的示例，要求只输出意图关键词

结果经过 `_normalize_intent()` 标准化：小写化、去标点、别名映射（`qa→kg_qa`、`cs→customer_service` 等）、有效性校验，最终兜底为 `chitchat`。

路由结果同时服务于三个下游：**工具选择提示**（见 3.3）、**缓存 key**、**模型分级**（见 3.5）。

### 3.3 ReAct 编排器 (`orchestration/react_orchestrator.py`)

主编排器，将 8 个 Agent 包装为 LangChain `Tool`，交给 `create_tool_calling_agent` + `AgentExecutor` 执行。

**三重执行保护**：

1. `max_iterations=5`（配置可调）——限制最大工具调用轮数
2. `max_execution_time=30s`——总执行超时
3. `RepeatDetectionCallback`——同一工具+相同参数连续调用超过 2 次直接中断（防止"调工具→看结果→再调同一工具"死循环）

**关键增强逻辑**：

- **意图提示注入**：关键词路由是确定性的，LLM 工具选择是概率性的（如"推荐"可能被导去 `search_products`）。路由命中业务意图时，向 `chat_history` 头部注入 SystemMessage 提示，强制 LLM 调用与路由一致的工具
- **幻觉兜底**：如果关键词路由已命中业务意图，但 ReAct 一轮工具都没调用（LLM 直接凭知识回答），强制改走路由对应的 Agent，确保回答基于真实数据
- **缓存优先**：先查三层缓存，命中则跳过整个 ReAct 循环
- **降级链**：ReAct 异常 → 固定路由编排器（`ECommerceOrchestrator`）

### 3.4 降级编排器 (`orchestration/graph.py`)

基于 LangGraph `StateGraph` 的状态机，作为 ReAct 失败时的 fallback，也是 `REACT_ENABLED=false` 时的主编排器：

```text
router → (conditional) → executor / clarify / fallback → finalize → END
```

- `router`：意图识别 + 简单澄清判断（输入 <3 字符且非闲聊 → 需要澄清）
- `executor`：按 `agent` 分发到对应子 Agent，`classify_agent` 会先提取商品名
- `clarify`：追问用户
- `fallback`：闲聊或未知意图，优先用 `chitchat_agent`，失败降级为 LLM 直接回复
- `finalize`：保存状态供元数据返回

### 3.5 成本优化 (`orchestration/model_router.py`)

三层成本控制：

**1) 模型分级 `ModelTierRouter`**

| 层级 | 模型 | max_tokens | temperature | 超时 | 适用意图 |
|------|------|-----------|-------------|------|----------|
| Tier 1 lite | deepseek-chat | 512 | 0.1 | 10s | chitchat / customer_service / order / classify |
| Tier 2 standard | deepseek-chat | 2048 | 0.3 | 30s | search / kg_qa |
| Tier 3 heavy | deepseek-reasoner (R1) | 4096 | 0.5 | 60s | recommend / analytics |

额外规则：短消息（<20 字符）+ 非 heavy 意图 → 自动降到 Tier 1。估算成本节省 30-50%。

**2) Prompt 压缩 `PromptCompressor`**：压缩连续空白、按场景截断超长 context（如搜索 2000 字符、KG 3000 字符）、压缩 JSON 移除空字段。

**3) 多级缓存 `MultiLevelCache`**

| 级别 | 存储 | TTL | 命中方式 |
|------|------|-----|----------|
| L1 | 内存 `TTLCache` | 5min | 精确 key 匹配 |
| L2 | 磁盘 JSON（md5 文件名） | 30min（6×TTL） | 精确 key 匹配，命中回填 L1 |
| L3 | 内存向量 | 5min | Embedding 余弦相似度 ≥0.92，上限 200 条 LRU 淘汰 |

缓存 key = `intent|query|params`；**闲聊结果不写缓存**（避免冷启动污染）；语义缓存需要注入 BGE embedder 才激活（延迟加载）。

### 3.6 记忆系统 (`orchestration/memory.py`)

**短期记忆 `ShortTermMemory`**：

- 滑动窗口保留最近 10 轮（`MEMORY_WINDOW`）
- 消息数超过 6（`MEMORY_SUMMARY_THRESHOLD`）时，对窗口外旧消息用 LLM 生成摘要
- **递归摘要**：已有摘要 + 新摘要超过 500 字符上限时再次压缩，保证摘要长度受控
- 无 LLM 时降级为简单截断摘要（保留用户问题前 50 字符）

**长期记忆 `LongTermMemory`**：

- 对每条 user-assistant 对话对，用 LLM 判断是否有关键信息（偏好/事实/决策），输出 JSON
- 内容经 BGE 向量化后存入 FAISS `IndexFlatIP`，落盘 `memories.json` + `memories.index`
- 写入前去重：相似度 >0.92 的已有记忆跳过
- 检索时按向量相似度召回 Top-K；无索引时降级为关键词匹配

**`MemoryManager`**：按 `session_id` 隔离会话，`build_context()` 输出 = 长期记忆上下文（system）+ 短期摘要（system）+ 最近窗口消息。

### 3.7 可观测性 (`orchestration/observability.py`)

- **结构化日志**：统一 JSON 格式，带时间戳与上下文
- **Token 追踪**：按模型聚合 token 用量与成本估算（DeepSeek 定价常量）
- **链路追踪**：`TraceSpan` 树（trace_id / span_id / parent），装饰器 `@obs.trace()` + 上下文管理器 `trace_context()`，`/api/trace` 可查看
- **LangSmith**：配置了 `TRACING_ENABLED=true` + API Key 时自动注入环境变量启用

### 3.8 注册中心 (`orchestration/registry.py`)

系统装配的依赖注入点：

- `create_llm()`：统一 DeepSeek ChatOpenAI 实例
- `create_neo4j_driver()`：连接失败不抛异常，返回后由 Agent 各自降级
- `create_mysql_engine()`：SQLAlchemy 连接池（pool_size=5、max_overflow=10、pool_pre_ping、1h recycle）
- **共享嵌入模型**：只加载一份 BGE（~400MB），在 SearchAgent 与 CSAgent 之间复用，避免内存翻倍
- `register_all_agents()`：返回 8 个 Agent 的字典

---

## 4. Agent 详解

所有 Agent 继承 `agents/tools/base.py` 的 `BaseAgentTool`，统一提供 `run(query, **kwargs)` 接口。

| Agent | 数据源 | 核心能力 | 降级策略 |
|-------|--------|----------|----------|
| `search_agent` | FAISS + Neo4j | 混合搜索 | 无索引时返回空/提示 |
| `kg_qa_agent` | Neo4j + LLM | Text2Cypher 问答 | 参数化兜底查询 → LLM 直接回答 |
| `classify_agent` | BERT 模型 | 15 分类 | 规则关键词分类 |
| `recommend_agent` | Neo4j + MySQL + LLM | 图协同 + Item-CF + LLM 重排 | 热门商品/LLM 直接推荐 |
| `order_agent` | MySQL | 订单/物流查询 | 无订单号时引导用户 |
| `cs_agent` | FAISS FAQ + LLM | 客服问答（RAG） | LLM 直接回答 |
| `analytics_agent` | MySQL + LLM | 销售分析 | LLM 常识回答 |
| `chitchat_agent` | LLM | 闲聊 | 预设问候语回复 |

### 4.1 商品搜索 `search_agent.py`

**混合检索三步**：

1. **向量召回**：BGE 编码 query → FAISS `IndexFlatIP` 检索 Top-2K
2. **关键词检索**：Neo4j 全文/模糊匹配（支持分类、价格区间过滤）
3. **RRF 融合**：`score = Σ 1/(k + rank)`，k=60，合并去重并按融合分排序

关键细节：FAISS 索引只存商品 ID，必须通过 `_product_meta` 映射补充名称/分类/品牌/价格，否则 LLM 拿到的只是编号。索引未加载时 `_vector_search` 直接返回空，靠 Neo4j 结果兜底。

### 4.2 知识图谱问答 `kg_qa_agent.py`

流程：LLM 基于 `KG_SCHEMA` 生成 Cypher → 四层安全校验 → 执行 → LLM 基于结果生成自然语言回答。

**Text2Cypher 四层防御**（`_entity_alignment`）：

1. **白名单**：必须以 `MATCH` / `OPTIONAL MATCH` 开头
2. **黑名单**：禁止 `DELETE`/`DROP`/`CREATE`/`MERGE`/`CALL` 等写操作与危险关键词
3. **复杂度限制**：长度 ≤800 字符、MATCH 子句 ≤5、必须含 `RETURN`、多 MATCH 必须有 `WHERE`（防笛卡尔积）
4. **结果集上限**：无 `LIMIT` 自动注入 `LIMIT 100`，读取时二次截断；查询超时 10s

### 4.3 商品推荐 `recommend_agent.py`

三段式推荐管线：

```text
图协同 (Neo4j: 同分类/同品牌偏好)  +  Item-CF (MySQL: 订单共现)
          → 合并去重
          → LLM 重排 (按用户需求/预算，输出 JSON 排序)
          → 格式化 Top-K
```

- 无候选时降级：Neo4j 宽泛查询（按分类提取）→ 热门商品 → LLM 直接推荐（明确告知非库内数据）
- 用户分类提取限定在 15 个一级分类内，避免 LLM 自由发挥

### 4.4 客服 `cs_agent.py`

FAQ RAG：FAISS FAQ 索引（BGE 编码）检索 Top-3 → LLM 基于检索到的 FAQ 生成回答（禁止编造）。启动时 `preload()` 预加载索引和模型，避免首次请求超时。

### 4.5 数据分析 `analytics_agent.py`

按关键词识别分析类型（销售趋势 / 商品排行 / 品类占比 / 用户行为），执行对应 SQL 并格式化；LLM 辅助分析兜底。

### 4.6 订单 `order_agent.py`

从用户输入提取订单号（字母数字混合或纯数字），查询 MySQL 订单与物流信息；未识别订单号时引导用户提供。

### 4.7 分类与闲聊

- `classify_agent`：BERT 分类，Top-K + 置信度；模型不可用时用 15 类关键词规则降级
- `chitchat_agent`：预设问候语映射（零成本）→ LLM 闲聊 → 兜底话术

---

## 5. 数据层

### Neo4j（知识图谱）

图谱 Schema（见 `KG_SCHEMA`）：

- **节点**：`Category1/2/3`（三级分类）、`SPU`（标准产品单元）、`SKU`（价格）、`Trademark`（品牌）、`BaseAttrName/BaseAttrValue`（属性）、`Tag`（标签）
- **关系**：`Belong`（SPU→Category3→Category2→Category1 归属链）、`Have`（SPU→SKU/Trademark/BaseAttrValue）

### MySQL（业务数据）

存储订单、商品、用户行为等业务数据（gmall），核心表包括 `order_info`、`order_detail`、`logistics_info` 等。订单查询与 Item-CF 推荐直接基于这些表。

### FAISS（向量索引）

| 索引 | 用途 | 构建脚本 |
|------|------|----------|
| `products.index` + `product_ids.npy` | 商品向量检索 | `scripts/build_faiss_index.py` |
| `faq/index.faiss` + `index.pkl` | 客服 FAQ 检索 | `scripts/build_faq_index.py` |
| `memory_store/memories.index` | 长期记忆 | 运行时自动构建 |

---

## 6. 安全设计

| 风险 | 防护 |
|------|------|
| LLM 生成恶意 Cypher | 四层校验（白名单/黑名单/复杂度/结果上限）+ 参数化查询 + 超时 |
| SQL 注入 | 全部使用参数化查询（`%s` 占位符） |
| 未授权访问 | 可选 `X-API-Key` 认证中间件 |
| 滥用/刷接口 | 内存滑动窗口限流（60 次/分钟/IP） |
| ReAct 死循环 | 迭代上限 + 总超时 + 重复调用检测 |
| 密钥泄露 | 全部通过环境变量注入（`.env` 不入库，提供 `.env.example`） |

---

## 7. 评估体系

| 脚本 | 覆盖 | 说明 |
|------|------|------|
| `tests/test_agents.py` | 单元测试（27 用例） | 路由/分类/订单/推荐/闲聊，Mock 外部依赖，CI 中运行 |
| `tests/rag_evaluation.py` | RAG 检索 + 生成 | Recall@K、NDCG、Faithfulness 等 |
| `tests/recommendation_eval.py` | 推荐排序 | NDCG/MAP/MRR、多样性、CTR/CVR、策略对比 |
| `tests/evaluate.py` | LLM-as-Judge | 多维度生成质量评分 |
| `tests/smoke_test.py` | 冒烟测试 | 模块导入与关键机制验证 |

---

## 8. 部署架构

`docker-compose.yml` 定义四个服务：

| 服务 | 镜像/构建 | 端口 |
|------|-----------|------|
| `agent-api` | 本地 Dockerfile | 8002 |
| `streamlit` | Dockerfile.streamlit | 8501 |
| `neo4j` | neo4j:5.26-community | 7474 / 7687 |
| `mysql` | mysql:8.0 | 3306 |

`data/` 与 `models/` 通过 volume 挂载（不打进镜像）；环境变量由 `.env.docker` 注入。数据导入脚本不随仓库分发，由 `scripts/import_taobao_data.py` 从原始数据集生成。

---

## 9. 常见追问与解答

**Q1：为什么不用 LangGraph 直接做工具调度，而是用 AgentExecutor？**

两者都在项目中：ReAct 循环用 LangChain 的 `create_tool_calling_agent` + `AgentExecutor`（工具调用、中间步骤、超时控制成熟），LangGraph 状态机作为降级路由保留。真实场景两者互补，避免单点依赖。

**Q2：ReAct 会失控吗？**

三重保护：最大迭代 5 次、总超时 30s、重复工具调用检测（相同参数连续 2 次即中断）。中断后不是崩溃，而是降级到路由 Agent 直接回答。

**Q3：语义缓存误命中怎么办？**

阈值 0.92 属于保守设置；另外只有非闲聊结果才写缓存，且命中后不绕过记忆记录。如果线上误命中率高，可以通过 `/api/stats` 的 `semantic_hit_rate` 监控并调整阈值。

**Q4：长期记忆会泄露隐私吗？**

记忆按 `session_id` 隔离，提供 `/api/session/{id}` 删除接口；记忆内容只来自用户与助手的对话。生产环境建议加定期清理策略。

**Q5：如果数据量翻 100 倍，哪里先扛不住？**

FAISS 内存索引（可换磁盘索引/分片）→ 单机 MySQL（可加只读副本）→ Neo4j 查询复杂度（依赖 `MAX_MATCH_CLAUSES` 与索引设计）→ 内存缓存与记忆存储（可迁移 Redis）。代码中这些容量参数都已集中在 `config/settings.py`，方便调优。
