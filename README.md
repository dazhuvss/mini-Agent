# Mini AI Agent

一个极简且功能完备的本地智能体系统。项目无外部重型框架依赖，仅使用基础科学计算库构建，旨在从零展示当前主流智能体的核心底层工作机制。

## 核心特性

* 推理大脑：完整实现 ReAct 思考与执行循环，支持动态任务规划与自我纠偏。
* 工具调用：内置网络搜索、代码执行、数学计算、时间获取与本地知识库检索等基础工具。
* 知识增强：原生实现文档读取、文本递归分块、批量向量化与检索链路。
* 记忆系统：短期记忆采用滑动窗口结合大模型摘要压缩技术，长期记忆支持自动静默提取并持久化用户的关键偏好与重要事实。
* 向量引擎：纯基于 Numpy 构建的高维向量数据库，支持余弦相似度计算与本地数据持久化。

---

## 运行方式

```bash
# 1. 安装依赖
pip install openai numpy duckduckgo-search

# 2. 设置 API Key（二选一）
#    方式 A：OpenAI
export OPENAI_API_KEY="sk-xxxxx"

#    方式 B：兼容 API（如 DeepSeek）
export OPENAI_API_KEY="sk-xxxxx"
export OPENAI_BASE_URL="https://api.deepseek.com/v1"
export CHAT_MODEL="deepseek-chat"

# 3. 启动
python main.py
```

---

## 运行效果演示

```
╔═══════════════════════════════════════════════════════╗
║              🤖  Mini AI Agent  v1.0                  ║
║            — "麻雀虽小，五脏俱全" —                    ║
╚═══════════════════════════════════════════════════════╝

正在初始化…
  → LLM 客户端
  → RAG 知识库
  → 已读取文档: ai_intro.txt  (2847 字符)
  → 正在为 9 个文档块生成 Embedding …
  ✓ 知识库索引构建完成 (9 个文档块)
  → 工具集
    已注册: ['calculator', 'current_time', 'python_executor', 'web_search', 'knowledge_search']
  → 记忆系统
✓ Agent 就绪!

───────────────────────────────────────────────────

👤 You > 计算一下圆周率的平方再乘以100，保留两位小数

═══════════════════════════════════════════════════
  🧠  Agent 推理过程
═══════════════════════════════════════════════════

  ──────────────────────────────────────────────
  📍 Step 1:
     Thought: 用户需要计算 π² × 100 并保留两位小数，我用计算器工具。
     Action: calculator
     Action Input: round(math.pi**2 * 100, 2)
  🔧 调用工具: calculator
     输入: round(math.pi**2 * 100, 2)
     结果: 986.96

  ──────────────────────────────────────────────
  📍 Step 2:
     Thought: 计算结果已经得到。
     Final Answer: π² × 100 ≈ **986.96**

═══════════════════════════════════════════════════
🤖 Agent > π² × 100 ≈ **986.96**

👤 You > Transformer 的自注意力机制是什么？

═══════════════════════════════════════════════════
  🧠  Agent 推理过程
═══════════════════════════════════════════════════

  ──────────────────────────────────────────────
  📍 Step 1:
     Thought: 这是关于 Transformer 架构的问题，我先在知识库中检索。
     Action: knowledge_search
     Action Input: Transformer 自注意力机制
  🔧 调用工具: knowledge_search
     输入: Transformer 自注意力机制
     结果: [片段1] (来源: ai_intro.txt, 相关度: 0.87) …

  ──────────────────────────────────────────────
  📍 Step 2:
     Thought: 知识库中找到了详细信息，可以回答了。
     Final Answer: Transformer 的自注意力机制（Self-Attention）
     允许模型在处理序列中每个位置时，关注输入序列中的
     所有其他位置……

═══════════════════════════════════════════════════
🤖 Agent > Transformer 的自注意力机制……（详细回答）
```

---

## 模块调用关系图

```
main.py
 │
 ├─ init_agent()
 │   ├─ LLM()                          ← llm.py
 │   ├─ RAG(llm)                       ← rag.py
 │   │   └─ VectorStore                ← vector_store.py
 │   │       └─ numpy cosine search
 │   ├─ Tools                          ← tools.py
 │   │   ├─ CalculatorTool
 │   │   ├─ CurrentTimeTool
 │   │   ├─ PythonExecutorTool
 │   │   ├─ WebSearchTool
 │   │   └─ KnowledgeSearchTool ──────────► RAG.retrieve()
 │   ├─ ShortTermMemory                ← memory.py
 │   └─ LongTermMemory                 ← memory.py
 │       └─ VectorStore                ← vector_store.py
 │
 └─ while True:
     └─ agent.run(user_input)           ← agent.py
         │
         │  ┌─────── ReAct Loop ───────┐
         │  │ 1. long_memory.recall()  │  ← 召回长期记忆
         │  │ 2. short_memory.get()    │  ← 获取对话上下文
         │  │ 3. LLM → Thought/Action  │  ← LLM 推理
         │  │ 4. tool.run()            │  ← 执行工具
         │  │ 5. Observation → LLM     │  ← 结果反馈
         │  │ 6. ... 循环直到 Final Answer
         │  └──────────────────────────┘
         │
         │  更新记忆:
         │  • short_memory.add()
         │  • short_memory.maybe_summarize()
         │  • long_memory.memorize()
         │
         └─ return final_answer
```

每个模块都可以独立替换或升级（比如将 `VectorStore` 换成 ChromaDB、将 `WebSearchTool` 换成 Tavily），而不影响其他模块——这就是手搓 Agent 框架相对于黑盒框架的学习价值所在。