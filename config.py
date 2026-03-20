"""
所有可配置项集中管理，通过环境变量覆盖默认值。
支持 OpenAI 兼容 API（如 Ollama、vLLM、DeepSeek 等）。
"""
import os
from pathlib import Path

# ═══════════════════ LLM 配置 ═══════════════════
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL   = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
CHAT_MODEL        = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM     = int(os.getenv("EMBEDDING_DIM", "1536"))

# ═══════════════════ 路径配置 ═══════════════════
BASE_DIR           = Path(__file__).parent
KNOWLEDGE_DIR      = BASE_DIR / "knowledge"
DATA_DIR           = BASE_DIR / "data"
RAG_STORE_PATH     = DATA_DIR / "rag_store.pkl"
MEMORY_STORE_PATH  = DATA_DIR / "memory_store.pkl"

# ═══════════════════ RAG 配置 ═══════════════════
CHUNK_SIZE    = 500     # 每个文档块的最大字符数
CHUNK_OVERLAP = 50      # 块间重叠字符数
RAG_TOP_K     = 3       # 检索返回的文档块数量

# ═══════════════════ Agent 配置 ═══════════════════
MAX_REACT_STEPS               = 10    # ReAct 最大推理步数
SHORT_MEMORY_MAX_TURNS        = 20    # 短期记忆保留的最大对话轮数
SHORT_MEMORY_SUMMARY_THRESHOLD = 12   # 超过此轮数触发摘要压缩

# ═══════════════════ 初始化目录 ═══════════════════
DATA_DIR.mkdir(exist_ok=True)
KNOWLEDGE_DIR.mkdir(exist_ok=True)
