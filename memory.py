"""
记忆系统:
  ShortTermMemory  — 滑动窗口 + LLM 摘要压缩（对话上下文）
  LongTermMemory   — 向量检索持久化（跨会话的用户偏好/事实）
"""
import time
from vector_store import VectorStore
from config import (
    MEMORY_STORE_PATH,
    SHORT_MEMORY_MAX_TURNS,
    SHORT_MEMORY_SUMMARY_THRESHOLD,
)


# ═══════════════════ 短期记忆 ═══════════════════

class ShortTermMemory:
    """
    最近 N 轮对话保持原文;
    超出阈值时，自动将早期对话用 LLM 压缩为摘要。
    """

    def __init__(self, max_turns: int = SHORT_MEMORY_MAX_TURNS):
        self.messages: list[dict] = []     # {"role": ..., "content": ...}
        self.summary: str = ""
        self.max_turns = max_turns

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def get_messages(self) -> list[dict]:
        """返回需注入 Prompt 的消息列表。"""
        result = []
        if self.summary:
            result.append(
                {"role": "system", "content": f"[之前的对话摘要]: {self.summary}"}
            )
        # 取最近 max_turns 轮（每轮 2 条）
        recent = self.messages[-(self.max_turns * 2) :]
        result.extend(recent)
        return result

    def maybe_summarize(self, llm):
        """当消息数超过阈值时，把早期消息压缩到 summary。"""
        threshold = SHORT_MEMORY_SUMMARY_THRESHOLD * 2  # 消息条数 = 轮数 × 2
        if len(self.messages) <= threshold:
            return

        # 需要压缩的部分
        cutoff = len(self.messages) - self.max_turns * 2
        if cutoff <= 0:
            return

        old_msgs = self.messages[:cutoff]
        conv_text = "\n".join(
            f"{m['role']}: {m['content'][:300]}" for m in old_msgs
        )

        prompt = (
            "请将以下对话历史压缩为简洁的摘要，保留所有关键信息:\n\n"
            f"已有摘要: {self.summary or '无'}\n\n"
            f"新对话:\n{conv_text}\n\n"
            "请输出合并后的摘要（不超过 300 字）:"
        )
        try:
            self.summary = llm.chat(
                [{"role": "user", "content": prompt}], temperature=0.2
            )
            self.messages = self.messages[cutoff:]
        except Exception as e:
            print(f"  ⚠ 摘要压缩失败: {e}")

    def clear(self):
        self.messages.clear()
        self.summary = ""


# ═══════════════════ 长期记忆 ═══════════════════

class LongTermMemory:
    """
    从每次对话中用 LLM 提取「值得记住」的信息，
    向量化后存入 VectorStore，下次对话时按相关性召回。
    """

    def __init__(self, llm):
        self.llm = llm
        self.store = VectorStore()
        self._load()

    def _load(self):
        if self.store.load(str(MEMORY_STORE_PATH)):
            print(f"  ✓ 已加载长期记忆 ({len(self.store)} 条)")

    def _save(self):
        self.store.save(str(MEMORY_STORE_PATH))

    # ────────────── 记忆写入 ──────────────

    def memorize(self, user_input: str, assistant_response: str):
        """让 LLM 提取值得长期保存的信息，向量化后存储。"""
        prompt = (
            "从以下对话中提取值得长期记住的关键信息"
            "（用户偏好、重要事实、决策结论等）。\n"
            "如果没有值得记住的内容，只需回复\"无\"。\n"
            "每条记忆单独一行，不要编号。\n\n"
            f"用户: {user_input}\n"
            f"助手: {assistant_response}\n\n"
            "提取的记忆:"
        )
        try:
            result = self.llm.chat(
                [{"role": "user", "content": prompt}], temperature=0.2
            )
            # 过滤
            if not result or ("无" in result and len(result) < 10):
                return

            memories = [
                line.strip("-– •·")
                for line in result.strip().split("\n")
                if line.strip() and line.strip() not in ("无", "-", "—")
            ]
            memories = [m for m in memories if len(m) > 4]

            if not memories:
                return

            embeddings = self.llm.embed_batch(memories)
            metas = [
                {"timestamp": time.time(), "source": "conversation"}
                for _ in memories
            ]
            self.store.add(memories, embeddings, metas)
            self._save()
        except Exception as e:
            print(f"  ⚠ 记忆存储失败: {e}")

    # ────────────── 记忆召回 ──────────────

    def recall(self, query: str, top_k: int = 3) -> list[str]:
        """根据当前问题，检索最相关的长期记忆。"""
        if len(self.store) == 0:
            return []
        try:
            q_emb = self.llm.embed(query)
            results = self.store.search(q_emb, top_k=top_k)
            return [r["text"] for r in results if r["score"] > 0.35]
        except Exception as e:
            print(f"  ⚠ 记忆召回失败: {e}")
            return []

    def clear(self):
        self.store.clear()
        self._save()
