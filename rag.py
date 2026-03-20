"""
RAG 完整流程:
  1. 加载 knowledge/ 下的文档
  2. 递归分块 (chunking)
  3. 向量化 (embedding)
  4. 存入 VectorStore
  5. 查询时做 Top-K 检索
"""
from pathlib import Path
from vector_store import VectorStore
from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RAG_TOP_K,
    RAG_STORE_PATH,
    KNOWLEDGE_DIR,
)


class RAG:
    def __init__(self, llm):
        self.llm = llm
        self.store = VectorStore()
        self._loaded = False

    # ────────────── 初始化 / 构建索引 ──────────────

    def load_or_build(self):
        """优先加载已有索引；不存在则扫描 knowledge/ 目录重建。"""
        if self.store.load(str(RAG_STORE_PATH)):
            print(f"  ✓ 已加载知识库索引 ({len(self.store)} 个文档块)")
            self._loaded = True
            return

        self.rebuild()

    def rebuild(self):
        """强制重新构建索引。"""
        self.store.clear()
        docs = self._load_documents()
        if not docs:
            print("  ⚠ knowledge/ 目录下没有找到文档文件（支持 .txt / .md）")
            return

        # 分块
        chunks, metas = [], []
        for name, text in docs:
            doc_chunks = self._chunk_text(text)
            chunks.extend(doc_chunks)
            metas.extend(
                [{"source": name, "chunk_idx": i} for i in range(len(doc_chunks))]
            )

        print(f"  → 正在为 {len(chunks)} 个文档块生成 Embedding …")
        embeddings = self._batch_embed(chunks)

        self.store.add(chunks, embeddings, metas)
        self.store.save(str(RAG_STORE_PATH))
        print(f"  ✓ 知识库索引构建完成 ({len(chunks)} 个文档块)")
        self._loaded = True

    # ────────────── 检索 ──────────────

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        if len(self.store) == 0:
            return []
        top_k = top_k or RAG_TOP_K
        q_emb = self.llm.embed(query)
        return self.store.search(q_emb, top_k=top_k)

    # ────────────── 文档加载 ──────────────

    @staticmethod
    def _load_documents() -> list[tuple[str, str]]:
        docs = []
        for ext in ("*.txt", "*.md"):
            for fp in Path(KNOWLEDGE_DIR).glob(ext):
                try:
                    text = fp.read_text(encoding="utf-8")
                    if text.strip():
                        docs.append((fp.name, text))
                        print(f"  → 已读取文档: {fp.name}  ({len(text)} 字符)")
                except Exception as e:
                    print(f"  ✗ 读取失败 {fp.name}: {e}")
        return docs

    # ────────────── 递归分块 ──────────────

    def _chunk_text(self, text: str) -> list[str]:
        separators = ["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", "；", " "]
        return self._split_recursive(text, separators, CHUNK_SIZE, CHUNK_OVERLAP)

    def _split_recursive(
        self, text: str, seps: list[str], size: int, overlap: int
    ) -> list[str]:
        text = text.strip()
        if not text:
            return []
        if len(text) <= size:
            return [text]

        # 找到文本中存在的最优分隔符
        chosen_sep = ""
        for sep in seps:
            if sep in text:
                chosen_sep = sep
                break

        if not chosen_sep:
            # 无可用分隔符 → 强制按长度切
            return [
                text[i : i + size].strip()
                for i in range(0, len(text), size - overlap)
                if text[i : i + size].strip()
            ]

        parts = text.split(chosen_sep)
        chunks, current = [], ""

        for part in parts:
            candidate = (chosen_sep.join([current, part]) if current else part)
            if len(candidate) <= size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current.strip())
                current = part

        if current.strip():
            chunks.append(current.strip())

        # 对仍超长的块进一步递归
        remaining_seps = seps[seps.index(chosen_sep) + 1 :] if chosen_sep in seps else []
        final = []
        for c in chunks:
            if len(c) > size and remaining_seps:
                final.extend(self._split_recursive(c, remaining_seps, size, overlap))
            else:
                final.append(c)
        return final

    # ────────────── 批量 Embedding ──────────────

    def _batch_embed(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        all_embs: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            all_embs.extend(self.llm.embed_batch(batch))
        return all_embs
