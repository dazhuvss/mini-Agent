"""
用 numpy 实现一个极简向量数据库:
  - 余弦相似度检索
  - pickle 持久化
  - 支持增量 add
无需任何第三方向量数据库依赖。
"""
import pickle
import numpy as np
from pathlib import Path


class VectorStore:
    def __init__(self):
        self.texts: list[str]       = []
        self.metadatas: list[dict]  = []
        self._embeddings: np.ndarray | None = None   # shape: (n, dim)

    # ────────────── 写入 ──────────────
    def add(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
    ):
        new_emb = np.array(embeddings, dtype=np.float32)

        if self._embeddings is None or len(self._embeddings) == 0:
            self._embeddings = new_emb
        else:
            self._embeddings = np.vstack([self._embeddings, new_emb])

        self.texts.extend(texts)
        self.metadatas.extend(metadatas or [{} for _ in texts])

    # ────────────── 检索（余弦相似度） ──────────────
    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        if self._embeddings is None or len(self.texts) == 0:
            return []

        q = np.array(query_embedding, dtype=np.float32)
        # 归一化
        emb_norm = self._embeddings / (
            np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-10
        )
        q_norm = q / (np.linalg.norm(q) + 1e-10)

        scores = emb_norm @ q_norm                     # 余弦相似度
        top_k = min(top_k, len(self.texts))
        idxs = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "text":     self.texts[i],
                "score":    float(scores[i]),
                "metadata": self.metadatas[i],
            }
            for i in idxs
        ]

    # ────────────── 持久化 ──────────────
    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "texts": self.texts,
                    "embeddings": self._embeddings,
                    "metadatas": self.metadatas,
                },
                f,
            )

    def load(self, path: str) -> bool:
        p = Path(path)
        if not p.exists():
            return False
        with open(p, "rb") as f:
            data = pickle.load(f)
        self.texts      = data["texts"]
        self._embeddings = data["embeddings"]
        self.metadatas   = data["metadatas"]
        return True

    def clear(self):
        self.texts = []
        self.metadatas = []
        self._embeddings = None

    def __len__(self):
        return len(self.texts)
