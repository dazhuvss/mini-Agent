"""
封装 OpenAI 兼容 API，提供 chat 和 embedding 两种能力。
任何兼容 OpenAI 接口的服务（DeepSeek、Ollama 等）均可直接使用。
"""
import openai
from config import OPENAI_API_KEY, OPENAI_BASE_URL, CHAT_MODEL, EMBEDDING_MODEL


class LLM:
    def __init__(self):
        self.client = openai.OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
        )
        self.chat_model = CHAT_MODEL
        self.embedding_model = EMBEDDING_MODEL

    # ────────────── Chat 对话 ──────────────
    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        stop: list[str] | None = None,
        max_tokens: int = 2048,
    ) -> str:
        """调用 LLM 进行对话，返回纯文本回复。"""
        try:
            resp = self.client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=temperature,
                stop=stop,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except openai.APIError as e:
            raise RuntimeError(f"LLM API 调用失败: {e}") from e

    # ────────────── Embedding 向量化 ──────────────
    def embed(self, text: str) -> list[float]:
        """将单条文本转为向量。"""
        resp = self.client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        return resp.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量向量化（API 单次最多 2048 条，此处不额外分批）。"""
        if not texts:
            return []
        resp = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        # 按输入顺序排列
        return [d.embedding for d in sorted(resp.data, key=lambda x: x.index)]
