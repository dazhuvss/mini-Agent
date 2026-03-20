"""
手搓工具体系:
  Tool 基类 → 5 个具体工具
  每个工具只需实现 name / description / run()
"""
import re
import math
import datetime
import io
import sys
from abc import ABC, abstractmethod


# ═══════════════════ 工具基类 ═══════════════════

class Tool(ABC):
    """所有工具的基类。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称（供 LLM 调用时引用）。"""

    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述（注入到 System Prompt，帮助 LLM 判断何时调用）。"""

    @abstractmethod
    def run(self, input_str: str) -> str:
        """执行工具，输入/输出均为字符串。"""

    def __repr__(self):
        return f"Tool({self.name})"


# ═══════════════════ 计算器 ═══════════════════

class CalculatorTool(Tool):
    @property
    def name(self):
        return "calculator"

    @property
    def description(self):
        return (
            "数学计算器。输入一个数学表达式，返回计算结果。"
            "支持 +、-、*、/、**、math.sqrt()、math.sin() 等。"
            "示例输入: 2**10 + math.sqrt(144)"
        )

    def run(self, input_str: str) -> str:
        try:
            safe_ns = {
                "__builtins__": {},
                "math": math,
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
                "sum": sum,
                "pow": pow,
                "int": int,
                "float": float,
            }
            result = eval(input_str.strip(), safe_ns)
            return str(result)
        except Exception as e:
            return f"计算错误: {e}"


# ═══════════════════ 当前时间 ═══════════════════

class CurrentTimeTool(Tool):
    @property
    def name(self):
        return "current_time"

    @property
    def description(self):
        return "获取当前日期和时间。输入任意内容即可（如 'now'）。"

    def run(self, _input_str: str) -> str:
        now = datetime.datetime.now()
        weekdays = "一二三四五六日"
        return now.strftime(f"当前时间: %Y-%m-%d %H:%M:%S 星期{weekdays[now.weekday()]}")


# ═══════════════════ Python 代码执行器 ═══════════════════

class PythonExecutorTool(Tool):
    @property
    def name(self):
        return "python_executor"

    @property
    def description(self):
        return (
            "执行 Python 代码并返回 print() 的输出。"
            "输入为 Python 代码字符串。用 print() 输出需要的结果。"
        )

    def run(self, input_str: str) -> str:
        code = input_str.strip()
        # 去除 Markdown 代码块标记
        code = re.sub(r"^```(?:python)?\s*\n?", "", code)
        code = re.sub(r"\n?```\s*$", "", code)

        old_stdout = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            exec(code, {"__builtins__": __builtins__, "math": math})
            output = buf.getvalue()
            return output if output.strip() else "(代码执行成功，无 print 输出)"
        except Exception as e:
            return f"执行错误: {e}"
        finally:
            sys.stdout = old_stdout


# ═══════════════════ 网络搜索 ═══════════════════

class WebSearchTool(Tool):
    @property
    def name(self):
        return "web_search"

    @property
    def description(self):
        return "搜索互联网获取最新信息。输入搜索关键词字符串。"

    def run(self, input_str: str) -> str:
        query = input_str.strip().strip("\"'")
        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3))
            if not results:
                return "未找到相关搜索结果。"
            lines = []
            for i, r in enumerate(results, 1):
                lines.append(f"{i}. {r['title']}\n   {r['body']}")
            return "\n\n".join(lines)
        except ImportError:
            return "错误: 未安装 duckduckgo-search，请运行 pip install duckduckgo-search"
        except Exception as e:
            return f"搜索出错: {e}"


# ═══════════════════ 知识库搜索（RAG 桥接） ═══════════════════

class KnowledgeSearchTool(Tool):
    """
    连接 RAG 模块的工具。
    Agent 调用此工具时，会在本地知识库中进行向量检索。
    """

    def __init__(self, rag_module=None):
        self._rag = rag_module

    def set_rag(self, rag_module):
        self._rag = rag_module

    @property
    def name(self):
        return "knowledge_search"

    @property
    def description(self):
        return (
            "在本地知识库中搜索相关内容。输入问题或关键词。"
            "适用于查询已导入 knowledge/ 目录下文档中的信息。"
        )

    def run(self, input_str: str) -> str:
        if self._rag is None:
            return "知识库未初始化。"
        results = self._rag.retrieve(input_str.strip())
        if not results:
            return "知识库中未找到相关内容。"
        parts = []
        for i, r in enumerate(results, 1):
            source = r["metadata"].get("source", "unknown")
            parts.append(
                f"[片段{i}] (来源: {source}, 相关度: {r['score']:.2f})\n{r['text']}"
            )
        return "\n\n".join(parts)
