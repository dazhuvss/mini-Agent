#!/usr/bin/env python3
"""
Mini AI Agent — 交互式命令行入口

命令:
  quit / exit   退出
  clear         清除所有记忆
  rebuild       重建知识库索引
  help          显示帮助
"""
import sys

from config import OPENAI_API_KEY
from llm import LLM
from tools import (
    CalculatorTool,
    CurrentTimeTool,
    PythonExecutorTool,
    WebSearchTool,
    KnowledgeSearchTool,
)
from rag import RAG
from memory import ShortTermMemory, LongTermMemory
from agent import Agent


# ─── ANSI 颜色 ───
class C:
    H = "\033[95m"; B = "\033[94m"; G = "\033[92m"
    Y = "\033[93m"; R = "\033[91m"; BOLD = "\033[1m"
    END = "\033[0m"; CYAN = "\033[96m"


BANNER = f"""
{C.CYAN}╔═══════════════════════════════════════════════════════╗
║              🤖  Mini AI Agent  v1.0                  ║
║            — "麻雀虽小，五脏俱全" —                    ║
║                                                       ║
║   模块: ReAct 推理 │ 工具调用 │ RAG │ 记忆系统          ║
║   工具: calculator │ web_search │ python_executor      ║
║         current_time │ knowledge_search               ║
╚═══════════════════════════════════════════════════════╝{C.END}
"""

HELP_TEXT = f"""
{C.Y}可用命令:{C.END}
  quit / exit   退出程序
  clear         清除短期 + 长期记忆
  rebuild       重建 knowledge/ 知识库索引
  help          显示本帮助

{C.Y}试试这些问题:{C.END}
  • 现在几点了？
  • 计算 (3.14 * 100**2) / 4 的结果
  • 帮我写个 Python 函数计算斐波那契数列前 10 项
  • 搜索一下 2024 年诺贝尔物理学奖
  • 知识库里有哪些关于 Transformer 的内容？
  • 我喜欢简洁的回答风格（测试长期记忆）
"""


def init_agent() -> Agent:
    """组装 Agent 的所有模块。"""
    print(f"{C.Y}正在初始化…{C.END}\n")

    # 1. LLM
    print("  → LLM 客户端")
    llm = LLM()

    # 2. RAG
    print("  → RAG 知识库")
    rag = RAG(llm)
    rag.load_or_build()

    # 3. Tools
    print("  → 工具集")
    tools = [
        CalculatorTool(),
        CurrentTimeTool(),
        PythonExecutorTool(),
        WebSearchTool(),
        KnowledgeSearchTool(rag),
    ]
    print(f"    已注册: {[t.name for t in tools]}")

    # 4. Memory
    print("  → 记忆系统")
    short_mem = ShortTermMemory()
    long_mem = LongTermMemory(llm)

    # 5. 组装
    agent = Agent(llm, tools, rag, short_mem, long_mem)
    print(f"\n{C.G}✓ Agent 就绪!{C.END}\n")
    return agent


def main():
    print(BANNER)

    # 检查 API Key
    if not OPENAI_API_KEY:
        print(f"{C.R}✗ 未检测到 OPENAI_API_KEY{C.END}")
        print("  请设置环境变量:")
        print("    export OPENAI_API_KEY='sk-...'")
        print("  如需使用兼容 API (Ollama/DeepSeek)，同时设置:")
        print("    export OPENAI_BASE_URL='http://localhost:11434/v1'")
        sys.exit(1)

    agent = init_agent()
    print(HELP_TEXT)
    print(f"{'─'*55}\n")

    while True:
        try:
            user_input = input(f"{C.G}👤 You >{C.END} ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{C.CYAN}再见 👋{C.END}")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("quit", "exit"):
            print(f"{C.CYAN}再见 👋{C.END}")
            break
        if cmd == "clear":
            agent.short_memory.clear()
            agent.long_memory.clear()
            print(f"{C.Y}✓ 记忆已清除{C.END}\n")
            continue
        if cmd == "rebuild":
            agent.rag.rebuild()
            print()
            continue
        if cmd == "help":
            print(HELP_TEXT)
            continue

        # ── 运行 Agent ──
        print(f"\n{C.B}{'═'*55}")
        print(f"  🧠  Agent 推理过程")
        print(f"{'═'*55}{C.END}")

        answer = agent.run(user_input)

        print(f"\n{C.B}{'═'*55}{C.END}")
        print(f"{C.CYAN}🤖 Agent >{C.END} {answer}\n")


if __name__ == "__main__":
    main()
