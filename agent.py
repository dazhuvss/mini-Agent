"""
Agent 核心: ReAct 推理循环
  Thought → Action → Observation → Thought → ... → Final Answer

整合: LLM + Tools + RAG + ShortTermMemory + LongTermMemory
"""
import re
from config import MAX_REACT_STEPS


class Agent:
    def __init__(self, llm, tools, rag, short_memory, long_memory):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.rag = rag
        self.short_memory = short_memory
        self.long_memory = long_memory

    # ═══════════════════ 主入口 ═══════════════════

    def run(self, user_input: str) -> str:
        """处理一次用户输入，返回最终回答。"""

        # ── Phase 1: 上下文准备 ──
        memories = self.long_memory.recall(user_input)
        context_msgs = self.short_memory.get_messages()

        # ── Phase 2: 构建 Prompt ──
        system_prompt = self._build_system_prompt(memories)
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(context_msgs)
        messages.append({"role": "user", "content": user_input})

        # ── Phase 3: ReAct 推理循环 ──
        final_answer = None

        for step in range(1, MAX_REACT_STEPS + 1):
            # 调用 LLM（遇到 Observation: 时停止，由我们填入真实结果）
            llm_output = self.llm.chat(
                messages, temperature=0.3, stop=["\nObservation:"]
            )

            self._print_step(step, llm_output)

            # 检查是否直接给出了最终答案
            if "Final Answer:" in llm_output:
                final_answer = self._extract_final_answer(llm_output)
                break

            # 解析 Action
            action_name, action_input = self._parse_action(llm_output)

            if action_name is None:
                # 没有 Action 也没有 Final Answer → 视为直接回答
                final_answer = llm_output.strip()
                break

            # 执行工具
            observation = self._execute_tool(action_name, action_input)

            # 将本轮 LLM 输出 + 工具观测追加到消息
            messages.append({"role": "assistant", "content": llm_output})
            messages.append({"role": "user", "content": f"Observation: {observation}"})

        if final_answer is None:
            final_answer = "抱歉，我在有限步骤内未能完成推理。请尝试简化问题或提供更多信息。"

        # ── Phase 4: 更新记忆 ──
        self.short_memory.add("user", user_input)
        self.short_memory.add("assistant", final_answer)
        self.short_memory.maybe_summarize(self.llm)

        # 长期记忆提取（允许静默失败）
        try:
            self.long_memory.memorize(user_input, final_answer)
        except Exception:
            pass

        return final_answer

    # ═══════════════════ System Prompt 构建 ═══════════════════

    def _build_system_prompt(self, memories: list[str]) -> str:
        tool_lines = "\n".join(
            f"  {i}. {name}: {tool.description}"
            for i, (name, tool) in enumerate(self.tools.items(), 1)
        )

        memory_section = ""
        if memories:
            mem_lines = "\n".join(f"  - {m}" for m in memories)
            memory_section = f"\n关于该用户的已知信息（长期记忆）:\n{mem_lines}\n"

        return f"""你是一个能力强大的 AI 助手，拥有工具调用和知识检索能力。

可用工具:
{tool_lines}
{memory_section}
═══ 推理格式（严格遵循） ═══

如果需要使用工具:
Thought: <你的思考过程>
Action: <工具名称>
Action Input: <工具的输入参数>

如果已经知道答案，或工具结果已够充分:
Thought: <你的思考过程>
Final Answer: <最终回答>

规则:
1. 每次只能调用一个工具。
2. Action 必须是上面列出的工具名称之一。
3. 不要自己编造 Observation，它会由系统自动填入。
4. 如果问题简单，可以不使用工具，直接给出 Final Answer。
5. 使用中文回答。"""

    # ═══════════════════ 解析 LLM 输出 ═══════════════════

    @staticmethod
    def _parse_action(text: str) -> tuple[str | None, str]:
        action_m = re.search(r"Action:\s*(.+?)(?:\n|$)", text)
        if not action_m:
            return None, ""

        action_name = action_m.group(1).strip()

        # Action Input 可能是多行（如 Python 代码）
        input_m = re.search(
            r"Action Input:\s*(.*?)(?=\nThought:|\nFinal Answer:|\nAction:|\Z)",
            text,
            re.DOTALL,
        )
        action_input = input_m.group(1).strip() if input_m else ""

        # 清理 Markdown 代码块
        action_input = re.sub(r"^```(?:\w+)?\s*\n?", "", action_input)
        action_input = re.sub(r"\n?```\s*$", "", action_input)
        # 清理首尾引号
        if (
            len(action_input) >= 2
            and action_input[0] in "\"'"
            and action_input[-1] == action_input[0]
        ):
            action_input = action_input[1:-1]

        return action_name, action_input

    @staticmethod
    def _extract_final_answer(text: str) -> str:
        m = re.search(r"Final Answer:\s*(.*)", text, re.DOTALL)
        return m.group(1).strip() if m else text.strip()

    # ═══════════════════ 工具执行 ═══════════════════

    def _execute_tool(self, name: str, input_str: str) -> str:
        if name not in self.tools:
            return f"错误: 未知工具 '{name}'。可用工具: {list(self.tools.keys())}"
        print(f"  🔧 调用工具: {name}")
        print(f"     输入: {input_str[:120]}{'…' if len(input_str)>120 else ''}")
        try:
            result = self.tools[name].run(input_str)
        except Exception as e:
            result = f"工具执行异常: {e}"
        print(f"     结果: {result[:200]}{'…' if len(result)>200 else ''}")
        return result

    # ═══════════════════ 调试输出 ═══════════════════

    @staticmethod
    def _print_step(step: int, text: str):
        print(f"\n  {'─'*46}")
        print(f"  📍 Step {step}:")
        for line in text.strip().split("\n"):
            print(f"     {line}")
