"""
LangGraph 降级编排器 — 状态机驱动的 ReAct Agent 工具调度 (ReAct 失败时的 fallback)

流程：
    用户输入 → Router(意图识别) → 条件分发
        → Executor(执行对应 Agent)
        → Clarify(需要澄清时追问)
        → Fallback(闲聊/兜底)
        → Finalize(返回结果)

学习参考:
    - Multi-AI-Agent4OnlineShopping 的 LangGraph State Machine
    - JoyAgent-JDGenie 的 DAG 执行引擎
    - llm-based-recommender 的 LangGraph workflow
"""

import asyncio
import operator
from typing import Annotated, Optional, Sequence, Tuple, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph


class AgentState(TypedDict):
    """Agent 状态定义"""

    messages: Annotated[Sequence[str], operator.add]
    intent: str
    agent: str
    current_response: str
    needs_clarification: bool
    tool_results: list
    input: str  # 当前用户输入（独立存储，避免 messages 累积影响取值）


class ECommerceOrchestrator:
    """
    主编排器 — LangGraph 状态机驱动

    管理 8 个子 Agent 的调度与编排：
    1. Router Agent 识别用户意图
    2. 根据意图分发到对应子 Agent 执行
    3. 需要澄清时追问用户
    4. 闲聊/兜底直接 LLM 回复
    5. 汇总结果返回
    """

    def __init__(self, llm: ChatOpenAI, agents: dict, router):
        """
        Args:
            llm: 统一 LLM 实例 (DeepSeek Chat)
            agents: Agent 实例字典 {"search_agent": SearchAgent(), ...}
            router: RouterAgent 路由器实例
        """
        self.llm = llm
        self.agents = agents
        self.router = router
        self._last_state: Optional[AgentState] = None
        self.graph = self._build_graph()

    # ------------------------------------------------------------------ #
    #  图构建
    # ------------------------------------------------------------------ #

    def _build_graph(self) -> StateGraph:
        """构建 LangGraph 状态机"""
        workflow = StateGraph(AgentState)

        # 添加节点
        workflow.add_node("router", self._route_intent)
        workflow.add_node("executor", self._execute_agent)
        workflow.add_node("clarify", self._clarify)
        workflow.add_node("fallback", self._fallback)
        workflow.add_node("finalize", self._finalize)

        # 定义边
        workflow.set_entry_point("router")
        workflow.add_conditional_edges(
            "router",
            self._after_route,
            {
                "execute": "executor",
                "clarify": "clarify",
                "fallback": "fallback",
            },
        )
        workflow.add_edge("executor", "finalize")
        workflow.add_edge("clarify", "finalize")
        workflow.add_edge("fallback", "finalize")
        workflow.add_edge("finalize", END)

        return workflow.compile()

    # ------------------------------------------------------------------ #
    #  节点实现
    # ------------------------------------------------------------------ #

    def _route_intent(self, state: AgentState) -> dict:
        """路由节点：识别用户意图（只返回需要更新的字段）"""
        user_input = state.get("input") or state["messages"][-1]
        route_info = self.router.route(user_input)

        # 简单澄清判断：输入过短且不是问候
        if len(user_input.strip()) < 3 and route_info["intent"] not in ("chitchat",):
            needs_clarify = True
        else:
            needs_clarify = False

        return {
            "intent": route_info["intent"],
            "agent": route_info["agent"],
            "input": user_input,
            "needs_clarification": needs_clarify,
        }

    def _execute_agent(self, state: AgentState) -> dict:
        """执行节点：调用对应子 Agent（只返回需要更新的字段）"""
        agent_name = state["agent"]
        user_input = state.get("input") or state["messages"][-1]

        if agent_name in self.agents:
            agent = self.agents[agent_name]
            try:
                # classify_agent 需要提取商品名，而不是传入完整问句
                if agent_name == "classify_agent":
                    query = self._extract_product_name(user_input)
                else:
                    query = user_input
                response = agent.run(query=query)
            except TypeError:
                # 某些 Agent 的 run 方法可能不接受 query 参数名
                response = agent.run(user_input)
            except Exception as e:
                response = f"执行 {agent_name} 时出错: {e}。请稍后重试。"
                print(f"[Orchestrator] Agent {agent_name} 执行失败: {e}")
        else:
            # Agent 不存在，降级为 LLM 直接回复
            try:
                response = self.llm.invoke(user_input).content
            except Exception as e:
                response = f"服务暂时不可用: {e}"

        return {"current_response": response}

    def _extract_product_name(self, user_input: str) -> str:
        """
        从用户问句中提取商品名称

        例: "iPhone 15 属于什么分类" → "iPhone 15"
            "小米手机 分类" → "小米手机"
            "蓝牙耳机" → "蓝牙耳机" (已经是商品名)
        """
        import re

        # 去掉常见的问句模板，保留商品名
        patterns = [
            r"(.*?)属于什么分类",
            r"(.*?)属于什么类别",
            r"(.*?)什么分类",
            r"(.*?)什么类别",
            r"(.*?)什么类",
            r"分类[:：\s]*(.*)",
            r"类别[:：\s]*(.*)",
        ]

        for pattern in patterns:
            match = re.match(pattern, user_input.strip(), re.IGNORECASE)
            if match:
                product = match.group(1).strip()
                # 去掉末尾的标点和空格
                product = re.sub(r"[，,。.!！?？\s]+$", "", product)
                if product:
                    return product

        return user_input.strip()

    def _after_route(self, state: AgentState) -> str:
        """条件边：决定走哪个分支"""
        if state.get("needs_clarification"):
            return "clarify"
        if state["intent"] in ("chitchat", "unknown"):
            return "fallback"
        return "execute"

    def _clarify(self, state: AgentState) -> dict:
        """澄清节点：需要更多信息时追问用户"""
        return {
            "current_response": (
                "您能再详细描述一下吗？比如您想了解哪个商品、什么分类，"
                "或者具体的问题？我可以帮您搜索商品、查订单、推荐好物等。"
            )
        }

    def _fallback(self, state: AgentState) -> dict:
        """兜底节点：闲聊或无法分类的意图"""
        agent_name = state["agent"]
        user_input = state.get("input") or state["messages"][-1]

        # 如果有 chitchat_agent，使用它
        if agent_name in self.agents:
            try:
                return {"current_response": self.agents[agent_name].run(query=user_input)}
            except TypeError:
                try:
                    return {"current_response": self.agents[agent_name].run(user_input)}
                except Exception:
                    pass
            except Exception:
                pass

        # 降级：LLM 直接回复
        try:
            return {"current_response": self.llm.invoke(user_input).content}
        except Exception as e:
            return {"current_response": f"抱歉，我暂时无法回复: {e}"}

    def _finalize(self, state: AgentState) -> dict:
        """终节点：保存最终状态用于元数据返回（注意：并发场景下会被覆盖）"""
        self._last_state = state
        return {}

    # ------------------------------------------------------------------ #
    #  对外接口
    # ------------------------------------------------------------------ #

    def invoke(self, user_input: str) -> tuple:
        """
        同步调用入口

        Args:
            user_input: 用户输入文本

        Returns:
            (回复字符串, 状态信息 dict)
        """
        result = self.graph.invoke(
            {
                "messages": [user_input],
                "input": user_input,
                "intent": "",
                "agent": "",
                "current_response": "",
                "needs_clarification": False,
                "tool_results": [],
            }
        )
        return result["current_response"], {
            "intent": result.get("intent", "unknown"),
            "agent": result.get("agent", "unknown"),
        }

    async def ainvoke(self, user_input: str) -> Tuple[str, dict]:
        """
        异步调用入口 — 使用 asyncio.to_thread 包装同步 graph.invoke，
        避免阻塞 asyncio 事件循环

        Args:
            user_input: 用户输入文本

        Returns:
            (回复字符串, 状态信息 dict)
        """
        initial_state = {
            "messages": [user_input],
            "input": user_input,
            "intent": "",
            "agent": "",
            "current_response": "",
            "needs_clarification": False,
            "tool_results": [],
        }
        # 在线程池中执行同步的 graph.invoke，不阻塞事件循环
        result = await asyncio.to_thread(self.graph.invoke, initial_state)
        return result["current_response"], {
            "intent": result.get("intent", "unknown"),
            "agent": result.get("agent", "unknown"),
        }

    def get_last_state(self) -> Optional[dict]:
        """获取最近一次执行的状态信息（已废弃，推荐使用 ainvoke 返回值）"""
        if not self._last_state:
            return {}
        return {
            "intent": self._last_state.get("intent", "unknown"),
            "agent": self._last_state.get("agent", "unknown"),
        }
