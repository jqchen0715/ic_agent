``` python
# -*- coding: utf-8 -*-
"""LangGraph 主链路：pre_tool_router -> tool_executor -> answer_generator。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict

try:
    from langgraph.graph import END, START, StateGraph

    _LANGGRAPH_AVAILABLE = True
except Exception:  # pragma: no cover - 仅在缺依赖环境触发
    END = "__end__" #如果没有 langgraph：就自己伪造一个 END。
    START = "__start__"
    StateGraph = None
    _LANGGRAPH_AVAILABLE = False

#定义一个“像 ToolRegistry （工具注册表）的东西”。
class _ToolRegistryLike(Protocol):
    def list_tool_names(self) -> list[str]:
        ...
    #要求：这个对象必须支持：异步调用工具。
    async def invoke(self, name: str, arguments: dict[str, Any]) -> str:
        ...#这里只定义接口，不写实现

#一个像 Model Router 的接口规范,任何对象，只要提供 chat() 方法，我就认为它是一个模型路由器。
class _ModelRouterLike(Protocol):
    async def chat(
        self,
        messages: list[dict[str, Any]],
        model_preference: str | None = None,
        **kwargs: Any,
    ) -> Any:
        ...

#TypedDict:有类型约束的 dict
class AgentState(TypedDict, total=False):# total=False:所有字段都不是必须的
    user_query: str
    messages: list[dict[str, Any]]
    model_preference: str | None
    temperature: float
    max_tokens: int | None
    route_reason: str
    selected_tools: list[str]
    needs_clarification: bool
    clarification: str
    strict_miss_marker: str
    tool_outputs: list[dict[str, Any]]
    final_answer: str
    model_id: str
    usage: dict[str, Any] | None


@dataclass
class LangGraphAgentResult:
    content: str
    model_id: str
    usage: dict[str, Any] | None = None
    selected_tools: list[str] = field(default_factory=list)
    tool_outputs: list[dict[str, Any]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    route_reason: str = ""
    needs_clarification: bool = False


class LangGraphICAgent:
    """
    IC 垂类 LangGraph 主链路：
    1) pre_tool_router：根据意图决定工具；
    2) tool_executor：执行工具并收集输出；
    3) answer_generator：将工具结果注入上下文，生成最终回复。
    """
    #轻量级意图分类规则
    _VERILOG_KEYWORDS = ("module", "always", "assign", "verilog", "rtl")
    _TIMING_KEYWORDS = ("sdc", "时序", "clock", "setup", "hold", "false path", "false_path")
    _KNOWLEDGE_CUES = (
        "什么",
        "为何",
        "为什么",
        "怎么",
        "如何",
        "区别",
        "原理",
        "方法",
        "有哪些",
        "优化",
        "what",
        "why",
        "how",
    )
    #为什么单独处理短query？因为很多用户的提问非常简短，缺乏上下文，导致工具路由和回答生成都很困难。比如用户可能只输入“乘法器时序”，这时候我们就可以直接判断这是一个时序相关的问题，并且建议用户补充更多细节，而不是盲目调用工具或生成答案。
    _SHORT_QUERY_EXACT = {"乘法器", "时序"}
    #RAG 防幻觉机制：如果检索工具返回的结果没有命中知识库中的可引用片段，就标记为严格模式未命中，后续回答生成会拒答并给出原因。
    STRICT_MISS_MARKER = "【严格模式未命中】"

    ‘’‘把：
    - 模型系统
    - 工具系统
    - 工作流图

    装配到 Agent 里’‘’

    def __init__(self, model_router: _ModelRouterLike, tool_registry: _ToolRegistryLike) -> None:
        self._model_router = model_router #LLM 调度器
        self._tools = tool_registry     #工具系统
        self._graph = self._build_graph() if _LANGGRAPH_AVAILABLE else None #构建工作流图。

    def _build_graph(self) -> Any:
        #没安装langgraph
        if StateGraph is None:
            return None

        #create a 状态图，状态类型是 AgentState（一个 TypedDict），每个节点对应一个方法，边上可以有条件判断。
        graph = StateGraph(AgentState) #全局共享状态
        #添加节点：pre_tool_router -> tool_executor -> answer_generator -> clarify
            ‘’‘分析用户意图
            决定下一步干什么’‘’
        graph.add_node("pre_tool_router", self._pre_tool_router)
        #调用工具
        graph.add_node("tool_executor", self._tool_executor)
        graph.add_node("answer_generator", self._answer_generator)
        graph.add_node("clarify", self._clarify)
        #添加起点：工作流开始后，第一步一定进入：pre_tool_router。

        graph.add_edge(START, "pre_tool_router")
        #条件分支：pre_tool_router 的输出决定下一步走哪条路，如果需要澄清就走 clarify，选了工具就走 tool_executor，否则直接走 answer_generator。动态工作流
        graph.add_conditional_edges(
            "pre_tool_router",
            self._route_after_pre_tool_router,
            {
                "clarify": "clarify",
                "tool_executor": "tool_executor",
                "answer_generator": "answer_generator",
            },
        )
        graph.add_edge("tool_executor", "answer_generator")
        graph.add_edge("answer_generator", END)
        graph.add_edge("clarify", END)
        ‘’‘编译工作流
            LangGraph 会：

                * 校验图
                * 校验边
                * 构建 runtime
                * 优化执行路径
            最终返回：
            可执行 Graph Runtime’‘’
        return graph.compile()

    async def run(
        self,
        *,
        messages: list[dict[str, Any]],#完整聊天历史。
        model_preference: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LangGraphAgentResult:
    #提取用户query
        query = self._extract_user_query(messages)
        #初始化
        initial: AgentState = {
            "user_query": query,
            "messages": messages,
            "model_preference": model_preference,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "selected_tools": [],
            "tool_outputs": [],
            "route_reason": "",
            #初始不需要澄清，除非后续逻辑判断需要。
            "needs_clarification": False,
            "clarification": "",
            "strict_miss_marker": "",
        }
        #类型标注：最终状态也是 AgentState 类型。
        final_state: AgentState
        #没有langgraph分支
        if self._graph is None:
            ##fallback 执行工作流图，得到最终状态。
            ‘’‘router
                ↓
                tool
                ↓
                answer’‘’
            final_state = await self._run_without_langgraph(initial)

        else:
           ’‘’START
                ↓
                router
                ↓
                conditional routing
                ↓
                tool
                ↓
                answer
                ↓
                END‘’‘
            final_state = await self._graph.ainvoke(initial)

        tool_events = list(final_state.get("tool_outputs") or [])
        return LangGraphAgentResult(
            content=str(final_state.get("final_answer", "")),
            model_id=str(final_state.get("model_id", "")),
            usage=final_state.get("usage"),
            selected_tools=list(final_state.get("selected_tools") or []),
            tool_outputs=tool_events,
            sources=self._extract_sources(tool_events),
            tool_events=tool_events,
            route_reason=str(final_state.get("route_reason", "")),
            needs_clarification=bool(final_state.get("needs_clarification", False)),
        )
    #Fallback Runtime（降级运行时），手动模拟 LangGraph 的执行流程。保证graph模式和非graph模式的一致性
    async def _run_without_langgraph(self, state: AgentState) -> AgentState:
        """langgraph 依赖缺失时的顺序执行降级，保持主链路行为一致。"""
        #拷贝state
        current: AgentState = dict(state) #防止污染源对象
        #执行 Router 节点
        current.update(await self._pre_tool_router(current)) #状态合并

        next_node = self._route_after_pre_tool_router(current)
        if next_node == "clarify":
            current.update(await self._clarify(current))
            return current

        if next_node == "tool_executor":
            current.update(await self._tool_executor(current))
        #没有return是因为，还需要answer_generator来生成最终答案，tool_executor 只是准备工具结果。

        current.update(await self._answer_generator(current))
        return current
                ‘’‘Agent 的“动态路由决策器”

                也就是：

                Router 节点执行完以后，
                下一步该去哪？

                这是整个 LangGraph 工作流里：

                Conditional Routing 的核心。’‘’
    def _route_after_pre_tool_router(self, state: AgentState) -> str:
        if state.get("needs_clarification"):
            return "clarify"
        if state.get("selected_tools"):
            return "tool_executor"
        return "answer_generator"

#在真正调用工具前，先判断用户问题是否有效、要不要继续走工具链。
    async def _pre_tool_router(self, state: AgentState) -> AgentState:
        query = str(state.get("user_query", "")).strip()
        #如果问题为空，就不要继续调用工具。
        if not query:
            return {
                "needs_clarification": True,
                "clarification": "请先描述你的问题，我再调用合适工具帮你分析。",
                "route_reason": "empty_query",
                "selected_tools": [],
            }

        if self._is_short_query(query):
            return {
                "needs_clarification": True,
                "clarification": (
                    "你的问题有点短。可以补充下场景吗？例如："
                    "“乘法器时序优化有哪些方法”或“SDC 里 setup/hold 约束怎么写”。"
                ),
                "route_reason": "short_query_clarify",
                "selected_tools": [],
            }

        selected: list[str] = []
        available = set(self._tools.list_tool_names())
        q_lower = query.lower()#把用户问题转成小写，方便匹配英文关键词。比如 Verilog、verilog、VERILOG 都能识别。
        #通过有无关键词，判断意图工具
        is_verilog = any(k in q_lower for k in self._VERILOG_KEYWORDS)
        is_timing = any(k in q_lower for k in self._TIMING_KEYWORDS)
        is_knowledge = any(k in query or k in q_lower for k in self._KNOWLEDGE_CUES)
        #判断是否有verilog代码：如果用户问题里有verilog代码块，或者出现了明显的verilog语法特征，就认为有verilog代码。因为有些用户可能直接贴代码而不说“这是verilog”，所以单纯靠关键词可能不够。
        ‘’‘两种表达式：
        1. _extract_verilog_code(query)：项目自己的代码提取逻辑
        2. 正则表达式：直接查 module / endmodule / always @ / assign xxx =
        ’‘’
        has_verilog_code = bool(self._extract_verilog_code(query)) or bool(re.search(r"\b(module|endmodule|always\s*@|assign\s+\w+\s*=)", query, flags=re.I))
        is_ic_domain = self._is_ic_domain_query(query)

        if not is_ic_domain:
            return {
                "selected_tools": [],
                "route_reason": "out_of_scope",
                "needs_clarification": False,
            }
        #根据前面识别的特征，把对应的工具加入待选列表。比如只要用户问题里有知识类的疑问，就选 ic_rag_search；只要有 verilog 相关的特征，就选 verilog_code_analyzer；只要有时序相关的特征，就选 timing_constraint_suggester。
        if (is_knowledge or is_verilog or is_timing or is_ic_domain) and "ic_rag_search" in available:
            selected.append("ic_rag_search")
        #调用代码分析工具的条件
        if is_verilog and has_verilog_code and "verilog_code_analyzer" in available:
            selected.append("verilog_code_analyzer")
        #判断是否为时序问题
        if is_timing and "timing_constraint_suggester" in available:
            selected.append("timing_constraint_suggester")
        #记录路由原因
        reason_parts: list[str] = []
        if is_knowledge:
            reason_parts.append("knowledge")
        if is_verilog:
            reason_parts.append("verilog")
        if is_timing:
            reason_parts.append("timing")
        if is_ic_domain and not reason_parts:
            reason_parts.append("ic_domain")
        route_reason = "|".join(reason_parts) if reason_parts else "default_rag"

        return {
            "selected_tools": selected,
            "route_reason": route_reason,
            "needs_clarification": False,
        }
        #判断是否属于ic领域问题
    def _is_ic_domain_query(self, query: str) -> bool:
        q = (query or "").lower()
        domain_terms = (
            "verilog", "hdl", "rtl", "asic", "fpga", "sdc", "eda", "vlsi",
            "module", "endmodule", "always", "assign", "initial", "wire", "reg",
            "fork", "join", "defparam", "parameter", "udp", "wait", "$display", "$time",
            "$stime", "$realtime", "$countdrivers", "posedge", "negedge",
            "综合", "仿真", "时序", "电路", "芯片", "寄存器", "触发器", "门级",
            "模块", "实例", "端口", "信号", "注释", "参数", "复位", "时钟", "单元库",
            "元件", "阻塞", "非阻塞", "硬件描述", "分级名字", "向上引用",
        )
        return any(term in q or term in query for term in domain_terms)
        #工具执行层    
        async def _tool_executor(self, state: AgentState) -> AgentState:
        query = str(state.get("user_query", "")).strip() #做字符串化和空格化
        selected_tools = list(state.get("selected_tools") or [])
        outputs: list[dict[str, Any]] = []
        #严格检索未命中标记
        strict_miss_marker = ""

        for tool_name in selected_tools:
            #给当前选择的工具构建参数
            args = self._build_tool_args(tool_name, query, state.get("messages") or [])
            try:
                #调用
                result = await self._tools.invoke(tool_name, args)
                outputs.append({"tool": tool_name, "arguments": args, "result": result, "ok": True})
                ‘’‘如果还没有 strict_miss_marker，就根据当前工具结果检查是否出现“严格检索未命中”。
                用于rag防止幻觉
                ’‘’
                if not strict_miss_marker:
                    marker = self._build_strict_miss_marker(tool_name, result)
                    if marker:
                        strict_miss_marker = marker
                #异常处理，如果工具调用失败了，也把这个事件记录下来，结果里标记 ok=False，后续回答生成时可以知道工具执行失败了。防止崩溃
            except Exception as exc:  # noqa: BLE001
                outputs.append(
                    {
                        "tool": tool_name,
                        "arguments": args,
                        "result": f"工具执行失败: {exc!s}",
                        "ok": False,
                    }
                )

        if strict_miss_marker:
            return {"tool_outputs": outputs, "strict_miss_marker": strict_miss_marker}
        return {"tool_outputs": outputs}

    async def _answer_generator(self, state: AgentState) -> AgentState:
        if state.get("needs_clarification"):
            return {"final_answer": str(state.get("clarification", ""))}
        
        #准备对话上下文：如果 state 里已经有历史消息，就用历史消息；如果没有，就用当前 query 临时构造一条 user message。
        query = str(state.get("user_query", "")).strip()
        messages = list(state.get("messages") or [])
        if not messages:
            messages = [{"role": "user", "content": query}]

        tool_outputs = list(state.get("tool_outputs") or [])
        if not tool_outputs:
            if str(state.get("route_reason", "")) == "out_of_scope":
                return {
                    "final_answer": self._strict_refusal_template(query, "问题不属于当前 IC/Verilog 知识库范围"),
                    "model_id": "strict_refusal",
                    "usage": None,
                }
            return {
                "final_answer": self._strict_refusal_template(query, "检索工具未返回结果"),
                "model_id": "strict_refusal",
                "usage": None,
            }

        tool_context = self._render_tool_context(tool_outputs)
        strict_marker = str(state.get("strict_miss_marker", "")).strip()
        #说明知识库没有找到可靠可引用片段
        if strict_marker.startswith(self.STRICT_MISS_MARKER):
            miss_reason = strict_marker[len(self.STRICT_MISS_MARKER) :].strip() or "知识库未命中可引用片段"
            return {
                "final_answer": self._strict_refusal_template(query, miss_reason),
                "model_id": "strict_refusal",
                "usage": None,
            }
      #在生成答案前，再次检查是否需要严格拒答（比如 RAG 明确未命中可引用片段，或者工具执行过程中出现了严重错误），如果需要，就直接返回拒答结果，不调用模型生成答案。
        strict_refusal = self._build_strict_refusal_if_needed(query, state, tool_outputs)
        if strict_refusal is not None:
            return {
                "final_answer": strict_refusal,
                "model_id": "strict_refusal",
                "usage": None,
            }
        #约束行为
        system_prompt = (
            "你是 IC/Verilog 知识库问答 Agent，必须严格基于 ic_rag_search 的检索片段回答。"
            "不要使用未在工具结果中出现的通用知识补答。"
            "每个事实结论都要能在检索片段中找到直接依据；证据不足时输出严格拒答。"
            "不要在正文中写文件名、页码、chunk_id 或参考资料区块，服务端会基于真实检索结果统一重写引用。"
            "答案应简洁，优先直接回答问题，再给出必要依据。"
        )
        #组装发送给llm的内容
        llm_messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "system",
                "content": (
                    f"pre_tool_router 选择工具: {state.get('selected_tools', [])}\n"
                    f"路由原因: {state.get('route_reason', '')}\n\n"
                    f"工具结果:\n{tool_context}"
                ),
            },
            *messages,
        ]

        try:
            resp = await self._model_router.chat(
                llm_messages,
                model_preference=state.get("model_preference"),
                temperature=state.get("temperature", 0.2),
                max_tokens=state.get("max_tokens"),
            )
            return {
                "final_answer": str(getattr(resp, "content", "") or ""),
                "model_id": str(getattr(resp, "model_id", "") or ""),
                "usage": getattr(resp, "usage", None),
            }
            ‘’‘依旧异常检测：如果 LLM 调用失败，不让整个 Agent 崩溃，而是生成一个兜底回答。
            比如模型 API 超时、密钥错误、网络失败，都可以走 fallback。’‘
        except Exception as exc:  # noqa: BLE001
            fallback = self._fallback_answer(query, tool_outputs, exc)
            return {"final_answer": fallback, "model_id": "", "usage": None}
#如果前面 router 判断需要澄清，就直接把 clarification 作为最终答案返回。
    async def _clarify(self, state: AgentState) -> AgentState:
        return {"final_answer": str(state.get("clarification", ""))}
#如果命中短查询白名单/集合，就认为是短问题。比如可能包含：
    def _is_short_query(self, query: str) -> bool:
        compact = re.sub(r"\s+", "", query)
        if compact in self._SHORT_QUERY_EXACT:
            return True
        #如果长度小于2，那就认为太短
        if len(compact) <= 2:
            return True
        #抽取词项
        terms = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", query)
        #如果只有一个词，而且总长度不超过 8，就认为问题太短。
        return len(terms) <= 1 and len(compact) <= 8
#提取用户新的问题
    def _extract_user_query(self, messages: list[dict[str, Any]]) -> str:
        #倒序抽取
        for item in reversed(messages or []):
            if str(item.get("role", "")).lower() == "user":
                return str(item.get("content", "")).strip()
        return str(messages[-1].get("content", "")).strip() if messages else ""
    #根据工具名构造工具参数
    def _build_tool_args(
        self,
        tool_name: str,
        query: str,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if tool_name == "ic_rag_search":
            return {"query": query}

        if tool_name == "verilog_code_analyzer":
            code_block = self._extract_verilog_code(query)
            if not code_block:
                code_block = self._extract_verilog_code_from_history(messages)
            return {"verilog_code": code_block or query}

        if tool_name == "timing_constraint_suggester":
            args: dict[str, Any] = {"query": query}
            module_name = self._extract_module_name(query)
            if module_name:
                args["module_name"] = module_name
            period = self._extract_clock_period(query)
            if period is not None:
                args["clock_period_ns"] = period
            return args

        return {"query": query}

    def _extract_verilog_code(self, text: str) -> str:
        match = re.search(r"```(?:verilog|sv)?\s*([\s\S]*?)```", text or "", flags=re.I)
        if match:
            return match.group(1).strip()
        return ""

    def _extract_verilog_code_from_history(self, messages: list[dict[str, Any]]) -> str:
        for item in reversed(messages or []):
            if str(item.get("role", "")).lower() != "user":
                continue
            content = str(item.get("content", ""))
            code = self._extract_verilog_code(content)
            if code:
                return code
        return ""

    def _extract_module_name(self, text: str) -> str | None:
        match = re.search(r"\bmodule\s+([A-Za-z_][A-Za-z0-9_]*)", text or "", flags=re.I)
        if match:
            return match.group(1)
        return None

    def _extract_clock_period(self, text: str) -> float | None:
        match = re.search(r"(\d+(?:\.\d+)?)\s*ns", (text or "").lower())
        if not match:
            return None
        try:
            value = float(match.group(1))
        except ValueError:
            return None
        return value if value > 0 else None
#把工具输出的内容变成llm可以读的上下文
    def _render_tool_context(self, tool_outputs: list[dict[str, Any]]) -> str:
        if not tool_outputs:
            return "无工具输出。"
        #没有工具输出的时候，返回一个空，避免prompt空着
        blocks: list[str] = []
        for idx, item in enumerate(tool_outputs, 1):
            tool_name = str(item.get("tool", "unknown_tool"))
            status = "ok" if item.get("ok", False) else "error"
            result = str(item.get("result", "")).strip()
            blocks.append(f"[T{idx}] {tool_name} ({status})\n{result}")
        return "\n\n".join(blocks)

    def _fallback_answer(
        self,
        query: str,
        tool_outputs: list[dict[str, Any]],
        error: Exception,
    ) -> str:
    #没有工具结果，没有检索依据直接拒绝回答
        if not tool_outputs:
            return self._strict_refusal_template(query, f"LLM 调用失败且无工具结果: {error!s}")

        sources = self._extract_sources(tool_outputs)
        if not sources:
            return self._strict_refusal_template(query, f"LLM 调用失败且检索结果不可用: {error!s}")

        snippets = []
        for item in sources[:3]:
            content = str(item.get("content", "")).strip()
            if content:
                snippets.append(content)
        #说明检索结果为空
        if not snippets:
            return self._strict_refusal_template(query, f"LLM 调用失败且检索片段为空: {error!s}")

        return "\n".join(snippets)

#未命中标记生成器
    def _build_strict_miss_marker(self, tool_name: str, result: Any) -> str | None:
        if tool_name != "ic_rag_search": #只检查 RAG 检索工具。其他工具，比如 Verilog 分析、时序约束，不参与这个严格未命中判断。
            return None

        text = str(result or "").strip()
        if not text:
            return f"{self.STRICT_MISS_MARKER}检索工具未返回结果"

        if self._is_rag_hit(text): #如果 _is_rag_hit(text) 判断检索成功，就不生成 marker。
            return None

        reason = "知识库未命中可引用片段"
        try:
            payload = json.loads(text)
        except Exception:
            payload = None
        if isinstance(payload, dict):
            reason = str(payload.get("reason", "")).strip() or reason

        return f"{self.STRICT_MISS_MARKER}{reason}"
#二次拒答
    def _build_strict_refusal_if_needed(
        self,
        query: str,
        state: AgentState,
        tool_outputs: list[dict[str, Any]],
    ) -> str | None:
        selected_tools = list(state.get("selected_tools") or [])
        if "ic_rag_search" not in selected_tools:
            return None

        rag_output = next((x for x in tool_outputs if x.get("tool") == "ic_rag_search"), None)
        if rag_output is None:
            return self._strict_refusal_template(query, "检索工具未返回结果")

        rag_text = str(rag_output.get("result", "")).strip()
        if not self._is_rag_hit(rag_text):
            return self._strict_refusal_template(query, "知识库未命中可引用片段")

        weak_reason = self._weak_evidence_reason(query, rag_text)
        if weak_reason:
            return self._strict_refusal_template(query, weak_reason)

        return None #表示证据通过检查，可以继续让 LLM 生成最终答案。

    def _weak_evidence_reason(self, query: str, rag_text: str) -> str | None:
        try:
            payload = json.loads(rag_text)
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None

        results = payload.get("results")
        if not isinstance(results, list) or not results:
            return "知识库未命中可引用片段"

        contents = [str(item.get("content", "")).strip() for item in results if isinstance(item, dict)]
        if not contents:
            return "检索结果缺少可引用正文"
        if all(len(content) < 40 for content in contents):
            return "检索片段过短，无法支撑可靠回答"

        return None

    def _core_query_terms(self, query: str) -> list[str]:
        terms = re.findall(r"[A-Za-z][A-Za-z0-9_]+|[一-鿿]{2,}", query or "")
        stopwords = {
            "在", "中", "是否", "可以", "什么", "分别", "主要", "用途", "含义", "推荐",
            "使用", "进行", "如果", "发生", "给定", "参考", "文本", "期间", "更改",
            "verilog", "hdl", "语言", "程序", "模块", "实例", "定义", "表达式", "单行", "多行",
            "语法", "建立", "模型", "元件", "单元", "执行",
        }
        out: list[str] = []
        seen: set[str] = set()
        for term in terms:
            key = term.lower()
            if key in seen or term.lower() in stopwords or term in stopwords:
                continue
            if len(term) < 2:
                continue
            seen.add(key)
            out.append(term)
        return out[:5]

    def _is_rag_hit(self, text: str) -> bool:
        content = (text or "").strip()
        if not content:
            return False

        try:
            payload = json.loads(content)
        except Exception:
            payload = None
        if isinstance(payload, dict):
            results = payload.get("results")
            if isinstance(results, list):
                return len(results) > 0

        no_hit_markers = (
            "知识库中未找到相关信息",
            "知识库中未找到足够相关的信息",
            "未找到相关信息",
            "未找到足够相关的信息",
        )
        if any(marker in content for marker in no_hit_markers):
            return False

        if re.search(r"\[R\d+\]\s*来源:\s*.+?\|\s*页码:", content):
            return True

        if all(token in content for token in ("content", "source", "page")):
            return True

        return False

    def _strict_refusal_template(self, query: str, reason: str) -> str:
        return (
            "【严格拒答】\n"
            "当前知识库未命中可引用证据，无法基于资料直接回答。\n"
            f"原因: {reason}\n"
            f"问题: {query}\n\n"
            "请补充更具体关键词，或更新/上传相关 PDF 后重试。"
        )

    def _extract_sources(self, tool_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        for event in tool_events:
            if str(event.get("tool", "")) != "ic_rag_search":
                continue

            raw = str(event.get("result", "")).strip()
            if not raw:
                continue

            from_json = self._extract_sources_from_json(raw)
            if from_json:
                sources.extend(from_json)
                continue

            from_text = self._extract_sources_from_text(raw)
            if from_text:
                sources.extend(from_text)

        # 去重（source/page/chunk_id/content）
        uniq: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str, str]] = set()
        for item in sources:
            key = (
                str(item.get("source", "")),
                str(item.get("page", "")),
                str(item.get("chunk_id", "")),
                str(item.get("content", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            uniq.append(item)
        return uniq

    def _extract_sources_from_json(self, raw: str) -> list[dict[str, Any]]:
        try:
            payload = json.loads(raw)
        except Exception:
            return []

        candidates: list[dict[str, Any]] = []
        if isinstance(payload, dict):
            if all(k in payload for k in ("content", "source", "page")):
                candidates.append(payload)
            elif isinstance(payload.get("results"), list):
                for item in payload["results"]:
                    if isinstance(item, dict):
                        candidates.append(item)
        elif isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    candidates.append(item)

        normalized: list[dict[str, Any]] = []
        for idx, item in enumerate(candidates, 1):
            normalized_item = self._normalize_source_item(item, idx)
            if normalized_item is not None:
                normalized.append(normalized_item)
        return normalized

    def _extract_sources_from_text(self, raw: str) -> list[dict[str, Any]]:
        pattern = re.compile(
            r"\[R\d+\]\s*来源:\s*(?P<source>.+?)\s*\|\s*页码:\s*(?P<page>.+?)\n片段:\s*(?P<content>[\s\S]*?)(?=\n\n\[R\d+\]|\Z)"
        )
        matches = list(pattern.finditer(raw))
        if not matches:
            return []

        out: list[dict[str, Any]] = []
        for idx, m in enumerate(matches, 1):
            source = m.group("source").strip()
            page = m.group("page").strip() or "页码未知"
            content = m.group("content").strip()
            if not source:
                continue
            out.append(
                {
                    "content": content,
                    "source": source,
                    "page": page,
                    "score": 0.0,
                    "chunk_id": f"{source}#{page}#r{idx}",
                }
            )
        return out

    def _normalize_source_item(self, item: dict[str, Any], idx: int) -> dict[str, Any] | None:
        source = str(item.get("source", "")).strip()
        if not source:
            return None

        page = str(item.get("page", "页码未知")).strip() or "页码未知"
        content = str(item.get("content", "")).strip()
        chunk_id = str(item.get("chunk_id", "")).strip() or f"{source}#{page}#r{idx}"

        score_raw = item.get("score", 0.0)
        try:
            score = float(score_raw)
        except (TypeError, ValueError):
            score = 0.0

        return {
            "content": content,
            "source": source,
            "page": page,
            "score": score,
            "chunk_id": chunk_id,
        }
