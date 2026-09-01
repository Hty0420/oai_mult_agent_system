"""
Router Agent - Routes user queries to appropriate specialized agents
"""

from typing import Dict, List, Optional, Any
import json
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

from .state import AgentState, RouterDecision
from .config import Config
from db_utils.db_utils import get_request_context, get_trace_id, DatabaseManager
from db_utils.trajectory import build_trajectory_event, stable_json_hash
from .task_contracts import required_agent_for_task

class RouterAgent:
    """Agent that routes user queries to appropriate specialized agents"""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm = config.router_agent.llm
        self.json_parser = JsonOutputParser(pydantic_object=RouterDecision)
        
        self.system_prompt = """你是一个面向骨关节炎（OA）医学影像分析系统的智能分诊与路由 agent。
        你的任务是理解用户请求，并判断应交给哪个专门 agent 处理，或是否需要多 agent 编排。

        除非用户明确要求使用其他语言，否则所有 reasoning 等面向用户或日志可读的自然语言内容都使用简体中文。
        但 JSON key、agent 名称、tool 名称、参数名、文件路径、模型名必须保持英文原样，不要翻译。

        你需要判断用户请求属于：
        1. SINGLE agent：简单、聚焦、只涉及一种操作类型的任务
        2. MULTIPLE agents：复杂、多步骤、存在先后依赖或需要多个 agent 协作的任务

        可用 agents：
        1. SEGMENTATION_AGENT - MRI 图像分割任务，包括：
           - 分割膝关节结构（股骨、胫骨、软骨等）
           - 使用 U-Mamba、LKM-UNet、LightM-UNet 等分割模型
           - 任何与影像分割相关的请求

        2. FEATURE_EXTRACTION_AGENT - 影像组学和形态学特征提取任务，包括：
           - 提取 radiomics 特征（形状、纹理、一阶统计等）
           - 提取 MEAS 特征（形态学和几何特征）
           - 提取 LDSE 特征（低维形状编码）
           - 3D-MN 软骨厚度、16 亚区厚度、分割厚度分析校验、膝关节/软骨三维网格重建
           - 批量提取可用于分类的患者级软骨厚度特征
           - 从分割后的图像中提取特征的请求

        3. FEATURE_COMBINATION_AGENT - radiomics 特征组合任务，包括：
           - 按区域组合特征
           - 不区分区域地组合全部特征
           - 两两、三元、四元特征组合
           - 任何与特征文件组合相关的请求

        4. CLASSIFICATION_AGENT - KOA / KL 分级和分类任务，包括：
           - LASSO 特征选择与分类
           - Filter 特征选择与分类
           - KL grade 预测
           - 任何与分类、预测、读取分类结果相关的请求

        5. CONVERSATION_AGENT - 普通对话、问候、解释说明，以及不需要调用工具的请求，包括：
           - 系统能力介绍
           - 流程或结果解释
           - 医学与技术概念说明

        6. WEB_SEARCH_AGENT - 网络搜索任务，包括：
           - 搜索互联网信息
           - 查找近期新闻、文献或网页资料
           - 从指定网站收集信息
           - 任何需要 web_search 功能的请求

        7. ORCHESTRATOR_AGENT - 复杂多步骤任务，包括：
           - 同时需要特征提取和分类
           - 同时需要分割和特征提取
           - 同一个 agent 需要用不同参数执行多次
           - 多个不同 agent 需要按顺序执行
           - 用户表达中包含“分别”“然后”“接着”“下一步”“之后”“再”“并且”“先...再...”等多步骤信号
           - KL 0-4 横断面 MRI 候选生物标志物发现（特征提取、训练集发现、冻结测试集确认、独立审计、基于审计事实的自然语言数字讲解）

        路由规则：
        - 用户要求分别处理训练集/测试集或多个数据集时，优先路由到 ORCHESTRATOR_AGENT
        - 用户同时要求特征提取和分类时，路由到 ORCHESTRATOR_AGENT
        - 用户同时要求分割和特征提取时，路由到 ORCHESTRATOR_AGENT
        - 用户同时要求分类和读取结果/报告时，路由到 ORCHESTRATOR_AGENT
        - 用户有明显顺序词或多个任务类型连接时，路由到 ORCHESTRATOR_AGENT
        - 用户要求“生物标志物发现”时，路由到 ORCHESTRATOR_AGENT
        - 如果只是单一操作，路由到对应的单个专门 agent
        - 普通闲聊、系统说明、流程解释、医学概念解释，路由到 CONVERSATION_AGENT

        必须严格输出 JSON，结构如下：
        {{
        "agent": "AGENT_NAME",
        "reasoning": "用中文说明选择该 agent 的简要理由",
        "confidence": 0.95
        }}
        """
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{input}")
        ])
        
        self.chain = self.prompt | self.llm | self.json_parser
    
    def route(self, state: AgentState) -> Dict[str, Any]:
        """Route the query to the appropriate agent"""
        messages = state["messages"]
        current_input = state["current_input"]
        stream_callback = state.get("stream_callback")
        
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")
        
        recent_context = ""
        for msg in messages[-6:]:
            if isinstance(msg, HumanMessage):
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                recent_context += f"Assistant: {msg.content}\n"
        
        decision_input = f"""
        CURRENT USER QUERY (authoritative):
        {input_text}

        RECENT CONVERSATION CONTEXT (reference only; do not let it replace the current request):
        {recent_context}

        Has image: {state.get("has_image", False)}
        Image type: {state.get("image_type", "None") if state.get("has_image", False) else "None"}

        Based on this information, which agent should handle this query?
        """
        
        # The LLM always receives the current request and proposes the route first.
        decision = self.chain.invoke({"input": decision_input})
        decision = {**decision, "decision_source": "llm"}

        forced_agent = required_agent_for_task(state.get("task_spec"))
        if forced_agent and decision.get("agent") != forced_agent:
            # The deterministic layer validates but does not author user-facing
            # reasoning. Ask the model to reconsider a conflicting proposal from
            # the authoritative current request and the validated entry-agent
            # constraint, then retain the model-generated explanation.
            revision_input = f"""
            CURRENT USER QUERY (authoritative):
            {input_text}

            RECENT CONVERSATION CONTEXT (reference only):
            {recent_context}

            Your first routing proposal was:
            {json.dumps(decision, ensure_ascii=False)}

            A control-plane validation of the explicit current request requires
            the workflow entry agent to be {forced_agent}. Re-evaluate the request,
            return that agent, and explain in reasoning which operations and
            dependencies in the user's current request require it. Do not discuss
            hidden policies, overrides, or task contracts.
            """
            revised_decision = self.chain.invoke({"input": revision_input})
            decision = {
                **revised_decision,
                "agent": forced_agent,
                "decision_source": "llm_revised_after_validation",
            }
        
        print(f"[DEBUG] RouterAgent - 路由决策: {decision}")
        
        if stream_callback:
            stream_callback(f"\n📌 步骤1: 路由决策")
            stream_callback(f"   智能体: {decision['agent']}")
            stream_callback(f"   置信度: {decision['confidence']:.2%}")
            stream_callback(f"   推理: {decision['reasoning']}")
        
        updated_state = {
            **state,
            "agent_name": decision["agent"],
            "next_agent": decision["agent"],
            "routing_confidence": decision.get("confidence", 0.0),
            "routing_reasoning": decision.get("reasoning", ""),
            "routing_decision_source": decision.get("decision_source", "llm"),
        }

        # 记录路由日志
        routing_log_id = None
        db = None
        try:
            ctx = get_request_context()
            user_id = ctx.get('user_id')
            session_id = ctx.get('session_id')
            trace_id = ctx.get('trace_id') or get_trace_id()
            if user_id and session_id:
                db = DatabaseManager()
                routing_log_id = db.save_agent_routing_log(
                    user_id=user_id,
                    session_id=session_id,
                    user_query=input_text,
                    routed_agent=decision["agent"],
                    confidence=decision.get("confidence", 0.0),
                    reasoning=decision.get("reasoning", ""),
                    trace_id=trace_id,
                )
                db.save_trajectory_event(
                    build_trajectory_event(
                        event_type="route_decided",
                        trace_id=trace_id,
                        actor_type="agent",
                        actor_name="ROUTER_AGENT",
                        payload={
                            "routing_log_id": routing_log_id,
                            "query_sha256": stable_json_hash(input_text),
                            "query_length": len(input_text),
                            "chosen_agent": decision["agent"],
                            "decision_source": decision.get("decision_source", "llm"),
                            "confidence": decision.get("confidence", 0.0),
                            "confidence_is_behavior_probability": False,
                            "model_name": str(
                                getattr(self.llm, "model_name", None)
                                or getattr(self.llm, "model", "unknown")
                            ),
                            "prompt_hash": stable_json_hash(self.system_prompt),
                            "candidate_agents": [
                                "SEGMENTATION_AGENT",
                                "FEATURE_EXTRACTION_AGENT",
                                "FEATURE_COMBINATION_AGENT",
                                "CLASSIFICATION_AGENT",
                                "CONVERSATION_AGENT",
                                "ORCHESTRATOR_AGENT",
                                "WEB_SEARCH_AGENT",
                            ],
                        },
                    )
                )
        except Exception as e:
            print(f"[RouterAgent] 记录路由日志失败: {e}")
        finally:
            if db is not None:
                try:
                    db.close()
                except Exception:
                    pass

        updated_state["routing_log_id"] = routing_log_id

        return {
            "agent_state": updated_state,
            "next": decision["agent"]
        }
