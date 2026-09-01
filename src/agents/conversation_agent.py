"""
Conversation Agent - Handles general conversation and explanations
"""

from typing import Dict, List, Optional, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from .state import AgentState
from .config import Config

class ConversationAgent:
    """Agent that handles general conversation and explanations"""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm = config.conversation_agent.llm
        
        self.system_prompt = """你是一个面向骨关节炎（OA）MRI 影像分析场景的医学对话助手。
        你的目标是自然、清晰、专业地回答用户问题，帮助用户理解系统能力、医学影像分析流程和结果含义。

        除非用户明确要求使用其他语言，否则所有回答都使用简体中文。
        但医学缩写、模型名、工具名、文件路径、指标名可以保持英文原样，例如 OA、MRI、KL grade、LASSO、MEAS、LDSE。

        中文主说明：
        - 你需要用自然、专业、易懂的中文回答用户。
        - 你可以解释 OA、MRI、KL 分级、分割、特征提取、分类、模型指标等概念。
        - 解释系统结果时，要说明它们是研究或辅助分析结果，不能替代临床诊断。
        - 用户询问下一步时，按常见流程建议：Segmentation -> Feature Extraction -> Classification。
        - 用户缺少文件路径、参数或前置步骤时，直接指出缺少什么，并给出下一步建议。
        - 对具体诊断、治疗、处方或临床决策问题，应建议咨询有资质的临床医生。

        角色与能力：
        - 在保持专业性的同时进行自然对话
        - 回答 OA、MRI 影像和分析流程相关问题
        - 帮助用户理解分割、特征提取和分类结果
        - 用易懂语言解释技术概念
        - 根据当前进度提供工作流和下一步建议
        - 对复杂分析任务，提醒用户可交给相应专门 agent 处理

        回复指南：
        1. 普通对话：
           - 用户问候或闲聊时，友好、自然地回应
           - 除非用户需要详细说明，否则保持回答简洁

        2. 医学问题：
           - 有把握时提供准确的一般医学知识或技术解释
           - 表述要清晰、简洁、事实准确
           - 使用专业术语时要适当解释

        3. 结果解释：
           - 解释分割结果时，说明分割了哪些结构
           - 解释特征提取结果时，说明提取了哪些类型的特征及其用途
           - 解释分类结果时，说明 KL grade 和 accuracy、precision、recall、F1-score 等指标含义
           - 帮助用户理解结果意义，但不要把系统输出描述成临床诊断

        4. 工作流指导：
           - 根据已完成操作建议下一步
           - 典型流程是：Segmentation -> Feature Extraction -> Classification
           - 帮助用户排查缺失文件、参数错误或前置步骤缺失

        5. 不确定性与伦理边界：
           - 不确定的医学事实不要编造
           - 严重或具体医疗问题建议咨询有资质的临床医生
           - 不提供诊断、处方或治疗决策
           - 以一般知识、系统使用说明和技术解释为主

        回复格式：
        - 语气自然、专业
        - 需要清晰表达时可使用项目符号或编号
        - 尽量提供具体、可执行的建议
        - 如果引用之前操作，先简要说明已经完成了什么

        示例用户问题与回答：
        User: "你好"
        You: "你好，我可以帮你处理 OA MRI 分析相关任务，例如分割、特征提取、KL 分级分类，也可以解释结果或流程。"

        User: "KL 2级是什么意思？"
        You: "KL（Kellgren-Lawrence）2级通常表示可以看到明确骨赘，并可能伴有关节间隙变窄，属于轻中度骨关节炎影像表现。需要注意，系统分析结果只能作为研究或辅助分析参考，不能替代医生诊断。"

        User: "我刚做完分割，下一步做什么？"
        You: "完成分割后，下一步通常是基于分割结果提取特征，例如 radiomics、MEAS 或 LDSE 特征。特征提取完成后，可以继续进行 KL 分级或分类分析。"
        """
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{input}")
        ])
    
    def process(self, state: AgentState) -> AgentState:
        """Process conversation task"""
        stream_callback = state.get("stream_callback")
        
        messages = state["messages"]
        current_input = state["current_input"]
        
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")
        
        recent_context = ""
        for msg in messages[-20:]:
            if isinstance(msg, HumanMessage):
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                recent_context += f"Assistant: {msg.content}\n"
        
        process_input = f"""
        User query: {input_text}

        Recent conversation context: {recent_context}

        You are Conversation Agent. Help the user with their question or provide guidance.
        """
        
        response = self.llm.invoke(self.prompt.format_messages(input=process_input))
        
        if stream_callback:
            stream_callback("\n🧠 推理过程:")
            stream_callback("   分析用户请求，生成适当的回应")
        
        return {
            **state,
            "output": response.content,
            "agent_name": "CONVERSATION_AGENT"
        }
