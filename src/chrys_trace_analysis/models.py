from __future__ import annotations

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Raw message models (mirror MongoDB session document structure)
# ---------------------------------------------------------------------------


class MessageContent(BaseModel):
    """A single content block within a raw message (text, tool_call, tool_result, etc.)."""
    type: str = ""
    text: str = ""
    tool_name: str = ""
    call_id: str = ""
    name: str = ""
    arguments: str = ""
    result: str = ""


class RawMessage(BaseModel):
    """A raw message as stored in MongoDB/chrys session JSON."""
    role: str  # user, assistant, system, tool
    contents: list[MessageContent] = []
    additional_properties: dict = {}
    message_id: str = ""


class SessionTurn(BaseModel):
    """A full conversational turn containing all raw messages."""
    turn_index: int
    messages: list[RawMessage]


class SessionRound(BaseModel):
    user_msg: str
    assistant_reply: str


class ApprovalRecord(BaseModel):
    """一次人工审批记录（chrys approvals/*.log）。

    工具调用前的人工检查点：turn_index 为对齐到的轮次（可能为空），
    approved 为用户是否批准，reason 为用户给出的理由（截断），
    prompt 为审批时刻的用户提示（用于轮次对齐），arguments_head 为
    提议的工具参数截断预览。
    """
    turn_index: int | None = None
    timestamp: str = ""
    tool_name: str = ""
    kind: str = ""
    prompt: str = ""
    approved: bool = True
    reason: str = ""
    arguments_head: str = ""


class SubAgentRecord(BaseModel):
    """一次子代理会话摘要（chrys sub_agents/sessions/*.json）。

    主会话只保留子代理的最终结果，这里保留子代理的结构化摘要：
    工具名、状态、任务预览、内部消息数、内部工具调用数与工具分布。
    turn_index 为按 parent call_id 对齐到的主会话轮次（可能为空）。
    """
    turn_index: int | None = None
    tool_name: str = ""
    status: str = ""
    parent_call_id: str = ""
    prompt_preview: str = ""
    message_count: int = 0
    tool_call_count: int = 0
    tool_usage: dict = {}


class Session(BaseModel):
    session_uuid: str
    session_abstract: list[SessionRound]
    mcp_tools: list[str] = []
    skills: list[str] = []
    # Full turn data with raw messages, for detailed per-turn analysis
    turns: list[SessionTurn] = []
    # Session metadata (meta section of the session.json envelope)
    meta: dict = {}
    # Human approval checkpoints (from approvals/*.log of the session directory)
    approvals: list[ApprovalRecord] = []
    # Sub-agent session summaries (from sub_agents/sessions/*.json)
    sub_agents: list[SubAgentRecord] = []


class Scenario(BaseModel):
    name: str
    description: str


class ScenarioWithTrace(Scenario):
    representative_session_uuid: str


class ClassifiedSession(BaseModel):
    session_uuid: str
    user_name: str
    scenario_name: str


class AnalysisResult(BaseModel):
    scenarios: list[Scenario]
    classified_sessions: list[ClassifiedSession]
    summary: dict


class DeviatedSession(BaseModel):
    """A session where agent execution deviated from user intent."""
    session_uuid: str
    user_name: str
    deviation_reason: str = ""
    problematic_turn_index: int | None = None
    category: str = ""


class TurnProblem(BaseModel):
    """Detailed analysis of a specific turn's deviation."""
    session_uuid: str
    user_name: str = ""
    turn_index: int
    deviation_action: str = ""
    turn_analysis: str = ""


class OffsetAnalysisResult(BaseModel):
    """Result of the offset analysis pipeline."""
    deviated_sessions: list[DeviatedSession]
    summary: dict
    turn_problems: list[TurnProblem] = []


class Intervention(BaseModel):
    """一次人工介入：所在轮次、类型与简要描述。

    turn_index 为结构化轮次号（1 起），position 保留模型给出的自由文本位置
    （可为空）；type 为聚合后的标准类型；type_raw 保留逐条分析阶段的原始类型。
    """
    turn_index: int | None = None
    type: str = ""
    type_raw: str = ""
    position: str = ""
    description: str = ""


class TraceTaskAnalysis(BaseModel):
    """单条轨迹的任务与人介入分析结果。

    task_category 为聚合后的标准任务分类（单个词）；task_category_raw 保留
    逐条分析阶段的原始分类词。features 为本地抽取的确定性特征（轮次数、
    工具调用次数、工具使用分布等）。failed 为 True 表示 LLM 分析失败
    （多次重试后仍失败），此时除 session_uuid/user_name/features 外均为空。
    """
    session_uuid: str
    user_name: str = ""
    task_category: str = ""
    task_category_raw: str = ""
    task_summary: str = ""
    human_intervention: bool = False
    interventions: list[Intervention] = []
    features: dict = {}
    failed: bool = False


class CategoryDefinition(BaseModel):
    """聚合后的固定分类（任务分类或人工介入类型）。"""
    name: str
    description: str = ""
    count: int = 0
    # 属于该分类的轨迹 uuid（任务分类）/ 介入所属轨迹 uuid（介入类型）
    session_uuids: list[str] = []


class TraceAnalysisResult(BaseModel):
    """Result of the trace analysis pipeline."""
    analyses: list[TraceTaskAnalysis]
    task_categories: list[CategoryDefinition]
    intervention_types: list[CategoryDefinition]
    task_category_mapping: dict[str, str] = {}
    intervention_type_mapping: dict[str, str] = {}
    summary: dict
