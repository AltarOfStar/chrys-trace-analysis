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


class Session(BaseModel):
    session_uuid: str
    session_abstract: list[SessionRound]
    mcp_tools: list[str] = []
    skills: list[str] = []
    # Full turn data with raw messages, for detailed per-turn analysis
    turns: list[SessionTurn] = []
    # Session metadata (meta section of the session.json envelope)
    meta: dict = {}


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
    """一次人工介入：类型、位置与简要描述。

    type 为聚合后的标准类型；type_raw 保留批次分析阶段的原始类型。
    """
    type: str = ""
    type_raw: str = ""
    position: str = ""
    description: str = ""


class TraceTaskAnalysis(BaseModel):
    """单条轨迹的任务与人介入分析结果。

    task_category 为聚合后的标准任务分类（单个词）；task_category_raw 保留
    批次分析阶段的原始分类词。
    """
    session_uuid: str
    user_name: str = ""
    task_category: str = ""
    task_category_raw: str = ""
    task_summary: str = ""
    human_intervention: bool = False
    interventions: list[Intervention] = []


class CategoryDefinition(BaseModel):
    """聚合后的固定分类（任务分类或人工介入类型）。"""
    name: str
    description: str = ""
    count: int = 0


class TraceAnalysisResult(BaseModel):
    """Result of the trace analysis pipeline."""
    analyses: list[TraceTaskAnalysis]
    task_categories: list[CategoryDefinition]
    intervention_types: list[CategoryDefinition]
    task_category_mapping: dict[str, str] = {}
    intervention_type_mapping: dict[str, str] = {}
    summary: dict
