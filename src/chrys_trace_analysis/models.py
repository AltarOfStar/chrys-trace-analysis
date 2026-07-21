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
    # Session metadata from MongoDB
    meta: dict = {}


class UserData(BaseModel):
    user_name: str
    total_add_lines: int
    groups: list[str] = []
    sessions: list[Session]


class Scenario(BaseModel):
    name: str
    description: str


class ScenarioWithTrace(Scenario):
    representative_session_uuid: str


class ClassifiedSession(BaseModel):
    session_uuid: str
    user_name: str
    scenario_name: str


class SimplifiedTrace(BaseModel):
    """单个会话的简化轨迹，不含原始消息（用于 on-disk 存储）。"""
    uuid: str
    user_name: str
    session_abstract: list[SessionRound]
    mcp_tools: list[str] = []
    skills: list[str] = []


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
