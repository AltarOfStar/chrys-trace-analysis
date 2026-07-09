from __future__ import annotations

from pydantic import BaseModel


class SessionRound(BaseModel):
    user_msg: str
    assistant_reply: str


class Session(BaseModel):
    session_uuid: str
    session_abstract: list[SessionRound]
    mcp_tools: list[str] = []
    skills: list[str] = []


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


class AnalysisResult(BaseModel):
    scenarios: list[Scenario]
    classified_sessions: list[ClassifiedSession]
    summary: dict
