"""Tests for session fetching and conversion to model objects."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

# ——————————————————————————————————————————————————————————————————————————————
# Set up import path (tests directory is outside src/)
# ——————————————————————————————————————————————————————————————————————————————

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ——————————————————————————————————————————————————————————————————————————————
# Imports from the package under test
# ——————————————————————————————————————————————————————————————————————————————

from chrys_trace_analysis.models import (  # noqa: E402
    DeviatedSession,
    MessageContent,
    OffsetAnalysisResult,
    RawMessage,
    Session,
    SessionRound,
    SessionTurn,
    TurnProblem,
    UserData,
)
from chrys_trace_analysis.mongo_loader import (  # noqa: E402
    _build_user_to_groups,
    _extract_texts,
    _parse_message_content,
    _parse_raw_message,
    _safe_int,
    build_session_abstract,
    build_session_turns,
    mongo_docs_to_user_data,
    parse_group_members,
)


# ==============================================================================
# Fixtures
# ==============================================================================


def _make_content(content_type: str, **kwargs: str) -> dict:
    return {"type": content_type, **kwargs}


def _make_msg(role: str, contents: list[dict], **kwargs: str | dict) -> dict:
    msg: dict = {"role": role, "contents": contents}
    msg.update(kwargs)
    return msg


@pytest.fixture
def sample_messages() -> list[dict]:
    """Realistic conversation with 2 turns and tool calls."""
    return [
        # ── Turn 1 ──
        _make_msg("system", [_make_content("text", text="You are a coding assistant.")]),
        _make_msg("user", [_make_content("text", text="帮我写一个 utils.py，包含一个 add 函数。")]),
        _make_msg(
            "assistant",
            [
                _make_content("text", text="好的，我先看看项目结构。"),
                _make_content(
                    "function_call",
                    tool_name="list_files",
                    call_id="call_001",
                    arguments='{"path": "/project"}',
                ),
            ],
        ),
        _make_msg(
            "tool",
            [
                _make_content(
                    "function_result",
                    call_id="call_001",
                    result='{"files": ["main.py"]}',
                ),
            ],
        ),
        _make_msg(
            "assistant",
            [_make_content("text", text="现在创建 utils.py。")],
        ),
        _make_msg(
            "assistant",
            [
                _make_content(
                    "function_call",
                    tool_name="write_file",
                    call_id="call_002",
                    arguments='{"path": "utils.py", "content": "def add(a, b): return a+b"}',
                ),
            ],
        ),
        _make_msg(
            "tool",
            [
                _make_content(
                    "function_result",
                    call_id="call_002",
                    result='{"success": true}',
                ),
            ],
        ),
        _make_msg(
            "assistant",
            [_make_content("text", text="已创建 utils.py，包含 add 函数。")],
        ),
        # boundary message
        _make_msg(
            "assistant",
            [],
            additional_properties={"_chrys_kind": "turn"},
            message_id="boundary_1",
        ),
        # ── Turn 2 ──
        _make_msg(
            "user",
            [
                _make_content("text", text="不对，我要的是乘法函数，不是加法。"),
                _make_content("text", text="函数名叫 multiply。"),
            ],
        ),
        _make_msg(
            "assistant",
            [_make_content("text", text="抱歉，我马上修正。")],
        ),
        _make_msg(
            "assistant",
            [
                _make_content(
                    "function_call",
                    tool_name="write_file",
                    call_id="call_003",
                    arguments='{"path": "utils.py", "content": "def multiply(a, b): return a*b"}',
                ),
            ],
        ),
        _make_msg(
            "tool",
            [
                _make_content(
                    "function_result",
                    call_id="call_003",
                    result='{"success": true}',
                ),
            ],
        ),
        _make_msg(
            "assistant",
            [_make_content("text", text="已修正为 multiply 函数。")],
        ),
        _make_msg(
            "assistant",
            [],
            additional_properties={"_chrys_kind": "turn"},
            message_id="boundary_2",
        ),
    ]


@pytest.fixture
def sample_mongo_doc(sample_messages: list[dict]) -> dict:
    """A single MongoDB session document."""
    return {
        "uuid": "sess-001",
        "meta": {
            "user_id": "user_zhangsan",
            "title": "创建工具函数",
            "primary_cwd": "/home/user/project",
        },
        "messages": sample_messages,
        "mcp_tools": [
            {"tool_name": "list_files", "arguments": {}, "result": "...", "call_id": "c1", "turn": 1},
            {"tool_name": "write_file", "arguments": {}, "result": "...", "call_id": "c2", "turn": 1},
        ],
        "skills": [
            {"skill_name": "python-coding", "description": "Python coding assistant"},
        ],
        "mr_relation": {"add_lines_hits": 25, "mounted_mr": "https://mr.example.com/123"},
        "state": {"turn_counter": 2},
    }


@pytest.fixture
def group_members_json() -> dict:
    """Sample group_members_stat.json content."""
    return {
        "groups": [
            {
                "group_id": 1001,
                "group_name": "后端一组",
                "members": [
                    {"user_id": "user_zhangsan", "department": "后端"},
                    {"user_id": "user_lisi", "department": "后端"},
                ],
            },
            {
                "group_id": 1002,
                "group_name": "前端一组",
                "members": [
                    {"user_id": "user_wangwu", "department": "前端"},
                ],
            },
        ],
    }


# ==============================================================================
# _safe_int
# ==============================================================================


class TestSafeInt:
    def test_valid_int(self):
        assert _safe_int(42) == 42

    def test_int_string(self):
        assert _safe_int("42") == 42

    def test_none_returns_default(self):
        assert _safe_int(None, default=5) == 5

    def test_invalid_string_returns_default(self):
        assert _safe_int("not a number", default=3) == 3

    def test_default_default_is_zero(self):
        assert _safe_int("nope") == 0


# ==============================================================================
# _extract_texts
# ==============================================================================


class TestExtractTexts:
    def test_single_text(self):
        msg = {"contents": [{"type": "text", "text": "hello"}]}
        assert _extract_texts(msg) == ["hello"]

    def test_multiple_texts(self):
        msg = {
            "contents": [
                {"type": "text", "text": "first line"},
                {"type": "text", "text": "second line"},
            ],
        }
        assert _extract_texts(msg) == ["first line", "second line"]

    def test_skips_non_text(self):
        msg = {
            "contents": [
                {"type": "function_call", "tool_name": "read_file"},
                {"type": "text", "text": "the answer"},
            ],
        }
        assert _extract_texts(msg) == ["the answer"]

    def test_fallback_to_content_field(self):
        """Fallback only triggers when contents is present but NOT a list."""
        msg = {"contents": "not_a_list", "content": "fallback text"}
        assert _extract_texts(msg) == ["fallback text"]

    def test_fallback_when_contents_not_list(self):
        msg = {"contents": "not a list", "content": "fallback"}
        assert _extract_texts(msg) == ["fallback"]

    def test_empty_message(self):
        assert _extract_texts({}) == []

    def test_empty_text_value_skipped(self):
        msg = {"contents": [{"type": "text", "text": ""}]}
        assert _extract_texts(msg) == []


# ==============================================================================
# _parse_message_content
# ==============================================================================


class TestParseMessageContent:
    def test_text_type(self):
        mc = _parse_message_content({"type": "text", "text": "Hello"})
        assert isinstance(mc, MessageContent)
        assert mc.type == "text"
        assert mc.text == "Hello"

    def test_tool_call_with_dict_arguments(self):
        mc = _parse_message_content({
            "type": "function_call",
            "tool_name": "read_file",
            "call_id": "abc",
            "arguments": {"path": "/foo.py"},
        })
        assert mc.type == "function_call"
        assert mc.tool_name == "read_file"
        parsed_args = json.loads(mc.arguments)
        assert parsed_args["path"] == "/foo.py"

    def test_tool_call_with_string_arguments(self):
        mc = _parse_message_content({
            "type": "function_call",
            "tool_name": "write_file",
            "arguments": '{"path":"bar.py"}',
        })
        assert "bar.py" in mc.arguments

    def test_tool_result_with_dict_result(self):
        mc = _parse_message_content({
            "type": "function_result",
            "call_id": "abc",
            "result": {"status": "ok"},
        })
        assert mc.type == "function_result"
        assert "ok" in mc.result

    def test_empty_content(self):
        mc = _parse_message_content({})
        assert mc.type == ""
        assert mc.text == ""

    def test_name_field(self):
        mc = _parse_message_content({"name": "my_tool"})
        assert mc.name == "my_tool"


# ==============================================================================
# _parse_raw_message
# ==============================================================================


class TestParseRawMessage:
    def test_basic_user_message(self):
        rm = _parse_raw_message({
            "role": "user",
            "contents": [{"type": "text", "text": "Hello"}],
            "message_id": "msg_1",
        })
        assert isinstance(rm, RawMessage)
        assert rm.role == "user"
        assert rm.message_id == "msg_1"
        assert len(rm.contents) == 1
        assert rm.contents[0].text == "Hello"

    def test_multiple_contents(self):
        rm = _parse_raw_message({
            "role": "assistant",
            "contents": [
                {"type": "text", "text": "Let me check."},
                {"type": "function_call", "tool_name": "read_file", "call_id": "c1"},
                {"type": "function_call", "tool_name": "write_file", "call_id": "c2"},
            ],
        })
        assert len(rm.contents) == 3
        assert rm.contents[1].tool_name == "read_file"

    def test_additional_properties(self):
        rm = _parse_raw_message({
            "role": "assistant",
            "contents": [],
            "additional_properties": {"_chrys_kind": "turn", "key": "val"},
        })
        assert rm.additional_properties == {"_chrys_kind": "turn", "key": "val"}

    def test_additional_properties_none(self):
        rm = _parse_raw_message({"role": "assistant", "contents": []})
        assert rm.additional_properties == {}

    def test_empty_contents(self):
        rm = _parse_raw_message({"role": "system"})
        assert rm.contents == []
        assert rm.message_id == ""


# ==============================================================================
# build_session_abstract
# ==============================================================================


class TestBuildSessionAbstract:
    def test_two_turns(self, sample_messages: list[dict]):
        abstract = build_session_abstract(sample_messages)
        assert len(abstract) == 2
        # Turn 1: user asks for add function
        assert "add" in abstract[0].user_msg
        assert "创建" in abstract[0].assistant_reply
        # Turn 2: user corrects to multiply
        assert "乘法" in abstract[1].user_msg
        assert "multiply" in abstract[1].assistant_reply

    def test_empty_messages(self):
        assert build_session_abstract([]) == []

    def test_none_input(self):
        assert build_session_abstract(None) == []  # type: ignore[arg-type]

    def test_no_turn_boundaries(self):
        """Without _chrys_kind markers, all messages are one turn."""
        msgs = [
            _make_msg("user", [_make_content("text", text="Hello")]),
            _make_msg("assistant", [_make_content("text", text="Hi")]),
        ]
        abstract = build_session_abstract(msgs)
        assert len(abstract) == 1
        assert abstract[0].user_msg == "Hello"
        assert abstract[0].assistant_reply == "Hi"

    def test_user_only_turn(self):
        """A turn with only user messages (agent hasn't replied yet)."""
        msgs = [
            _make_msg("user", [_make_content("text", text="Question")]),
            _make_msg("assistant", [], additional_properties={"_chrys_kind": "turn"}),
        ]
        abstract = build_session_abstract(msgs)
        assert len(abstract) == 1
        assert abstract[0].user_msg == "Question"
        assert abstract[0].assistant_reply == ""

    def test_assistant_only_turn(self):
        """A turn where user msg comes from system context."""
        msgs = [
            _make_msg("assistant", [_make_content("text", text="Auto reply")]),
            _make_msg("assistant", [], additional_properties={"_chrys_kind": "turn"}),
        ]
        abstract = build_session_abstract(msgs)
        assert len(abstract) == 1
        assert abstract[0].user_msg == ""
        assert abstract[0].assistant_reply == "Auto reply"

    def test_filters_non_dict_items(self):
        msgs: list = [
            _make_msg("user", [_make_content("text", text="Hi")]),
            "not a dict",
            42,
            _make_msg("assistant", [_make_content("text", text="Hey")]),
        ]
        abstract = build_session_abstract(msgs)
        assert len(abstract) == 1
        assert abstract[0].user_msg == "Hi"
        assert abstract[0].assistant_reply == "Hey"


# ==============================================================================
# build_session_turns
# ==============================================================================


class TestBuildSessionTurns:
    def test_two_turns(self, sample_messages: list[dict]):
        turns = build_session_turns(sample_messages)
        assert len(turns) == 2
        assert turns[0].turn_index == 1
        assert turns[1].turn_index == 2

    def test_turn_indices_are_sequential(self, sample_messages: list[dict]):
        turns = build_session_turns(sample_messages)
        for i, turn in enumerate(turns, 1):
            assert turn.turn_index == i

    def test_all_messages_are_raw_message(self, sample_messages: list[dict]):
        turns = build_session_turns(sample_messages)
        for turn in turns:
            for msg in turn.messages:
                assert isinstance(msg, RawMessage)

    def test_boundary_messages_excluded(self, sample_messages: list[dict]):
        """Messages with _chrys_kind='turn' should not appear in any turn."""
        turns = build_session_turns(sample_messages)
        for turn in turns:
            for msg in turn.messages:
                assert "_chrys_kind" not in msg.additional_properties

    def test_turn_contents_match_abstract(self, sample_messages: list[dict]):
        """Turns and abstract should describe the same conversation."""
        turns = build_session_turns(sample_messages)
        abstract = build_session_abstract(sample_messages)
        assert len(turns) == len(abstract)

    def test_empty_messages(self):
        assert build_session_turns([]) == []

    def test_none_input(self):
        assert build_session_turns(None) == []  # type: ignore[arg-type]

    def test_no_turn_boundaries(self):
        msgs = [
            _make_msg("user", [_make_content("text", text="Hello")]),
            _make_msg("assistant", [_make_content("text", text="Hi")]),
        ]
        turns = build_session_turns(msgs)
        assert len(turns) == 1
        assert len(turns[0].messages) == 2

    def test_preserves_tool_calls_and_results(self, sample_messages: list[dict]):
        turns = build_session_turns(sample_messages)
        turn1_types = [c.type for msg in turns[0].messages for c in msg.contents]
        assert "function_call" in turn1_types
        assert "function_result" in turn1_types

    def test_message_roles_are_preserved(self, sample_messages: list[dict]):
        turns = build_session_turns(sample_messages)
        roles_in_turn1 = [msg.role for msg in turns[0].messages]
        assert "system" in roles_in_turn1
        assert "user" in roles_in_turn1
        assert "assistant" in roles_in_turn1
        assert "tool" in roles_in_turn1


# ==============================================================================
# parse_group_members
# ==============================================================================


class TestParseGroupMembers:
    def test_parses_groups_correctly(self, group_members_json: dict):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8",
        ) as f:
            json.dump(group_members_json, f)
            tmp_path = f.name

        try:
            groups = parse_group_members(tmp_path)
            assert len(groups) == 2
            assert groups[0]["group_name"] == "后端一组"
            assert len(groups[0]["members"]) == 2
            assert groups[0]["members"][0]["user_id"] == "user_zhangsan"
            assert groups[1]["group_name"] == "前端一组"
            assert len(groups[1]["members"]) == 1
        finally:
            Path(tmp_path).unlink()


# ==============================================================================
# _build_user_to_groups
# ==============================================================================


class TestBuildUserToGroups:
    def test_builds_correct_mapping(self):
        groups_data = [
            {
                "group_id": "1001",
                "group_name": "后端一组",
                "members": [
                    {"user_id": "user_a", "department": "后端"},
                    {"user_id": "user_b", "department": "后端"},
                ],
            },
            {
                "group_id": "1002",
                "group_name": "前端一组",
                "members": [
                    {"user_id": "user_b", "department": "前端"},
                    {"user_id": "user_c", "department": "前端"},
                ],
            },
        ]
        mapping = _build_user_to_groups(groups_data)
        assert mapping == {
            "user_a": ["后端一组"],
            "user_b": ["后端一组", "前端一组"],
            "user_c": ["前端一组"],
        }

    def test_empty_groups(self):
        assert _build_user_to_groups([]) == {}


# ==============================================================================
# mongo_docs_to_user_data
# ==============================================================================


class TestMongoDocsToUserData:
    def test_converts_single_user_with_one_session(
        self, sample_mongo_doc: dict,
    ):
        sessions_by_user = {"user_zhangsan": [sample_mongo_doc]}
        result = mongo_docs_to_user_data(sessions_by_user)

        assert len(result) == 1
        user = result[0]
        assert isinstance(user, UserData)
        assert user.user_name == "user_zhangsan"
        assert user.total_add_lines == 25
        assert user.groups == []
        assert len(user.sessions) == 1

    def test_session_has_all_fields(self, sample_mongo_doc: dict):
        sessions_by_user = {"user_zhangsan": [sample_mongo_doc]}
        result = mongo_docs_to_user_data(sessions_by_user)
        session = result[0].sessions[0]

        assert session.session_uuid == "sess-001"
        assert len(session.session_abstract) == 2
        assert session.mcp_tools == ["list_files", "write_file"]
        assert session.skills == ["python-coding"]
        assert session.meta == {
            "user_id": "user_zhangsan",
            "title": "创建工具函数",
            "primary_cwd": "/home/user/project",
        }
        assert len(session.turns) == 2

    def test_turns_and_abstract_are_consistent(self, sample_mongo_doc: dict):
        sessions_by_user = {"user_zhangsan": [sample_mongo_doc]}
        result = mongo_docs_to_user_data(sessions_by_user)
        session = result[0].sessions[0]
        assert len(session.turns) == len(session.session_abstract)

    def test_user_groups_filled(self, sample_mongo_doc: dict):
        sessions_by_user = {"user_zhangsan": [sample_mongo_doc]}
        user_to_groups = {"user_zhangsan": ["后端一组", "核心组"]}
        result = mongo_docs_to_user_data(sessions_by_user, user_to_groups)

        assert result[0].groups == ["后端一组", "核心组"]

    def test_user_groups_none_uses_empty(self, sample_mongo_doc: dict):
        sessions_by_user = {"user_zhangsan": [sample_mongo_doc]}
        result = mongo_docs_to_user_data(sessions_by_user, None)
        assert result[0].groups == []

    def test_multiple_users(self, sample_mongo_doc: dict):
        doc2 = dict(sample_mongo_doc, uuid="sess-002")
        sessions_by_user = {
            "user_zhangsan": [sample_mongo_doc],
            "user_lisi": [doc2],
        }
        result = mongo_docs_to_user_data(sessions_by_user)
        assert len(result) == 2

    def test_user_with_no_sessions(self):
        sessions_by_user = {"user_empty": []}
        result = mongo_docs_to_user_data(sessions_by_user)
        assert len(result) == 1
        assert result[0].sessions == []
        assert result[0].total_add_lines == 0

    def test_mr_relation_missing(self, sample_mongo_doc: dict):
        doc = dict(sample_mongo_doc)
        doc.pop("mr_relation", None)
        sessions_by_user = {"user_zhangsan": [doc]}
        result = mongo_docs_to_user_data(sessions_by_user)
        assert result[0].total_add_lines == 0

    def test_mcp_tools_not_list(self, sample_mongo_doc: dict):
        doc = dict(sample_mongo_doc)
        doc["mcp_tools"] = "not a list"
        sessions_by_user = {"user_zhangsan": [doc]}
        result = mongo_docs_to_user_data(sessions_by_user)
        assert result[0].sessions[0].mcp_tools == []

    def test_skills_not_list(self, sample_mongo_doc: dict):
        doc = dict(sample_mongo_doc)
        doc["skills"] = None
        sessions_by_user = {"user_zhangsan": [doc]}
        result = mongo_docs_to_user_data(sessions_by_user)
        assert result[0].sessions[0].skills == []

    def test_meta_not_dict(self, sample_mongo_doc: dict):
        doc = dict(sample_mongo_doc)
        doc["meta"] = "not a dict"
        sessions_by_user = {"user_zhangsan": [doc]}
        result = mongo_docs_to_user_data(sessions_by_user)
        assert result[0].sessions[0].meta == {}

    def test_messages_empty(self, sample_mongo_doc: dict):
        doc = dict(sample_mongo_doc)
        doc["messages"] = []
        sessions_by_user = {"user_zhangsan": [doc]}
        result = mongo_docs_to_user_data(sessions_by_user)
        assert result[0].sessions[0].session_abstract == []
        assert result[0].sessions[0].turns == []


# ==============================================================================
# Model backward compatibility
# ==============================================================================


class TestModelBackwardCompatibility:
    """Ensure new optional fields don't break existing code."""

    def test_session_defaults(self):
        s = Session(session_uuid="u1", session_abstract=[])
        assert s.turns == []
        assert s.meta == {}
        assert s.mcp_tools == []
        assert s.skills == []

    def test_deviated_session_without_turn_index(self):
        ds = DeviatedSession(
            session_uuid="u1",
            user_name="test",
            deviation_reason="测试偏离",
        )
        assert ds.problematic_turn_index is None
        assert ds.category == ""

    def test_deviated_session_with_turn_index(self):
        ds = DeviatedSession(
            session_uuid="u2",
            user_name="test",
            deviation_reason="工具执行错误",
            problematic_turn_index=3,
            category="系统异常与执行中断",
        )
        assert ds.problematic_turn_index == 3
        assert ds.category == "系统异常与执行中断"

    def test_offset_analysis_result_without_turn_problems(self):
        r = OffsetAnalysisResult(
            deviated_sessions=[],
            summary={},
        )
        assert r.turn_problems == []

    def test_turn_problem_defaults(self):
        tp = TurnProblem(
            session_uuid="u1",
            user_name="test",
            turn_index=2,
        )
        assert tp.deviation_action == ""
        assert tp.turn_analysis == ""

    def test_message_content_defaults(self):
        mc = MessageContent()
        assert mc.type == ""
        assert mc.text == ""
        assert mc.tool_name == ""

    def test_raw_message_only_role(self):
        rm = RawMessage(role="user")
        assert rm.contents == []
        assert rm.additional_properties == {}
        assert rm.message_id == ""


# ==============================================================================
# Full round-trip test
# ==============================================================================


class TestRoundTrip:
    """Verify data flows end-to-end through the conversion pipeline."""

    def test_full_conversion_then_serialize(self, sample_mongo_doc: dict):
        sessions_by_user = {"user_zhangsan": [sample_mongo_doc]}
        users = mongo_docs_to_user_data(sessions_by_user)

        # Should serialize without error
        for user in users:
            data = user.model_dump()
            assert data["user_name"] == "user_zhangsan"

            for session in user.sessions:
                sd = session.model_dump()
                assert len(sd["turns"]) == len(sd["session_abstract"])
                # Every turn has valid raw messages
                for turn in sd["turns"]:
                    assert isinstance(turn["turn_index"], int)
                    assert isinstance(turn["messages"], list)
                    for msg in turn["messages"]:
                        assert "role" in msg

    def test_deviated_session_roundtrip(self):
        ds = DeviatedSession(
            session_uuid="s1",
            user_name="u1",
            deviation_reason="工具调用参数错误",
            problematic_turn_index=2,
            category="代码逻辑缺陷与生成错误",
        )
        data = ds.model_dump()
        assert data["problematic_turn_index"] == 2
        assert data["category"] == "代码逻辑缺陷与生成错误"

        tp = TurnProblem(
            session_uuid="s1",
            user_name="u1",
            turn_index=2,
            deviation_action="write_file使用了错误的文件路径",
            turn_analysis="第3条消息调用了错误的文件路径",
        )
        tp_data = tp.model_dump()
        assert tp_data["deviation_action"] == "write_file使用了错误的文件路径"

        result = OffsetAnalysisResult(
            deviated_sessions=[ds],
            summary={"total": 1},
            turn_problems=[tp],
        )
        result_data = result.model_dump()
        assert len(result_data["turn_problems"]) == 1
