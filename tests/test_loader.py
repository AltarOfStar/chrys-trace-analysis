"""Tests for session loading: message parsing, envelope adaptation, and {uuid}.json loading."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ——————————————————————————————————————————————————————————————————————————————
# Set up import path (tests directory is outside src/)
# ——————————————————————————————————————————————————————————————————————————————

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# ——————————————————————————————————————————————————————————————————————————————
# Imports from the package under test
# ——————————————————————————————————————————————————————————————————————————————

from chrys_trace_analysis.loader import (  # noqa: E402
    _collect_names,
    _extract_texts,
    _parse_message_content,
    _parse_raw_message,
    _session_uuid,
    build_session_abstract,
    build_session_turns,
    load_sessions,
    to_reconstruct_envelope,
)
from chrys_trace_analysis.models import (  # noqa: E402
    MessageContent,
    RawMessage,
)


# ==============================================================================
# Helpers / fixtures
# ==============================================================================


def _make_content(content_type: str, **kwargs: str) -> dict:
    return {"type": content_type, **kwargs}


def _make_msg(role: str, contents: list[dict], **kwargs: str | dict) -> dict:
    msg: dict = {"role": role, "contents": contents}
    msg.update(kwargs)
    return msg


def _turn_marker(turn_num: int) -> dict:
    return _make_msg(
        "assistant", [],
        additional_properties={"_chrys_kind": "turn", "_turn": turn_num},
    )


def _summary_msg(ctx_id: str) -> dict:
    return _make_msg(
        "assistant",
        [_make_content("text", text=f"[Compressed context: {ctx_id}]\nSummary: ...")],
        additional_properties={"_chrys_kind": "summary", "_block_id": ctx_id},
    )


def _block(ctx_id: str, messages: list[dict], turn_range: tuple[int, int]) -> dict:
    return {
        "compressed_context_id": ctx_id,
        "messages": messages,
        "summary_text": "Summary",
        "marker_id": f"t{turn_range[1]}",
        "turn_range": list(turn_range),
        "created_at": "2026-01-01T00:00:00+00:00",
    }


def _envelope(
    live: list[dict],
    blocks: list[dict],
    session_id: str = "sess-001",
    user_id: str = "user_a",
    redundant_fields: dict | None = None,
) -> dict:
    """chrys 本地 session.json envelope 形态（可带冗余根字段）。"""
    doc = {
        "meta": {"session_id": session_id, "user_id": user_id, "title": "测试会话"},
        "state": {"messages": live, "compressed_msgs": blocks, "turn_counter": 3},
    }
    if redundant_fields:
        doc.update(redundant_fields)
    return doc


def _sample_envelope() -> dict:
    """一个含压缩块的 envelope：前两轮在压缩块内，第三轮为 live 消息。"""
    compressed_turns = [
        _make_msg("user", [_make_content("text", text="写一个 add 函数。")]),
        _make_msg("assistant", [_make_content("text", text="好的，已创建。")]),
        _turn_marker(1),
        _make_msg("user", [_make_content("text", text="改成乘法函数。")]),
        _make_msg("assistant", [_make_content("text", text="已修改为 multiply。")]),
        _turn_marker(2),
    ]
    live_turn = [
        _summary_msg("ctx_aa11bb22"),
        _make_msg("user", [_make_content("text", text="再加一个取模函数。")]),
        _make_msg("assistant", [_make_content("text", text="完成。")]),
        _turn_marker(3),
    ]
    return _envelope(live=live_turn, blocks=[_block("ctx_aa11bb22", compressed_turns, (1, 2))])


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

    def test_additional_properties(self):
        rm = _parse_raw_message({
            "role": "assistant",
            "contents": [],
            "additional_properties": {"_chrys_kind": "turn", "key": "val"},
        })
        assert rm.additional_properties == {"_chrys_kind": "turn", "key": "val"}

    def test_empty_contents(self):
        rm = _parse_raw_message({"role": "system"})
        assert rm.contents == []
        assert rm.message_id == ""


# ==============================================================================
# build_session_abstract
# ==============================================================================


class TestBuildSessionAbstract:
    def test_abstract_from_sample_envelope(self):
        envelope = _sample_envelope()
        # 展开压缩块前无法得到完整轮次；这里直接验证重建后的消息
        from chrys_trace_analysis.reconstruct import reconstruct_messages

        restored = reconstruct_messages(envelope).restored_messages
        abstract = build_session_abstract(restored)
        assert len(abstract) == 3
        assert abstract[0].user_msg == "写一个 add 函数。"
        assert abstract[1].user_msg == "改成乘法函数。"
        assert abstract[2].user_msg == "再加一个取模函数。"

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
    def test_turn_indices_are_sequential(self):
        envelope = _sample_envelope()
        from chrys_trace_analysis.reconstruct import reconstruct_messages

        restored = reconstruct_messages(envelope).restored_messages
        turns = build_session_turns(restored)
        assert len(turns) == 3
        for i, turn in enumerate(turns, 1):
            assert turn.turn_index == i
        roles_in_turn1 = [msg.role for msg in turns[0].messages]
        assert "user" in roles_in_turn1
        assert "assistant" in roles_in_turn1

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


# ==============================================================================
# to_reconstruct_envelope
# ==============================================================================


class TestToReconstructEnvelope:
    def test_envelope_passthrough(self):
        doc = {"meta": {}, "state": {"messages": [{"role": "user"}], "compressed_msgs": []}}
        out = to_reconstruct_envelope(doc)
        assert out["state"]["messages"] == [{"role": "user"}]

    def test_flat_form_merges_both(self):
        doc = {
            "uuid": "s1",
            "messages": [{"role": "user"}],
            "compressed_msgs": [{"compressed_context_id": "ctx_1"}],
        }
        out = to_reconstruct_envelope(doc)
        assert out["state"]["messages"] == [{"role": "user"}]
        assert out["state"]["compressed_msgs"][0]["compressed_context_id"] == "ctx_1"

    def test_no_messages_keeps_state(self):
        doc = {"uuid": "s1", "state": {"turn_counter": 1}}
        out = to_reconstruct_envelope(doc)
        assert out["state"]["turn_counter"] == 1


# ==============================================================================
# _session_uuid
# ==============================================================================


class TestSessionUuid:
    def test_root_uuid_preferred(self):
        assert _session_uuid({"uuid": "u1", "meta": {"session_id": "u2"}}, "file") == "u1"

    def test_meta_session_id(self):
        assert _session_uuid({"meta": {"session_id": "u2"}}, "file") == "u2"

    def test_filename_fallback(self):
        assert _session_uuid({"meta": {}}, "file-stem") == "file-stem"

    def test_empty_values_fall_through(self):
        assert _session_uuid({"uuid": "", "meta": {"session_id": ""}}, "stem") == "stem"


# ==============================================================================
# load_sessions
# ==============================================================================


class TestLoadSessions:
    def test_expands_compressed_msgs(self, tmp_path: Path):
        envelope = _sample_envelope()
        (tmp_path / "sess-001.json").write_text(json.dumps(envelope), encoding="utf-8")

        traces, diagnostics = load_sessions(tmp_path)
        assert len(traces) == 1
        user_name, session = traces[0]
        assert user_name == "user_a"
        assert session.session_uuid == "sess-001"
        # 3 轮：2 轮来自压缩块，1 轮来自 live 消息
        assert len(session.session_abstract) == 3
        assert session.session_abstract[0].user_msg == "写一个 add 函数。"
        assert session.session_abstract[2].user_msg == "再加一个取模函数。"
        assert len(session.turns) == 3
        assert session.meta["title"] == "测试会话"
        assert diagnostics["spliced_block_count"] == 1
        assert diagnostics["restored_message_count"] > 0
        assert diagnostics["reconstruction_failures"] == 0

    def test_tolerates_redundant_root_fields(self, tmp_path: Path):
        envelope = _sample_envelope()
        envelope["redundant_field"] = {"any": "thing"}
        envelope["mcp_tools"] = [{"tool_name": "write_file"}]
        (tmp_path / "s.json").write_text(json.dumps(envelope), encoding="utf-8")

        traces, _ = load_sessions(tmp_path)
        assert len(traces) == 1
        assert traces[0][1].mcp_tools == ["write_file"]

    def test_skips_malformed_files(self, tmp_path: Path):
        (tmp_path / "ok.json").write_text(
            json.dumps(_sample_envelope()), encoding="utf-8",
        )
        (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
        (tmp_path / "list.json").write_text("[1, 2]", encoding="utf-8")

        traces, diagnostics = load_sessions(tmp_path)
        assert len(traces) == 1
        assert diagnostics["total_documents"] == 3

    def test_missing_directory_raises(self, tmp_path: Path):
        import pytest

        with pytest.raises(FileNotFoundError, match="Session directory not found"):
            load_sessions(tmp_path / "nope")

    def test_uuid_from_filename_when_no_session_id(self, tmp_path: Path):
        envelope = _sample_envelope()
        envelope["meta"] = {"user_id": "user_a"}
        (tmp_path / "uuid-fallback.json").write_text(json.dumps(envelope), encoding="utf-8")

        traces, _ = load_sessions(tmp_path)
        assert traces[0][1].session_uuid == "uuid-fallback"

    def test_tolerates_malformed_documents(self, tmp_path: Path):
        (tmp_path / "bad-state.json").write_text(
            json.dumps({"meta": {}, "state": {"messages": "oops"}}), encoding="utf-8",
        )
        traces, diagnostics = load_sessions(tmp_path)
        assert len(traces) == 1
        assert traces[0][1].session_abstract == []
        assert diagnostics["reconstruction_failures"] == 0


# ==============================================================================
# _collect_names
# ==============================================================================


class TestCollectNames:
    def test_extracts_names(self):
        items = [
            {"tool_name": "a", "args": {}},
            {"tool_name": "b"},
            "junk",
            {"tool_name": ""},
        ]
        assert _collect_names(items, "tool_name") == ["a", "b"]

    def test_non_list(self):
        assert _collect_names("junk", "tool_name") == []
