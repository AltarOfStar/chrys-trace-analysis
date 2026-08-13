"""Tests for session.json trace reconstruction (compressed_msgs → messages)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# ——————————————————————————————————————————————————————————————————————————————
# Set up import path (tests directory is outside src/)
# ——————————————————————————————————————————————————————————————————————————————

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# ——————————————————————————————————————————————————————————————————————————————
# Imports from the package under test
# ——————————————————————————————————————————————————————————————————————————————

from chrys_trace_analysis.reconstruct import (  # noqa: E402
    block_turn_range,
    blocks_for_turn,
    extract_state,
    is_summary_message,
    load_session_file,
    parse_compressed_blocks,
    reconstruct_messages,
    split_turns,
    summary_context_id,
)


# ==============================================================================
# Helpers / fixtures
# ==============================================================================


def _msg(role: str, contents: list[dict] | None = None, **extra: object) -> dict:
    m: dict = {"role": role, "contents": contents or []}
    m.update(extra)
    return m


def _turn_marker(turn_num: int, turn_id: str) -> dict:
    return _msg(
        "assistant",
        additional_properties={"_chrys_kind": "turn", "_turn": turn_num, "_turn_id": turn_id},
    )


def _summary_msg(ctx_id: str, summary: str = "Summary text") -> dict:
    return _msg(
        "assistant",
        [{"type": "text", "text": f"[Compressed context: {ctx_id}]\nSummary: {summary}"}],
        additional_properties={"_chrys_kind": "summary", "_block_id": ctx_id},
    )


def _block(
    ctx_id: str,
    messages: list[dict],
    turn_range: tuple[int, int],
    summary: str = "Summary text",
    marker_id: str = "t2",
    created_at: str = "2026-01-01T00:00:00+00:00",
) -> dict:
    return {
        "compressed_context_id": ctx_id,
        "messages": messages,
        "summary_text": summary,
        "marker_id": marker_id,
        "turn_range": list(turn_range),
        "created_at": created_at,
    }


def _envelope(live: list[dict], blocks: list[dict], extra_state: dict | None = None) -> dict:
    state: dict = {"messages": live, "compressed_msgs": blocks, "turn_counter": 5}
    if extra_state:
        state.update(extra_state)
    return {"meta": {"session_id": "sess-001", "title": "测试"}, "state": state}


@pytest.fixture
def compressed_turn1_2() -> list[dict]:
    """第一/二轮被压缩进块的原始消息（含两条轮次边界标记）。"""
    return [
        _msg("system", [{"type": "text", "text": "You are a coding assistant."}]),
        _msg("user", [{"type": "text", "text": "写一个 add 函数。"}]),
        _msg(
            "assistant",
            [
                {"type": "text", "text": "好的。"},
                {"type": "function_call", "tool_name": "write_file", "call_id": "c1",
                 "arguments": '{"path": "utils.py", "content": "def add(a,b): return a+b"}'},
            ],
        ),
        _msg("tool", [{"type": "function_result", "call_id": "c1", "result": '{"success": true}'}]),
        _turn_marker(1, "t1"),
        _msg("user", [{"type": "text", "text": "改成乘法。"}]),
        _msg("assistant", [{"type": "text", "text": "已改为 multiply。"}]),
        _turn_marker(2, "t2"),
    ]


@pytest.fixture
def live_turn3_4() -> list[dict]:
    """未被压缩的 live 消息（第三/四轮）。"""
    return [
        _summary_msg("ctx_aa11bb22"),
        _msg("user", [{"type": "text", "text": "再加一个取模函数。"}]),
        _msg("assistant", [{"type": "text", "text": "完成。"}]),
        _turn_marker(3, "t3"),
        _msg("user", [{"type": "text", "text": "测试一下。"}]),
        _msg("assistant", [{"type": "text", "text": "测试通过。"}]),
        _turn_marker(4, "t4"),
    ]


# ==============================================================================
# load_session_file / extract_state
# ==============================================================================


class TestLoadSessionFile:
    def test_loads_valid_json(self, tmp_path: Path):
        fp = tmp_path / "session.json"
        fp.write_text(json.dumps({"meta": {}, "state": {}}), encoding="utf-8")
        data = load_session_file(fp)
        assert data == {"meta": {}, "state": {}}

    def test_tolerates_bom(self, tmp_path: Path):
        fp = tmp_path / "session.json"
        fp.write_text("\ufeff" + json.dumps({"a": 1}), encoding="utf-8")
        assert load_session_file(fp) == {"a": 1}

    def test_invalid_json_raises(self, tmp_path: Path):
        fp = tmp_path / "session.json"
        fp.write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError, match="invalid JSON"):
            load_session_file(fp)

    def test_non_object_root_raises(self, tmp_path: Path):
        fp = tmp_path / "session.json"
        fp.write_text("[1, 2]", encoding="utf-8")
        with pytest.raises(ValueError, match="must be a JSON object"):
            load_session_file(fp)

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(OSError):
            load_session_file(tmp_path / "nope.json")


class TestExtractState:
    def test_envelope_state(self):
        state, source = extract_state({"meta": {}, "state": {"messages": []}})
        assert source == "envelope.state"
        assert state == {"messages": []}

    def test_flat_top_level_fallback(self):
        state, source = extract_state({"messages": [1], "compressed_msgs": []})
        assert source == "top-level"
        assert state == {"messages": [1], "compressed_msgs": []}

    def test_non_dict_state_falls_back(self):
        state, source = extract_state({"state": "not a dict", "messages": []})
        assert source == "top-level"


# ==============================================================================
# parse_compressed_blocks
# ==============================================================================


class TestParseCompressedBlocks:
    def test_parses_valid_blocks(self):
        blocks = parse_compressed_blocks({"compressed_msgs": [_block("ctx_1", [_msg("user")], (1, 1))]})
        assert len(blocks) == 1
        assert blocks[0]["compressed_context_id"] == "ctx_1"

    def test_missing_key_is_empty(self):
        assert parse_compressed_blocks({}) == []

    def test_non_list_is_empty(self):
        assert parse_compressed_blocks({"compressed_msgs": "nope"}) == []

    def test_skips_non_dict_entries(self):
        from chrys_trace_analysis.reconstruct import ReconstructionDiagnostics

        diag = ReconstructionDiagnostics()
        blocks = parse_compressed_blocks(
            {"compressed_msgs": [_block("ctx_1", [_msg("user")], (1, 1)), "junk", 42]},
            diag,
        )
        assert len(blocks) == 1
        assert diag.skipped_malformed_block_count == 2

    def test_skips_block_without_messages_list(self):
        from chrys_trace_analysis.reconstruct import ReconstructionDiagnostics

        diag = ReconstructionDiagnostics()
        blocks = parse_compressed_blocks({"compressed_msgs": [{"compressed_context_id": "ctx_2"}]}, diag)
        assert blocks == []
        assert diag.skipped_malformed_block_count == 1
        assert any("no messages list" in note for note in diag.notes)

    def test_filters_non_dict_messages_inside_block(self):
        blocks = parse_compressed_blocks(
            {"compressed_msgs": [_block("ctx_1", [_msg("user"), "junk"], (1, 1))]}
        )
        assert len(blocks[0]["messages"]) == 1

    def test_redundant_block_fields_ignored(self):
        block = _block("ctx_1", [_msg("user")], (1, 1))
        block["extra_field"] = {"anything": [1, 2, 3]}
        blocks = parse_compressed_blocks({"compressed_msgs": [block]})
        assert blocks[0]["extra_field"] == {"anything": [1, 2, 3]}


# ==============================================================================
# summary detection
# ==============================================================================


class TestSummaryDetection:
    def test_summary_context_id_from_props(self):
        msg = _summary_msg("ctx_aa11bb22")
        assert summary_context_id(msg) == "ctx_aa11bb22"
        assert is_summary_message(msg)

    def test_summary_context_id_from_text_fallback(self):
        msg = _msg("assistant", [{"type": "text", "text": "[Compressed context: ctx_deadbeef]\nSummary: x"}])
        assert summary_context_id(msg) == "ctx_deadbeef"
        assert is_summary_message(msg)

    def test_summary_kind_without_id(self):
        msg = _msg("assistant", additional_properties={"_chrys_kind": "summary"})
        assert summary_context_id(msg) == ""
        assert is_summary_message(msg)

    def test_regular_messages_not_summary(self):
        assert not is_summary_message(_msg("user", [{"type": "text", "text": "hi"}]))
        assert not is_summary_message(_turn_marker(1, "t1"))


# ==============================================================================
# block_turn_range / blocks_for_turn
# ==============================================================================


class TestBlockQueries:
    def test_valid_range(self):
        assert block_turn_range({"turn_range": [1, 3]}) == (1, 3)

    def test_malformed_ranges_default_to_sentinel(self):
        assert block_turn_range({}) == (0, 0)
        assert block_turn_range({"turn_range": "1-3"}) == (0, 0)
        assert block_turn_range({"turn_range": [1, "3"]}) == (0, 0)
        assert block_turn_range({"turn_range": [1]}) == (0, 0)

    def test_blocks_for_turn(self):
        blocks = [
            _block("ctx_old", [], (1, 3)),
            _block("ctx_new", [], (4, 5)),
            _block("ctx_unknown", [], (0, 0)),
        ]
        assert [b["compressed_context_id"] for b in blocks_for_turn(blocks, 2)] == ["ctx_old"]
        assert [b["compressed_context_id"] for b in blocks_for_turn(blocks, 5)] == ["ctx_new"]
        assert blocks_for_turn(blocks, 9) == []
        assert blocks_for_turn(blocks, "2") == []


# ==============================================================================
# split_turns
# ==============================================================================


class TestSplitTurns:
    def test_splits_on_markers_and_drops_them(self):
        msgs = [
            _msg("user", [{"type": "text", "text": "q1"}]),
            _turn_marker(1, "t1"),
            _msg("user", [{"type": "text", "text": "q2"}]),
            _turn_marker(2, "t2"),
        ]
        turns = split_turns(msgs)
        assert len(turns) == 2
        assert turns[0][0]["role"] == "user"
        assert turns[1][0]["role"] == "user"

    def test_no_markers_single_turn(self):
        turns = split_turns([_msg("user"), _msg("assistant")])
        assert len(turns) == 1
        assert len(turns[0]) == 2

    def test_trailing_turn_without_marker(self):
        turns = split_turns([_turn_marker(1, "t1"), _msg("user")])
        assert len(turns) == 1
        assert turns[0][0]["role"] == "user"

    def test_empty(self):
        assert split_turns([]) == []


# ==============================================================================
# reconstruct_messages
# ==============================================================================


class TestReconstructMessages:
    def test_basic_splice(self, compressed_turn1_2, live_turn3_4):
        block = _block("ctx_aa11bb22", compressed_turn1_2, (1, 2), marker_id="t2")
        env = _envelope(live_turn3_4, [block])
        result = reconstruct_messages(env)

        diag = result.diagnostics
        assert diag.state_source == "envelope.state"
        assert diag.live_message_count == len(live_turn3_4)
        assert diag.block_count == 1
        assert diag.spliced_block_count == 1
        assert diag.orphan_block_count == 0
        assert diag.unresolved_summary_count == 0

        # 块消息原位替换占位符，顺序完整
        restored = result.restored_messages
        assert len(restored) == len(compressed_turn1_2) + (len(live_turn3_4) - 1)
        assert restored[0]["role"] == "system"
        assert restored[0]["contents"][0]["text"] == "You are a coding assistant."
        # 边界标记也随块还原
        assert restored[4]["additional_properties"]["_chrys_kind"] == "turn"
        assert restored[4]["additional_properties"]["_turn"] == 1
        # 块之后紧跟 live 剩余消息
        assert restored[len(compressed_turn1_2)]["contents"][0]["text"] == "再加一个取模函数。"
        assert restored[-1]["additional_properties"]["_turn"] == 4

    def test_input_not_mutated(self, compressed_turn1_2, live_turn3_4):
        env = _envelope(live_turn3_4, [_block("ctx_aa11bb22", compressed_turn1_2, (1, 2))])
        before = json.dumps(env, sort_keys=True, ensure_ascii=False)
        reconstruct_messages(env)
        after = json.dumps(env, sort_keys=True, ensure_ascii=False)
        assert before == after

    def test_orphan_block_positioned_by_turn(self, compressed_turn1_2):
        live = [
            _msg("user", [{"type": "text", "text": "第四轮问题。"}]),
            _msg("assistant", [{"type": "text", "text": "第四轮回答。"}]),
            _turn_marker(4, "t4"),
        ]
        env = _envelope(live, [_block("ctx_orphan", compressed_turn1_2, (1, 2))])
        result = reconstruct_messages(env)

        assert result.diagnostics.orphan_block_count == 1
        assert result.diagnostics.spliced_block_count == 0
        restored = result.restored_messages
        # live 无任何 <= 2 的轮次信息：块整体前置
        assert restored[0]["role"] == "system"
        marker_t4_idx = [i for i, m in enumerate(restored)
                         if m.get("additional_properties", {}).get("_turn") == 4][0]
        # 块整体前置后，live 的 user/assistant/t4 标记依次跟在块后面
        assert marker_t4_idx == len(compressed_turn1_2) + 2
        assert "第四轮问题。" in restored[len(compressed_turn1_2)]["contents"][0]["text"]

    def test_orphan_block_anchored_after_duplicate_marker(self):
        block = _block("ctx_orphan", [_msg("user", [{"type": "text", "text": "被折叠的轮次"}])], (1, 2))
        live = [
            _turn_marker(2, "t2"),
            _msg("user", [{"type": "text", "text": "重复标记之后的消息。"}]),
        ]
        env = _envelope(live, [block])
        result = reconstruct_messages(env)

        def first_text(msg: dict) -> str:
            for content in msg.get("contents", []):
                if isinstance(content, dict) and content.get("text"):
                    return content["text"]
            return ""

        texts = [first_text(m) for m in result.restored_messages]
        # 块插在最后一条 _turn <= 2 的 live 消息（t2 标记）之后
        assert texts == ["", "被折叠的轮次", "重复标记之后的消息。"]

    def test_orphan_block_prepended_without_turn_info(self, compressed_turn1_2):
        live = [
            _msg("user", [{"type": "text", "text": "无轮次信息的问题。"}]),
            _msg("assistant", [{"type": "text", "text": "无轮次信息的回答。"}]),
        ]
        env = _envelope(live, [_block("ctx_orphan", compressed_turn1_2, (1, 2))])
        result = reconstruct_messages(env)
        restored = result.restored_messages
        assert restored[0]["role"] == "system"
        assert restored[-1]["role"] == "assistant"
        assert restored[-1]["contents"][0]["text"] == "无轮次信息的回答。"
        assert result.diagnostics.orphan_block_count == 1

    def test_orphan_blocks_ordered_by_turn_range(self):
        block_old = _block("ctx_old", [_msg("user", [{"type": "text", "text": "旧轮次"}])], (1, 1),
                           created_at="2026-01-01T00:00:00+00:00")
        block_new = _block("ctx_new", [_msg("user", [{"type": "text", "text": "新轮次"}])], (2, 2),
                           created_at="2026-01-02T00:00:00+00:00")
        env = _envelope([], [block_new, block_old])
        result = reconstruct_messages(env)
        texts = [m["contents"][0]["text"] for m in result.restored_messages]
        assert texts == ["旧轮次", "新轮次"]
        assert result.diagnostics.orphan_block_count == 2

    def test_duplicate_block_ids_first_wins(self):
        live = [_summary_msg("ctx_dup"), _msg("user", [{"type": "text", "text": "live tail"}])]
        env = _envelope(
            live,
            [
                _block("ctx_dup", [_msg("user", [{"type": "text", "text": "第一个块"}])], (1, 1)),
                _block("ctx_dup", [_msg("user", [{"type": "text", "text": "重复块"}])], (2, 2)),
            ],
        )
        result = reconstruct_messages(env)
        assert result.diagnostics.duplicate_block_id_count == 1
        texts = [m["contents"][0]["text"] for m in result.restored_messages]
        assert texts == ["第一个块", "live tail"]

    def test_duplicate_summary_placeholder_kept(self, compressed_turn1_2):
        live = [_summary_msg("ctx_dup"), _summary_msg("ctx_dup"), _msg("user")]
        env = _envelope(live, [_block("ctx_dup", compressed_turn1_2, (1, 2))])
        result = reconstruct_messages(env)
        assert result.diagnostics.duplicate_summary_placeholder_count == 1
        assert result.diagnostics.spliced_block_count == 1
        roles = [m["role"] for m in result.restored_messages]
        assert roles.count("system") == 1
        # 块内 2 条 user + live 保留 1 条
        assert roles.count("user") == 3

    def test_unresolved_summary_kept_in_place(self):
        live = [_summary_msg("ctx_ghost"), _msg("user", [{"type": "text", "text": "之后的消息"}])]
        env = _envelope(live, [])
        result = reconstruct_messages(env)
        assert result.diagnostics.unresolved_summary_count == 1
        assert result.restored_messages[0]["additional_properties"]["_block_id"] == "ctx_ghost"
        assert result.restored_messages[1]["contents"][0]["text"] == "之后的消息"

    def test_no_compressed_msgs_returns_live(self):
        live = [_msg("user", [{"type": "text", "text": "hi"}]), _turn_marker(1, "t1")]
        env = _envelope(live, [])
        result = reconstruct_messages(env)
        assert result.restored_messages == live
        assert result.diagnostics.block_count == 0

    def test_missing_compressed_msgs_key_returns_live(self):
        """compressed_msgs 键完全不存在时，还原结果应原样等于 live。"""
        live = [_msg("user", [{"type": "text", "text": "hi"}]),
                _msg("assistant", [{"type": "text", "text": "bye"}])]
        env = {"meta": {"session_id": "s1"}, "state": {"messages": live}}
        result = reconstruct_messages(env)
        assert result.restored_messages == live
        assert result.diagnostics.block_count == 0
        assert result.diagnostics.live_message_count == 2

    def test_missing_state_key_returns_empty(self):
        """整个 state 缺失时顶层退化，无消息可还原，优雅返回空列表。"""
        result = reconstruct_messages({"meta": {"session_id": "s1"}})
        assert result.restored_messages == []
        assert result.diagnostics.state_source == "top-level"

    def test_spliced_output_has_no_summary_placeholder(self, compressed_turn1_2):
        """还原后不应残留 _chrys_kind=summary 占位消息，轮次标记序列完整。"""
        live = [
            _summary_msg("ctx_aa11bb22"),
            _msg("user", [{"type": "text", "text": "turn3 问题。"}]),
            _turn_marker(3, "t3"),
        ]
        env = _envelope(live, [_block("ctx_aa11bb22", compressed_turn1_2, (1, 2))])
        result = reconstruct_messages(env)
        kinds = [m.get("additional_properties", {}).get("_chrys_kind") for m in result.restored_messages]
        assert "summary" not in kinds
        turns = [k for k in kinds if k == "turn"]
        assert turns == ["turn", "turn", "turn"]

    def test_flat_layout_without_state(self, compressed_turn1_2, live_turn3_4):
        env = {
            "meta": {"session_id": "s1"},
            "messages": live_turn3_4,
            "compressed_msgs": [_block("ctx_aa11bb22", compressed_turn1_2, (1, 2))],
            "redundant_top_key": {"x": 1},
        }
        result = reconstruct_messages(env)
        assert result.diagnostics.state_source == "top-level"
        assert result.diagnostics.spliced_block_count == 1
        assert result.restored_messages[0]["role"] == "system"

    def test_redundant_fields_ignored_everywhere(self, compressed_turn1_2, live_turn3_4):
        block = _block("ctx_aa11bb22", compressed_turn1_2, (1, 2))
        block["redundant_block_field"] = [1, 2, 3]
        live = live_turn3_4
        live[1]["redundant_msg_field"] = {"deep": "value"}
        env = _envelope(live, [block], extra_state={"redundant_state_field": 42})
        env["redundant_envelope_field"] = "yes"
        result = reconstruct_messages(env)
        assert result.diagnostics.spliced_block_count == 1
        assert len(result.restored_messages) == len(compressed_turn1_2) + (len(live) - 1)

    def test_malformed_entries_skipped(self, live_turn3_4):
        env = _envelope(
            [live_turn3_4[0], "junk", live_turn3_4[1], 42],
            [_block("ctx_aa11bb22", [_msg("user")], (1, 2)), "bad_block"],
        )
        result = reconstruct_messages(env)
        assert result.diagnostics.skipped_malformed_message_count == 2
        assert result.diagnostics.skipped_malformed_block_count == 1
        # 有效块照常拼接
        assert result.restored_messages[0]["role"] == "user"

    def test_messages_not_a_list(self):
        env = _envelope("not a list", [])
        result = reconstruct_messages(env)
        assert result.restored_messages == []
        assert any("not a list" in note for note in result.diagnostics.notes)

    def test_excluded_and_spill_notice_diagnostics(self, compressed_turn1_2):
        spill_msg = _msg(
            "tool",
            [{"type": "function_result", "call_id": "c9",
              "result": "[Full output saved to: C:/sessions/x/tool_results/shell_ab12cd34.txt\n"
                        "1200 lines, ~8000 tokens. Use read_file or shell tools to inspect it.]"}],
            additional_properties={"_excluded": True},
        )
        block = _block("ctx_aa11bb22", compressed_turn1_2 + [spill_msg], (1, 2))
        env = _envelope([_summary_msg("ctx_aa11bb22")], [block])
        result = reconstruct_messages(env)
        diag = result.diagnostics
        assert diag.restored_excluded_count == 1
        assert diag.restored_spill_notice_count == 1
        assert diag.restored_message_count == len(compressed_turn1_2) + 1

    def test_result_to_dict_and_blocks_index(self, compressed_turn1_2):
        env = _envelope([_summary_msg("ctx_aa11bb22")], [_block("ctx_aa11bb22", compressed_turn1_2, (1, 2))])
        result = reconstruct_messages(env)
        payload = result.to_dict()
        assert set(payload) == {"restored_messages", "blocks", "diagnostics"}
        assert payload["blocks"][0]["compressed_context_id"] == "ctx_aa11bb22"
        assert payload["blocks"][0]["turn_range"] == [1, 2]
        assert payload["blocks"][0]["message_count"] == len(compressed_turn1_2)
        assert payload["diagnostics"]["spliced_block_count"] == 1

    def test_from_file_path(self, tmp_path: Path, compressed_turn1_2):
        env = _envelope([_summary_msg("ctx_aa11bb22")], [_block("ctx_aa11bb22", compressed_turn1_2, (1, 2))])
        fp = tmp_path / "session.json"
        fp.write_text(json.dumps(env, ensure_ascii=False), encoding="utf-8")
        result = reconstruct_messages(fp)
        assert result.diagnostics.spliced_block_count == 1
        assert result.restored_messages[0]["role"] == "system"

    def test_nested_summary_inside_block_noted(self):
        inner = _summary_msg("ctx_inner")
        block = _block("ctx_outer", [inner, _msg("user")], (1, 2))
        env = _envelope([_summary_msg("ctx_outer")], [block])
        result = reconstruct_messages(env)
        assert any("not recursively expanded" in note for note in result.diagnostics.notes)

    def test_blocks_for_turn_after_reconstruction(self, compressed_turn1_2, live_turn3_4):
        env = _envelope(live_turn3_4, [_block("ctx_aa11bb22", compressed_turn1_2, (1, 2))])
        result = reconstruct_messages(env)
        matched = blocks_for_turn(result.blocks, 2)
        assert [b["compressed_context_id"] for b in matched] == ["ctx_aa11bb22"]
        assert blocks_for_turn(result.blocks, 5) == []

    def test_restored_split_turns_round_trip(self, compressed_turn1_2, live_turn3_4):
        env = _envelope(live_turn3_4, [_block("ctx_aa11bb22", compressed_turn1_2, (1, 2))])
        result = reconstruct_messages(env)
        turns = split_turns(result.restored_messages)
        # 4 轮完整还原，边界标记消息被丢弃
        assert len(turns) == 4
        assert turns[0][0]["role"] == "system"
        assert turns[3][0]["contents"][0]["text"] == "测试一下。"
