"""轨迹初步还原：从 session.json（可能被修改、含冗余字段）中通过 compressed_msgs 还原完整 messages。

本模块为独立业务逻辑，不依赖 chrys 源码包与分析流水线，输入可以是：

- chrys 会话 envelope（``{"meta": {...}, "state": {...}}``）；
- 扁平布局（``messages`` / ``compressed_msgs`` 直接位于顶层，无 ``state``）；
- MongoDB 文档形态（顶层 ``messages``，无 ``compressed_msgs``）。

还原规则（对应 chrys ``_compress_state`` 的持久化结构）：

- ``state.compressed_msgs`` 每个块包含被折叠轮次的完整消息深拷贝，以及
  ``compressed_context_id`` / ``summary_text`` / ``marker_id`` / ``turn_range`` /
  ``created_at``；
- live ``state.messages`` 中的压缩摘要占位消息形如
  ``[Compressed context: ctx_xxxxxxxx]\\nSummary: ...``，携带
  ``additional_properties._chrys_kind="summary"`` 与
  ``additional_properties._block_id="ctx_xxxxxxxx"``；
- 还原 = 把每个占位消息原位替换为对应块的 ``messages``；找不到占位符的孤立块
  按 ``turn_range`` 锚定回时间线（退化为整体前置/末尾追加）。

容错：忽略一切未知/冗余字段；跳过并计数畸形条目；块或占位符无法匹配时保留
原样并在诊断中记录，绝不因个别异常中止整体还原。

用法::

    python -m chrys_trace_analysis.reconstruct <session.json> [--out out.json]
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import re
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_BLOCK_ID_KEY = "_block_id"
_KIND_KEY = "_chrys_kind"
_TURN_KEY = "_turn"
_EXCLUDED_KEY = "_excluded"

_SUMMARY_KIND = "summary"
_TURN_MARKER_KIND = "turn"

# chrys 生成的压缩上下文 id 恒为 "ctx_" + 8 位小写十六进制；宽容匹配任意长度。
_SUMMARY_TEXT_RE = re.compile(r"\[Compressed context:\s*(ctx_[0-9a-f]+)\]", re.IGNORECASE)

# 工具结果采集期截断时的 notice 前缀（全文仅存在于会话目录 tool_results/ 的 spill 文件）。
_SPILL_NOTICE_MARKER = "[Full output saved to:"


# ---------------------------------------------------------------------------
# 结果模型
# ---------------------------------------------------------------------------


@dataclass
class ReconstructionDiagnostics:
    """还原过程的统计与说明信息（容错不抛错，一切异常都落到这里）。"""

    state_source: str = ""
    live_message_count: int = 0
    block_count: int = 0
    restored_message_count: int = 0
    spliced_block_count: int = 0
    orphan_block_count: int = 0
    duplicate_block_id_count: int = 0
    duplicate_summary_placeholder_count: int = 0
    unresolved_summary_count: int = 0
    skipped_malformed_message_count: int = 0
    skipped_malformed_block_count: int = 0
    restored_excluded_count: int = 0
    restored_spill_notice_count: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReconstructionResult:
    """还原结果：完整消息列表 + 块索引 + 诊断信息。"""

    restored_messages: list[dict[str, Any]]
    blocks: list[dict[str, Any]]
    diagnostics: ReconstructionDiagnostics

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# 读取与定位
# ---------------------------------------------------------------------------


def load_session_file(path: str | Path) -> dict[str, Any]:
    """读取 session.json（容忍 BOM），校验根节点为 JSON 对象。"""
    fp = Path(path)
    try:
        text = fp.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise OSError(f"cannot read session file {fp}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in session file {fp}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"session file {fp} root must be a JSON object, got {type(data).__name__}")
    return data


def extract_state(data: Mapping[str, Any]) -> tuple[Mapping[str, Any], str]:
    """定位 state 数据源。

    优先 ``envelope.state``；缺失或非对象时退化为顶层扁平布局（此时整个
    文档被当作 state 使用）。返回 ``(state, source_name)``。
    """
    raw_state = data.get("state")
    if isinstance(raw_state, Mapping):
        return raw_state, "envelope.state"
    return data, "top-level"


def parse_live_messages(
    state: Mapping[str, Any],
    diagnostics: ReconstructionDiagnostics | None = None,
) -> list[dict[str, Any]]:
    """提取 live 消息列表，跳过非对象条目并计入诊断。"""
    raw = state.get("messages")
    messages: list[dict[str, Any]] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, Mapping):
                messages.append(dict(item))
            elif diagnostics is not None:
                diagnostics.skipped_malformed_message_count += 1
    elif raw is not None and diagnostics is not None:
        diagnostics.notes.append(
            f"state['messages'] is not a list ({type(raw).__name__}); treated as empty"
        )
    return messages


def parse_compressed_blocks(
    state: Mapping[str, Any],
    diagnostics: ReconstructionDiagnostics | None = None,
) -> list[dict[str, Any]]:
    """提取压缩块列表（容忍冗余字段）。

    跳过：非对象条目、缺少 ``messages`` 列表的块。返回的块内 ``messages``
    已过滤为纯 dict 列表。
    """
    raw = state.get("compressed_msgs")
    blocks: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        return blocks
    for item in raw:
        if not isinstance(item, Mapping):
            if diagnostics is not None:
                diagnostics.skipped_malformed_block_count += 1
            continue
        block = dict(item)
        if not isinstance(block.get("messages"), list):
            if diagnostics is not None:
                diagnostics.skipped_malformed_block_count += 1
                diagnostics.notes.append(
                    f"compressed block {block.get('compressed_context_id', '<no-id>')!r} "
                    "has no messages list; skipped"
                )
            continue
        block["messages"] = [dict(m) for m in block["messages"] if isinstance(m, Mapping)]
        blocks.append(block)
    return blocks


# ---------------------------------------------------------------------------
# 消息 / 块识别
# ---------------------------------------------------------------------------


def _additional_properties(msg: Mapping[str, Any]) -> dict[str, Any]:
    props = msg.get("additional_properties")
    return dict(props) if isinstance(props, Mapping) else {}


def _message_texts(msg: Mapping[str, Any]) -> list[str]:
    texts: list[str] = []
    contents = msg.get("contents")
    if isinstance(contents, list):
        for content in contents:
            if isinstance(content, Mapping) and isinstance(content.get("text"), str):
                texts.append(content["text"])
    return texts


def summary_context_id(msg: Mapping[str, Any]) -> str:
    """提取消息携带的压缩上下文 id，找不到返回 ""。

    优先 ``additional_properties._block_id``，其次匹配摘要正文
    ``[Compressed context: ctx_xxxxxxxx]``。
    """
    props = _additional_properties(msg)
    block_id = props.get(_BLOCK_ID_KEY)
    if isinstance(block_id, str) and block_id:
        return block_id
    for text in _message_texts(msg):
        match = _SUMMARY_TEXT_RE.search(text)
        if match:
            return match.group(1)
    return ""


def is_summary_message(msg: Mapping[str, Any]) -> bool:
    """是否为压缩摘要占位消息（kind 标记或正文指纹命中其一）。"""
    return summary_context_id(msg) != "" or _additional_properties(msg).get(_KIND_KEY) == _SUMMARY_KIND


def _is_turn_marker(msg: Mapping[str, Any]) -> bool:
    return msg.get("role") == "assistant" and _additional_properties(msg).get(_KIND_KEY) == _TURN_MARKER_KIND


def _msg_turn(msg: Mapping[str, Any]) -> int | None:
    """读取消息的 ``additional_properties._turn``，非 int 视为缺失。"""
    raw = _additional_properties(msg).get(_TURN_KEY)
    if type(raw) is int:
        return raw
    return None


def _excluded_flag(msg: Mapping[str, Any]) -> bool:
    props = _additional_properties(msg)
    return bool(props.get(_EXCLUDED_KEY)) or bool(msg.get(_EXCLUDED_KEY))


def _contains_marker(value: Any, marker: str) -> bool:
    if isinstance(value, str):
        return marker in value
    if isinstance(value, Mapping):
        return any(_contains_marker(item, marker) for item in value.values())
    if isinstance(value, list):
        return any(_contains_marker(item, marker) for item in value)
    return False


# ---------------------------------------------------------------------------
# 块查询
# ---------------------------------------------------------------------------


def block_turn_range(block: Mapping[str, Any]) -> tuple[int, int]:
    """解析块的 ``turn_range``；畸形或缺失时返回 ``(0, 0)`` 哨兵（同 chrys）。"""
    raw = block.get("turn_range")
    if isinstance(raw, list | tuple) and len(raw) == 2:
        low, high = raw
        if type(low) is int and type(high) is int:
            return low, high
    return (0, 0)


def _block_created_at(block: Mapping[str, Any]) -> str:
    raw = block.get("created_at")
    return raw if isinstance(raw, str) else ""


def blocks_for_turn(blocks: list[Mapping[str, Any]], turn_number: int) -> list[dict[str, Any]]:
    """返回覆盖指定轮次的压缩块（按 ``turn_range`` 判定，跳过 ``(0,0)`` 哨兵）。

    用于从还原结果中定位"指定对话轮次"的上下文来源块。
    """
    if type(turn_number) is not int:
        return []
    matched: list[dict[str, Any]] = []
    for block in blocks:
        low, high = block_turn_range(block)
        if (low, high) == (0, 0):
            continue
        if low <= turn_number <= high:
            matched.append(dict(block))
    return matched


def split_turns(messages: list[Mapping[str, Any]]) -> list[list[dict[str, Any]]]:
    """按轮次边界（assistant + ``_chrys_kind="turn"``）切分消息，边界消息丢弃。"""
    turns: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for msg in messages:
        if _is_turn_marker(msg):
            if current:
                turns.append(current)
                current = []
            continue
        current.append(dict(msg))
    if current:
        turns.append(current)
    return turns


# ---------------------------------------------------------------------------
# 还原主流程
# ---------------------------------------------------------------------------


def _sanitized_block(block: dict[str, Any]) -> dict[str, Any]:
    return {
        "compressed_context_id": block.get("compressed_context_id", ""),
        "summary_text": block.get("summary_text", ""),
        "marker_id": block.get("marker_id", ""),
        "turn_range": list(block_turn_range(block)),
        "created_at": _block_created_at(block),
        "message_count": len(block["messages"]),
    }


def _anchor_after_live_turn(messages: list[dict[str, Any]], turn_number: int) -> int:
    """返回孤儿块的插入点：最后一条 ``_turn <= turn_number`` 的 live 消息之后。

    轮次标记位于轮次末尾（标记关闭轮次），块的 ``turn_range`` 含其收尾标记，
    因此块应插在"仍属于 <= high 轮次"的最后一条 live 消息之后；live 中不存在
    任何 <= high 的轮次信息时返回 0（规范模型中块恒旧于 live 历史，整体前置）。
    """
    anchor = 0
    for index, msg in enumerate(messages):
        turn = _msg_turn(msg)
        if turn is not None and turn <= turn_number:
            anchor = index + 1
    return anchor


def reconstruct_messages(source: Path | str | Mapping[str, Any]) -> ReconstructionResult:
    """从 session.json 还原完整消息列表。

    Args:
        source: session.json 文件路径，或已解析的 envelope dict（此时不拷贝原 dict，
                输出消息均为深拷贝，不会修改输入）。

    Returns:
        ReconstructionResult: ``restored_messages``（完整消息）、``blocks``（块索引）、
        ``diagnostics``（统计与说明）。
    """
    if isinstance(source, Mapping):
        data = dict(source)
    else:
        data = load_session_file(source)

    diagnostics = ReconstructionDiagnostics()
    state, state_source = extract_state(data)
    diagnostics.state_source = state_source

    live = parse_live_messages(state, diagnostics)
    blocks = parse_compressed_blocks(state, diagnostics)
    diagnostics.live_message_count = len(live)
    diagnostics.block_count = len(blocks)

    # 按 compressed_context_id 建立块索引（首个生效，重复 id 只计数）。
    blocks_by_id: dict[str, dict[str, Any]] = {}
    ordered_ids: list[str] = []
    for block in blocks:
        cid = block.get("compressed_context_id")
        if not isinstance(cid, str) or not cid:
            diagnostics.skipped_malformed_block_count += 1
            diagnostics.notes.append(f"compressed block without a usable id skipped: {block!r}")
            continue
        if cid in blocks_by_id:
            diagnostics.duplicate_block_id_count += 1
            diagnostics.notes.append(f"duplicate compressed_context_id {cid!r}; first block wins")
            continue
        blocks_by_id[cid] = block
        ordered_ids.append(cid)

    # 原位展开：摘要占位消息 → 对应块的消息。
    restored: list[dict[str, Any]] = []
    spliced_ids: set[str] = set()
    for msg in live:
        cid = summary_context_id(msg)
        block = blocks_by_id.get(cid) if cid else None
        if block is not None and cid not in spliced_ids:
            restored.extend(copy.deepcopy(block["messages"]))
            spliced_ids.add(cid)
            diagnostics.spliced_block_count += 1
            nested = sum(1 for m in block["messages"] if is_summary_message(m))
            if nested:
                diagnostics.notes.append(
                    f"block {cid!r} contains {nested} summary-like message(s); not recursively expanded"
                )
            continue
        if block is not None:
            diagnostics.duplicate_summary_placeholder_count += 1
            diagnostics.notes.append(f"summary placeholder for {cid!r} duplicated in live messages; kept in place")
            restored.append(copy.deepcopy(msg))
            continue
        if is_summary_message(msg):
            diagnostics.unresolved_summary_count += 1
        restored.append(copy.deepcopy(msg))

    # 孤立块：live 中无占位符引用，按 (turn_range.high, created_at) 升序插回时间线。
    orphan_ids = [cid for cid in ordered_ids if cid not in spliced_ids]
    if orphan_ids:
        diagnostics.orphan_block_count = len(orphan_ids)
        orphans = sorted(
            (blocks_by_id[cid] for cid in orphan_ids),
            key=lambda block: (block_turn_range(block)[1], _block_created_at(block)),
        )
        # 插入点随 high 单调不减，同一插入点上的块保持升序。
        anchors: dict[int, list[dict[str, Any]]] = {}
        for block in orphans:
            high = block_turn_range(block)[1]
            anchor = _anchor_after_live_turn(live, high)
            anchors.setdefault(anchor, []).append(block)
            diagnostics.notes.append(
                f"orphan block {block.get('compressed_context_id')!r} "
                f"(turns {block_turn_range(block)}) inserted at live index {anchor}"
            )
        restored = []
        for index, msg in enumerate(live):
            for block in anchors.get(index, []):
                restored.extend(copy.deepcopy(block["messages"]))
            restored.append(copy.deepcopy(msg))
        for block in anchors.get(len(live), []):
            restored.extend(copy.deepcopy(block["messages"]))

    diagnostics.restored_message_count = len(restored)
    diagnostics.restored_excluded_count = sum(1 for msg in restored if _excluded_flag(msg))
    diagnostics.restored_spill_notice_count = sum(1 for msg in restored if _contains_marker(msg, _SPILL_NOTICE_MARKER))

    return ReconstructionResult(
        restored_messages=restored,
        blocks=[_sanitized_block(blocks_by_id[cid]) for cid in ordered_ids],
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# 独立 CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m chrys_trace_analysis.reconstruct",
        description="从 session.json 的 compressed_msgs 还原完整 messages 轨迹。",
    )
    parser.add_argument("input", help="chrys session.json 文件路径")
    parser.add_argument("--out", help="还原结果写入该路径；缺省输出到 stdout")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    try:
        result = reconstruct_messages(args.input)
    except (OSError, ValueError) as exc:
        print(f"reconstruct failed: {exc}", file=sys.stderr)
        return 1

    payload = result.to_dict()
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)

    diag = result.diagnostics
    print(
        f"restored {diag.restored_message_count} messages "
        f"(state={diag.state_source}, live={diag.live_message_count}, "
        f"blocks={diag.block_count}, spliced={diag.spliced_block_count}, "
        f"orphans={diag.orphan_block_count}, unresolved summaries={diag.unresolved_summary_count})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
