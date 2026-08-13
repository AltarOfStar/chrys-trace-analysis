"""Tests for the trace analysis pipeline (batching and aggregation)."""

from __future__ import annotations

import sys
from pathlib import Path

# ——————————————————————————————————————————————————————————————————————————————
# Set up import path (tests directory is outside src/)
# ——————————————————————————————————————————————————————————————————————————————

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# ——————————————————————————————————————————————————————————————————————————————
# Imports from the package under test
# ——————————————————————————————————————————————————————————————————————————————

from chrys_trace_analysis.models import (  # noqa: E402
    CategoryDefinition,
    Intervention,
    Session,
    TraceTaskAnalysis,
)
from chrys_trace_analysis.trace_analysis import (  # noqa: E402
    _extract_categories,
    _extract_mapping,
    apply_aggregation,
    chunk_traces,
)


# ==============================================================================
# Helpers
# ==============================================================================


def _analysis(
    session_uuid: str,
    category: str = "编码",
    interventions: list[dict] | None = None,
    human_intervention: bool | None = None,
) -> TraceTaskAnalysis:
    return TraceTaskAnalysis(
        session_uuid=session_uuid,
        user_name="user_a",
        task_category=category,
        task_summary="任务概括",
        human_intervention=bool(human_intervention or interventions),
        interventions=[Intervention.model_validate(iv) for iv in (interventions or [])],
    )


# ==============================================================================
# chunk_traces
# ==============================================================================


class TestChunkTraces:
    def _sessions(self, n: int) -> list[tuple[str, Session]]:
        return [
            (f"user_{i}", Session(session_uuid=f"s{i}", session_abstract=[]))
            for i in range(n)
        ]

    def test_empty(self):
        assert chunk_traces([], 20) == []

    def test_less_than_batch_size_single_group(self):
        batches = chunk_traces(self._sessions(10), 20)
        assert len(batches) == 1
        assert len(batches[0]) == 10

    def test_exact_multiple(self):
        batches = chunk_traces(self._sessions(40), 20)
        assert [len(b) for b in batches] == [20, 20]

    def test_remainder_merged_into_last_batch(self):
        batches = chunk_traces(self._sessions(65), 20)
        assert [len(b) for b in batches] == [20, 20, 25]

    def test_remainder_two(self):
        batches = chunk_traces(self._sessions(45), 20)
        assert [len(b) for b in batches] == [20, 25]

    def test_every_group_meets_batch_size(self):
        batches = chunk_traces(self._sessions(83), 20)
        assert [len(b) for b in batches] == [20, 20, 20, 23]
        assert all(len(b) >= 20 for b in batches)

    def test_order_preserved(self):
        batches = chunk_traces(self._sessions(65), 20)
        flat = [s.session_uuid for b in batches for _, s in b]
        assert flat == [f"s{i}" for i in range(65)]


# ==============================================================================
# apply_aggregation
# ==============================================================================


class TestApplyAggregation:
    def test_maps_categories_and_types(self):
        analyses = [
            _analysis(
                "s1", category="编码",
                interventions=[{"type": "指出错误", "position": "第2轮", "description": "函数名写错"}],
            ),
            _analysis(
                "s2", category="写代码",
                interventions=[{"type": "用户纠错", "position": "第3轮", "description": "方向不对"}],
            ),
        ]
        aggregation = {
            "task_categories": [{"name": "代码编写", "description": "编写或修改代码"}],
            "task_category_mapping": {"编码": "代码编写", "写代码": "代码编写"},
            "intervention_types": [{"name": "指出错误", "description": "用户指出 agent 的错误"}],
            "intervention_type_mapping": {"指出错误": "指出错误", "用户纠错": "指出错误"},
        }
        task_cats, type_cats, task_mapping, type_mapping = apply_aggregation(
            analyses, aggregation,
        )

        assert task_cats == [
            CategoryDefinition(name="代码编写", description="编写或修改代码", count=2)
        ]
        assert type_cats == [
            CategoryDefinition(name="指出错误", description="用户指出 agent 的错误", count=2)
        ]
        assert task_mapping == {"编码": "代码编写", "写代码": "代码编写"}
        assert type_mapping == {"指出错误": "指出错误", "用户纠错": "指出错误"}

        # 原始分类保留在 *_raw 字段
        assert analyses[0].task_category == "代码编写"
        assert analyses[0].task_category_raw == "编码"
        assert analyses[0].interventions[0].type == "指出错误"
        assert analyses[0].interventions[0].type_raw == "指出错误"
        assert analyses[1].interventions[0].type == "指出错误"
        assert analyses[1].interventions[0].type_raw == "用户纠错"

    def test_unmapped_category_keeps_original(self):
        analyses = [_analysis("s1", category="未知分类")]
        aggregation = {
            "task_categories": [{"name": "代码编写", "description": ""}],
            "task_category_mapping": {},
        }
        task_cats, _, _, _ = apply_aggregation(analyses, aggregation)
        assert analyses[0].task_category == "未知分类"
        assert analyses[0].task_category_raw == "未知分类"
        assert [c.name for c in task_cats] == ["未知分类"]

    def test_none_aggregation_falls_back_to_identity(self):
        analyses = [
            _analysis("s1", category="编码"),
            _analysis("s2", category="调试"),
        ]
        task_cats, _, _, _ = apply_aggregation(analyses, None)
        assert {c.name for c in task_cats} == {"编码", "调试"}
        assert analyses[0].task_category == "编码"
        assert analyses[0].task_category_raw == "编码"

    def test_untouched_analyses(self):
        a = _analysis("s1", category="", human_intervention=False)
        task_cats, type_cats, _, _ = apply_aggregation([a], None)
        assert a.task_category == ""
        assert a.task_category_raw == ""
        assert task_cats == []
        assert type_cats == []


# ==============================================================================
# _extract_mapping / _extract_categories
# ==============================================================================


class TestExtractHelpers:
    def test_extract_mapping(self):
        raw = {"task_category_mapping": {"编码": "代码编写", "x": ""}}
        assert _extract_mapping(raw, "task_category_mapping") == {"编码": "代码编写"}

    def test_extract_mapping_missing(self):
        assert _extract_mapping({"a": 1}, "b") == {}
        assert _extract_mapping(None, "b") == {}
        assert _extract_mapping({"b": "not dict"}, "b") == {}

    def test_extract_categories(self):
        raw = {"task_categories": [{"name": "代码编写", "description": "desc"}, {"x": 1}, 3]}
        assert _extract_categories(raw, "task_categories") == [
            {"name": "代码编写", "description": "desc"}
        ]

    def test_extract_categories_missing(self):
        assert _extract_categories({"a": 1}, "b") == []
        assert _extract_categories(None, "b") == []
