"""Tests for the trace analysis pipeline (digest, per-trace analysis, aggregation)."""

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

from chrys_trace_analysis.config import (  # noqa: E402
    Config,
    DeviationAnalysisConfig,
    LLMConfig,
    PathsConfig,
    PipelineConfig,
    TraceAnalysisConfig,
)
from chrys_trace_analysis.models import (  # noqa: E402
    ApprovalRecord,
    CategoryDefinition,
    Intervention,
    MessageContent,
    RawMessage,
    Session,
    SessionRound,
    SessionTurn,
    SubAgentRecord,
    TraceTaskAnalysis,
)
from chrys_trace_analysis.trace_analysis import (  # noqa: E402
    AGGREGATION_SYSTEM_PROMPT,
    TRACE_ANALYSIS_SYSTEM_PROMPT,
    _aggregate_categories,
    _analyze_trace,
    _extract_categories,
    _extract_mapping,
    _format_digest,
    _truncate,
    apply_aggregation,
    build_summary,
    build_trace_digest,
    render_markdown_overview,
    run_trace_analysis,
    run_trace_analyses,
    write_markdown_overview,
)


# ==============================================================================
# Fake LLM client
# ==============================================================================


class FakeClient:
    """Deterministic chat_json stub: per-uuid analysis responses + aggregation."""

    def __init__(
        self,
        analysis_responses: dict[str, dict] | None = None,
        aggregation: dict | None = None,
        default_analysis: dict | None = None,
    ):
        self.analysis_responses = analysis_responses or {}
        self.aggregation = aggregation or {
            "task_categories": [{"name": "代码编写", "description": "编写或修改代码"}],
            "task_category_mapping": {"编码": "代码编写", "写代码": "代码编写"},
            "intervention_types": [{"name": "指出错误", "description": "用户指出 agent 的错误"}],
            "intervention_type_mapping": {"指出错误": "指出错误", "用户纠错": "指出错误"},
        }
        self.default_analysis = default_analysis or {
            "task_category": "编码",
            "task_summary": "任务概括",
            "interventions": [],
        }
        self.trace_calls = 0
        self.aggregation_calls = 0

    def chat_json(self, system_prompt: str, user_prompt: str, max_tokens: int | None = None):
        if system_prompt == TRACE_ANALYSIS_SYSTEM_PROMPT:
            self.trace_calls += 1
            uuid = self._uuid_from_prompt(user_prompt)
            return self.analysis_responses.get(uuid, self.default_analysis)
        if system_prompt == AGGREGATION_SYSTEM_PROMPT:
            self.aggregation_calls += 1
            return self.aggregation
        raise AssertionError(f"unexpected system prompt: {system_prompt[:60]}")

    @staticmethod
    def _uuid_from_prompt(user_prompt: str) -> str:
        for line in user_prompt.splitlines():
            if line.startswith("Trajectory "):
                return line[len("Trajectory "):].strip()
        return ""


# ==============================================================================
# Session builders
# ==============================================================================


def _session(
    uuid: str,
    rounds: list[tuple[str, str]],
    tools: list[list[str]] | None = None,
) -> Session:
    """Build a Session from (user_text, assistant_text) rounds and per-turn tool names."""
    tools = tools or [[] for _ in rounds]
    turns = []
    for i, (user_text, asst_text) in enumerate(rounds, 1):
        messages = [RawMessage(role="user", contents=[MessageContent(type="text", text=user_text)])]
        for name in tools[i - 1]:
            messages.append(RawMessage(
                role="assistant",
                contents=[MessageContent(type="function_call", name=name, arguments="{}")],
            ))
        if asst_text:
            messages.append(RawMessage(
                role="assistant",
                contents=[MessageContent(type="text", text=asst_text)],
            ))
        turns.append(SessionTurn(turn_index=i, messages=messages))
    return Session(
        session_uuid=uuid,
        session_abstract=[SessionRound(user_msg=u, assistant_reply=a) for u, a in rounds],
        turns=turns,
    )


# ==============================================================================
# _truncate
# ==============================================================================


class TestTruncate:
    def test_short_text_kept(self):
        assert _truncate("abc", 10) == "abc"

    def test_long_text_truncated_with_ellipsis(self):
        assert _truncate("a" * 100, 10) == "a" * 10 + "…"
        assert len(_truncate("a" * 100, 10)) == 11

    def test_whitespace_stripped(self):
        assert _truncate("  abc  ", 10) == "abc"


# ==============================================================================
# build_trace_digest
# ==============================================================================


class TestBuildTraceDigest:
    def test_basic_compression(self):
        session = _session(
            "s1",
            [("用户请求", "助手长答复" + "x" * 500), ("追问", "回答")],
            tools=[["grep", "read_file", "grep"], ["powershell"]],
        )
        digest = build_trace_digest(session, user_name="u1", max_assistant_chars=50)
        assert digest.session_uuid == "s1"
        assert digest.user_name == "u1"
        assert len(digest.turns) == 2
        assert digest.truncated_turns is False
        # 长答复被截断
        assert digest.turns[0].assistant_text.endswith("…")
        assert len(digest.turns[0].assistant_text) == 51
        # 工具调用按出现顺序保留（含重复），tool_names 属性去重视图
        assert [c.name for c in digest.turns[0].tool_calls] == ["grep", "read_file", "grep"]
        assert digest.turns[0].tool_names == ["grep", "read_file", "grep"]
        assert digest.turns[1].tool_names == ["powershell"]
        # 特征抽取
        assert digest.features["turn_count"] == 2
        assert digest.features["tool_call_count"] == 4
        assert digest.features["tool_usage"] == {"grep": 2, "read_file": 1, "powershell": 1}
        assert digest.features["message_count"] == 8
        assert digest.features["approval_count"] == 0
        assert digest.features["rejection_count"] == 0
        assert digest.features["sub_agent_count"] == 0

    def test_tool_args_preview_truncated(self):
        session = _session(
            "s1",
            [("q", "a")],
            tools=[["read_file"]],
        )
        session.turns[0].messages[1].contents[0].arguments = '{"path": "' + "x" * 300 + '"}'
        digest = build_trace_digest(session, max_tool_args_chars=50)
        args = digest.turns[0].tool_calls[0].args_head
        assert args.endswith("…")
        assert len(args) == 51

    def test_approvals_and_sub_agents_in_digest(self):
        session = _session("s1", [("q1", "a1")])
        session.approvals = [
            ApprovalRecord(turn_index=1, tool_name="powershell", approved=True,
                           reason="只读命令"),
            ApprovalRecord(turn_index=None, tool_name="write_file", approved=False,
                           reason="不要改"),
        ]
        session.sub_agents = [
            SubAgentRecord(turn_index=1, tool_name="explore_agent", status="completed",
                           message_count=30, tool_call_count=5,
                           tool_usage={"read_file": 4, "glob": 1},
                           prompt_preview="Find code"),
            SubAgentRecord(turn_index=None, tool_name="explore_agent", status="failed",
                           message_count=2, tool_call_count=0, tool_usage={}),
        ]
        digest = build_trace_digest(session)
        assert len(digest.turns[0].approvals) == 1  # 只保留对齐到本轮的
        assert digest.turns[0].approvals[0].approved is True
        assert len(digest.turns[0].sub_agents) == 1
        assert digest.features["approval_count"] == 2
        assert digest.features["rejection_count"] == 1
        assert digest.features["sub_agent_count"] == 2
        assert digest.features["sub_agent_tool_calls"] == 5
        text = _format_digest(digest)
        assert "Checkpoints: APPROVED powershell (只读命令)" in text
        assert "Sub-agent explore_agent: completed, 30 msgs, 5 tool calls" in text
        assert "read_file×4" in text

    def test_rejected_approval_rendered_prominently(self):
        session = _session("s1", [("q1", "a1")])
        session.approvals = [
            ApprovalRecord(turn_index=1, tool_name="write_file", approved=False,
                           reason="不要修改这个文件！"),
        ]
        digest = build_trace_digest(session)
        text = _format_digest(digest)
        assert "Checkpoints: REJECTED write_file (不要修改这个文件！)" in text

    def test_max_turns_cap(self):
        session = _session(
            "s1", [(f"q{i}", f"a{i}") for i in range(10)],
        )
        digest = build_trace_digest(session, max_turns=3)
        assert len(digest.turns) == 3
        assert digest.truncated_turns is True
        assert digest.turns[0].turn_index == 1
        assert digest.turns[2].turn_index == 3

    def test_format_digest_mentions_truncation(self):
        session = _session("s1", [("q1", "a1"), ("q2", "a2")])
        digest = build_trace_digest(session, max_turns=1)
        text = _format_digest(digest)
        assert "Trajectory s1" in text
        assert "only the first 1" in text

    def test_format_digest_caps_tool_calls(self):
        session = _session(
            "s1", [("q1", "a1")],
            tools=[["grep"] * 10],
        )
        digest = build_trace_digest(session)
        text = _format_digest(digest, max_tool_calls_shown=3)
        assert "+7 more" in text


# ==============================================================================
# _analyze_trace
# ==============================================================================


class TestAnalyzeTrace:
    def _digest(self):
        session = _session("s1", [("q1", "a1")], tools=[["grep"]])
        return build_trace_digest(session)

    def test_success(self):
        client = FakeClient(analysis_responses={
            "s1": {
                "task_category": "调试",
                "task_summary": "修复了 bug",
                "interventions": [
                    {"turn_index": 1, "type": "指出错误", "description": "路径写错了"},
                ],
            },
        })
        result = _analyze_trace(client, self._digest())
        assert result.failed is False
        assert result.task_category == "调试"
        assert result.task_summary == "修复了 bug"
        assert result.human_intervention is True
        assert result.interventions[0].turn_index == 1
        assert result.interventions[0].type == "指出错误"
        assert result.features["tool_call_count"] == 1

    def test_no_intervention(self):
        client = FakeClient(analysis_responses={"s1": {"interventions": []}})
        result = _analyze_trace(client, self._digest())
        assert result.human_intervention is False
        assert result.interventions == []

    def test_bad_turn_index_cleared(self):
        client = FakeClient(analysis_responses={
            "s1": {"interventions": [{"turn_index": "1", "type": "x", "description": "d"}]},
        })
        result = _analyze_trace(client, self._digest())
        assert result.interventions[0].turn_index is None
        assert result.interventions[0].type == "x"

    def test_retries_then_failed(self):
        class AlwaysBad:
            def chat_json(self, *args, **kwargs):
                raise ValueError("parse failure")

        result = _analyze_trace(AlwaysBad(), self._digest(), retries=2)
        assert result.failed is True
        assert result.session_uuid == "s1"
        assert result.task_category == ""
        assert result.interventions == []


# ==============================================================================
# run_trace_analyses (parallel + cache)
# ==============================================================================


class TestRunTraceAnalyses:
    def _digests(self):
        return [
            build_trace_digest(_session("s1", [("q1", "a1")])),
            build_trace_digest(_session("s2", [("q2", "a2")])),
            build_trace_digest(_session("s3", [("q3", "a3")])),
        ]

    def test_parallel_results_in_order(self, tmp_path):
        client = FakeClient(analysis_responses={
            "s1": {"task_category": "编码"},
            "s2": {"task_category": "调试"},
            "s3": {"task_category": "测试"},
        })
        analyses, failed = run_trace_analyses(
            client, self._digests(), tmp_path, max_workers=3,
        )
        assert failed == []
        assert [a.session_uuid for a in analyses] == ["s1", "s2", "s3"]
        assert [a.task_category for a in analyses] == ["编码", "调试", "测试"]
        # 每条轨迹都有且仅有一条
        assert len(analyses) == 3

    def test_cache_reuse_skips_llm(self, tmp_path):
        client = FakeClient(analysis_responses={
            "s1": {"task_category": "编码"}, "s2": {"task_category": "调试"},
        })
        run_trace_analyses(client, self._digests()[:2], tmp_path, max_workers=2)
        calls_first = client.trace_calls
        assert calls_first == 2
        # 中间文件已存在，第二次运行不再调用 LLM
        analyses, failed = run_trace_analyses(client, self._digests()[:2], tmp_path, max_workers=2)
        assert client.trace_calls == calls_first
        assert [a.task_category for a in analyses] == ["编码", "调试"]

    def test_failed_analysis_kept_and_reported(self, tmp_path):
        class FailForS2:
            def chat_json(self, system_prompt, user_prompt, max_tokens=None):
                if "Trajectory s2" in user_prompt:
                    raise ValueError("boom")
                return {"task_category": "编码"}

        analyses, failed = run_trace_analyses(
            FailForS2(), self._digests(), tmp_path, max_workers=3, retries=1,
        )
        assert failed == ["s2"]
        assert len(analyses) == 3  # 不静默缺失
        by_uuid = {a.session_uuid: a for a in analyses}
        assert by_uuid["s2"].failed is True
        assert by_uuid["s2"].task_category == ""
        assert by_uuid["s1"].failed is False


# ==============================================================================
# apply_aggregation
# ==============================================================================


def _analysis(
    session_uuid: str,
    category: str = "编码",
    interventions: list[dict] | None = None,
    failed: bool = False,
) -> TraceTaskAnalysis:
    return TraceTaskAnalysis(
        session_uuid=session_uuid,
        user_name="user_a",
        task_category=category,
        task_summary="任务概括",
        human_intervention=bool(interventions),
        interventions=[Intervention.model_validate(iv) for iv in (interventions or [])],
        failed=failed,
    )


class TestApplyAggregation:
    def test_maps_categories_and_types(self):
        analyses = [
            _analysis(
                "s1", category="编码",
                interventions=[{"type": "指出错误", "turn_index": 2, "description": "函数名写错"}],
            ),
            _analysis(
                "s2", category="写代码",
                interventions=[{"type": "用户纠错", "turn_index": 3, "description": "方向不对"}],
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
            CategoryDefinition(name="代码编写", description="编写或修改代码",
                               count=2, session_uuids=["s1", "s2"])
        ]
        assert type_cats == [
            CategoryDefinition(name="指出错误", description="用户指出 agent 的错误",
                               count=2, session_uuids=["s1", "s2"])
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

    def test_failed_entries_excluded(self):
        analyses = [
            _analysis("s1", category="编码"),
            _analysis("s2", category="调试", failed=True),
        ]
        task_cats, _, _, _ = apply_aggregation(analyses, None)
        assert [c.name for c in task_cats] == ["编码"]
        assert task_cats[0].count == 1  # 失败条目不计数
        assert analyses[1].task_category == "调试"  # 失败条目保持原样、不参与映射

    def test_untouched_analyses(self):
        a = _analysis("s1", category="")
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


# ==============================================================================
# _aggregate_categories
# ==============================================================================


class TestAggregateCategories:
    def test_returns_aggregation(self):
        client = FakeClient()
        analyses = [
            _analysis("s1", category="编码"),
            _analysis("s2", category="写代码"),
        ]
        result = _aggregate_categories(client, analyses)
        assert result is not None
        assert result["task_category_mapping"]["编码"] == "代码编写"
        assert client.aggregation_calls == 1

    def test_none_on_persistent_failure(self):
        class AlwaysBad:
            def chat_json(self, *args, **kwargs):
                raise ValueError("bad")

        assert _aggregate_categories(AlwaysBad(), [], retries=2) is None


# ==============================================================================
# build_summary
# ==============================================================================


class TestBuildSummary:
    def test_distributions_use_real_counts(self):
        analyses = [
            _analysis("s1", category="代码编写",
                      interventions=[{"type": "指出错误", "description": "d1"}]),
            _analysis("s2", category="代码编写",
                      interventions=[{"type": "指出错误", "description": "d2"},
                                     {"type": "补充信息", "description": "d3"}]),
            _analysis("s3", category="调试"),
            _analysis("s4", category="调试", failed=True),
        ]
        summary = build_summary(
            analyses, ["s4"], {"total_documents": 4}, total_trajectories=4,
        )
        assert summary["total_trajectories"] == 4
        assert summary["analyzed_trajectories"] == 3
        assert summary["failed_trajectories"] == 1
        assert summary["intervention_sessions"] == 2
        assert summary["per_task_category_distribution"] == {"代码编写": 2, "调试": 1}
        assert summary["per_intervention_type_distribution"] == {"指出错误": 2, "补充信息": 1}
        assert summary["intervention_rate"] == "66.7%"

    def test_feature_overview(self):
        a1 = _analysis("s1", category="编码")
        a1.features = {"turn_count": 3, "tool_call_count": 4,
                       "tool_usage": {"grep": 3, "read_file": 1}}
        a2 = _analysis("s2", category="调试")
        a2.features = {"turn_count": 5, "tool_call_count": 2,
                       "tool_usage": {"grep": 1, "powershell": 1}}
        summary = build_summary([a1, a2], [], {}, total_trajectories=2)
        assert summary["feature_overview"]["total_tool_calls"] == 6
        assert summary["feature_overview"]["top_tools"] == {"grep": 4, "read_file": 1, "powershell": 1}
        assert summary["feature_overview"]["avg_turns"] == 4.0
        assert summary["feature_overview"]["max_turns"] == 5

    def test_empty(self):
        summary = build_summary([], [], {"total_documents": 0}, total_trajectories=0)
        assert summary["analyzed_trajectories"] == 0
        assert summary["per_task_category_distribution"] == {}
        assert summary["intervention_rate"] == "0%"


# ==============================================================================
# End-to-end via run_trace_analysis with fake client
# ==============================================================================


class TestEndToEnd:
    def _write_session(self, sessions_dir: Path, uuid: str, user_msg: str):
        doc = {
            "meta": {"session_id": uuid, "user_id": f"user_{uuid}"},
            "state": {
                "messages": [
                    {"role": "user", "contents": [{"type": "text", "text": user_msg}]},
                    {"role": "assistant", "contents": [{"type": "text", "text": "完成。"}],
                     "additional_properties": {"_chrys_kind": "turn"}},
                ]
            },
        }
        (sessions_dir / f"{uuid}.json").write_text(
            json.dumps(doc, ensure_ascii=False), encoding="utf-8",
        )

    def _config(self, tmp_path: Path) -> Config:
        return Config(
            llm=LLMConfig(api_endpoint="http://fake", api_key="k", model_name="m"),
            pipeline=PipelineConfig(),
            deviation_analysis=DeviationAnalysisConfig(),
            trace_analysis=TraceAnalysisConfig(max_workers=2),
            paths=PathsConfig(
                output_dir=tmp_path / "out",
                sessions_dir=tmp_path / "sessions",
            ),
        )

    def test_full_pipeline(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        self._write_session(sessions_dir, "s1", "帮我写排序")
        self._write_session(sessions_dir, "s2", "帮我排查报错")

        client = FakeClient(analysis_responses={
            "s1": {"task_category": "编码", "task_summary": "写排序", "interventions": []},
            "s2": {
                "task_category": "调试",
                "task_summary": "排查报错",
                "interventions": [{"turn_index": 1, "type": "指出错误", "description": "日志路径错"}],
            },
        })
        config = self._config(tmp_path)
        result = run_trace_analysis(config, client=client)

        assert len(result.analyses) == 2
        by_uuid = {a.session_uuid: a for a in result.analyses}
        assert by_uuid["s1"].task_category == "代码编写"  # 已聚合映射
        assert by_uuid["s1"].task_category_raw == "编码"
        assert by_uuid["s2"].human_intervention is True
        assert by_uuid["s2"].interventions[0].type == "指出错误"
        assert by_uuid["s2"].interventions[0].type_raw == "指出错误"

        # 分类总览：s1 的"编码"被映射为"代码编写"，s2 的"调试"无映射保留原样
        assert {c.name for c in result.task_categories} == {"代码编写", "调试"}
        by_name = {c.name: c.count for c in result.task_categories}
        assert by_name == {"代码编写": 1, "调试": 1}
        assert result.intervention_types[0].count == 1
        assert result.summary["per_task_category_distribution"] == {"代码编写": 1, "调试": 1}

        # 中间产物落盘
        out_dir = tmp_path / "out" / "trace_analysis"
        assert (out_dir / "trajectories" / "s1.json").exists()
        assert (out_dir / "aggregation.json").exists()
        assert (out_dir / "trace_analysis_result.json").exists()

        # 结果文件可解析且结构完整
        dumped = json.loads((out_dir / "trace_analysis_result.json").read_text(encoding="utf-8"))
        assert len(dumped["analyses"]) == 2
        assert dumped["summary"]["total_trajectories"] == 2

    def test_overview_md_written(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        self._write_session(sessions_dir, "s1", "帮我写排序")
        config = self._config(tmp_path)
        client = FakeClient(analysis_responses={
            "s1": {
                "task_category": "编码", "task_summary": "写排序",
                "interventions": [{"turn_index": 1, "type": "指出错误", "description": "d"}],
            },
        })
        result = run_trace_analysis(config, client=client)
        out_dir = tmp_path / "out" / "trace_analysis"
        overview = (out_dir / "overview.md")
        assert overview.exists()
        text = overview.read_text(encoding="utf-8")
        # 汇总数量
        assert "# 轨迹分析总览" in text
        assert "## 任务分类总览" in text
        assert "代码编写" in text
        # 每条轨迹的具体分析
        assert "## 逐条轨迹分析" in text
        assert "s1" in text
        assert "写排序" in text
        assert "指出错误" in text

    def test_category_members_listed(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        self._write_session(sessions_dir, "s1", "任务1")
        self._write_session(sessions_dir, "s2", "任务2")
        config = self._config(tmp_path)
        client = FakeClient(analysis_responses={
            "s1": {"task_category": "编码"},
            "s2": {"task_category": "写代码"},
        })
        result = run_trace_analysis(config, client=client)
        code_cat = next(c for c in result.task_categories if c.name == "代码编写")
        assert code_cat.count == 2
        assert set(code_cat.session_uuids) == {"s1", "s2"}


# ==============================================================================
# render_markdown_overview
# ==============================================================================


class TestRenderMarkdownOverview:
    def _result(self) -> TraceAnalysisResult:
        from chrys_trace_analysis.models import TraceAnalysisResult
        a1 = _analysis("s1", category="代码编写",
                       interventions=[{"type": "指出错误", "turn_index": 2,
                                       "description": "函数名写错"}])
        a1.features = {"turn_count": 4, "tool_call_count": 6,
                       "approval_count": 2, "rejection_count": 0}
        a2 = _analysis("s2", category="调试")
        a2.features = {"turn_count": 2, "tool_call_count": 1}
        return TraceAnalysisResult(
            analyses=[a1, a2],
            task_categories=[CategoryDefinition(
                name="代码编写", description="编写或修改代码", count=1,
                session_uuids=["s1"],
            )],
            intervention_types=[CategoryDefinition(
                name="指出错误", description="用户指出 agent 的错误", count=1,
                session_uuids=["s1"],
            )],
            task_category_mapping={"编码": "代码编写"},
            intervention_type_mapping={"指出错误": "指出错误"},
            summary=build_summary([a1, a2], [], {}, total_trajectories=2),
        )

    def test_render_contains_summary_and_details(self):
        text = render_markdown_overview(self._result())
        assert "## 任务分类总览" in text
        assert "代码编写" in text and "（1 条）" in text
        assert "## 逐条轨迹分析" in text
        assert "### s1" in text
        assert "第2轮" in text and "指出错误" in text and "函数名写错" in text
        assert "人工介入：无" in text

    def test_write_markdown_overview(self, tmp_path):
        out = write_markdown_overview(self._result(), tmp_path / "overview.md")
        assert out.read_text(encoding="utf-8").startswith("# 轨迹分析总览")
