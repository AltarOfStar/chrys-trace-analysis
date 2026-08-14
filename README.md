# chrys-trace-analysis

Coding agent 轨迹分析工具：对用户在 Coding Agent（chrys）中的会话轨迹进行分析，当前提供两条流水线：

- **deviation_analysis（偏移分析）**：检测 agent 执行偏离用户意图的会话，并按固定类别归类、做轮次级细粒度分析。
- **trace_analysis（轨迹分析）**：展开会话的 `compressed_msgs` 还原完整轨迹，按批次分析每条轨迹的任务类型与人工介入情况，并跨批次聚合为固定分类。

## 安装

```bash
uv sync          # 或 pip install -e .
```

## 配置

`config.yaml` 中可配置 LLM 接入信息、会话数据目录与各流水线参数：

```yaml
llm:
  api_endpoint: ...
  api_key: ...
  model_name: ...

deviation_analysis:
  detection_batch_size: 30
  categorization_batch_size: 30

trace_analysis:
  max_workers: 8            # 逐条并行分析并发调用 LLM 的线程数（无 batch 语义）
  max_turns: 60             # 每条轨迹摘要最多保留的轮次数
  max_user_chars: 600       # 每轮用户消息截断上限（字符）
  max_assistant_chars: 1000 # 每轮助手答复截断上限（字符）
  max_tool_args_chars: 200  # 每个工具调用参数预览截断上限（字符）
  max_tool_calls_shown: 12  # 每轮摘要最多展示的工具调用条数

paths:
  data_dir: "./data"
  output_dir: "./output"
  # sessions_dir 支持两种布局：
  #   扁平：目录下直接放 {uuid}.json
  #   嵌套：目录下放 {uuid}/session.json（chrys 本地 sessions 目录的真实布局，
  #         自动读取同级 approvals/ 审批记录与 sub_agents/ 子代理会话摘要）
  sessions_dir: "./data/sessions"   # 会话目录：存放 {uuid}.json 或 {uuid}/session.json
```

## 数据输入

输入为 `paths.sessions_dir` 指定目录下的会话，支持两种布局：

- **扁平**：`{uuid}.json`，每个文件对应一条会话；
- **嵌套**（chrys 本地 sessions 目录的真实布局）：`{uuid}/session.json`，
  会话 UUID 取目录名，并自动读取同级伴生信息：
  - `approvals/*.log`：工具调用前的人工审批检查点（人工交互的直接证据，
    含批准/拒绝与理由，按用户提示或时间戳对齐到轮次）；
  - `sub_agents/sessions/*.json`：子代理完整会话（主会话只保留最终结果，
    这里保留子代理的内部工具调用统计与任务预览，按 call_id 对齐到轮次）。

文件格式参考 chrys 本地 `session.json` 的 envelope 结构：

```json
{
  "meta": {"session_id": "...", "user_id": "...", "title": "..."},
  "state": {
    "messages": [ ...live 消息... ],
    "compressed_msgs": [ ...压缩块（含完整轮次消息）... ]
  }
}
```

根字段可能存在冗余字段，一律容忍；加载时会自动通过 reconstruct 模块将
`compressed_msgs` 原位展开为完整消息（顶层 `messages` / `compressed_msgs`
的扁平形态同样兼容）。会话 UUID 依次取根字段 `uuid`、`meta.session_id`、
目录名/文件名。

可用 `scripts/session_request.py <uuid>` 从统计服务下载单个会话生成
`{uuid}.json`。

## deviation_analysis 流水线

```bash
python -m chrys_trace_analysis.main --pipeline deviation_analysis
```

三步流程：

1. **偏离检测**：按批次判断每条会话是否偏离用户意图，输出偏离原因与问题出现的轮次；
2. **偏离归类**：将偏离会话归入 5 个固定类别（需求理解偏差、执行幻觉、代码逻辑缺陷、系统异常、其他）；
3. **轮次细粒度分析**：拉取问题轮次的完整原始消息，定位具体的错误动作并给出分析。

可通过 `--start-step` 从中间步骤恢复（中间文件落在 `output/deviation_analysis/`）。

## trace_analysis 流水线

```bash
python -m chrys_trace_analysis.main --pipeline trace_analysis
```

处理流程：

1. **展开压缩消息**：加载 `paths.sessions_dir` 目录下的会话（扁平
   `{uuid}.json` 或嵌套 `{uuid}/session.json`），通过 `reconstruct_messages`
   将 `compressed_msgs` 原位展开为完整消息；嵌套布局额外读取
   `approvals/` 审批记录与 `sub_agents/` 子代理摘要并对齐到轮次；
2. **本地摘要压缩**：每条轨迹压缩为紧凑摘要——每轮保留截断后的用户消息、
   最终助手答复、工具调用（含参数预览）、人工审批检查点与子代理摘要
   （`max_user_chars` / `max_assistant_chars` / `max_tool_args_chars` /
   `max_tool_calls_shown` 控制截断，`max_turns` 控制轮次上限），并抽取
   确定性特征（轮次数、工具调用次数、审批/拒绝次数、子代理统计等）。
   由于每次 LLM 调用只包含一条轨迹（无 batch 拼接），截断上限可以放宽，
   每条轨迹保留更多内容用于分析；
3. **逐条并行分析**：每条轨迹独立调用一次 LLM（`max_workers` 个线程并发），
   输入为单条轨迹摘要、输出上限 4096 tokens，token 天然不会超限；
   输出任务分类（单个词）、任务概括与每次人工介入（结构化轮次号 + 类型 +
   描述），结果落盘 `trajectories/{uuid}.json`，崩溃后可复用续跑；
   提示词中明确告知模型：批准类审批检查点本身不是人工介入，但审批理由
   中的重定向（如"只读即可，不要修改文件"）与拒绝类检查点属于人工介入；
4. **跨轨迹聚合**：将自由形式的任务分类与人工介入类型聚合成固定种类
   （固定种类定义 + 分布 + 成员轨迹列表 + 原始→标准映射），替换每条轨迹/
   每次介入的分类；
5. **输出**：分类总览（`overview.md`，人类可读）、每条轨迹的类别与特征、
   每次人工介入的类别。

输出位于 `output/trace_analysis/`：

- `trajectories/{uuid}.json`：每条轨迹的逐条分析结果（崩溃后可复用续跑）；
- `aggregation.json`：跨轨迹聚合结果（固定种类与原始→标准映射）；
- `trace_analysis_result.json`：完整机器可读结果——每条轨迹的任务/介入分析
  （含 `task_category_raw` / `type_raw` 原始分类与聚合后标准分类、`features`
  特征、`failed` 标记）、固定种类（每个分类带 `count` 计数与
  `session_uuids` 成员轨迹列表）、映射关系与汇总统计
  （`per_task_category_distribution` 等为真实计数分布）；
- `overview.md`：人类可读的分类总览——汇总数量 + 每个分类的成员轨迹 +
  每条轨迹的任务分类/概括/介入详情。

## 测试

```bash
python -m pytest tests -q
```
