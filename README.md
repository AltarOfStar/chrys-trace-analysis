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
  batch_size: 20      # 每个批次的轨迹数量（默认 20）

paths:
  data_dir: "./data"
  output_dir: "./output"
  sessions_dir: "./data/sessions"   # 会话目录：存放 {uuid}.json 文件
```

## 数据输入

本工具不再从 MongoDB 拉取数据。输入为 `paths.sessions_dir` 指定目录下的
`{uuid}.json` 会话文件，每个文件对应一条会话。

文件格式参考 chrys 本地会话 `session.json` 的 envelope 结构：

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
文件名（不含扩展名）。

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

1. **展开压缩消息**：加载 `paths.sessions_dir` 目录下的 `{uuid}.json` 会话，
   通过 `reconstruct_messages` 将 `compressed_msgs` 原位展开为完整消息，
   还原每条轨迹（轮次摘要 + 完整轮次）；
2. **分批**：将轨迹按 `batch_size`（默认 20）划分批次，每个批次数量不低于 `batch_size`，
   如有剩余（不足一个批次），将其并入最后一个批次（例如 65 条轨迹、batch_size=20
   划分为 `[20, 20, 25]`）；
3. **批次分析**：对每个批次组装提示词，要求模型逐条轨迹输出两个维度的分析：

   - **任务维度**：`task_category`（单个词的任务分类）+ `task_summary`（一小段话概括任务内容）；
   - **人工介入维度**：`human_intervention`（是否存在人工介入）；若存在，逐次输出
     `interventions`，每次包含介入类型（指出错误、补充信息、需求变更等）、介入位置
     （如"第2轮用户消息"）与简要描述。

4. **跨批次聚合**：将所有批次自由形式的任务分类与人工介入类型聚合成几个固定种类，
   将每个批次的分析结果替换为聚合后的标准分类，输出最终聚合信息（固定种类定义 +
   分布）与每条轨迹的对应信息。

输出位于 `output/trace_analysis/`：

- `batch_analysis/batch_{n}.json`：每个批次的分析结果（崩溃后可复用续跑）；
- `aggregation.json`：跨批次聚合结果（固定种类与原始→标准映射）；
- `trace_analysis_result.json`：最终结果，含每条轨迹的任务/介入分析、固定种类、
  映射关系与汇总统计。

## 测试

```bash
python -m pytest tests -q
```
