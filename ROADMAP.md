# Roadmap

fusion-store 演进路线。当前 **0.2.0-rc.1**（0.2 线 RC 基线，受控商用 / 技术预览）。本文件记架构演进方向与容量边界，不承诺版本号时间。

## 容量边界（当前 0.1.x）

单层 NSW 无分层/分片捷径，单 namespace 向量数有软上限。生产建议 + 实测水位：

| 维度 | 软上限 | 实测（1M×768） | 触发信号 |
|------|--------|---------------|----------|
| 单 namespace 向量数 | **≤1-2M** | 1M 达标，余量收窄 | `/stats` `vector_count` |
| 检索 p99 | <5ms SLA | 1544us（余量 ~70%） | `/metrics` `fusion_store_ops_latency_us` p99 |
| 写入吞吐 | 批 100 ≥10K vecs/s | 单批 100 ~16K/s；1M 稳态 48 vecs/s | `/metrics` `fusion_store_ops_total` rate |
| 图常驻 RAM | 物理内存 ~40% | 415MB@1M | `/stats` `graph_memory_bytes` |

**超限动作**：`vector_count` 接近 1M 时运维应决策扩容（见下「NSW 规模演进」分片/分层路径），非自动迁移。当前版本超限 = 检索延迟劣化逼近 SLA + 写入超线性衰减，无数据正确性风险（仍可写可查，仅性能降级）。`/stats` 暴露水位供监控告警（grafana 面板见 `ops/grafana-dashboard.json`）。

## NSW 规模演进（F-ARCH-3 深度解）

当前 `NswGraph` 是**单层 NSW**（邻接图，无多层跳表），检索复杂度近似 O(N^log M)。N 持续增长线性劣化检索延迟 + 写入超线性衰减。两条演进路径，按改动量从小到大：

### 路径 A — 分片多 namespace（推荐，改动小）

消费方侧分片，引擎不改核心算法：

- 一个逻辑库拆 K 个 namespace（各开独立 `Engine::open` 目录），向量按 id hash 或业务键路由分片。
- 检索 fan-out：并行查 K 个 namespace 的 `search_knn`，归并 top-k。延迟 = max(分片延迟) + fan-out 开销，单分片 N/K 小 → 单分片 p99 显著降。
- 写入：按分片路由，单分片构建量降 → 写入吞吐回升。
- **引擎侧支持**：当前已支持多 `Engine` 实例（A4 单 namespace 设计的副作用正好服务此路径）。需补的是**可选的 `ShardedEngine` 封装**（fan-out 检索 + 路由写），属新增薄层非核心改动。
- 触发：`vector_count` 超 1M 即可评估切分片。无需改 `NswGraph`。

### 路径 B — 分层 HNSW（改动大，彻底解）

引擎核心算法升级，引入多层跳表：

- `NswGraph` → `HnswGraph`：节点按指数衰减概率分配层级，顶层 entry point，逐层缩小候选集 → 检索复杂度 O(log N)（非 O(N^log M)）。
- 检索延迟随 N 增长 **对数** 劣化（非线性），1M→10M 延迟仅温和升。
- **代价**：图结构变复杂（多层邻接 + 层级分配），snapshot 格式变（需持久化层级），构建成本升（每节点算层级 + 多层连边）。属核心重构，需独立 milestone + 全规模基准重验。
- 触发：单 namespace 单分片需 5M+ 且 fan-out 分片不可接受（强一致/低延迟场景）时排。

### 决策

0.1.x ~ 0.2.x：**路径 A**（分片）为主——零核心改动，消费方可控，覆盖 ≤10M 量级。`ShardedEngine` 薄层排入 0.2.x。
1.0+：若出现单分片 5M+ 强需求，评估路径 B（分层 HNSW）核心重构，独立 milestone。当前不排期，避免发布门禁阶段引入不可控算法变更（Rule 2/3/10）。

## 分片多 namespace（A4 演进）

当前单 `Engine` = 单 namespace（A4 有意设计）。多 namespace 是消费方职责（各开独立 `Engine::open`）。演进：0.2.x 提供 `ShardedEngine` 薄层封装（fan-out 检索 + 路由写 + 归并 top-k），减消费方样板代码。不引入单引擎内多 namespace 路由（保持隔离简单性，A4 文档化立场不变）。

## 列式类型扩展（F-COL-1 演进）

当前 Arrow 列式覆盖定长原语（Int32/Int64/Float32/Float64，M3 范围）。演进：按上游 fusion-mlx Tensor buffer 实际需求扩 Int8/Int16/Float16/定长字符串/布尔。`columnar/store.rs` 类型 match 加分支即可，无架构改动。触发：fusion-mlx 需 fp16/bf16 量化 tensor 零拷贝对齐时排。

## map_size 自动 grow（F-PERF-4 演进）

当前 KV/列式/vec_meta LMDB map_size 固定命名常量（KV 2GB / COL 2GB / VEC_META 256MB，P2-12）。`MapFull` 时返错拒写，需运维手动改常量重编或扩容。演进：`MapFull` 捕获后自动 `env.resize` 扩 map_size 重试（heed 支持 resize）。触发：生产 `MapFull` 频发时排。当前 2GB KV 容量充足（单 namespace ≤1-2M 向量规模下元数据量远低于此）。
