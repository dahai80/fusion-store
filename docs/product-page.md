---
doc_id: fusion-store/product-page
product: fusion-store
class: core
entry: service
doc_type: product-page
title: { zh: "fusion-store 产品页", en: "fusion-store product page" }
audience: engineer
status: draft
version: 0.2.0-rc.4
updated: 2026-11-07
owner: @dahai
source_of_truth:
  - README.md
  - crates/fs-core/src/lib.rs
  - Cargo.toml
---

# fusion-store 产品页（Product Page）

单机零拷贝存储与索引引擎，Fusion 生态 L1 基础设施层持久化底座。Rust 实现（edition 2021，toolchain 1.94），Apache-2.0，当前版本 0.2.0-rc.4。

## 定位

将三类存储原语统一在一个引擎内，进程内零拷贝读取，消除消费方的序列化/反序列化样板：

| 能力 | 实现 | 消费方不再需要写 |
|------|------|------------------|
| KV 状态 | mmap 段 + heed（LMDB） | 自建 KV 落盘、崩溃恢复、多进程读协调 |
| 向量索引 | 单层 NSW 图常驻 RAM + 单 mmap 段快照 | 自实现 ANN 图、持久化、SIMD 距离计算（NEON 位等价于标量） |
| 列式数据 | Apache Arrow 定长原语列（Int32/Int64/Float32/Float64） | 自管理列式内存布局与投影读取 |
| 崩溃安全 | WAL 唯一 crash-safe 同步点 + 幂等 seq/applied_seq | 自写 WAL、torn-frame 容错、kill -9 后重放逻辑 |
| 多进程并发 | fs2 flock 写互斥 + 段 append-only 不可变 | 自设计文件锁协议与并发读方案 |
| 分片编排 | `ShardedEngine`（`fs-core::sharded`）K 分片 fan-out 检索 + 路由写 | 自写分片路由、并行归并 top-k、跨分片生命周期编排 |

## 访问形态（三种）

1. **进程内库（默认形态）**：Rust 消费方直接依赖 `fs-core`，`Engine::open` 后经 `FusionStoreEngine` trait 读写，零拷贝仅在此形态成立（A7）。
2. **HTTP daemon**：`fs-serve`（axum + tokio，端口 11463），管理/监控面；C/Python/Swift 等跨语言消费方经此访问，读路径强制拷贝。
3. **FFI 绑定**：C-ABI（`fs-ffi-c`，12 符号，头文件 `crates/fs-ffi-c/fs_store.h`）供 C/Swift 链接；PyO3 绑定（`fs-ffi-py`，模块 `fusion_store`）供 Python 使用。

## 约束与边界（消费方必读）

- **单层 NSW，非 HNSW**（A2）：检索复杂度近似 O(N^log M)，单 namespace 向量数软上限 **≤1-2M**（F-ARCH-3）。1M×768 实测 p99 1544us（SLA <5ms）、图常驻 RAM 415MB；超限后检索延迟线性劣化、写入超线性衰减，无数据正确性风险。`/stats` `vector_count` 暴露水位。
- **零拷贝仅限 Rust 进程内**（A7）：C-ABI/Python 读路径强制拷贝到调用方缓冲区，不暴露 mmap 指针视图。
- **单 namespace = 单 Engine 实例**（A4）：引擎内不做多租户/多 namespace 路由；多 namespace 是消费方职责（各开独立 `Engine::open`），可选 `ShardedEngine` 薄层封装。
- **timeout 语义分层**（F-ERR-5）：put_kv/delete_kv 实现阻塞 flock 超时；insert_vector/search_knn/get_kv_zero_copy 超时未完全实现（传 None 无限等待）；get_vector/list_vector_ids/delete_vector 忽略 timeout。
- **compact 磁盘峰值约 2× 在库数据**（F-PERF-5）：COW 新段写入期间旧段延迟回收（60s `reclaim_safe`），大库 compact 需预留双倍磁盘余量。
- **flock 是 advisory 锁**（F-SEC-4）：依赖所有写者遵守协议；绕过 Engine 直写 mmap 段文件不受锁保护，消费方不应绕过 API。
- **C-ABI `fs_store_close` 返回 void**（F-SEC-5）：close 落盘失败仅记日志；关键数据先显式调 `fs_store_checkpoint`（返回错误码）再 close。
- **C-ABI 不暴露列式**（F-API-2）：12 符号仅覆盖 KV + 向量原语；C/Swift 需列式走 fs-serve HTTP `/columnar`（Arrow IPC base64）或 Python 绑定。
- **LMDB map_size 固定**（F-PERF-4）：KV 2GB / 列式 2GB / vec_meta 256MB 命名常量，`MapFull` 返错拒写，需改常量重编。
- **无静态数据加密**：本地磁盘明文，依赖 OS 文件权限；多租户隔离是消费方职责（A4）。

## 服务等级（实测来源 `crates/fs-core/examples/`）

| 指标 | SLA | 实测 |
|------|-----|------|
| 单条 put_kv 吞吐 | ≥5K ops/s | ~183K ops/s（WAL 落地后） |
| insert_batch 向量写入（batch 100） | ≥10K vecs/s | ~16K vecs/s；batch 500 → 5.9K，batch 1000 → 4K（NSW 构建固有成本，大 batch 遵循 R1 用 ~100） |
| KNN p99 | <5ms | 200K×768 388us；1M×768 1544us |
| 召回率（top-10 vs 暴力） | ≥0.95 | 0.995（1M×768，ef_s=200） |
| 图常驻 RAM | — | 83MB@200K、415MB@1M（768 维） |

## 与生态的关系

- **上游消费方**：`fusion-memory`（认知图谱读写向量 + KV）；fusion-store 只持有原始向量/KV 页，不做上下文选择与语义关联。
- **对端**：`fusion-mlx` Tensor buffer 布局 — Arrow 定长列与之对齐以支持零拷贝交接。
- **运维**：`deploy/start.sh` 启停、`deploy/Dockerfile` 容器化、`deploy/fusion-store.plist` launchd、`ops/grafana-dashboard.json` 监控面板（背压队列 8000 黄 / 10000 红阈值）。

## 不做什么（业务边界，来自 PRD）

上下文选择 / 语义关联（归 `fusion-memory`）、业务 SQL 查询优化、跨网络分布式存储（单机 local-first，所有 I/O 限 127.0.0.1，无云依赖、无出站网络）。
