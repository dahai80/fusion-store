---
doc_id: fusion-store/architecture
product: fusion-store
class: core
entry: service
doc_type: architecture
title: { zh: "fusion-store 架构", en: "fusion-store architecture" }
audience: engineer
status: draft
version: 0.2.0-rc.4
updated: 2026-11-07
owner: "@dahai"
source_of_truth:
  - crates/fs-core/src/lib.rs
  - crates/fs-core/src/engine.rs
  - crates/fs-core/src/sharded.rs
  - README.md
  - Cargo.toml
---

# fusion-store 架构（Architecture）

## 1. 总体分层

```
fs-cli (CLI, 子命令 init/put/get/stats/compact/recover/serve)
  └─> fs-serve (axum + tokio HTTP daemon :11463, RBAC/背压/prometheus)
        └─> fs-core (引擎核心: mmap 段 + KV + NSW 向量 + Arrow 列式 + WAL/recover/compact)
fs-ffi-c  (C-ABI staticlib+cdylib, 12 符号, 读强制拷贝)   ──> fs-core
fs-ffi-py (PyO3 模块 fusion_store, 读强制拷贝)            ──> fs-core
```

依赖方向单向无环（fs-cli → fs-serve → fs-core）。workspace 单版本（`Cargo.toml` `[workspace.package]`），toolchain 1.94（`rust-toolchain.toml`），release profile opt-level 3 + thin LTO + strip + codegen-units=1（F-OPS-5），不加 panic=abort（cdylib FFI 边界保留 unwind）。

## 2. fs-core 模块（crates/fs-core/src/）

| 模块 | 职责 |
|------|------|
| `mem/` | mmap 段管理 + 块分配器；段 append-only 不可变，满即 seal（只读）并开新段，杜绝 SIGBUS 与并发撕裂 |
| `store/` | KV 存储：heed（LMDB）+ namespace 配额（`QuotaExceeded`） |
| `vector/` | 单层 NSW 图（`NswGraph`，M=16/ef_c=200/ef_s=200）常驻 RAM；向量存 mmap 段，id→locator + 图边；NEON SIMD 与标量位等价（R4，dot + L2 双路径）；软删除 |
| `columnar/` | Apache Arrow 定长原语列（Int32/Int64/Float32/Float64，M3 范围）；投影读取；不支持的类型拒收 |
| `engine.rs` / `engine_impl.rs` | `FusionStoreEngine` trait + 实现 + `ZeroCopyBuffer`（ptr + `Arc<MmapHandle>` 保活） |
| `sharded.rs` | `ShardedEngine` 分片薄层：K 个独立 `Engine`，`ShardRouter` 路由（默认 `HashRouter`：id % K / key fnv-1a % K），`std::thread::scope` fan-out 并行检索归并 top-k |
| `compact.rs` | COW 原子切换 + 旧段延迟回收（60s `reclaim_safe`） |
| `error.rs` | `StoreError`（thiserror） |

公开契约：`get_kv_zero_copy` 返回 mmap 段内切片，`Arc<MmapHandle>` 保证映射存活、段不可变保证指针有效；不复制字节除非显式 `to_owned_slice()`。

## 3. 持久化与崩溃恢复

- **WAL 是唯一 crash-safe 同步点**：put_kv/insert/delete 先 `wal.append + fsync` 再落子模块；mmap 段 + heed 元数据延迟刷（heed 用 `MDB_NOSYNC`）。这是单条 put_kv 达 ~183K ops/s 的原因（WAL 前 ~264 ops/s）。
- **幂等重放**：WAL 帧带 `seq`，引擎维护 `applied_seq`；`Engine::open` 自动 `build_recover_plan → replay_op → finalize_recover`（insert 重复跳过、put_kv 覆盖、delete 幂等），torn-frame（不完整尾帧）容错。
- **崩溃验收**：真实写 API → 模拟 kill -9 → `Engine::open` 自动恢复无数据丢失（`tests/test_crash_recovery.rs` 6 例）。
- **checkpoint**：图快照落单 mmap 段（H1），重开时恢复图结构；C-ABI/Python 侧 close 返 void（F-SEC-5），关键数据须先显式 checkpoint。

## 4. 并发模型

- 多进程写互斥：`fs2` flock（advisory，F-SEC-4）；`lock_exclusive` 阻塞可带超时（put_kv/delete_kv），`try_lock` 快速失败 `LockBusy`。
- 锁序不变量（E6）：`vector/store.rs` 模块文档维护全局锁序表（insert/search/compact 三路径 pool/graph/locators 加锁顺序固定）；R1 修复后 search 不再持 pool.write。
- 进程内检索并发：`/knn` 经 Semaphore 限流（R4）。
- 单 namespace = 单 Engine = 独立目录 + WAL + heed env + flock（A4）；多 namespace 由消费方开多 Engine 或用 `ShardedEngine`。

## 5. fs-serve HTTP 面

axum 路由：`/health` `/stats` `/metrics`（GET，开放；`/stats` 非环回需 ≥Readonly，F-OPS-8）、`/kv` `/vector` `/columnar` `/knn`（POST）、`/admin/compact`（POST，Admin）。中间件：RBAC token（三角色 Admin/Readwrite/Readonly，常量时间比较）→ 有界写队列（满则 503，A2）→ 16MB body limit（413）→ Prometheus 真实 metrics（E5）+ audit log（F-SEC-7）。`/health` 在 compact 期间报 `unavailable` 供上游熔断（E11），队列 80% 报 `degraded`。

## 6. FFI 边界

- `fs-ffi-c`：staticlib + cdylib，手写头 `crates/fs-ffi-c/fs_store.h` 12 符号；读路径强制拷贝到调用方预分配缓冲区（E3），不暴露 mmap 指针视图；列式不跨 C-ABI（F-API-2）。
- `fs-ffi-py`：PyO3，模块 `fusion_store`；输入 numpy f32 连续数组 `PyBuffer` 零拷贝读，输出 owned 对象；产物 arm64 wheel（tag 触发 `wheel.yml`）。
- 跨 FFI panic 安全性未评估，保留 unwind 可观测（release profile 注释）。

## 7. 容量与演进边界

| 项 | 边界 | 演进路径 |
|----|------|----------|
| 单 namespace 向量数 | ≤1-2M（单层 NSW 无分层捷径，检索 O(N^log M)） | Path A `ShardedEngine` 已落地（≤10M 量级）；Path B 分层 HNSW 核心 1.0+ 评估（ROADMAP.md） |
| LMDB map_size | KV 2GB / 列式 2GB / vec_meta 256MB 固定 | `MapFull` 自动 resize（F-PERF-4，未排期） |
| 列式类型 | 定长原语 4 种 | 按 fusion-mlx Tensor 需求扩 Int8/Float16 等（F-COL-1） |
| 跨分片原子 | 无（2PC 超薄层范围） | 消费方按业务容忍 |

## 8. 测试与质量门禁

- `cargo test --workspace --exclude fs-ffi-py`：141 Rust 测试（debug + release 双绿），含 9 例 proptest fuzz（F-TEST-2）、非环回无 token 拒启动门禁（F-SEC-1）、9 例 ShardedEngine 集成测试。
- `python crates/fs-ffi-py/tests/test_binding.py`：7 例绑定测试（需 maturin + numpy，独立于 cargo test）。
- lint：`cargo clippy --workspace --all-targets --exclude fs-ffi-py -- -D warnings`。
- 全规模基准：`crates/fs-core/examples/full_scale_bench.rs`（验收门禁 recall ≥0.95、p99 <5ms；1M×768 需 release + ~4GB RAM，~5.8h，手动/定期跑，非 CI 阻塞）。
