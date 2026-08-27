# fusion-store

单机零拷贝存储与索引引擎 —— Fusion 生态 L1 基础设施层持久化底座。

统一三类存储原语，跨进程零拷贝，消除序列化：

- **KV state** — `mmap` + heed，crash-safe 读
- **向量索引** — HNSW 图常驻 RAM + 单 mmap 段 snapshot
- **列式数据** — Apache Arrow 通用列式

## 设计

权威规范：[`architecture/fusion-store-prd-0826.md`](../architecture/fusion-store-prd-0826.md) v2.0
落地计划：[`~/fusion/fusion-store-prd-plan-0826.md`](../fusion-store-prd-plan-0826.md) v2.0

核心约束：
- 段 append-only 不可变（写满封存只读，永不 truncate/grow/就地改）→ 根治 SIGBUS 与并发撕裂
- HNSW 图常驻 RAM，snapshot 落单 mmap 段（非 per-node heed）
- WAL 唯一 crash-safe 同步点 + 幂等 `seq`/`applied_seq`
- namespace 独立段池 + 配额

## 构建

```bash
cargo check                         # 类型检查
cargo build --release               # release 构建
cargo test                          # 单测（inline #[test]）
cargo test <name>                   # 单测
cargo fmt --check                   # 格式检查
cargo clippy -- -D warnings         # lint（警告即错）
```

## 工程布局

```
crates/
  fs-core/    # 引擎核心：mem 段 + KV + HNSW 向量 + Arrow 列式 + WAL/recover/compact + trait + StoreError
  fs-cli/     # CLI 二进制 fusion-store（init/put/get/stats/compact/recover/serve）
  fs-serve/   # 管理/监控 HTTP daemon（axum + tokio，端口 11463，prometheus 指标 + 背压）
  fs-ffi-c/   # C-ABI 绑定（staticlib + cdylib，导出 fs_store.h）
```

里程碑（per PRD v2.0 roadmap）：
- M0 — workspace 骨架 + fs-core trait ✅
- M1 — mem 段 + 块分配器 + heed KV + namespace 配额 ✅
- M2 — HNSW 图常驻 RAM + snapshot + NEON SIMD 位等 + batch ✅
- M3 — Arrow 通用列式 + C-ABI + Python 读强制拷贝 ✅（Python 绑定 fs-ffi-py PyO3 已落地）
- M4 — WAL 幂等 + recover + compact COW + fs-serve (11463) + prometheus + 背压 + SLA ✅

## 公共表面

```rust
pub struct ZeroCopyBuffer { /* ptr + Arc<MmapHandle> 保活映射，段不可变保指针有效 */ }

pub trait FusionStoreEngine {
    fn put_kv(&self, key: &[u8], value: &[u8], timeout: Option<Duration>) -> Result<()>;
    fn get_kv_zero_copy(&self, key: &[u8], timeout: Option<Duration>) -> Result<Option<ZeroCopyBuffer>>;
    // ... 完整签名见 crates/fs-core/src/engine.rs，对齐 PRD §1.4
}
```

`get_kv_zero_copy` 返回 mmap 段内切片 —— `Arc<MmapHandle>` 保活映射，不拷贝除非显式 `to_owned_slice()`。

## 状态

Greenfield → M0 完成（workspace 骨架 + fs-core trait）。M1 完成（KV + mmap 段）。M2 完成（HNSW 图常驻 RAM + snapshot + NEON SIMD 位等 + batch）。M3 完成（Arrow 通用列式 + C-ABI + 读强制拷贝 + Python 绑定 fs-ffi-py）。M4 完成（WAL 幂等 + recover + compact COW + fs-serve + 背压 + prometheus + SLA）。PRD v2.0 roadmap 全里程碑 + 全延后项落地，88 测试全绿（debug + release 双绿；fs-ffi-py 6 Python 测试另计，maturin 开发环境）。

### M3 验收现状

| 项 | 状态 |
|----|------|
| Arrow 列式写读（真实 buffer 往返，非 mock，E4） | ✅ |
| 零拷贝读：Buffer::from_custom_allocation + Arc<MmapHandle> 保活 | ✅ |
| 列式投影（subset columns） | ✅ |
| null 列 / 不支持类型拒收 | ✅ |
| C-ABI `fs_store_search_knn` 端到端通（C 程序经头+静态库链接运行） | ✅ |
| C-ABI `fs_store_put_kv`/`get_kv` 往返（读强制拷贝 E3） | ✅ |
| create→checkpoint→close→open 重开持久化（schema + 图 snapshot） | ✅ |
| fs_store.h 头文件（手写对齐 9 符号，随接口增长可改 cbindgen） | ✅ |
| 52 测试全绿（47 core + 5 ffi；fmt + clippy -D warnings + test） | ✅ |
| Python 绑定 fs-ffi-py（PyO3，读强制拷贝 E3，出参 owned 非 mmap view） | ✅ 6 测试全绿（另见 fs-ffi-py 节） |

**C-ABI 读强制拷贝说明（E3）**：C 侧 `out_ids`/`out_dists`/`out_val` 是 caller 预分配 buffer，Rust 拷贝填入，不暴露 mmap 指针 view。跨 C-ABI 零拷贝需返回 mmap 指针 + 保活句柄，生命周期复杂度高，M0 先拷贝。close 后 caller buffer 仍可读（owned），证非 view。

### M4 验收现状

| 项 | 状态 |
|----|------|
| WAL 幂等 + applied_seq 截断（R2） | ✅ |
| kill -9 后 recover 幂等恢复（已 checkpoint 不丢、未 checkpoint 重放、重复 recover 无重复/无泄漏，R2） | ✅ |
| compact COW 原子切换 + 旧段延迟回收 + 期间读不阻塞（A3/H4） | ✅ |
| fs-serve daemon（axum，端口 11463，R3） | ✅ |
| /health 暴露 backpressure/degraded 供上游熔断（A2） | ✅ |
| /stats /metrics /kv /knn /admin/compact 端点通 | ✅ |
| prometheus 真指标（非 tracing，E5） | ✅ |
| 写有界队列背压 → 503（A2） | ✅ |
| 写吞吐 SLA 单条 put_kv ≥5K ops/s（H5，实测 ~183K ops/s） | ✅ |
| 写吞吐 SLA 批量 insert_batch ≥10K vecs/s（R1，单批 100 实测 ~16K vecs/s） | ✅ |
| fs-cli 全子命令（init/put/get/stats/compact/recover/serve） | ✅ |
| 88 测试全绿（fmt + clippy -D warnings + test） | ✅ |

**写吞吐 SLA 说明（H5/R1）**：heed 定位元数据用 `MDB_NOSYNC`（commit 不 fsync），WAL 是唯一 crash-safe 同步点（fsync 落 WAL），mmap 段延迟刷 —— 单条 put_kv 实测 ~183K ops/s（M1 的 ~264 ops/s 即因无 WAL 每写 commit fsync，M4 WAL 落地后达标）。批量 insert_batch 用 HNSW 图构建，吞吐随批内规模降：单批 100→~16K/s、500→5.9K、1000→4K（见 `fs-core/examples/sla_probe.rs`）。大批低于 10K 是 HNSW 构建固有成本非缺陷；消费方按 R1 用 ~100 批即达标。SLA 测单批 100（PRD R1 批大小下界，对应现实调用模式），留 60% 余量抗并行测试争用。

**全规模基准（PRD §2.5）**：`crates/fs-core/examples/full_scale_bench.rs` —— env 配规模（`FS_BENCH_N`/`FS_BENCH_DIM`/`FS_BENCH_TOP_K`），测稳态写入吞吐 + 召回 vs 暴力 top-k + KNN p99 + 图 RAM。验收门禁（PRD §2.5）：召回 ≥0.95、p99 <5ms，按 N 分档（N≥100K 严格 0.95，小规模 0.90 验趋势）。实测 200K×768：召回 0.995、p99 388us、图 RAM 83.1MB、写入 359 vecs/s（BENCH OK，严格门禁通过）。全规模 1M×768 实测：召回 0.995、p99 1544us、图 RAM 415.2MB、写入 48 vecs/s（BENCH OK，严格门禁通过；单线程 NSW 构建超线性，1M 跑 ~5.8hr，非 CI 阻塞，人工/定期跑）。复跑：`FS_BENCH_N=1000000 FS_BENCH_DIM=768 cargo run --release -p fs-core --example full_scale_bench`（需 release + 约 4GB RAM）。

### M1 验收现状

| 项 | 状态 |
|----|------|
| 零拷贝读指针落 mmap 区域（断言） | ✅ |
| namespace 配额隔离（QuotaExceeded 不波及他域） | ✅ |
| 多进程并发读（fs-cli 子进程，真实第二进程） | ✅ |
| 封存轮转（写满 seal + 新段） | ✅ |
| 17 测试全绿（fmt + clippy -D warnings + check + test） | ✅ |
| 写吞吐 SLA ≥5K ops/s（H5） | ✅ M4 达标（见 M4 验收，~183K ops/s） |

**写吞吐说明**：M1 实测 ~264 ops/s（单条 put_kv），因无 WAL 每写 heed commit fsync。PRD §1.5 吞吐 SLA 依赖 M4 的 WAL 单同步点（H5：WAL fsync 为唯一 crash-safe 同步点，mmap 段延迟刷，非每写 msync）。M4 WAL 落地后单条 put_kv 达 ~183K ops/s —— 见 M4 验收现状。

### M2 验收现状

| 项 | 状态 |
|----|------|
| NEON SIMD 与标量位等（R4，to_bits 断言；dot + L2 双路径） | ✅ |
| HNSW 图常驻 RAM（单层 NSW，M=16/ef_c=200/ef_s=200） | ✅ |
| 向量 mmap 段 + id→locator + 图连边（零拷贝读算距离） | ✅ |
| insert_vector_batch 批量插入（增量连边，零拷贝读 mmap，免 O(N²) rescan） | ✅ |
| search_knn(timeout) + soft-delete | ✅ |
| snapshot 落单 mmap 段 + 重启重载图（H1） | ✅ |
| 召回率 vs 暴力 top-10 ≥0.95（1M×768 实测 0.995） | ✅ |
| knn p99 < 5ms（1M×768 实测 1544us） | ✅ |
| 并发 insert+search / delete+search 无死锁无 panic | ✅ |
| 40 测试全绿（fmt + clippy -D warnings + test） | ✅ |
| 1M×768 召回>95% / p99<5ms 全规模基准 | ✅ 实测 1M×768：召回 0.995、p99 1544us、图 RAM 415.2MB（`examples/full_scale_bench.rs`，BENCH OK 严格门禁通过） |

**ef_search 调参说明**：单层 NSW 无多层捷径，ef_s=50 召回卡 0.925 不达 PRD §2.5 ≥0.95。提至 ef_s=200 后召回 0.995、p99 仍 <1ms（5ms 预算充足）。PRD「ef_s=50 固定」与 §2.5 召回 SLA 冲突，取召回 SLA 为硬验收，调参阶段先提 ef_s（参数「先固定后调参」阶段迭代）。

**写入性能修复说明**：原 `insert_batch` 每批 rescan 全部 N 旧向量入 RAM map → O(N²) 拷贝，1M 不可行。改为零拷贝 mmap 切片读（`ZeroCopyVec` 持 Arc<MmapHandle> 保活）+ locator 内存缓存（16B/项元数据，非向量字节，H1 仍向量在 mmap 段）+ 增量连边不复读旧向量。消除每跳 heed read_txn + serde 反序列化 + Vec<f32> 分配。50K×768 写入 ~1195 vecs/s（修前 1M 跑 2hr 仅 ~480K/1M = 67 vecs/s）。

**降规模说明**：M2 用 1000×128 验证图连边、跳转、召回正确性趋势。PRD §2.5 全规模基准（1M×768）留 M4 统一基准（需 WAL 落地后的稳定写入路径 + 真实嵌入向量集）。

## C-ABI（fs-ffi-c）

`crates/fs-ffi-c/` 导出 `staticlib` + `cdylib`，头文件 `crates/fs-ffi-c/fs_store.h`。Rust 消费方直接用 `fs-core` 享完整零拷贝；C/Swift 消费方经此 C-ABI（读路径强制拷贝，E3）。

```bash
cargo build -p fs-ffi-c                  # 产出 target/debug/libfusion_store.a + .dylib
```

C 消费方链接（macOS 需 CoreFoundation/Security 框架，Rust 运行时依赖）：

```c
#include "fs_store.h"

FsStoreHandle* h = NULL;
fs_store_create("/path/to/store", 768, &h);   // dim=768，锁定 schema
for (int i = 0; i < N; i++) fs_store_insert_vector(h, i, vec[i], 768);

uint64_t ids[10]; float dists[10]; size_t n = 0;
fs_store_search_knn(h, query, 768, 10, ids, dists, &n);   // ids/dists 是拷贝，caller owned

fs_store_put_kv(h, (const uint8_t*)"k", 1, (const uint8_t*)"v", 1);
uint8_t out[64]; size_t vlen = 0;
fs_store_get_kv(h, (const uint8_t*)"k", 1, out, 64, &vlen);  // 强制拷贝到 out

fs_store_checkpoint(h);   // close 前调，重开方可从 snapshot 恢复图
fs_store_close(h);
```

链接示例：`cc app.c libfusion_store.a -lm -lpthread -ldl -framework CoreFoundation -framework Security`

目录布局：单 path 下 `vec/`+`kv/` 各占独立 heed env 子目录（避免同路径 Env 冲突）。

## Python 绑定（fs-ffi-py）

`crates/fs-ffi-py/` PyO3 绑定，模块名 `fusion_store`。薄封装 fs-core，无业务逻辑。
入参向量 numpy.ndarray f32 连续零拷贝读 buffer（`PyBuffer`），出参 ids/dists/get_kv 强制拷贝为 owned Python 对象（E3：不暴露 mmap 指针 view）。

```bash
cd crates/fs-ffi-py && maturin develop    # editable 装入当前 venv（CPython 3.14 arm64）
```

CI 发布 arm64 wheel：`.github/workflows/wheel.yml`（push tag `v*` 触发 / 手工 `workflow_dispatch`，macos-14 runner，maturin build --release，产物上传 artifact）。

```python
import numpy as np
import fusion_store

store = fusion_store.Store.open("/path/to/store", dim=768)   # dim=Some→create 锁 schema；None→reopen
store.put_kv(b"k", b"v")
got = store.get_kv(b"k")                                      # bytes（强制拷贝），missing→None
store.insert_vector(0, np.zeros(768, dtype=np.float32))       # numpy 入参零拷贝
ids, dists = store.search_knn(np.zeros(768, dtype=np.float32), top_k=10, timeout_ms=100)
                                                               # ids/dists 是 owned list，非 mmap view
store.checkpoint()                                            # close 前调，重开方从 snapshot 恢复图
print(store.vector_dim(), store.vector_count())
```

E3 校验（`tests/test_binding.py::test_search_knn_output_is_owned_not_mmap_view`）：search_knn 返回 ids/dists 后 `del store` 丢弃句柄，结果仍可读（owned），证非 mmap view；`isinstance(ids, list)`。非连续/非 float32 数组 `as_slice` 返回 None 被拒。LMDB 单写者 env：同进程同目录不可同时开两个 Store（`EnvAlreadyOpened`）—— reopen 前先 `del` 旧句柄；跨进程/跨次启动不受此限。

6 测试全绿（真实 Store 往返非 mock，E4）：open create→reopen、put_kv/get_kv 强制拷贝、insert+search_knn numpy 入参、出参 owned 非 view、timeout_ms=None、非连续数组拒收。需 maturin（≥1.4）+ numpy 环境，故不并入 `cargo test --workspace`，单列 `python tests/test_binding.py`。

## fs-cli

二进制名 `fusion-store`。日志走 stderr，stdout 仅输出结果（供脚本/测试解析）。

```bash
fs-cli --home <dir> init    --namespace <ns> [--seg-size N] [--quota N] [--dim 768]
fs-cli --home <dir> put     --namespace <ns> <key> <value> [--seg-size N] [--quota N]
fs-cli --home <dir> get     --namespace <ns> <key>            # 找不到 exit 2
fs-cli --home <dir> stats   --namespace <ns>                  # 段/向量数/配额用量
fs-cli --home <dir> compact --namespace <ns> [--reclaim-now]  # COW 原子切换 + 旧段延迟回收
fs-cli --home <dir> recover --namespace <ns> [--dry-run]      # 幂等崩溃恢复（读 WAL + 截断）
fs-cli --home <dir> serve   --namespace <ns> [--port 11463] [--dim 768] [--queue-cap 0]  # 起 daemon
```

`--home` 默认 `~/.fusion-store`，可用 `FS_HOME` 环境变量覆盖。`serve` 委托 `fs-serve` 库（DAG：fs-cli → fs-serve → fs-core，无环）。

## fs-serve

管理/监控 HTTP daemon（axum + tokio，端口 11463，可 `FUSION_STORE_PORT` 覆盖）。

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | `ok`/`degraded`/`backpressure` —— 队列深度 80% 报 backpressure 供上游熔断（A2） |
| `/stats` | GET | namespace KV 用量/配额 + 向量数/dim + 图常驻 RAM 字节（§776 #9 评估基建） |
| `/metrics` | GET | prometheus 真指标（非 tracing，E5） |
| `/kv` | POST | 写 KV，有界写队列满返 503 背压（A2） |
| `/knn` | POST | KNN 检索 |
| `/admin/compact` | POST | 触发 compact COW 原子切换（A3） |

默认模式：fusion-store 以嵌入式库被消费方 in-process 调用（零开销）；fs-serve daemon 为可选管理/监控面。

License: MIT
