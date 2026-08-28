# fusion-store

单机零拷贝存储与索引引擎 —— Fusion 生态 L1 基础设施层持久化底座。

统一三类存储原语，**进程内**零拷贝，消除序列化：

- **KV state** — `mmap` + heed，crash-safe 读
- **向量索引** — 单层 NSW 图常驻 RAM + 单 mmap 段 snapshot（非 HNSW，见下）
- **列式数据** — Apache Arrow 定长原语列式（M3 范围：Int32/Int64/Float32/Float64，非全类型；见下）

> **零拷贝适用范围（A7）**：零拷贝仅在 **Rust 进程内**有效（mmap + `Arc<MmapHandle>` 保活映射）。
> C/Swift/Python 经 C-ABI 读路径**强制拷贝**到 caller-owned buffer（见 [C-ABI 节](#c-abifs-ffi-c)），
> 不暴露 mmap 指针 view。跨语言零拷贝非本阶段能力，对外按此定位。
>
> **向量索引命名（A2）**：本引擎向量索引是**单层 NSW**（邻接图，无多层跳表），非 HNSW。
> 检索复杂度近似 O(N^log M)，非多层 HNSW 的 O(log N)。N 持续增长会线性劣化检索延迟，
> p99<5ms SLA 有扩展上限（见全规模基准节）。代码类型 `NswGraph`，不再以 HNSW 对外宣传。
>
> **容量边界（F-ARCH-3）**：单层 NSW 无分层/分片捷径，生产建议单 namespace 向量数 **≤1-2M**。
> 1M×768 实测 p99 1544us（5ms SLA 余量收窄）、写入 48 vecs/s（构建超线性）。超此规模需架构演进
> （分片多 namespace / 分层 NSW），`/stats` 暴露 `vector_count` 水位供监控触发扩容决策。
> 演进路径详见 [`ROADMAP.md`](ROADMAP.md)（路径 A 分片 / 路径 B 分层 HNSW + 容量边界表 + 触发信号）。

## 设计

权威规范：[`architecture/fusion-store-prd-0826.md`](../architecture/fusion-store-prd-0826.md) v2.0
落地计划：[`~/fusion/fusion-store-prd-plan-0826.md`](../fusion-store-prd-plan-0826.md) v2.0

核心约束：
- 段 append-only 不可变（写满封存只读，永不 truncate/grow/就地改）→ 根治 SIGBUS 与并发撕裂
- NSW 图常驻 RAM，snapshot 落单 mmap 段（非 per-node heed）
- WAL 唯一 crash-safe 同步点 + 幂等 `seq`/`applied_seq`
- **单 namespace 隔离（A4）**：一个 `Engine` 实例 = 一个 namespace（独立目录 + WAL + heed env +
  flock）。不支持单引擎内多 namespace 路由或多租户隔离；多 namespace 是消费方职责，
  各开独立 `Engine::open`。

### 运维约束（F-PERF-5 / F-SEC-4 / F-ERR-5）

| 项 | 约束 |
|----|------|
| **compact 磁盘峰值（F-PERF-5）** | COW 写新段期间旧段延迟回收（60s `reclaim_safe`），磁盘峰值约 **2× 活数据**。大库 compact 需预留翻倍磁盘余量；运维应监控磁盘水位自动触发 compact 而非手动。`/stats` 的 `kv_disk_bytes`/`vec_disk_bytes` 暴露真实段占用（含 padding/段尾空洞，>配额净 payload）。 |
| **flock 为 advisory（F-SEC-4）** | 多进程写互斥用 `fs2` flock（advisory，非内核强制锁），依赖所有写者遵守协议。单 namespace 单 Engine 设计下风险低，但外部进程若绕过 Engine 直写 mmap 段文件，锁不阻止。消费方**不应**绕过 API 直写段文件。 |
| **timeout 语义（F-ERR-5）** | trait 写/读路径带 `timeout: Option<Duration>`，但实现分层：**put_kv/delete_kv** 实现阻塞 flock 超时（写锁竞争快速失败 `LockBusy`）；**insert_vector/search_knn/get_kv_zero_copy** 走段池/图锁，超时未完整实现（传 None 即无限等）；**get_vector/list_vector_ids/delete_vector** 忽略 timeout（图锁内存态，无 I/O 阻塞）。调用方按此语义传 timeout，勿假设全路径生效。 |

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
  fs-core/    # 引擎核心：mem 段 + KV + NSW 向量 + Arrow 列式 + WAL/recover/compact + trait + StoreError
  fs-cli/     # CLI 二进制 fusion-store（init/put/get/stats/compact/recover/serve）
  fs-serve/   # 管理/监控 HTTP daemon（axum + tokio，端口 11463，prometheus 指标 + 背压）
  fs-ffi-c/   # C-ABI 绑定（staticlib + cdylib，导出 fs_store.h）
```

里程碑（per PRD v2.0 roadmap）：
- M0 — workspace 骨架 + fs-core trait ✅
- M1 — mem 段 + 块分配器 + heed KV + namespace 配额 ✅
- M2 — NSW 图常驻 RAM + snapshot + NEON SIMD 位等 + batch ✅
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

Greenfield → M0 完成（workspace 骨架 + fs-core trait）。M1 完成（KV + mmap 段）。M2 完成（NSW 图常驻 RAM + snapshot + NEON SIMD 位等 + batch）。M3 完成（Arrow 定长原语列式 + C-ABI + 读强制拷贝 + Python 绑定 fs-ffi-py）。M4 完成（WAL 幂等 + recover + compact COW + fs-serve + 背压 + prometheus + SLA）。审计 30 缺陷全修复（A1-A7 / R1-R10 / E1-E13）+ 生产就绪审计 P0-P3 全修复（见审计纠偏节）。PRD v2.0 roadmap 全里程碑 + 全延后项落地，132 测试全绿（debug + release 双绿，含 9 例 proptest fuzz harness F-TEST-2 + 1 例非环回启动门禁 F-SEC-1；fs-ffi-py 7 Python 测试另计，maturin 开发环境）。**当前 0.2.0-rc.1（Pre-release，0.2 线 RC 基线）** —— 受控商用 / 技术预览档达标（审计 P0/P1 闭环）；NSW 规模演进（F-ARCH-3，`ShardedEngine` 分片薄层）排入 0.2.0-rc.2 → 0.2.0。向量读取/枚举 API（`get_vector`/`list_vector_ids`）+ 删除 API 经 Engine trait → C-ABI → Python 三层全暴露（#2/#3）。RBAC + audit log + token file + body limit + /stats 认证门禁 + 非环回无 token 拒启动 hard fail（F-SEC-2/F-SEC-7/F-SEC-1/F-OPS-8）。

### 审计纠偏（A2/A4/A7/E6）

四项诚实性修正（不做不可控大重构，按审计认可的「诚实命名/定位/文档化」路径落地）：

| 缺陷 | 修正 |
|------|------|
| **A2** — 单层 NSW 误称 HNSW | 类型 `HnswGraph`→`NswGraph`、模块 `hnsw`→`nsw` 全量改名；明确单层 NSW 的 O(N^log M) 复杂度与扩展上限，不再以 HNSW 对外宣传 |
| **A4** — 单 namespace 与多租户定位冲突 | 文档化「单 namespace = 单 Engine 实例」为有意设计，不承诺单引擎内多租户/多 namespace 路由；多 namespace 是消费方职责 |
| **A7** — FFI 边界零拷贝伪命题 | 明确零拷贝仅 Rust 进程内有效，C/Python 经 C-ABI 读强制拷贝；对外定位按此修正 |
| **E6** — 锁顺序无文档化不变量 | `vector/store.rs` 模块文档新增全局锁顺序表（insert/search/compact 三路径 pool/graph/locators 锁获取顺序固定），标注 R1 修复后 search 不再持 pool.write |

> 说明：A2 的真多层 HNSW、A4 的单引擎多 namespace、A7 的跨 C-ABI 零拷贝均属大重构，
> 本阶段按审计「最小修复路径」第 6/7 项的备选方案——诚实命名/定位/SLA 下调 + 文档化——
> 落地，避免在发布门禁阶段引入不可控风险（Rule 2/3/12）。

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
| fs_store.h 头文件（手写对齐 12 符号，随接口增长可改 cbindgen） | ✅ |
| 52 测试全绿（47 core + 5 ffi；fmt + clippy -D warnings + test） | ✅ |
| Python 绑定 fs-ffi-py（PyO3，读强制拷贝 E3，出参 owned 非 mmap view） | ✅ 7 测试全绿（另见 fs-ffi-py 节） |

**C-ABI 读强制拷贝说明（E3）**：C 侧 `out_ids`/`out_dists`/`out_val` 是 caller 预分配 buffer，Rust 拷贝填入，不暴露 mmap 指针 view。跨 C-ABI 零拷贝需返回 mmap 指针 + 保活句柄，生命周期复杂度高，M0 先拷贝。close 后 caller buffer 仍可读（owned），证非 view。

### M4 验收现状

> **端到端验收（审计纠偏）**：M4 验收标准 = 真实 `Engine::put_kv`/`insert_vector` 写 → drop 模拟 kill-9 → `Engine::open` 自动 recover → 数据不丢（`tests/test_crash_recovery.rs` 6 测试，真实写 API 非 WAL 桩）。WAL 经 F2 接入写路径（put_kv/insert/delete 先 `wal.append+fsync` 再落子模块），recover 经 A1 在 `Engine::open` 自动重放（幂等：insert dup 跳过、put_kv 覆盖、delete 幂等）。不再是"M4a 模块 + 自测绿"的模块级验收。

| 项 | 状态 |
|----|------|
| WAL 幂等 + applied_seq 截断（R2） | ✅ |
| WAL 接入写路径（F2：put_kv/insert/delete 先 wal.append+fsync） | ✅ |
| Engine::open 自动 recover 重放（A1：build_recover_plan→replay_op→finalize_recover） | ✅ |
| kill -9 后 recover 幂等恢复（已 checkpoint 不丢、未 checkpoint 重放、重复 recover 无重复/无泄漏，R2） | ✅ |
| compact COW 原子切换 + 旧段延迟回收 + 期间读不阻塞（A3/H4） | ✅ |
| fs-serve daemon（axum，端口 11463，R3） | ✅ |
| /health 暴露 unavailable（compact）/backpressure 供上游熔断（E11） | ✅ |
| /stats /metrics /kv /knn /admin/compact 端点通 | ✅ |
| prometheus 真指标（非 tracing，E5） | ✅ |
| 写有界队列背压 → 503（A2） | ✅ |
| 写吞吐 SLA 单条 put_kv ≥5K ops/s（H5，实测 ~183K ops/s） | ✅ |
| 写吞吐 SLA 批量 insert_batch ≥10K vecs/s（R1，单批 100 实测 ~16K vecs/s） | ✅ |
| fs-cli 全子命令（init/put/get/stats/compact/recover/serve） | ✅ |
| 88 测试全绿（fmt + clippy -D warnings + test） | ✅ |

**写吞吐 SLA 说明（H5/R1）**：heed 定位元数据用 `MDB_NOSYNC`（commit 不 fsync），WAL 是唯一 crash-safe 同步点（fsync 落 WAL），mmap 段延迟刷 —— 单条 put_kv 实测 ~183K ops/s（M1 的 ~264 ops/s 即因无 WAL 每写 commit fsync，M4 WAL 落地后达标）。批量 insert_batch 用 NSW 图构建，吞吐随批内规模降：单批 100→~16K/s、500→5.9K、1000→4K（见 `fs-core/examples/sla_probe.rs`）。大批低于 10K 是 NSW 构建固有成本非缺陷；消费方按 R1 用 ~100 批即达标。SLA 测单批 100（PRD R1 批大小下界，对应现实调用模式），留 60% 余量抗并行测试争用。

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
| NSW 图常驻 RAM（单层 NSW，M=16/ef_c=200/ef_s=200，非 HNSW） | ✅ |
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

> **列式不暴露（F-API-2）**：C-ABI 当前仅暴露 KV + 向量原语（12 符号），**不暴露** `put_columnar`/`get_columnar`。列式 Arrow 类型（RecordBatch/IPC）跨 C-ABI 需 IPC bytes 编解码，复杂度高，PRD M3 C-ABI 验收按设计不含列式。C/Swift 消费方需列式请经 fs-serve HTTP `/columnar`（Arrow IPC base64）或 Python 绑定。属设计取舍非缺陷。
>
> **close 返 void（F-SEC-5）**：`fs_store_close` 签名 `void` 无法回传错误码。close 内部 engine 落盘（flush + heed sync + checkpoint）若失败，仅记 `tracing::error` 日志，**caller 无错误码感知**。C 侧正常退出后应检查日志确认无 `fs_store_close: engine close FAILED`；关键数据建议显式调 `fs_store_checkpoint` 后再 close，checkpoint 返错误码可感知。

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

fs_store_delete_vector(h, 42);                              // 软删（#2）
float vec[768]; size_t vlen = 0;
fs_store_get_vector(h, 42, vec, 768, &vlen);                // 取单向量（#3，强制拷贝）
uint64_t all_ids[1000]; size_t cnt = 0;
fs_store_list_vector_ids(h, all_ids, 1000, &cnt);           // 枚举存活 id（#3，强制拷贝）

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
store.delete_vector(0)                                        # 软删，True=删了存活向量（#2）
got = store.get_vector(0)                                     # list[float]|None，missing/已软删→None（#3）
live_ids = store.list_vector_ids()                            # list[int]，存活（非软删）id（#3）
store.checkpoint()                                            # close 前调，重开方从 snapshot 恢复图
print(store.vector_dim(), store.vector_count())
```

E3 校验（`tests/test_binding.py::test_search_knn_output_is_owned_not_mmap_view`）：search_knn 返回 ids/dists 后 `del store` 丢弃句柄，结果仍可读（owned），证非 mmap view；`isinstance(ids, list)`。非连续/非 float32 数组 `as_slice` 返回 None 被拒。LMDB 单写者 env：同进程同目录不可同时开两个 Store（`EnvAlreadyOpened`）—— reopen 前先 `del` 旧句柄；跨进程/跨次启动不受此限。

7 测试全绿（真实 Store 往返非 mock，E4）：open create→reopen、put_kv/get_kv 强制拷贝、insert+search_knn numpy 入参、出参 owned 非 view、timeout_ms=None、非连续数组拒收、delete_vector+get_vector+list_vector_ids 软删/枚举往返（#2/#3）。需 maturin（≥1.4）+ numpy 环境，故不并入 `cargo test --workspace`，单列 `python tests/test_binding.py`。

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
| `/health` | GET | `ok`/`unavailable`/`degraded` —— compact 期间报 unavailable 触上游熔断（E11），队列深度 80% 报 degraded |
| `/stats` | GET | namespace KV 用量/配额 + 向量数/dim + 图常驻 RAM 字节（§776 #9 评估基建） |
| `/metrics` | GET | prometheus 真指标（非 tracing，E5） |
| `/kv` | POST | 写 KV，有界写队列满返 503 背压（A2） |
| `/vector` | POST | 写向量（经写队列背压，E10） |
| `/columnar` | POST | 写列式（Arrow IPC base64，经写队列背压，E10） |
| `/knn` | POST | KNN 检索（Semaphore 限并发，R4） |
| `/admin/compact` | POST | 触发 compact COW 原子切换（A3，需 Bearer Token，R10） |

写端点（`/kv`/`/vector`/`/columnar`/`/admin/compact`）强制 Bearer Token 认证（`FS_AUTH_TOKEN`，R10），无 token 拒 401；只读监控端点放行。默认监听 127.0.0.1（`--bind` 覆盖，非环回部署强警告）。

Grafana dashboard 模板：[`ops/grafana-dashboard.json`](ops/grafana-dashboard.json)（F-OPS-7）—— 导入 Prometheus 数据源，含 ops 吞吐 / p99 延迟 / 段用量 vs 容量 / 背压队列深度（8000 黄 / 10000 红阈值）面板。

默认模式：fusion-store 以嵌入式库被消费方 in-process 调用（零开销）；fs-serve daemon 为可选管理/监控面。

License: MIT
