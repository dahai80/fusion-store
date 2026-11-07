---
doc_id: fusion-store/api-reference
product: fusion-store
class: core
entry: service
doc_type: api-reference
title: { zh: "fusion-store API 参考", en: "fusion-store API reference" }
audience: engineer
status: draft
version: 0.2.0-rc.4
updated: 2026-11-07
owner: "@dahai"
source_of_truth:
  - crates/fs-core/src/engine.rs
  - crates/fs-core/src/engine_impl.rs
  - crates/fs-serve/src/lib.rs
  - crates/fs-ffi-c/fs_store.h
  - crates/fs-ffi-py/src/lib.rs
---

# fusion-store API 参考（API Reference）

四种访问形态：Rust 进程内（`fs-core`）、HTTP daemon（`fs-serve`）、C-ABI（`fs-ffi-c`）、Python（`fs-ffi-py`）。零拷贝仅第一种成立；其余读路径强制拷贝（A7）。

## 1. Rust 进程内（fs-core）

依赖 `fs-core = { path = "../fusion-store/crates/fs-core" }`。核心契约在 `crates/fs-core/src/engine.rs`。

```rust
pub struct ZeroCopyBuffer { /* ptr + Arc<MmapHandle>，段不可变保证指针有效 */ }

pub trait FusionStoreEngine {
    fn put_kv(&self, key: &[u8], value: &[u8], timeout: Option<Duration>) -> Result<()>;
    fn get_kv_zero_copy(&self, key: &[u8], timeout: Option<Duration>) -> Result<Option<ZeroCopyBuffer>>;
    // insert_vector / search_knn / get_vector / list_vector_ids / delete_vector
    // 完整签名见 crates/fs-core/src/engine.rs（与 PRD §1.4 对齐）
}
```

- `get_kv_zero_copy` 返回 mmap 段内切片，`Arc<MmapHandle>` 保活；不复制字节，除非显式 `to_owned_slice()`。
- **timeout 语义分层**（F-ERR-5）：put_kv/delete_kv 实现阻塞 flock 超时（争用快速失败 `LockBusy`）；insert_vector/search_knn/get_kv_zero_copy 经段池/图锁，超时未完全实现（传 None = 无限等待）；get_vector/list_vector_ids/delete_vector 忽略 timeout（图锁在内存，无 I/O 阻塞）。
- 单 namespace = 单 `Engine` 实例（A4）：`Engine::open(&dir, Some(schema), quota)`，一个实例对应独立目录 + WAL + heed env + flock。

### ShardedEngine（fs-core::sharded）

```rust
use fs_core::sharded::ShardedEngine;
use fs_core::vector::schema::{VectorSchema, MetricKind};

let schema = VectorSchema::new(768, MetricKind::Cosine);
let eng = ShardedEngine::open(&home, 4, Some(schema), 0)?;   // 4 分片

eng.insert_vector(id, &v, None)?;        // 按 id % 4 路由，消费方透明
let results = eng.search_knn(&query, 10, None)?;  // fan-out 并行查 4 分片，归并全局 top-k
eng.put_kv(b"k", b"v", None)?;           // 按 key fnv-1a % 4 路由
eng.checkpoint()?;                        // 全分片顺序 checkpoint
eng.close()?;                             // 全分片顺序 close
```

- 路由可注入：`ShardRouter` trait，默认 `HashRouter`；注入业务键路由（如按租户固定分片）可免 fan-out。
- 无跨分片原子：`insert_vector_batch` 按 id 分摊到各分片独立 group commit，2PC 超薄层范围。
- K 增长时 fd/mmap 随之增长，macOS 系统上限由消费方控制。

### 批量与基准工具

- `insert_vector_batch`：增量连边批量插入（不做 O(N²) 重扫）；大 batch 有 NSW 构建固有成本（batch 100 ~16K vecs/s，500 → 5.9K，1000 → 4K），建议 ~100 一批（R1）。
- `crates/fs-core/examples/full_scale_bench.rs`：`FS_BENCH_N` / `FS_BENCH_DIM` / `FS_BENCH_TOP_K` 配置规模，测稳态写吞吐 + 召回 + KNN p99 + 图 RAM。
- `crates/fs-core/examples/sla_probe.rs`：批大小-吞吐曲线探针。

## 2. HTTP daemon（fs-serve，端口 11463）

| 端点 | 方法 | 请求体 | 认证 | 说明 |
|------|------|--------|------|------|
| `/health` | GET | — | 开放 | `ok` / `unavailable`（compact 期间，供上游熔断 E11）/ `degraded`（队列 80% 水位） |
| `/stats` | GET | — | 环回开放；非环回需 ≥Readonly（F-OPS-8） | namespace KV 用量/配额、向量数/维度、图常驻 RAM 字节、`vector_count` 水位 |
| `/metrics` | GET | — | 开放 | Prometheus 指标（真实 metrics，非 tracing，E5） |
| `/kv` | POST | `{"key":"...","value":"..."}` | ≥Readwrite，401/403 | 写 KV；有界写队列满返 503 背压（A2） |
| `/vector` | POST | `{"id":0,"vector":[0.0]}` | ≥Readwrite | 写向量（经写队列背压，E10） |
| `/columnar` | POST | `{"table_id":"...","ipc_base64":"..."}` | ≥Readwrite | 写列式，Arrow IPC base64（E10） |
| `/knn` | POST | `{"query":[0.0],"top_k":5}` | 开放 | KNN 检索，Semaphore 并发限流（R4）；维度须与建库 dim 一致 |
| `/admin/compact` | POST | — | Admin（R10） | 触发 compact COW 原子切换 |

错误约定：无 token 401、角色不足 403、队列满 503、请求体超 16MB 413。RBAC 三角色 `Admin`(2) ≥ `Readwrite`(1) ≥ `Readonly`(0)。

## 3. C-ABI（fs-ffi-c）

产物 `libfusion_store.a` + `.dylib`，头文件 `crates/fs-ffi-c/fs_store.h`（12 符号，手工维护）。读路径强制拷贝到调用方预分配缓冲区（E3）。

```c
#include "fs_store.h"

FsStoreHandle* h = NULL;
fs_store_create("/path/to/store", 768, &h);   // dim 锁定 schema
fs_store_insert_vector(h, i, vec[i], 768);
fs_store_search_knn(h, query, 768, 10, ids, dists, &n);   // out_ids/out_dists 调用方预分配
fs_store_put_kv(h, key, klen, val, vlen);
fs_store_get_kv(h, key, klen, out, out_cap, &vlen);       // 强制拷贝入 out
fs_store_get_vector(h, 42, vec, 768, &vlen);              // 单向量读取（强制拷贝）
fs_store_list_vector_ids(h, ids, cap, &cnt);              // 存活 id 枚举
fs_store_delete_vector(h, 42);                            // 软删除
fs_store_checkpoint(h);   // close 前显式调用（F-SEC-5：close 返 void 无错误码）
fs_store_close(h);
```

链接示例：`cc app.c libfusion_store.a -lm -lpthread -ldl -framework CoreFoundation -framework Security`（macOS 需 CoreFoundation/Security）。完整符号表以 `fs_store.h` 为准。列式不经 C-ABI 暴露（F-API-2），需列式走 HTTP `/columnar` 或 Python 绑定。

## 4. Python（fs-ffi-py，模块 fusion_store）

PyO3 薄封装，无业务逻辑。输入向量 numpy f32 连续数组（PyBuffer 零拷贝读）；输出强制拷贝为 owned Python 对象。

```python
import numpy as np
import fusion_store

store = fusion_store.Store.open("/path/to/store", dim=768)  # dim=None 重开；Some 锁 schema
store.put_kv(b"k", b"v")
store.get_kv(b"k")                    # bytes | None（强制拷贝）
store.insert_vector(0, np.zeros(768, dtype=np.float32))
ids, dists = store.search_knn(np.zeros(768, dtype=np.float32), top_k=10, timeout_ms=100)
store.delete_vector(0)                # 软删除，True=删除了存活向量
store.get_vector(0)                   # list[float] | None
store.list_vector_ids()               # list[int]，存活 id
store.checkpoint()                    # close 前调用，重开恢复图快照
store.vector_dim(), store.vector_count()
```

- 非 f32 / 非连续数组被拒绝（`as_slice` 返 None）。
- 同进程同目录不能同时开两个 Store（LMDB `EnvAlreadyOpened`）：先 `del` 旧句柄再重开；跨进程/跨运行不受影响。
- 安装：`cd crates/fs-ffi-py && maturin develop`；CI 出 arm64 wheel（tag `v*` 触发 `.github/workflows/wheel.yml`）。

## 5. CLI（fs-cli）

日志走 stderr，stdout 只输出结果；`get` 未命中退出码 2。

```bash
fs-cli --home <dir> init    --namespace <ns> [--seg-size N] [--quota N] [--dim 768]
fs-cli --home <dir> put     --namespace <ns> <key> <value> [--seg-size N] [--quota N]
fs-cli --home <dir> get     --namespace <ns> <key>
fs-cli --home <dir> stats   --namespace <ns>
fs-cli --home <dir> compact --namespace <ns> [--reclaim-now]
fs-cli --home <dir> recover --namespace <ns> [--dry-run]
fs-cli --home <dir> serve   --namespace <ns> [--port 11463] [--dim 768] [--queue-cap 0]
```

`serve` 子命令委托给 `fs-serve` 库（依赖方向 fs-cli → fs-serve → fs-core，无环）。
