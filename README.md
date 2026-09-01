# fusion-store

> **Language**: **English** | [中文](README_CN.md)

Single-machine zero-copy storage and indexing engine — the L1 infrastructure-layer persistence base for the Fusion ecosystem.

Unifies three storage primitives, zero-copy **in-process**, eliminating serialization:

- **KV state** — `mmap` + heed, crash-safe reads
- **Vector index** — single-layer NSW graph resident in RAM + single mmap-segment snapshot (NOT HNSW, see below)
- **Columnar data** — Apache Arrow fixed-width primitive columns (M3 scope: Int32/Int64/Float32/Float64, not full types; see below)

> **Zero-copy scope (A7)**: zero-copy is only valid **inside the Rust process** (mmap + `Arc<MmapHandle>` keeps the mapping alive).
> C/Swift/Python via the C-ABI read path **force a copy** into a caller-owned buffer (see [C-ABI section](#c-abi-fs-ffi-c)),
> no mmap pointer view is exposed. Cross-language zero-copy is not a capability at this stage; this is the public positioning.
>
> **Vector index naming (A2)**: this engine's vector index is a **single-layer NSW** (proximity graph, no multi-layer skip list), NOT HNSW.
> Search complexity is approximately O(N^log M), not the multi-layer HNSW O(log N). Continuous N growth linearly degrades search latency,
> so the p99<5ms SLA has a scaling ceiling (see full-scale benchmark section). The code type is `NswGraph`; it is no longer advertised as HNSW.
>
> **Capacity boundary (F-ARCH-3)**: single-layer NSW has no tiering/sharding shortcut; production recommends **≤1-2M** vectors per namespace.
> 1M×768 measured p99 1544us (5ms SLA margin narrowing), write 48 vecs/s (super-linear construction). Beyond this scale requires architectural evolution
> (sharded multi-namespace / tiered NSW); `/stats` exposes the `vector_count` watermark for monitoring-driven scale-out decisions.
> Evolution paths are detailed in [`ROADMAP.md`](ROADMAP.md) (Path A sharding / Path B tiered HNSW + capacity boundary table + trigger signals).

## Design

Authoritative spec: [`architecture/fusion-store-prd-0826.md`](../architecture/fusion-store-prd-0826.md) v2.0
Landing plan: [`~/fusion/fusion-store-prd-plan-0826.md`](../fusion-store-prd-plan-0826.md) v2.0

Core constraints:
- Segment append-only immutable (sealed read-only when full, never truncate/grow/in-place mutate) → eradicates SIGBUS and concurrent tearing
- NSW graph resident in RAM, snapshot in a single mmap segment (not per-node heed)
- WAL is the sole crash-safe sync point + idempotent `seq`/`applied_seq`
- **Single-namespace isolation (A4)**: one `Engine` instance = one namespace (independent dir + WAL + heed env + flock).
  No in-engine multi-namespace routing or multi-tenant isolation; multi-namespace is the consumer's responsibility,
  each opening an independent `Engine::open`.

### Operations constraints (F-PERF-5 / F-SEC-4 / F-ERR-5)

| Item | Constraint |
|------|------------|
| **compact disk peak (F-PERF-5)** | During COW new-segment writes, old segments are reclaimed with delay (60s `reclaim_safe`); disk peak is about **2× live data**. Large-store compact needs reserved doubled disk headroom; ops should monitor disk watermark to auto-trigger compact rather than manual. `/stats` `kv_disk_bytes`/`vec_disk_bytes` expose true segment occupancy (incl. padding/segment-tail holes, > quota net payload). |
| **flock is advisory (F-SEC-4)** | Multi-process write mutual exclusion uses `fs2` flock (advisory, not kernel-enforced lock), relying on all writers honoring the protocol. Risk is low under the single-namespace single-Engine design, but an external process bypassing the Engine to write mmap segment files directly is not blocked by the lock. Consumers **should not** bypass the API to write segment files directly. |
| **timeout semantics (F-ERR-5)** | The trait read/write paths carry `timeout: Option<Duration>`, but the implementation is layered: **put_kv/delete_kv** implement blocking flock timeout (write-lock contention fast-fails with `LockBusy`); **insert_vector/search_knn/get_kv_zero_copy** go through the segment-pool/graph lock, timeout not fully implemented (pass None = wait indefinitely); **get_vector/list_vector_ids/delete_vector** ignore timeout (graph lock is in-memory, no I/O blocking). Callers should pass timeout per this semantics; do not assume it is effective on all paths. |

## Build

```bash
cargo check                         # type check
cargo build --release               # release build
cargo test                          # unit tests (inline #[test])
cargo test <name>                   # single test
cargo fmt --check                   # format check
cargo clippy -- -D warnings         # lint (warnings = errors)
```

## Layout

```
crates/
  fs-core/    # engine core: mem segments + KV + NSW vectors + Arrow columnar + WAL/recover/compact + trait + StoreError + ShardedEngine sharding thin layer
  fs-cli/     # CLI binary fusion-store (init/put/get/stats/compact/recover/serve)
  fs-serve/   # management/monitoring HTTP daemon (axum + tokio, port 11463, prometheus metrics + backpressure)
  fs-ffi-c/   # C-ABI bindings (staticlib + cdylib, exports fs_store.h)
```

Milestones (per PRD v2.0 roadmap):
- M0 — workspace skeleton + fs-core trait ✅
- M1 — mem segments + block allocator + heed KV + namespace quota ✅
- M2 — NSW graph resident in RAM + snapshot + NEON SIMD bit-equivalence + batch ✅
- M3 — Arrow generic columnar + C-ABI + Python forced-copy read ✅ (Python binding fs-ffi-py PyO3 landed)
- M4 — WAL idempotency + recover + compact COW + fs-serve (11463) + prometheus + backpressure + SLA ✅

## Public surface

```rust
pub struct ZeroCopyBuffer { /* ptr + Arc<MmapHandle> keeps mapping alive, segment immutability keeps pointer valid */ }

pub trait FusionStoreEngine {
    fn put_kv(&self, key: &[u8], value: &[u8], timeout: Option<Duration>) -> Result<()>;
    fn get_kv_zero_copy(&self, key: &[u8], timeout: Option<Duration>) -> Result<Option<ZeroCopyBuffer>>;
    // ... full signature in crates/fs-core/src/engine.rs, aligned with PRD §1.4
}
```

`get_kv_zero_copy` returns a slice inside the mmap segment — `Arc<MmapHandle>` keeps the mapping alive; no copy unless an explicit `to_owned_slice()`.

## Status

Greenfield → M0 done (workspace skeleton + fs-core trait). M1 done (KV + mmap segments). M2 done (NSW graph resident in RAM + snapshot + NEON SIMD bit-equivalence + batch). M3 done (Arrow fixed-width primitive columnar + C-ABI + forced-copy read + Python binding fs-ffi-py). M4 done (WAL idempotency + recover + compact COW + fs-serve + backpressure + prometheus + SLA). Audit 30 defects all fixed (A1-A7 / R1-R10 / E1-E13) + production-readiness audit P0-P3 all fixed (see audit-correction section). PRD v2.0 roadmap all milestones + all deferred items landed, 141 tests green (debug + release double-green, incl. 9 proptest fuzz harness F-TEST-2 + 1 non-loopback startup hard-fail gate F-SEC-1 + 9 ShardedEngine sharding tests; fs-ffi-py 7 Python tests counted separately, maturin dev env). **Currently 0.2.0-rc.2 (Pre-release, 0.2-line RC)** — controlled commercial / tech-preview tier met (audit P0/P1 closed); NSW scaling evolution Path A `ShardedEngine` sharding thin layer landed (see section below, rc.2 candidate), next step 0.2.0 stable. Vector read/enumerate APIs (`get_vector`/`list_vector_ids`) + delete API exposed through the three layers Engine trait → C-ABI → Python (#2/#3). RBAC + audit log + token file + body limit + /stats auth gate + non-loopback no-token startup hard-fail (F-SEC-2/F-SEC-7/F-SEC-1/F-OPS-8).

### Audit corrections (A2/A4/A7/E6)

Four honesty corrections (no uncontrollable large refactor; landed via the audit-approved "honest naming/positioning/documentation" path):

| Defect | Correction |
|--------|------------|
| **A2** — single-layer NSW misnamed HNSW | Type `HnswGraph`→`NswGraph`, module `hnsw`→`nsw` full rename; clarified single-layer NSW O(N^log M) complexity and scaling ceiling; no longer advertised as HNSW |
| **A4** — single-namespace vs multi-tenant positioning conflict | Documented "single namespace = single Engine instance" as intentional design; no in-engine multi-tenant/multi-namespace routing promised; multi-namespace is the consumer's responsibility |
| **A7** — FFI-boundary zero-copy false premise | Clarified zero-copy is only valid in the Rust process; C/Python read via C-ABI forces copy; public positioning corrected accordingly |
| **E6** — undocumented lock-order invariant | `vector/store.rs` module docs added a global lock-order table (insert/search/compact three-path pool/graph/locators lock acquisition order fixed), noting that after the R1 fix search no longer holds pool.write |

> Note: A2's true multi-layer HNSW, A4's in-engine multi-namespace, and A7's cross-C-ABI zero-copy are all large refactors.
> This stage landed them via the audit "minimal fix path" items 6/7 alternatives — honest naming/positioning/SLA downgrade + documentation —
> to avoid introducing uncontrollable risk at the release-gate stage (Rule 2/3/12).

### M3 acceptance status

| Item | Status |
|------|--------|
| Arrow columnar write/read (real buffer roundtrip, not mock, E4) | ✅ |
| Zero-copy read: Buffer::from_custom_allocation + Arc<MmapHandle> keepalive | ✅ |
| Columnar projection (subset columns) | ✅ |
| null column / unsupported type rejected | ✅ |
| C-ABI `fs_store_search_knn` end-to-end (C program links header + static lib and runs) | ✅ |
| C-ABI `fs_store_put_kv`/`get_kv` roundtrip (forced-copy read E3) | ✅ |
| create→checkpoint→close→open reopen persistence (schema + graph snapshot) | ✅ |
| fs_store.h header (hand-written aligned 12 symbols; switch to cbindgen as the interface grows) | ✅ |
| 52 tests green (47 core + 5 ffi; fmt + clippy -D warnings + test) | ✅ |
| Python binding fs-ffi-py (PyO3, forced-copy read E3, output owned not mmap view) | ✅ 7 tests green (see fs-ffi-py section) |

**C-ABI forced-copy read note (E3)**: On the C side `out_ids`/`out_dists`/`out_val` are caller-preallocated buffers; Rust copies into them and does not expose an mmap pointer view. Cross-C-ABI zero-copy would need to return an mmap pointer + keepalive handle, with high lifecycle complexity; M0 copies first. After close, the caller buffer is still readable (owned), proving it is not a view.

### M4 acceptance status

> **End-to-end acceptance (audit correction)**: M4 acceptance criteria = real `Engine::put_kv`/`insert_vector` writes → drop simulating kill -9 → `Engine::open` auto-recovers → no data loss (`tests/test_crash_recovery.rs` 6 tests, real write APIs not WAL stubs). WAL is wired into the write path via F2 (put_kv/insert/delete do `wal.append+fsync` before landing in the submodule); recover auto-replays via A1 in `Engine::open` (idempotent: insert dup skipped, put_kv overwrites, delete idempotent). No longer a "M4a module + self-test green" module-level acceptance.

| Item | Status |
|------|--------|
| WAL idempotency + applied_seq truncation (R2) | ✅ |
| WAL wired into write path (F2: put_kv/insert/delete do wal.append+fsync first) | ✅ |
| Engine::open auto-recover replay (A1: build_recover_plan→replay_op→finalize_recover) | ✅ |
| Recover idempotent after kill -9 (checkpointed not lost, uncheckpointed replayed, repeated recover no dup/no leak, R2) | ✅ |
| compact COW atomic switch + old-segment delayed reclaim + reads not blocked during (A3/H4) | ✅ |
| fs-serve daemon (axum, port 11463, R3) | ✅ |
| /health exposes unavailable (compact)/backpressure for upstream circuit-breaking (E11) | ✅ |
| /stats /metrics /kv /knn /admin/compact endpoints working | ✅ |
| prometheus real metrics (not tracing, E5) | ✅ |
| Bounded write queue backpressure → 503 (A2) | ✅ |
| Write-throughput SLA single put_kv ≥5K ops/s (H5, measured ~183K ops/s) | ✅ |
| Write-throughput SLA batch insert_batch ≥10K vecs/s (R1, batch 100 measured ~16K vecs/s) | ✅ |
| fs-cli all subcommands (init/put/get/stats/compact/recover/serve) | ✅ |
| 88 tests green (fmt + clippy -D warnings + test) | ✅ |

**Write-throughput SLA note (H5/R1)**: heed uses `MDB_NOSYNC` for metadata (commit does not fsync); the WAL is the sole crash-safe sync point (fsync to WAL), mmap segments flush lazily — single put_kv measured ~183K ops/s (M1's ~264 ops/s was due to no WAL, every write commit fsync; after M4 WAL landed it meets the SLA). Batch insert_batch uses NSW graph construction; throughput drops with in-batch size: batch 100→~16K/s, 500→5.9K, 1000→4K (see `fs-core/examples/sla_probe.rs`). Large batches below 10K is inherent NSW construction cost, not a defect; consumers follow R1 with ~100 batches to meet the SLA. SLA tested at batch 100 (PRD R1 batch-size lower bound, matching realistic call patterns), leaving 60% headroom against parallel test contention.

**Full-scale benchmark (PRD §2.5)**: `crates/fs-core/examples/full_scale_bench.rs` — env-configurable size (`FS_BENCH_N`/`FS_BENCH_DIM`/`FS_BENCH_TOP_K`), measures steady-state write throughput + recall vs brute-force top-k + KNN p99 + graph RAM. Acceptance gate (PRD §2.5): recall ≥0.95, p99 <5ms, tiered by N (N≥100K strict 0.95, small scale 0.90 to verify trend). Measured 200K×768: recall 0.995, p99 388us, graph RAM 83.1MB, write 359 vecs/s (BENCH OK, strict gate passed). Full-scale 1M×768 measured: recall 0.995, p99 1544us, graph RAM 415.2MB, write 48 vecs/s (BENCH OK, strict gate passed; single-thread NSW construction is super-linear, 1M takes ~5.8hr, not a CI blocker, run manually/periodically). Re-run: `FS_BENCH_N=1000000 FS_BENCH_DIM=768 cargo run --release -p fs-core --example full_scale_bench` (needs release + ~4GB RAM).

### M1 acceptance status

| Item | Status |
|------|--------|
| Zero-copy read pointer lands in mmap region (asserted) | ✅ |
| Namespace quota isolation (QuotaExceeded does not spill to other domains) | ✅ |
| Multi-process concurrent read (fs-cli subprocess, real second process) | ✅ |
| Seal rotation (full → seal + new segment) | ✅ |
| 17 tests green (fmt + clippy -D warnings + check + test) | ✅ |
| Write-throughput SLA ≥5K ops/s (H5) | ✅ met at M4 (see M4 acceptance, ~183K ops/s) |

**Write-throughput note**: M1 measured ~264 ops/s (single put_kv), because without WAL every write does a heed commit fsync. PRD §1.5 throughput SLA depends on M4's WAL single sync point (H5: WAL fsync is the sole crash-safe sync point, mmap segments flush lazily, not msync per write). After M4 WAL landed, single put_kv reaches ~183K ops/s — see M4 acceptance status.

### M2 acceptance status

| Item | Status |
|------|--------|
| NEON SIMD bit-equivalent to scalar (R4, to_bits assert; dot + L2 dual path) | ✅ |
| NSW graph resident in RAM (single-layer NSW, M=16/ef_c=200/ef_s=200, not HNSW) | ✅ |
| Vector mmap segment + id→locator + graph edges (zero-copy read for distance) | ✅ |
| insert_vector_batch batch insert (incremental edge-linking, zero-copy mmap read, no O(N²) rescan) | ✅ |
| search_knn(timeout) + soft-delete | ✅ |
| Snapshot to single mmap segment + graph reload on restart (H1) | ✅ |
| Recall vs brute-force top-10 ≥0.95 (1M×768 measured 0.995) | ✅ |
| knn p99 < 5ms (1M×768 measured 1544us) | ✅ |
| Concurrent insert+search / delete+search no deadlock no panic | ✅ |
| 40 tests green (fmt + clippy -D warnings + test) | ✅ |
| 1M×768 recall>95% / p99<5ms full-scale benchmark | ✅ measured 1M×768: recall 0.995, p99 1544us, graph RAM 415.2MB (`examples/full_scale_bench.rs`, BENCH OK strict gate passed) |

**ef_search tuning note**: Single-layer NSW has no multi-layer shortcut; ef_s=50 recall stuck at 0.925, not meeting PRD §2.5 ≥0.95. Raising to ef_s=200 gives recall 0.995, p99 still <1ms (5ms budget ample). PRD "ef_s=50 fixed" conflicts with §2.5 recall SLA; take the recall SLA as hard acceptance, tune ef_s first at the tuning stage ("fix then tune" iterative stage).

**Write-performance fix note**: The original `insert_batch` rescanned all N old vectors into a RAM map per batch → O(N²) copy, infeasible at 1M. Changed to zero-copy mmap slice read (`ZeroCopyVec` holds Arc<MmapHandle> keepalive) + locator in-memory cache (16B/item metadata, not vector bytes; H1 still vectors in mmap segment) + incremental edge-linking without re-reading old vectors. Eliminated per-hop heed read_txn + serde deserialization + Vec<f32> allocation. 50K×768 write ~1195 vecs/s (before the fix 1M ran 2hr to only ~480K/1M = 67 vecs/s).

**Downscale note**: M2 used 1000×128 to verify graph edge-linking, hopping, recall-correctness trends. PRD §2.5 full-scale benchmark (1M×768) deferred to a unified M4 benchmark (needs a stable write path after WAL landed + a real embedding set).

## ShardedEngine (NSW scaling evolution Path A)

Single-layer NSW single-namespace soft ceiling ≤1-2M vectors. `ShardedEngine` (`fs-core::sharded`) adds a thin sharding layer above `Engine` without changing the `NswGraph` core: one logical store is split into K namespaces (each an independent `Engine::open` dir + WAL + flock), vectors are routed by id shard, and search fans out in parallel across K shards then merges the global top-k.

```rust
use fs_core::sharded::ShardedEngine;
use fs_core::vector::schema::{VectorSchema, MetricKind};

// 4 shards, each builds a vector index with the same schema
let schema = VectorSchema::new(768, MetricKind::Cosine);
let eng = ShardedEngine::open(&home, 4, Some(schema), 0)?;

// Vectors routed by id % 4, transparent to the consumer
for (id, v) in vectors { eng.insert_vector(id, &v, None)?; }
// Fan-out parallel search across 4 shards, merge global top-10
let results = eng.search_knn(&query, 10, None)?;
// KV routed by key fnv hash % 4
eng.put_kv(b"k", b"v", None)?;

eng.checkpoint()?;   // sequential checkpoint across all shards
eng.close()?;        // sequential close across all shards
```

- **Injectable routing**: `ShardRouter` trait, default `HashRouter` (id % K / key fnv-1a % K). Injecting a business-key router (e.g. fixed shard by tenant) can skip fan-out.
- **Fan-out parallelism**: `std::thread::scope` searches K shards in parallel, merges by sort and takes the top_k; latency = max(shard latency), not sum.
- **Full impl of `FusionStoreEngine`**: transparent to the consumer; one handle operates the entire sharded store. checkpoint/recover/close called sequentially across all shards.
- **No cross-shard atomicity**: `insert_vector_batch` is bucketed by id into per-shard independent group commits; cross-shard 2PC is beyond the thin-layer scope (Rule 2), consumers tolerate per business.
- **A4 stance unchanged**: a single `Engine` is still a single namespace; ShardedEngine is consumer-side multi-Engine orchestration, not in-engine multi-namespace routing. As K grows, fd/mmap grow accordingly; hitting the macOS system limit is controlled by the consumer.

Capacity-evolution decisions are in [`ROADMAP.md`](ROADMAP.md) (Path A sharding ≤10M / Path B tiered HNSW single-shard 5M+).

## C-ABI (fs-ffi-c)

`crates/fs-ffi-c/` exports `staticlib` + `cdylib`, header `crates/fs-ffi-c/fs_store.h`. Rust consumers use `fs-core` directly for full zero-copy; C/Swift consumers go through this C-ABI (read path forces a copy, E3).

> **Columnar not exposed (F-API-2)**: the C-ABI currently exposes only KV + vector primitives (12 symbols), **not** `put_columnar`/`get_columnar`. Columnar Arrow types (RecordBatch/IPC) across the C-ABI need IPC-bytes encoding/decoding with high complexity; PRD M3 C-ABI acceptance by design does not include columnar. C/Swift consumers needing columnar should go through fs-serve HTTP `/columnar` (Arrow IPC base64) or the Python binding. This is a design tradeoff, not a defect.
>
> **close returns void (F-SEC-5)**: `fs_store_close` has signature `void` and cannot return an error code. If the internal engine flush (flush + heed sync + checkpoint) fails on close, it only logs `tracing::error`, and the **caller has no error-code awareness**. After normal exit the C side should check the logs to confirm no `fs_store_close: engine close FAILED`; for critical data, explicitly call `fs_store_checkpoint` before close — checkpoint returns an error code that can be observed.

```bash
cargo build -p fs-ffi-c                  # produces target/debug/libfusion_store.a + .dylib
```

C consumer linking (macOS needs CoreFoundation/Security frameworks, Rust runtime dependency):

```c
#include "fs_store.h"

FsStoreHandle* h = NULL;
fs_store_create("/path/to/store", 768, &h);   // dim=768, locks the schema
for (int i = 0; i < N; i++) fs_store_insert_vector(h, i, vec[i], 768);

uint64_t ids[10]; float dists[10]; size_t n = 0;
fs_store_search_knn(h, query, 768, 10, ids, dists, &n);   // ids/dists are copies, caller owned

fs_store_put_kv(h, (const uint8_t*)"k", 1, (const uint8_t*)"v", 1);
uint8_t out[64]; size_t vlen = 0;
fs_store_get_kv(h, (const uint8_t*)"k", 1, out, 64, &vlen);  // forced copy into out

fs_store_delete_vector(h, 42);                              // soft delete (#2)
float vec[768]; size_t vlen = 0;
fs_store_get_vector(h, 42, vec, 768, &vlen);                // fetch single vector (#3, forced copy)
uint64_t all_ids[1000]; size_t cnt = 0;
fs_store_list_vector_ids(h, all_ids, 1000, &cnt);           // enumerate live ids (#3, forced copy)

fs_store_checkpoint(h);   // call before close, so reopen can restore the graph from snapshot
fs_store_close(h);
```

Link example: `cc app.c libfusion_store.a -lm -lpthread -ldl -framework CoreFoundation -framework Security`

Directory layout: under a single path, `vec/` + `kv/` occupy independent heed env subdirectories (to avoid Env conflicts at the same path).

## Python binding (fs-ffi-py)

`crates/fs-ffi-py/` PyO3 binding, module name `fusion_store`. Thin wrapper over fs-core, no business logic.
Input vectors are numpy.ndarray f32 contiguous zero-copy read buffer (`PyBuffer`); output ids/dists/get_kv are forced-copied into owned Python objects (E3: no mmap pointer view exposed).

```bash
cd crates/fs-ffi-py && maturin develop    # editable install into the current venv (CPython 3.14 arm64)
```

CI publishes arm64 wheels: `.github/workflows/wheel.yml` (push tag `v*` triggers / manual `workflow_dispatch`, macos-14 runner, maturin build --release, artifact uploaded).

```python
import numpy as np
import fusion_store

store = fusion_store.Store.open("/path/to/store", dim=768)   # dim=Some→create locks schema; None→reopen
store.put_kv(b"k", b"v")
got = store.get_kv(b"k")                                      # bytes (forced copy), missing→None
store.insert_vector(0, np.zeros(768, dtype=np.float32))       # numpy input zero-copy
ids, dists = store.search_knn(np.zeros(768, dtype=np.float32), top_k=10, timeout_ms=100)
                                                               # ids/dists are an owned list, not an mmap view
store.delete_vector(0)                                        # soft delete, True=deleted a live vector (#2)
got = store.get_vector(0)                                     # list[float]|None, missing/soft-deleted→None (#3)
live_ids = store.list_vector_ids()                            # list[int], live (non-soft-deleted) ids (#3)
store.checkpoint()                                            # call before close, so reopen restores the graph from snapshot
print(store.vector_dim(), store.vector_count())
```

E3 verification (`tests/test_binding.py::test_search_knn_output_is_owned_not_mmap_view`): after search_knn returns ids/dists, `del store` drops the handle and the result is still readable (owned), proving it is not an mmap view; `isinstance(ids, list)`. Non-contiguous/non-float32 arrays return None from `as_slice` and are rejected. LMDB single-writer env: two Stores cannot be opened at the same directory in the same process (`EnvAlreadyOpened`) — `del` the old handle before reopen; cross-process/cross-run is not affected.

7 tests green (real Store roundtrip, not mock, E4): open create→reopen, put_kv/get_kv forced copy, insert+search_knn numpy input, output owned not view, timeout_ms=None, non-contiguous array rejected, delete_vector+get_vector+list_vector_ids soft-delete/enumerate roundtrip (#2/#3). Requires maturin (≥1.4) + numpy env, so not merged into `cargo test --workspace`; listed separately as `python tests/test_binding.py`.

## fs-cli

Binary name `fusion-store`. Logs go to stderr; stdout emits only results (for script/test parsing).

```bash
fs-cli --home <dir> init    --namespace <ns> [--seg-size N] [--quota N] [--dim 768]
fs-cli --home <dir> put     --namespace <ns> <key> <value> [--seg-size N] [--quota N]
fs-cli --home <dir> get     --namespace <ns> <key>            # not found exit 2
fs-cli --home <dir> stats   --namespace <ns>                  # segments/vector count/quota usage
fs-cli --home <dir> compact --namespace <ns> [--reclaim-now]  # COW atomic switch + old-segment delayed reclaim
fs-cli --home <dir> recover --namespace <ns> [--dry-run]      # idempotent crash recovery (read WAL + truncate)
fs-cli --home <dir> serve   --namespace <ns> [--port 11463] [--dim 768] [--queue-cap 0]  # start daemon
```

`--home` defaults to `~/.fusion-store`, overridable by the `FS_HOME` env var. `serve` delegates to the `fs-serve` library (DAG: fs-cli → fs-serve → fs-core, acyclic).

## fs-serve

Management/monitoring HTTP daemon (axum + tokio, port 11463, overridable by `FUSION_STORE_PORT`).

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | `ok`/`unavailable`/`degraded` — reports unavailable during compact to trigger upstream circuit-breaking (E11), degraded at 80% queue depth |
| `/stats` | GET | namespace KV usage/quota + vector count/dim + graph-resident RAM bytes (§776 #9 evaluation infra) |
| `/metrics` | GET | prometheus real metrics (not tracing, E5) |
| `/kv` | POST | write KV, bounded write queue full returns 503 backpressure (A2) |
| `/vector` | POST | write vector (via write-queue backpressure, E10) |
| `/columnar` | POST | write columnar (Arrow IPC base64, via write-queue backpressure, E10) |
| `/knn` | POST | KNN search (Semaphore concurrency limit, R4) |
| `/admin/compact` | POST | trigger compact COW atomic switch (A3, requires Bearer Token, R10) |

Write endpoints (`/kv`/`/vector`/`/columnar`/`/admin/compact`) enforce Bearer Token auth (`FS_AUTH_TOKEN`, R10), reject 401 without token; read-only monitoring endpoints are open. Listens on 127.0.0.1 by default (overridable with `--bind`, non-loopback deployment strongly warned).

Grafana dashboard template: [`ops/grafana-dashboard.json`](ops/grafana-dashboard.json) (F-OPS-7) — import the Prometheus datasource, with ops throughput / p99 latency / segment usage vs capacity / backpressure queue depth (8000 yellow / 10000 red thresholds) panels.

Default mode: fusion-store is consumed in-process as an embedded library (zero overhead); the fs-serve daemon is an optional management/monitoring surface.

License: Apache-2.0
