# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

**Greenfield.** No source exists yet — only `.remember/`. This file documents the *intended* design so the first lines of code land on the right architecture. Canonical spec: `architecture/fusion-store-prd-0826.md` (read it before implementing).

## What fusion-store Is

Zero-copy storage & indexing engine — the **infrastructure-layer** persistence base for the Fusion ecosystem. Sits in the same layer as `fusion-core`, `fusion-mlx`, `fusion-gateway` (see `architecture/fusion-union-architecture.md`).

Unifies three storage primitives over macOS mmap + Apple Silicon unified memory, eliminating inter-process serialization:

- **KV state** — `mmap` + LMDB, crash-safe reads
- **Vector index** — HNSW (Rust-native), incremental insert, SIMD distance
- **Columnar data** — Apache Arrow, memory layout aligned with fusion-mlx Tensor buffers

Written in **Rust** for C-ABI-compatible memory layout (FFI to Swift/Python consumers).

## Business Boundary (from PRD)

In-scope: mmap block alloc/read-write, incremental HNSW vector index + persistence, multi-process concurrency locks + crash recovery.

Out-of-scope: context selection / semantic association (that's `fusion-memory`), business SQL query optimization (keep storage primitives minimal).

`fusion-memory` is the primary consumer — it owns the cognitive graph; fusion-store only holds the raw vector/KV pages. Don't let storage-layer concerns leak up into memory logic.

## Core API Contract (PRD §核心机制)

The `FusionStoreEngine` trait and `ZeroCopyBuffer` struct are the public surface — design crates around them:

```rust
pub struct ZeroCopyBuffer {
    ptr: *const u8,
    len: usize,
    mmap_handle: Arc<Mmap>,
}

pub trait FusionStoreEngine {
    fn put_kv(&self, key: &[u8], value: &[u8]) -> Result<()>;
    fn get_kv_zero_copy(&self, key: &[u8]) -> Result<Option<ZeroCopyBuffer>>;
    fn insert_vector(&self, id: u64, vector: &[f32]) -> Result<()>;
    fn search_knn(&self, query_vector: &[f32], top_k: usize) -> Result<Vec<(u64, f32)>>;
}
```

`get_kv_zero_copy` returns a slice into the mmap'd region — the `Arc<Mmap>` keeps the mapping alive. Never copy bytes out unless the caller explicitly needs owned data.

## Build & Test (establish once Cargo project exists)

Follow sibling Rust conventions (`fusion-cli`, `fusion-design`):

```bash
cargo check                         # type-check
cargo build --release               # release build
cargo test                          # unit tests (inline #[test] in src/)
cargo test <test_name>              # single test
cargo fmt --check                   # format check
cargo clippy -- -D warnings         # lint (treat warnings as errors)
```

No commands are runnable yet — create `Cargo.toml` first (see "Bootstrapping" below).

## Code Style (match `fusion-design` — the established Rust pattern in this monorepo)

- Rust edition 2021, minimum toolchain 1.94 (pin in `rust-toolchain.toml`)
- Indentation: **4 spaces** (multiples of 4 — never 5/9/11)
- **No docstrings** on functions; use `//!` module docs and inline `//` comments for intent
- Logging: `tracing` crate (`tracing::info!`, `tracing::error!`) — all runtime paths logged
- Error handling: `anyhow` for the binary/application, `thiserror` for library error types
- Serialization: `serde` + `serde_json`
- Async: `tokio` runtime if needed (storage primitives are likely sync — mmap is sync)
- Fail visibly: propagate errors, never swallow

## Suggested Crate Layout

Single crate for the engine (the PRD treats KV/vector/columnar as one engine, not a workspace). If it grows, split into a workspace mirroring `fusion-design`'s leaf-first dependency graph.

Likely modules: `mmap` (block alloc, `ZeroCopyBuffer`), `kv` (LMDB-backed `put_kv`/`get_kv_zero_copy`), `vector` (HNSW `insert_vector`/`search_knn`), `columnar` (Arrow), `lock` (multi-process concurrency + crash recovery).

## Bootstrapping (first implementation task)

1. `cargo init --lib` (or `--name fusion-store`), set `edition = "2021"` in `Cargo.toml`
2. Add `rust-toolchain.toml` pinning channel `1.94`
3. Add deps per PRD选型: `memmap2` (mmap), `lmdb`/`heed` (LMDB), `arrow` (Apache Arrow), a Rust HNSW impl (e.g. `hora` or hand-rolled), `thiserror`, `tracing`, `serde`
4. Scaffold `ZeroCopyBuffer` + `FusionStoreEngine` trait first — that's the contract the rest of the ecosystem depends on
5. Generate `README.md` after the first working module (monorepo rule: update README when code changes)

## Ecosystem Context

- **Upstream consumers**: `fusion-memory` (cognitive graph reads/writes vectors + KV here), potentially `fusion-kb`, `fusion-rag`
- **Peer**: `fusion-mlx` (Tensor buffer layout — Arrow columnar must align with it for zero-copy handoff)
- **IPC**: ecosystem services talk over Unix Domain Socket + Metal shared memory (see union architecture §1.1); fusion-store's zero-copy buffers feed that shared-memory path
- When an upstream issue blocks you: file an issue first, then a PR, then land code — don't edit other projects' code (global rule)
