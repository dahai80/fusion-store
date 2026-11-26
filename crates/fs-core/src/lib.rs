//! fusion-store core engine library — single-node zero-copy storage and indexing base
//!
//! Public surface: [`ZeroCopyBuffer`] / [`MmapHandle`] / [`VectorSchema`] /
//! the [`FusionStoreEngine`] trait / [`StoreError`].
//! See `~/fusion/fusion-store-prd-plan-0826.md` v2.0 for the detailed architecture.

pub mod columnar;
pub mod compact;
pub mod engine;
pub mod engine_impl;
pub mod error;
pub mod mem;
pub mod sharded;
pub mod store;
pub mod vector;

/// Shared constant for heed EnvOpenOptions max_dbs (F-PERF-8).
/// KV / Vector / Columnar each open their own heed env with max_dbs capped at 8;
/// named as a constant to prevent magic-number drift across the three sites
/// (lowering any one below the actual db count makes create_database panic).
pub const HEED_MAX_DBS: u32 = 8;

pub use compact::{reclaim, run_compact, CompactResult};
pub use engine::FusionStoreEngine;
pub use engine_impl::Engine;
pub use error::{Result, StoreError};
pub use mem::mmap::{MmapHandle, ZeroCopyBuffer};
pub use store::KvStore;
pub use vector::schema::{MetricKind, VectorSchema};
