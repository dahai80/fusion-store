//! fusion-store 核心引擎库 —— 单机零拷贝存储与索引底座
//!
//! 公共表面：[`ZeroCopyBuffer`] / [`MmapHandle`] / [`VectorSchema`] /
//! [`FusionStoreEngine`] trait / [`StoreError`]。
//! 详细架构见 `~/fusion/fusion-store-prd-plan-0826.md` v2.0。

pub mod columnar;
pub mod compact;
pub mod engine;
pub mod engine_impl;
pub mod error;
pub mod mem;
pub mod store;
pub mod vector;

/// heed EnvOpenOptions 的 max_dbs 共享常量（F-PERF-8）。
/// KV / Vector / Columnar 三 store 各开独立 heed env，max_dbs 上限统一为 8，
/// 提名为常量防三处魔数漂移（任一处改小 → create_database 漏库 panic）。
pub const HEED_MAX_DBS: u32 = 8;

pub use compact::{reclaim, run_compact, CompactResult};
pub use engine::FusionStoreEngine;
pub use engine_impl::Engine;
pub use error::{Result, StoreError};
pub use mem::mmap::{MmapHandle, ZeroCopyBuffer};
pub use store::KvStore;
pub use vector::schema::{MetricKind, VectorSchema};
