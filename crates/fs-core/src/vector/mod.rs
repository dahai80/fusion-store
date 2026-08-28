//! L-INDEX 向量索引层（单层 NSW 图常驻 RAM + NEON SIMD）[v2 H1/R4]
//!
//! A2 诚实命名：本层实现是**单层 NSW**（邻接图，无多层跳表），非真 HNSW。
//! 检索复杂度近似 O(N^log M)，非多层 HNSW 的 O(log N)。N 持续增长会击穿
//! p99<5ms SLA（README 全规模基准节有扩展上限说明）。命名从 `HnswGraph`/`hnsw`
//! 改为 `NswGraph`/`nsw` 以匹配实现，不再对外宣传 HNSW。

pub mod nsw;
pub mod schema;
pub mod simd;
pub mod snapshot;
pub mod store;
