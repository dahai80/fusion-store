//! L-INDEX 向量索引层（HNSW 图常驻 RAM + NEON SIMD）[v2 H1/R4]

pub mod hnsw;
pub mod schema;
pub mod simd;
pub mod snapshot;
pub mod store;
