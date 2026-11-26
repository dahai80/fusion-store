//! L-INDEX vector index layer (single-layer NSW graph resident in RAM + NEON SIMD) [v2 H1/R4]
//!
//! A2 honest naming: this layer implements a **single-layer NSW** (adjacency
//! graph, no multi-layer skip list), not a true HNSW. Search complexity is
//! approximately O(N^log M), not multi-layer HNSW's O(log N). Unbounded growth
//! of N will eventually breach the p99<5ms SLA (see the full-scale benchmark
//! section in the README for scaling limits). Names changed from
//! `HnswGraph`/`hnsw` to `NswGraph`/`nsw` to match the implementation, so we no
//! longer advertise HNSW externally.

pub mod nsw;
pub mod schema;
pub mod simd;
pub mod snapshot;
pub mod store;
