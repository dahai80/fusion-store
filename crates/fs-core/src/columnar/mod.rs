//! L-COLUMNAR columnar storage layer — generic Arrow columnar + zero-copy reads [v2 H3/E3]
//!
//! PRD §2.6: put_columnar writes RecordBatch column Buffers into col mmap segments
//! (append-only), and the column schema (field name/dtype/offset/row count) lands in
//! heed meta. get_columnar_zero_copy reads the heed schema, maps the sealed col segment,
//! rebuilds the RecordBatch zero-copy via arrow Buffer::from_custom_allocation, and wraps
//! it in a ZeroCopyArrowBatch (holding Arc<MmapHandle> for keepalive).
//!
//! Scope (M3 simplification): fixed-width primitive columns
//! (Int32/Int64/Float32/Float64/Boolean), a single data buffer per column, no null
//! bitmap (M3 is all non-null). Variable-length columns are future work. No MLX
//! alignment (H3 downgrade).

pub mod store;
pub mod types;

pub use store::ColumnarStore;
pub use types::ColType;
