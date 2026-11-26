//! L-API engine trait — external contract, aligned with PRD §1.4

use std::time::Duration;

use crate::error::Result;
use crate::mem::mmap::ZeroCopyBuffer;
use crate::vector::schema::VectorSchema;

#[cfg(feature = "columnar")]
use arrow::record_batch::RecordBatch as ArrowRecordBatch;

/// Columnar zero-copy batch — holds Arc<MmapHandle> for keepalive; immutable segments keep pointers valid [v2 H4]
#[cfg(feature = "columnar")]
pub struct ZeroCopyArrowBatch {
    pub batch: ArrowRecordBatch,
    _mmap_keepalive: std::sync::Arc<crate::mem::mmap::MmapHandle>,
}

#[cfg(feature = "columnar")]
impl ZeroCopyArrowBatch {
    pub fn new(
        batch: ArrowRecordBatch,
        keepalive: std::sync::Arc<crate::mem::mmap::MmapHandle>,
    ) -> Self {
        Self {
            batch,
            _mmap_keepalive: keepalive,
        }
    }
}

/// Engine public surface — the fusion-store external contract
///
/// timeout parameter: prevents Agent orchestration from blocking on slow operations
/// [v2 A1]; None means no timeout.
pub trait FusionStoreEngine {
    // —— KV primitives ——
    fn put_kv(&self, key: &[u8], value: &[u8], timeout: Option<Duration>) -> Result<()>;
    fn get_kv_zero_copy(
        &self,
        key: &[u8],
        timeout: Option<Duration>,
    ) -> Result<Option<ZeroCopyBuffer>>;
    fn delete_kv(&self, key: &[u8], timeout: Option<Duration>) -> Result<bool>;

    // —— Vector index lifecycle (schema locked at creation) ——
    fn create_vector_index(&self, name: &str, schema: VectorSchema) -> Result<()>;
    fn open_vector_index(&self, name: &str) -> Result<VectorSchema>;

    // —— Vector primitives ——
    // Dimension not matching schema.dim returns DimensionMismatch, never silent [v2 E2]
    fn insert_vector(&self, id: u64, vector: &[f32], timeout: Option<Duration>) -> Result<()>;
    // Batch insert: multiple Agents accumulate then commit once, reducing flock contention [v2 R1/R5]
    fn insert_vector_batch(&self, items: &[(u64, &[f32])], timeout: Option<Duration>)
        -> Result<()>;
    fn search_knn(
        &self,
        query_vector: &[f32],
        top_k: usize,
        timeout: Option<Duration>,
    ) -> Result<Vec<(u64, f32)>>;
    fn delete_vector(&self, id: u64, timeout: Option<Duration>) -> Result<bool>;

    // —— Vector read/enumeration (consumer retrieve_context fallback + reconcile audit, #3) ——
    // Missing id or soft-deleted returns None; list excludes soft-deleted
    fn get_vector(&self, id: u64, timeout: Option<Duration>) -> Result<Option<Vec<f32>>>;
    fn list_vector_ids(&self, timeout: Option<Duration>) -> Result<Vec<u64>>;

    // —— Columnar primitives (generic Arrow, no MLX involvement) [v2 H3] ——
    #[cfg(feature = "columnar")]
    fn put_columnar(
        &self,
        table_id: &str,
        batch: &ArrowRecordBatch,
        timeout: Option<Duration>,
    ) -> Result<()>;
    #[cfg(feature = "columnar")]
    fn get_columnar_zero_copy(
        &self,
        table_id: &str,
        columns: &[&str],
        timeout: Option<Duration>,
    ) -> Result<Option<ZeroCopyArrowBatch>>;

    // —— Lifecycle ——
    fn checkpoint(&self) -> Result<()>;
    fn recover(&self) -> Result<()>;
    // A6: orderly shutdown — flush active segments + heed sync + final WAL marker,
    // so a normal exit never loses data
    fn close(&self) -> Result<()>;
}
