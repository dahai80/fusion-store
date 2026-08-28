//! L-API 引擎 trait —— 对外契约，对齐 PRD §1.4

use std::time::Duration;

use crate::error::Result;
use crate::mem::mmap::ZeroCopyBuffer;
use crate::vector::schema::VectorSchema;

#[cfg(feature = "columnar")]
use arrow::record_batch::RecordBatch as ArrowRecordBatch;

/// 列式零拷贝 Batch —— 持 Arc<MmapHandle> 保活，段不可变保指针有效 [v2 H4]
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

/// 引擎公共表面 —— fusion-store 对外契约
///
/// timeout 参数：避免 Agent 编排在慢操作下阻塞 [v2 A1]，None 表示无超时。
pub trait FusionStoreEngine {
    // —— KV 原语 ——
    fn put_kv(&self, key: &[u8], value: &[u8], timeout: Option<Duration>) -> Result<()>;
    fn get_kv_zero_copy(
        &self,
        key: &[u8],
        timeout: Option<Duration>,
    ) -> Result<Option<ZeroCopyBuffer>>;
    fn delete_kv(&self, key: &[u8], timeout: Option<Duration>) -> Result<bool>;

    // —— 向量库生命周期（schema 建库时锁定）——
    fn create_vector_index(&self, name: &str, schema: VectorSchema) -> Result<()>;
    fn open_vector_index(&self, name: &str) -> Result<VectorSchema>;

    // —— 向量原语 ——
    // 维度不符 schema.dim → DimensionMismatch，不静默 [v2 E2]
    fn insert_vector(&self, id: u64, vector: &[f32], timeout: Option<Duration>) -> Result<()>;
    // 批量插入：多 Agent 攒批后单次提交，减少 flock 竞争 [v2 R1/R5]
    fn insert_vector_batch(&self, items: &[(u64, &[f32])], timeout: Option<Duration>)
        -> Result<()>;
    fn search_knn(
        &self,
        query_vector: &[f32],
        top_k: usize,
        timeout: Option<Duration>,
    ) -> Result<Vec<(u64, f32)>>;
    fn delete_vector(&self, id: u64, timeout: Option<Duration>) -> Result<bool>;

    // —— 向量读取/枚举（消费方 retrieve_context 兜底 + reconcile 审计，#3）——
    // id 不存在或已软删 → None；list 排除软删
    fn get_vector(&self, id: u64, timeout: Option<Duration>) -> Result<Option<Vec<f32>>>;
    fn list_vector_ids(&self, timeout: Option<Duration>) -> Result<Vec<u64>>;

    // —— 列式原语（通用 Arrow，不涉 MLX）[v2 H3] ——
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

    // —— 生命周期 ——
    fn checkpoint(&self) -> Result<()>;
    fn recover(&self) -> Result<()>;
    // A6：有序关闭——flush active 段 + heed sync + WAL 最终 marker，免正常退出丢数据
    fn close(&self) -> Result<()>;
}
