//! L-COLUMNAR 列式存储层 —— 通用 Arrow 列式 + 零拷贝读 [v2 H3/E3]
//!
//! PRD §2.6：put_columnar 写 RecordBatch 列 Buffer 到 col mmap 段（append-only），
//! 列 schema（字段名/dtype/偏移/行数）落 heed meta；get_columnar_zero_copy 读 heed schema，
//! 映射封存 col 段，经 arrow Buffer::from_custom_allocation 零拷贝重构 RecordBatch，
//! 包成 ZeroCopyArrowBatch（持 Arc<MmapHandle> 保活）。
//!
//! 范围（M3 简化）：定宽 primitive 列（Int32/Int64/Float32/Float64/Boolean），
//! 单 data buffer 每列，无 null bitmap（M3 全 non-null）。变长列远期。无 MLX 对齐（H3 降级）。

pub mod store;
pub mod types;

pub use store::ColumnarStore;
pub use types::ColType;
