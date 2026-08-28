//! 错误类型 —— thiserror，失败可见不兜底

use thiserror::Error;

pub type Result<T> = std::result::Result<T, StoreError>;

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("heed error: {0}")]
    Heed(#[from] heed::Error),

    #[error("serde json error: {0}")]
    SerdeJson(#[from] serde_json::Error),

    #[error("key not found")]
    NotFound,

    #[error("vector id {0} already exists")]
    DuplicateVector(u64),

    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("namespace quota exceeded")]
    QuotaExceeded,

    #[error("write backpressure: queue full")]
    Busy,

    #[error("mmap segment full")]
    SegmentFull,

    #[error("lock contention: writer held by another process")]
    LockBusy,

    #[error("corrupt: {0}")]
    Corrupt(String),

    // E8：heed/LMDB map_size 写满（MapFull）。caller 据此决定扩 map_size 或告警，
    // 非 panic。当前 map_size=2GB（KV locator 12B/项 ≈ 1.8 亿 key），企业级容量预警。
    #[error("heed map full: metadata exceeded map_size ({current} bytes, limit {limit})")]
    MapFull { current: u64, limit: u64 },

    #[error("operation timed out")]
    Timeout,

    // F3：payload/offset 超 u32::MAX 拒绝，免静默截断损坏
    #[error("value too large: {0} bytes exceeds u32 max")]
    ValueTooLarge(usize),

    // L5：锁中毒（写侧持锁 panic）显式报错，不 unwrap panic 扩散
    #[error("lock poisoned: writer panicked while holding lock")]
    LockPoisoned,

    #[cfg(feature = "columnar")]
    #[error("arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
}
