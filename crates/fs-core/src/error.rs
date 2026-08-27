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

    #[error("operation timed out")]
    Timeout,

    #[cfg(feature = "columnar")]
    #[error("arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
}
