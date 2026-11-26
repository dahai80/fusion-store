//! Error types — thiserror, fail visibly with no silent fallbacks

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

    // E8: heed/LMDB map_size full (MapFull). Caller decides whether to grow map_size
    // or raise an alert — not a panic. Current map_size=2GB (KV locator 12B/entry,
    // roughly 180M keys), sized for enterprise capacity headroom.
    #[error("heed map full: metadata exceeded map_size ({current} bytes, limit {limit})")]
    MapFull { current: u64, limit: u64 },

    #[error("operation timed out")]
    Timeout,

    // F3: reject payload/offset above u32::MAX to avoid silent truncation corruption
    #[error("value too large: {0} bytes exceeds u32 max")]
    ValueTooLarge(usize),

    // L5: lock poisoning (writer panicked while holding the lock) reported explicitly
    // instead of spreading via unwrap panic
    #[error("lock poisoned: writer panicked while holding lock")]
    LockPoisoned,

    // F-TEST-2: reject empty keys and keys above LMDB MDB_MAXKEYSIZE explicitly
    // instead of the cryptic BadValSize (Rule 12: fail visibly)
    #[error("invalid key: {0}")]
    InvalidKey(&'static str),

    #[cfg(feature = "columnar")]
    #[error("arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
}
