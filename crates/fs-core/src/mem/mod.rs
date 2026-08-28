//! L-MEM 零拷贝 mmap 内存管理层
//!
//! 段 append-only 不可变 (H2/H4)：写满封存只读，永不改大小，根治 SIGBUS/并发撕裂。

pub mod mmap;
pub mod recover;
pub mod segment;
pub mod wal;

pub use recover::{build_recover_plan, finalize_recover, RecoverPlan};
pub use segment::{SegmentPool, ValueLocator};
pub use wal::{CheckpointMarker, Wal, WalEntry, WalOp};
