//! L-MEM zero-copy mmap memory management layer
//!
//! Segments are append-only and immutable (H2/H4): sealed full segments become read-only
//! and never resize, eliminating SIGBUS and concurrent tearing at the root.

pub mod mmap;
pub mod recover;
pub mod segment;
pub mod wal;

pub use recover::{build_recover_plan, finalize_recover, RecoverPlan};
pub use segment::{SegmentPool, ValueLocator};
pub use wal::{CheckpointMarker, Wal, WalEntry, WalOp};
