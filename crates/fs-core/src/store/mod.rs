//! L-STORE persistence layer — heed KV + segment pool + namespace quota [v2 E1/E6]
//!
//! KV values always land in mmap payload segments (zero-copy read pointers point into
//! the mmap region, matching the §1.4 assertion). heed stores only (key -> ValueLocator)
//! plus namespace quota counters.
//! Multi-process single writer: flock fails fast (R1).

pub mod kv;

pub use kv::KvStore;
