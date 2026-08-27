//! L-STORE 持久化层 —— heed KV + 段池 + namespace 配额 [v2 E1/E6]
//!
//! KV value 一律落 mmap payload 段（零拷贝读指针落 mmap 区域，对齐 §1.4 断言）。
//! heed 只存 (key -> ValueLocator) + namespace 配额计数。
//! 多进程单写：flock 快速失败（R1）。

pub mod kv;

pub use kv::KvStore;
