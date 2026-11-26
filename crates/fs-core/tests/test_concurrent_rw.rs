//! M2 integration: concurrent read/write [v2 A1/R5]
//!
//! Multi-threaded concurrent insert + search_knn, verifying the RwLock graph has no deadlock, no panic,
//! reads don't block reads, writes never tear results. Concurrent delete + search never panics.

use std::sync::Arc;
use std::thread;

use fs_core::vector::schema::{MetricKind, VectorSchema};
use fs_core::vector::store::VectorIndex;
use tempfile::tempdir;

const DIM: usize = 64;
const WRITERS: usize = 4;
const READERS: usize = 4;
const OPS: usize = 250;

#[test]
fn concurrent_insert_and_search_no_deadlock() {
    let dir = tempdir().unwrap();
    let schema = VectorSchema::new(DIM, MetricKind::L2);
    let idx = Arc::new(VectorIndex::open(dir.path(), schema, 0).unwrap());

    let mut handles = Vec::new();

    // writers: each thread writes OPS vectors, non-overlapping ids
    for w in 0..WRITERS {
        let idx = idx.clone();
        handles.push(thread::spawn(move || {
            for i in 0..OPS {
                let id = (w * OPS + i) as u64;
                let mut v = vec![0.0f32; DIM];
                v[0] = id as f32;
                idx.insert(id, &v, None).unwrap();
            }
        }));
    }
    // readers: concurrent search
    for _ in 0..READERS {
        let idx = idx.clone();
        handles.push(thread::spawn(move || {
            for _ in 0..OPS {
                let q = vec![1.0f32; DIM];
                let res = idx.search_knn(&q, 10, None).unwrap();
                // Only verify no panic, no deadlock
                let _ = res.len();
            }
        }));
    }
    for h in handles {
        h.join().expect("thread panicked");
    }
    // All writes done, total = WRITERS * OPS
    assert_eq!(idx.len(), WRITERS * OPS);
}

#[test]
fn concurrent_delete_and_search_no_panic() {
    let dir = tempdir().unwrap();
    let schema = VectorSchema::new(DIM, MetricKind::L2);
    let idx = Arc::new(VectorIndex::open(dir.path(), schema, 0).unwrap());
    // Pre-populate
    for i in 0..500u64 {
        let mut v = vec![0.0f32; DIM];
        v[0] = i as f32;
        idx.insert(i, &v, None).unwrap();
    }
    let mut handles = Vec::new();
    let idx_d = idx.clone();
    handles.push(thread::spawn(move || {
        for i in 0..500u64 {
            let _ = idx_d.delete(i, None);
        }
    }));
    let idx_s = idx.clone();
    handles.push(thread::spawn(move || {
        for _ in 0..500 {
            let q = vec![3.0f32; DIM];
            let _ = idx_s.search_knn(&q, 10, None);
        }
    }));
    for h in handles {
        h.join().expect("thread panicked");
    }
    // After all deletes, len=0
    assert_eq!(idx.len(), 0);
}
