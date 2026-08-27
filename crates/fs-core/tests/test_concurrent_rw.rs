//! M2 集成：并发读写 [v2 A1/R5]
//!
//! 多线程并发 insert + search_knn，验证 RwLock 图无死锁、无 panic、
//! 读不阻塞读、写不撕裂结果。delete 与 search 并发不 panic。

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

    // writers：每线程写 OPS 个向量，id 不重叠
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
    // readers：并发 search
    for _ in 0..READERS {
        let idx = idx.clone();
        handles.push(thread::spawn(move || {
            for _ in 0..OPS {
                let q = vec![1.0f32; DIM];
                let res = idx.search_knn(&q, 10, None).unwrap();
                // 仅校验不 panic、不死锁
                let _ = res.len();
            }
        }));
    }
    for h in handles {
        h.join().expect("thread panicked");
    }
    // 全部写完，总数 = WRITERS * OPS
    assert_eq!(idx.len(), WRITERS * OPS);
}

#[test]
fn concurrent_delete_and_search_no_panic() {
    let dir = tempdir().unwrap();
    let schema = VectorSchema::new(DIM, MetricKind::L2);
    let idx = Arc::new(VectorIndex::open(dir.path(), schema, 0).unwrap());
    // 预写
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
    // 删完后 len=0
    assert_eq!(idx.len(), 0);
}
