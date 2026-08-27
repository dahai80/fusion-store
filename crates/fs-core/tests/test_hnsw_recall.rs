//! M2 集成：HNSW 召回率 + 延迟 [v2 R4/M2 验收]
//!
//! 召收率：HNSW KNN vs 暴力 KNN，top-10 重合率。
//! 延迟：单次 search_knn p99 < 5ms（PRD 降规模验证趋势，1M×768 留 M4 基准）。
//! 规模：1000×128 f32，足证图连边 + 跳转正确性。

use std::time::{Duration, Instant};

use fs_core::vector::schema::{MetricKind, VectorSchema};
use fs_core::vector::simd;
use fs_core::vector::store::VectorIndex;
use tempfile::tempdir;

const N: usize = 1000;
const DIM: usize = 128;
const TOP_K: usize = 10;

fn gen_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    // 简单确定性伪随机（不依赖 Math.random，可复现）
    let mut v = Vec::with_capacity(n);
    let mut s = seed;
    for i in 0..n {
        let mut row = Vec::with_capacity(dim);
        for j in 0..dim {
            // xorshift
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let f = ((s % 1000) as f32) / 100.0 + (i as f32 * 0.001) + (j as f32 * 0.0001);
            row.push(f);
        }
        v.push(row);
    }
    v
}

#[test]
fn hnsw_recall_vs_bruteforce_top10() {
    let dir = tempdir().unwrap();
    let schema = VectorSchema::new(DIM, MetricKind::L2);
    let idx = VectorIndex::open(dir.path(), schema, 0).unwrap();
    let vecs = gen_vectors(N, DIM, 42);
    let items: Vec<(u64, &[f32])> = vecs
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u64, v.as_slice()))
        .collect();
    idx.insert_batch(&items, None).unwrap();

    // 暴力 top-10
    let query = &vecs[123];
    let mut brute: Vec<(u64, f32)> = (0..N)
        .map(|i| (i as u64, simd::distance(MetricKind::L2, query, &vecs[i])))
        .collect();
    brute.sort_by(|a, b| a.1.total_cmp(&b.1));
    let brute_top: Vec<u64> = brute.iter().take(TOP_K).map(|(id, _)| *id).collect();

    let hnsw_res = idx.search_knn(query, TOP_K, None).unwrap();
    let hnsw_top: Vec<u64> = hnsw_res.iter().map(|(id, _)| *id).collect();

    let overlap = hnsw_top.iter().filter(|id| brute_top.contains(id)).count();
    let recall = overlap as f32 / TOP_K as f32;
    eprintln!(
        "recall@{} = {:.2} (brute={:?}, hnsw={:?})",
        TOP_K, recall, brute_top, hnsw_top
    );
    assert!(recall >= 0.90, "recall {} < 0.90", recall);
}

#[test]
fn knn_latency_p99_under_5ms() {
    let dir = tempdir().unwrap();
    let schema = VectorSchema::new(DIM, MetricKind::L2);
    let idx = VectorIndex::open(dir.path(), schema, 0).unwrap();
    let vecs = gen_vectors(N, DIM, 7);
    let items: Vec<(u64, &[f32])> = vecs
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u64, v.as_slice()))
        .collect();
    idx.insert_batch(&items, None).unwrap();

    let mut latencies = Vec::with_capacity(100);
    for q in &vecs[..100] {
        let t0 = Instant::now();
        let _ = idx
            .search_knn(q, TOP_K, Some(Duration::from_millis(5)))
            .unwrap();
        latencies.push(t0.elapsed().as_micros() as f64);
    }
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p99 = latencies[(latencies.len() as f64 * 0.99) as usize];
    eprintln!("knn p99 = {:.0}us (n={}, dim={})", p99, N, DIM);
    assert!(p99 < 5000.0, "p99 {}us >= 5000us", p99);
}
