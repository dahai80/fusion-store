//! M2 integration: NSW recall + latency [v2 R4/M2 acceptance]
//!
//! Recall: NSW KNN vs brute-force KNN, top-10 overlap rate.
//! Latency: single search_knn p99 < 5ms (PRD reduced-scale trend check; 1M×768 left to the M4 benchmark).
//! Scale: 1000×128 f32, sufficient to prove graph linking + traversal correctness.
//! A2: this engine's vector index is a single-layer NSW (not HNSW). The latency SLA is a release metric; debug builds
//! are unoptimized and parallel-test contention causes p99 jitter false negatives → debug only measures, release (--release / CI) asserts strictly.

use std::time::{Duration, Instant};

use fs_core::vector::schema::{MetricKind, VectorSchema};
use fs_core::vector::simd;
use fs_core::vector::store::VectorIndex;
use tempfile::tempdir;

const N: usize = 1000;
const DIM: usize = 128;
const TOP_K: usize = 10;

fn gen_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    // Simple deterministic pseudo-random (no Math.random dependency, reproducible)
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
fn nsw_recall_vs_bruteforce_top10() {
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

    // Brute-force top-10
    let query = &vecs[123];
    let mut brute: Vec<(u64, f32)> = (0..N)
        .map(|i| (i as u64, simd::distance(MetricKind::L2, query, &vecs[i])))
        .collect();
    brute.sort_by(|a, b| a.1.total_cmp(&b.1));
    let brute_top: Vec<u64> = brute.iter().take(TOP_K).map(|(id, _)| *id).collect();

    let nsw_res = idx.search_knn(query, TOP_K, None).unwrap();
    let nsw_top: Vec<u64> = nsw_res.iter().map(|(id, _)| *id).collect();

    let overlap = nsw_top.iter().filter(|id| brute_top.contains(id)).count();
    let recall = overlap as f32 / TOP_K as f32;
    eprintln!(
        "recall@{} = {:.2} (brute={:?}, nsw={:?})",
        TOP_K, recall, brute_top, nsw_top
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
    // Latency SLA is a release metric; unoptimized debug + parallel-test CPU contention cause p99 jitter false negatives.
    // Debug only measures, release (--release / CI) asserts strictly (measured ~3340us, ~50% headroom).
    #[cfg(not(debug_assertions))]
    assert!(p99 < 5000.0, "p99 {}us >= 5000us", p99);
    #[cfg(debug_assertions)]
    eprintln!("knn p99: debug build, latency assertion skipped (run --release to enforce)");
}

#[test]
fn get_vector_and_list_vector_ids_roundtrip() {
    // #3: get_vector (present/missing/deleted) + list_vector_ids (soft-deleted excluded)
    let dir = tempdir().unwrap();
    let schema = VectorSchema::new(8, MetricKind::L2);
    let idx = VectorIndex::open(dir.path(), schema, 0).unwrap();
    let vecs = gen_vectors(20, 8, 7);
    let items: Vec<(u64, &[f32])> = vecs
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u64, v.as_slice()))
        .collect();
    idx.insert_batch(&items, None).unwrap();

    // List 20
    let mut ids = idx.list_vector_ids().unwrap();
    ids.sort();
    assert_eq!(ids, (0..20u64).collect::<Vec<_>>());

    // Get a live vector: value == inserted
    let got = idx.get_vector(5).unwrap();
    assert_eq!(got.as_ref().unwrap().as_slice(), vecs[5].as_slice());

    // Get missing -> None
    assert!(idx.get_vector(999).unwrap().is_none());

    // Soft-delete id=5
    assert!(idx.delete(5, None).unwrap());
    // Get after delete -> None
    assert!(idx.get_vector(5).unwrap().is_none());
    // List excludes id=5
    let mut ids2 = idx.list_vector_ids().unwrap();
    ids2.sort();
    assert_eq!(ids2, (0..20u64).filter(|i| *i != 5).collect::<Vec<_>>());
}
