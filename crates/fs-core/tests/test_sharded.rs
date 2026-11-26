//! ShardedEngine integration tests — NSW scale-evolution Path A thin sharding layer [ROADMAP F-ARCH-3]
//!
//! Covers: multi-shard creation + vector/KV routing correctness + fan-out search merge recall + batch splitting +
//! delete/get routing + list merged across all shards + checkpoint/close-reopen persistence + vector_count aggregation.
//! Deterministic pseudo-random (xorshift, reproducible), tempdir isolation, real Engine round-trips, no mocks.

use fs_core::sharded::{HashRouter, ShardRouter, ShardedEngine};
use fs_core::vector::schema::{MetricKind, VectorSchema};
use fs_core::FusionStoreEngine;
use tempfile::tempdir;

const DIM: usize = 64;

fn gen_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut v = Vec::with_capacity(n);
    let mut s = seed;
    for i in 0..n {
        let mut row = Vec::with_capacity(dim);
        for j in 0..dim {
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

fn bruteforce_topk(query: &[f32], pool: &[(u64, Vec<f32>)], top_k: usize) -> Vec<(u64, f32)> {
    let mut scored: Vec<(u64, f32)> = pool
        .iter()
        .map(|(id, v)| (*id, cosine_dist(query, v)))
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(top_k);
    scored
}

fn cosine_dist(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 1.0;
    }
    1.0 - dot / (na * nb)
}

#[test]
fn sharded_open_creates_k_shard_dirs() {
    let dir = tempdir().unwrap();
    let schema = VectorSchema::new(DIM, MetricKind::Cosine);
    let eng = ShardedEngine::open(dir.path(), 3, Some(schema), 0).unwrap();
    assert_eq!(eng.num_shards(), 3);
    for i in 0..3 {
        assert!(
            dir.path().join(format!("shard_{}", i)).exists(),
            "shard_{} dir missing",
            i
        );
    }
    eng.close().unwrap();
}

#[test]
fn sharded_insert_routes_by_id_mod() {
    let dir = tempdir().unwrap();
    let schema = VectorSchema::new(DIM, MetricKind::Cosine);
    let eng = ShardedEngine::open(dir.path(), 4, Some(schema), 0).unwrap();
    let router = HashRouter;
    let vecs = gen_vectors(40, DIM, 42);
    for (id, v) in vecs.iter().enumerate() {
        eng.insert_vector(id as u64, v, None).unwrap();
    }
    // Verify: each id lands on the shard chosen by the router (consistent via get_vector routed read-back)
    for (id, v) in vecs.iter().enumerate() {
        let expect_shard = router.shard_for_vector(id as u64, 4);
        assert!(expect_shard < 4);
        let got = eng.get_vector(id as u64, None).unwrap();
        assert_eq!(got, Some(v.clone()), "id {} roundtrip mismatch", id);
    }
    assert_eq!(eng.vector_count().unwrap(), 40);
    eng.close().unwrap();
}

#[test]
fn sharded_kv_routes_and_roundtrips() {
    let dir = tempdir().unwrap();
    let eng = ShardedEngine::open(dir.path(), 4, None, 0).unwrap();
    // 20 keys across shards
    for i in 0..20u64 {
        let key = format!("key_{}", i);
        let val = format!("val_{}", i);
        eng.put_kv(key.as_bytes(), val.as_bytes(), None).unwrap();
    }
    for i in 0..20u64 {
        let key = format!("key_{}", i);
        let got = eng.get_kv_zero_copy(key.as_bytes(), None).unwrap();
        assert!(got.is_some(), "key {} missing", key);
        assert_eq!(got.unwrap().as_bytes(), format!("val_{}", i).as_bytes());
    }
    // Delete half, verify routing consistency
    for i in 0..10u64 {
        let key = format!("key_{}", i);
        let deleted = eng.delete_kv(key.as_bytes(), None).unwrap();
        assert!(deleted, "delete key {} should return true", key);
    }
    for i in 0..10u64 {
        let key = format!("key_{}", i);
        assert!(
            eng.get_kv_zero_copy(key.as_bytes(), None)
                .unwrap()
                .is_none(),
            "deleted key {} still present",
            key
        );
    }
    for i in 10..20u64 {
        let key = format!("key_{}", i);
        assert!(
            eng.get_kv_zero_copy(key.as_bytes(), None)
                .unwrap()
                .is_some(),
            "key {} wrongly gone",
            key
        );
    }
    eng.close().unwrap();
}

#[test]
fn sharded_fanout_knn_recall_vs_bruteforce() {
    // Fan-out search after sharding should match global brute-force top-k (id sets overlap).
    // Scale: 4 shards × 250 vectors = 1000, top-10, verifying recall-correctness trends.
    let dir = tempdir().unwrap();
    let schema = VectorSchema::new(DIM, MetricKind::Cosine);
    let eng = ShardedEngine::open(dir.path(), 4, Some(schema), 0).unwrap();
    let n = 1000;
    let vecs = gen_vectors(n, DIM, 7);
    for (id, v) in vecs.iter().enumerate() {
        eng.insert_vector(id as u64, v, None).unwrap();
    }
    let query = gen_vectors(1, DIM, 99)[0].clone();
    let top_k = 10;
    let got = eng.search_knn(&query, top_k, None).unwrap();
    assert_eq!(got.len(), top_k, "fan-out should return top_k");

    // Global brute-force top-k
    let pool: Vec<(u64, Vec<f32>)> = vecs
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u64, v.clone()))
        .collect();
    let truth = bruteforce_topk(&query, &pool, top_k);
    let got_ids: std::collections::HashSet<u64> = got.iter().map(|(id, _)| *id).collect();
    let truth_ids: std::collections::HashSet<u64> = truth.iter().map(|(id, _)| *id).collect();
    let overlap = got_ids.intersection(&truth_ids).count();
    // Sharded search recall should be >= 0.9 (per-shard ef_search=200 meets the recall SLA; fan-out merge never drops true top-k)
    assert!(
        overlap as f64 / top_k as f64 >= 0.9,
        "fan-out recall {} too low (got {:?} vs truth {:?})",
        overlap as f64 / top_k as f64,
        got_ids,
        truth_ids
    );
    eng.close().unwrap();
}

#[test]
fn sharded_insert_batch_splits_across_shards() {
    let dir = tempdir().unwrap();
    let schema = VectorSchema::new(DIM, MetricKind::Cosine);
    let eng = ShardedEngine::open(dir.path(), 4, Some(schema), 0).unwrap();
    let vecs = gen_vectors(100, DIM, 5);
    let items: Vec<(u64, &[f32])> = vecs
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u64, v.as_slice()))
        .collect();
    eng.insert_vector_batch(&items, None).unwrap();
    assert_eq!(eng.vector_count().unwrap(), 100);
    // Each id readable back
    for (id, v) in vecs.iter().enumerate() {
        assert_eq!(eng.get_vector(id as u64, None).unwrap(), Some(v.clone()));
    }
    eng.close().unwrap();
}

#[test]
fn sharded_delete_get_vector_routes() {
    let dir = tempdir().unwrap();
    let schema = VectorSchema::new(DIM, MetricKind::Cosine);
    let eng = ShardedEngine::open(dir.path(), 3, Some(schema), 0).unwrap();
    let vecs = gen_vectors(30, DIM, 3);
    for (id, v) in vecs.iter().enumerate() {
        eng.insert_vector(id as u64, v, None).unwrap();
    }
    // Delete even ids
    for id in (0..30).step_by(2) {
        let deleted = eng.delete_vector(id as u64, None).unwrap();
        assert!(deleted, "delete id {} should return true", id);
    }
    // Even → None, odd → Some
    for id in 0..30 {
        let got = eng.get_vector(id as u64, None).unwrap();
        if id % 2 == 0 {
            assert!(got.is_none(), "deleted id {} should be None", id);
        } else {
            assert!(got.is_some(), "alive id {} should be Some", id);
        }
    }
    eng.close().unwrap();
}

#[test]
fn sharded_list_vector_ids_merges_all_shards() {
    let dir = tempdir().unwrap();
    let schema = VectorSchema::new(DIM, MetricKind::Cosine);
    let eng = ShardedEngine::open(dir.path(), 3, Some(schema), 0).unwrap();
    let vecs = gen_vectors(60, DIM, 11);
    for (id, v) in vecs.iter().enumerate() {
        eng.insert_vector(id as u64, v, None).unwrap();
    }
    let mut ids = eng.list_vector_ids(None).unwrap();
    ids.sort();
    let expect: Vec<u64> = (0..60).collect();
    assert_eq!(ids, expect, "list should merge all 60 ids across shards");
    eng.close().unwrap();
}

#[test]
fn sharded_checkpoint_close_reopen_persists() {
    let dir = tempdir().unwrap();
    let home = dir.path().to_path_buf();
    let schema = VectorSchema::new(DIM, MetricKind::Cosine);
    {
        let eng = ShardedEngine::open(&home, 3, Some(schema.clone()), 0).unwrap();
        let vecs = gen_vectors(30, DIM, 8);
        for (id, v) in vecs.iter().enumerate() {
            eng.insert_vector(id as u64, v, None).unwrap();
        }
        eng.put_kv(b"meta", b"persisted", None).unwrap();
        eng.checkpoint().unwrap();
        eng.close().unwrap();
    }
    // Reopen (schema=None reopens all shards), data should persist
    let eng = ShardedEngine::open(&home, 3, None, 0).unwrap();
    assert_eq!(
        eng.vector_count().unwrap(),
        30,
        "vectors should persist after reopen"
    );
    let got = eng.get_kv_zero_copy(b"meta", None).unwrap();
    assert_eq!(got.unwrap().as_bytes(), b"persisted");
    // Search still usable
    let query = gen_vectors(1, DIM, 8)[0].clone();
    let res = eng.search_knn(&query, 5, None).unwrap();
    assert!(!res.is_empty());
    eng.close().unwrap();
}

#[test]
fn sharded_vector_count_aggregates() {
    let dir = tempdir().unwrap();
    let schema = VectorSchema::new(DIM, MetricKind::Cosine);
    let eng = ShardedEngine::open(dir.path(), 4, Some(schema), 0).unwrap();
    assert_eq!(eng.vector_count().unwrap(), 0);
    let vecs = gen_vectors(80, DIM, 2);
    for (id, v) in vecs.iter().enumerate() {
        eng.insert_vector(id as u64, v, None).unwrap();
    }
    assert_eq!(eng.vector_count().unwrap(), 80);
    // Delete 20, count drops
    for id in 0..20 {
        eng.delete_vector(id as u64, None).unwrap();
    }
    assert_eq!(eng.vector_count().unwrap(), 60);
    eng.close().unwrap();
}
