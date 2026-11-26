use std::time::Instant;

use fs_core::vector::schema::{MetricKind, VectorSchema};
use fs_core::vector::store::VectorIndex;
use tempfile::tempdir;

fn main() {
    // Fresh idx per N (avoids id collisions), measuring graph-build throughput vs scale
    for &n in &[100usize, 500, 1000, 2000, 5000] {
        let dir = tempdir().unwrap();
        let schema = VectorSchema::new(768, MetricKind::L2);
        let idx = VectorIndex::open(dir.path(), schema, 0).unwrap();
        let vecs: Vec<Vec<f32>> = (0..n as u64).map(|i| vec![i as f32; 768]).collect();
        let batch: Vec<(u64, &[f32])> = (0..n as u64)
            .map(|i| (i, vecs[i as usize].as_slice()))
            .collect();
        let s = Instant::now();
        idx.insert_batch(&batch, None).unwrap();
        let e = s.elapsed().as_secs_f64();
        println!("n={} {:.3}s = {:.0} vecs/s", n, e, n as f64 / e);
    }
}
