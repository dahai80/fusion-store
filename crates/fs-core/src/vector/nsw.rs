//! NSW graph resident in RAM — single-layer adjacency graph (A2 honest naming, not HNSW) [v2 H1/R5]
//!
//! This implementation is a **single-layer NSW** (Navigable Small World
//! adjacency graph) with no multi-layer skip-list structure. Search complexity
//! is approximately O(N^log M), not a true HNSW's O(log N). Unbounded growth
//! of N degrades search latency linearly, so the p99<5ms SLA has a scaling
//! ceiling (see the full-scale benchmark section in the README).
//!
//! The graph structure is entirely memory-resident (not per-node in heed —
//! the H1 root fix). Search is pure in-memory pointer chasing, zero disk IO.
//! Fixed parameters M=16, ef_construction=200, ef_search=200 (PRD risk
//! measure: fix first, tune later).
//! Soft delete: nodes marked deleted=true are skipped by KNN; physical
//! reclamation happens in the M4 compact.
//! Persistence: snapshots land in a single mmap segment (snapshot.rs, M2 follow-up).
//!
//! Pure topology: holds no vector bytes; distances are injected by the caller
//! (insert/search pass a dist closure). This lets VectorIndex read vectors
//! zero-copy from mmap to compute distances, with no per-hop copying through
//! Box<dyn Fn>.

use std::collections::{BinaryHeap, HashMap};

/// Fixed NSW parameters (PRD M2 risk measure: fix first, tune later)
pub const M: usize = 16;
pub const EF_CONSTRUCTION: usize = 200;
// ef_search defaults to 200: a single-layer NSW has no multi-layer shortcuts, and ef=50
// plateaus recall at 0.925, failing the PRD §2.5 ≥0.95 requirement.
// The PRD's "ef_search=50 fixed" conflicts with the §2.5 recall SLA (Rule 7); the recall SLA
// is taken as the hard acceptance criterion. During tuning, ef_search is raised first to
// protect recall; the 5ms p99 budget has ample headroom (50K measured: ef=200 still <1ms).
// F-ARCH-2: ef_search is no longer hardcoded; it is locked at index creation via
// VectorSchema.ef_search (serde default falls back to this value).
pub const DEFAULT_EF_SEARCH: usize = 200;

/// Node id
pub type NodeId = u64;

/// Graph node — neighbor list lives on the heap (id is the HashMap key, not duplicated)
#[derive(Debug, Clone)]
struct Node {
    deleted: bool,
    neighbors: Vec<NodeId>,
}

impl Node {
    fn new() -> Self {
        Self {
            deleted: false,
            neighbors: Vec::with_capacity(M),
        }
    }
}

/// Single-layer NSW graph (RAM-resident, pure topology)
///
/// Node id → neighbor list + soft-delete flag. Vector data and distances are injected by the caller.
pub struct NswGraph {
    nodes: HashMap<NodeId, Node>,
    entry_point: Option<NodeId>,
}

impl NswGraph {
    pub fn new() -> Self {
        tracing::info!(
            M,
            EF_CONSTRUCTION,
            DEFAULT_EF_SEARCH,
            "nsw graph created (single-layer NSW)"
        );
        Self {
            nodes: HashMap::new(),
            entry_point: None,
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.values().filter(|n| !n.deleted).count()
    }

    /// All non-deleted node ids (for batch edge-building vector prefetch)
    pub fn all_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.nodes
            .iter()
            .filter(|(_, n)| !n.deleted)
            .map(|(id, _)| *id)
    }

    /// All soft-deleted node ids (used by compact filtering)
    pub fn all_deleted_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.nodes
            .iter()
            .filter(|(_, n)| n.deleted)
            .map(|(id, _)| *id)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // RAM footprint estimate of the resident graph (H1 derivative: >10M tiered loading to be
    // assessed against infra §776 #9).
    // Vector data lives in mmap segments, not RAM; this covers only graph topology:
    // HashMap bucket overhead + Node + neighbor Vecs.
    // Estimate: HashMap entry ≈ 48B (key+value+slot) + Node{deleted:1B,neighbors ptr+len+cap} ≈ 32B
    //          + per-neighbor Vec capacity * size_of::<NodeId>()(8B).
    pub fn memory_usage(&self) -> usize {
        let node_overhead = std::mem::size_of::<Node>() + 48; // Node + amortized HashMap slot
        let mut bytes = self.nodes.len() * node_overhead;
        for node in self.nodes.values() {
            bytes += node.neighbors.capacity() * std::mem::size_of::<NodeId>();
        }
        bytes
    }

    /// Insert a node (with edge building) — single-layer NSW: connect to the ef_construction nearest.
    /// dist: closure giving the distance from the new vector to any existing node id (caller reads vectors).
    pub fn insert(&mut self, id: NodeId, dist: &dyn Fn(NodeId) -> f32) -> crate::Result<()> {
        if self.nodes.contains_key(&id) {
            return Err(crate::StoreError::DuplicateVector(id));
        }
        let mut node = Node::new();
        // First node becomes the entry point
        if self.entry_point.is_none() {
            self.entry_point = Some(id);
            self.nodes.insert(id, node);
            tracing::debug!(id, "nsw insert first node (entry point)");
            return Ok(());
        }
        let entry = self.entry_point.unwrap();
        // Search for ef_construction nearest candidates (insert has no deadline; build path
        // never times out)
        let candidates = self.search_layer(dist, EF_CONSTRUCTION, entry, None);
        // Connect to the top M
        let neighbors: Vec<NodeId> = candidates.into_iter().take(M).map(|(nid, _)| nid).collect();
        node.neighbors = neighbors.clone();
        // Bidirectional edges: selected neighbors also point back to this node (truncated to
        // M*2 to bound growth)
        for &nb in &neighbors {
            if let Some(nb_node) = self.nodes.get_mut(&nb) {
                nb_node.neighbors.push(id);
                if nb_node.neighbors.len() > M * 2 {
                    nb_node.neighbors.truncate(M * 2);
                }
            }
        }
        self.nodes.insert(id, node);
        tracing::debug!(id, neighbors = neighbors.len(), "nsw insert node");
        Ok(())
    }

    /// Greedy in-layer search — returns (id, distance) ascending, skipping deleted.
    /// query_dist: closure giving the distance from the query vector to any node id.
    /// deadline: on timeout, traversal terminates early and returns the subset visited so far
    /// (R5: checked mid-traversal, not after the fact).
    fn search_layer(
        &self,
        query_dist: &dyn Fn(NodeId) -> f32,
        ef: usize,
        entry: NodeId,
        deadline: Option<std::time::Instant>,
    ) -> Vec<(NodeId, f32)> {
        // Candidate heap (min-heap by distance) + result heap (max-heap keeping the ef nearest)
        let mut visited: HashMap<NodeId, ()> = HashMap::new();
        let mut candidates: BinaryHeap<(std::cmp::Reverse<OrderDist>, NodeId)> = BinaryHeap::new();
        let mut results: BinaryHeap<(OrderDist, NodeId)> = BinaryHeap::new();

        let d0 = query_dist(entry);
        candidates.push((std::cmp::Reverse(OrderDist(d0)), entry));
        visited.insert(entry, ());
        if !self.is_deleted(entry) {
            results.push((OrderDist(d0), entry));
        }

        // R5: throttle deadline checks — inspect the deadline every 256 pops to avoid the
        // per-hop Instant::now cost
        let mut pop_count: u32 = 0;
        while let Some((std::cmp::Reverse(OrderDist(cd)), cid)) = candidates.pop() {
            pop_count = pop_count.wrapping_add(1);
            if pop_count & 0xFF == 0 {
                if let Some(dl) = deadline {
                    if std::time::Instant::now() > dl {
                        tracing::warn!(
                            ef,
                            pop_count,
                            "search_layer timeout mid-traversal, returning partial"
                        );
                        break;
                    }
                }
            }
            // Result set full and current candidate is farther than the farthest result → stop
            if results.len() >= ef {
                if let Some((OrderDist(farthest), _)) = results.peek() {
                    if cd > *farthest {
                        break;
                    }
                }
            }
            // Traverse neighbors — borrow &node.neighbors directly to iterate, no clone
            // (saves a heap allocation per hop)
            // Safe: the loop only writes to local candidates/results/visited, never borrows
            // self mutably
            let Some(node) = self.nodes.get(&cid) else {
                continue;
            };
            for &nb in &node.neighbors {
                if visited.contains_key(&nb) {
                    continue;
                }
                visited.insert(nb, ());
                let d = query_dist(nb);
                candidates.push((std::cmp::Reverse(OrderDist(d)), nb));
                if !self.is_deleted(nb) {
                    if results.len() < ef {
                        results.push((OrderDist(d), nb));
                    } else if let Some((OrderDist(farthest), _)) = results.peek() {
                        if d < *farthest {
                            results.pop();
                            results.push((OrderDist(d), nb));
                        }
                    }
                }
            }
        }
        // Results sorted by distance ascending
        let mut out: Vec<(NodeId, f32)> = results
            .into_iter()
            .map(|(OrderDist(d), id)| (id, d))
            .collect();
        out.sort_by(|a, b| a.1.total_cmp(&b.1));
        out
    }

    /// Whether id is soft-deleted (compact reshuffling skips deleted; KNN skips deleted)
    pub fn is_deleted(&self, id: NodeId) -> bool {
        self.nodes.get(&id).is_none_or(|n| n.deleted)
    }

    /// KNN search — with timeout (returns the best subset visited before the deadline, A1).
    /// query_dist: closure giving the distance from the query vector to any node id
    /// (caller reads vectors zero-copy to compute it).
    /// ef_search: candidate-set width (locked via VectorSchema per F-ARCH-2, configurable at
    /// index creation); the larger of it and top_k is used.
    pub fn search_knn(
        &self,
        query_dist: &dyn Fn(NodeId) -> f32,
        top_k: usize,
        ef_search: usize,
        deadline: Option<std::time::Instant>,
    ) -> crate::Result<Vec<(NodeId, f32)>> {
        let entry = match self.entry_point {
            Some(e) => e,
            None => return Ok(vec![]),
        };
        let ef = ef_search.max(top_k);
        let mut found = self.search_layer(query_dist, ef, entry, deadline);
        // R5: traversal already terminated early on deadline (checked every 256 hops); check
        // once more here as a backstop
        if let Some(dl) = deadline {
            if std::time::Instant::now() > dl {
                tracing::warn!(top_k, "knn exceeded deadline, returning partial best");
            }
        }
        found.truncate(top_k);
        Ok(found)
    }

    /// Soft delete — mark deleted=true, skipped by KNN (A3)
    pub fn delete(&mut self, id: NodeId) -> bool {
        if let Some(node) = self.nodes.get_mut(&id) {
            let was = !node.deleted;
            node.deleted = true;
            tracing::debug!(id, "nsw soft-delete node");
            was
        } else {
            false
        }
    }

    /// Snapshot all nodes (for CSR serialization, snapshot.rs)
    pub fn snapshot_nodes(&self) -> Vec<(NodeId, Vec<NodeId>)> {
        self.nodes
            .iter()
            .map(|(id, n)| (*id, n.neighbors.clone()))
            .collect()
    }

    /// Snapshot all nodes including soft-delete flags (L7: lets snapshots persist deleted state)
    pub fn snapshot_nodes_with_deleted(&self) -> Vec<(NodeId, Vec<NodeId>, bool)> {
        self.nodes
            .iter()
            .map(|(id, n)| (*id, n.neighbors.clone(), n.deleted))
            .collect()
    }

    /// Snapshot the entry point (used in the snapshot.rs header)
    pub fn snapshot_entry_point(&self) -> Option<NodeId> {
        self.entry_point
    }

    /// Restore a single node from a snapshot (inserted directly, bypassing edge-building logic)
    pub fn restore_node(&mut self, id: NodeId, neighbors: Vec<NodeId>, deleted: bool) {
        self.nodes.insert(id, Node { deleted, neighbors });
    }

    /// Restore the entry point from a snapshot
    pub fn restore_entry_point(&mut self, id: NodeId) {
        if self.nodes.contains_key(&id) {
            self.entry_point = Some(id);
        }
    }

    // E4: graph-structure integrity check — called after snapshot reload.
    // 1) Every edge's neighbor id must exist in the node set (no dangling edges, otherwise
    //    search would jump to a nonexistent node)
    // 2) A non-empty graph's entry point must exist and not be soft-deleted (otherwise KNN has
    //    no starting point)
    // Any failure → Err(Corrupt); the caller rebuilds the graph instead of loading a bad one.
    pub fn validate_edges(&self) -> crate::Result<()> {
        for (id, node) in &self.nodes {
            for &nb in &node.neighbors {
                if !self.nodes.contains_key(&nb) {
                    return Err(crate::StoreError::Corrupt(format!(
                        "dangling edge: node {} -> {} (target not in graph)",
                        id, nb
                    )));
                }
            }
        }
        if !self.nodes.is_empty() {
            match self.entry_point {
                Some(ep) if !self.is_deleted(ep) => {}
                Some(ep) => {
                    return Err(crate::StoreError::Corrupt(format!(
                        "entry point {} is soft-deleted, no valid search start",
                        ep
                    )));
                }
                None => {
                    return Err(crate::StoreError::Corrupt(
                        "non-empty graph has no entry point".into(),
                    ));
                }
            }
        }
        Ok(())
    }
}

impl Default for NswGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Distance wrapper (BinaryHeap needs Ord; total_cmp prevents NaN misordering)
#[derive(Debug, Clone, Copy)]
struct OrderDist(f32);

impl PartialEq for OrderDist {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for OrderDist {}
impl PartialOrd for OrderDist {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrderDist {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::schema::MetricKind;
    use crate::vector::simd;

    // Distance closure factory: fixed vector table + query vector → |node_id| -> f32
    fn dist_fn<'a>(
        vectors: &'a HashMap<NodeId, Vec<f32>>,
        query: &'a [f32],
        metric: MetricKind,
    ) -> impl Fn(NodeId) -> f32 + 'a {
        move |nid| {
            let v = vectors.get(&nid).expect("node vector missing");
            simd::distance(metric, query, v)
        }
    }

    #[test]
    fn insert_first_sets_entry_point() {
        let mut v = HashMap::new();
        v.insert(1u64, vec![1.0f32; 8]);
        let mut g = NswGraph::new();
        let df = dist_fn(&v, &v[&1], MetricKind::L2);
        g.insert(1, &df).unwrap();
        assert_eq!(g.len(), 1);
    }

    #[test]
    fn insert_duplicate_rejects() {
        let mut v = HashMap::new();
        v.insert(1, vec![1.0f32; 8]);
        let mut g = NswGraph::new();
        let df = dist_fn(&v, &v[&1], MetricKind::L2);
        g.insert(1, &df).unwrap();
        let err = g.insert(1, &df).unwrap_err();
        assert!(matches!(err, crate::StoreError::DuplicateVector(1)));
    }

    #[test]
    fn knn_returns_nearest() {
        let mut v = HashMap::new();
        // 10 eight-dim vectors, first dim = id
        for i in 0..10u64 {
            let mut vec = vec![0.0f32; 8];
            vec[0] = i as f32;
            v.insert(i, vec);
        }
        let mut g = NswGraph::new();
        for i in 0..10u64 {
            let df = dist_fn(&v, &v[&i], MetricKind::L2);
            g.insert(i, &df).unwrap();
        }
        // Query near id=3
        let q = vec![3.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let df = dist_fn(&v, &q, MetricKind::L2);
        let res = g.search_knn(&df, 3, DEFAULT_EF_SEARCH, None).unwrap();
        assert!(!res.is_empty());
        // Nearest should be id=3
        assert_eq!(res[0].0, 3);
    }

    #[test]
    fn soft_delete_skips_in_knn() {
        let mut v = HashMap::new();
        for i in 0..10u64 {
            let mut vec = vec![0.0f32; 8];
            vec[0] = i as f32;
            v.insert(i, vec);
        }
        let mut g = NswGraph::new();
        for i in 0..10u64 {
            let df = dist_fn(&v, &v[&i], MetricKind::L2);
            g.insert(i, &df).unwrap();
        }
        // Soft-delete id=3
        assert!(g.delete(3));
        let q = vec![3.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let df = dist_fn(&v, &q, MetricKind::L2);
        let res = g.search_knn(&df, 3, DEFAULT_EF_SEARCH, None).unwrap();
        // Results must not contain 3
        assert!(!res.iter().any(|(id, _)| *id == 3));
    }
}
