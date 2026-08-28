//! NSW 图常驻 RAM —— 单层邻接图（A2 诚实命名，非 HNSW）[v2 H1/R5]
//!
//! 本实现是**单层 NSW**（Navigable Small World 邻接图），无多层跳表结构。
//! 检索复杂度近似 O(N^log M)，非真 HNSW 的 O(log N)。N 持续增长会线性劣化
//! 检索延迟，p99<5ms SLA 有扩展上限（README 全规模基准节）。
//!
//! 图结构整体常驻内存（非逐节点入 heed，H1 根治）。
//! 检索全程内存指针跳转，零磁盘 IO。
//! M=16, ef_construction=200, ef_search=200 固定参数（PRD 风险对策：先固定后调参）。
//! 软删除：节点标 deleted=true，KNN 跳过；物理清理在 M4 compact。
//! 持久化：snapshot 落单 mmap 段（snapshot.rs，M2 后续）。
//!
//! 纯拓扑：不持向量字节，距离由 caller 注入（insert/search 传 dist 闭包）。
//! 这样 VectorIndex 可零拷贝读 mmap 向量算距离，无需 Box<dyn Fn> 每跳拷贝。

use std::collections::{BinaryHeap, HashMap};

/// NSW 固定参数（PRD M2 风险对策：先固定后调参）
pub const M: usize = 16;
pub const EF_CONSTRUCTION: usize = 200;
// ef_search 默认 200：单层 NSW 无多层捷径，ef=50 召回卡 0.925 不达 PRD §2.5 ≥0.95。
// PRD「ef_search=50 固定」与 §2.5 召回 SLA 冲突（Rule 7），取召回 SLA 为硬验收，
// 调参阶段先提 ef_search 保召回；p99 预算 5ms 充足（50K 实测 ef=200 仍 <1ms）。
// F-ARCH-2：ef_search 不再硬编码，建库时经 VectorSchema.ef_search 锁定（serde default 回落此值）。
pub const DEFAULT_EF_SEARCH: usize = 200;

/// 节点 id
pub type NodeId = u64;

/// 图节点 —— 邻居链表在堆（id 为 HashMap key，不冗余存）
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

/// NSW 单层图（常驻 RAM，纯拓扑）
///
/// 节点 id→邻居链表 + 软删除标记。向量数据与距离由 caller 注入。
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

    /// 所有未删节点 id（供批量连边预取向量）
    pub fn all_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.nodes
            .iter()
            .filter(|(_, n)| !n.deleted)
            .map(|(id, _)| *id)
    }

    /// 所有软删节点 id（compact 过滤用）
    pub fn all_deleted_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.nodes
            .iter()
            .filter(|(_, n)| n.deleted)
            .map(|(id, _)| *id)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // 图常驻 RAM 估算（H1 衍生：>10M 分层加载评估基建 §776 #9）。
    // 向量数据在 mmap 段非 RAM；此处仅图拓扑：HashMap 桶开销 + Node + 邻居 Vec。
    // 估算：HashMap 条目约 48B（key+value+槽）+ Node{deleted:1B,neighbors ptr+len+cap} 约 32B
    //      + 每邻居 Vec 容量 * size_of::<NodeId>()(8B)。
    pub fn memory_usage(&self) -> usize {
        let node_overhead = std::mem::size_of::<Node>() + 48; // Node + HashMap 槽分摊
        let mut bytes = self.nodes.len() * node_overhead;
        for node in self.nodes.values() {
            bytes += node.neighbors.capacity() * std::mem::size_of::<NodeId>();
        }
        bytes
    }

    /// 插入节点（连边）—— 单层 NSW：选 ef_construction 近邻建边。
    /// dist: 新向量到任意现存节点 id 的距离闭包（caller 读向量算）。
    pub fn insert(&mut self, id: NodeId, dist: &dyn Fn(NodeId) -> f32) -> crate::Result<()> {
        if self.nodes.contains_key(&id) {
            return Err(crate::StoreError::DuplicateVector(id));
        }
        let mut node = Node::new();
        // 第一个节点为入口点
        if self.entry_point.is_none() {
            self.entry_point = Some(id);
            self.nodes.insert(id, node);
            tracing::debug!(id, "nsw insert first node (entry point)");
            return Ok(());
        }
        let entry = self.entry_point.unwrap();
        // 搜 ef_construction 近邻候选（insert 无 deadline，构建不走超时）
        let candidates = self.search_layer(dist, EF_CONSTRUCTION, entry, None);
        // 取前 M 个连边
        let neighbors: Vec<NodeId> = candidates.into_iter().take(M).map(|(nid, _)| nid).collect();
        node.neighbors = neighbors.clone();
        // 双向连边：被选邻居也加本节点（截断到 M*2 防爆）
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

    /// 层内贪心搜索 —— 返回 (id, distance) 升序，跳过 deleted。
    /// query_dist: 查询向量到任意节点 id 的距离闭包。
    /// deadline: 超时则提前终止遍历返回已遍历子集（R5：遍历中途检查，非事后）。
    fn search_layer(
        &self,
        query_dist: &dyn Fn(NodeId) -> f32,
        ef: usize,
        entry: NodeId,
        deadline: Option<std::time::Instant>,
    ) -> Vec<(NodeId, f32)> {
        // 候选堆（小顶，按距离）+ 结果堆（大顶，保 ef 个最近）
        let mut visited: HashMap<NodeId, ()> = HashMap::new();
        let mut candidates: BinaryHeap<(std::cmp::Reverse<OrderDist>, NodeId)> = BinaryHeap::new();
        let mut results: BinaryHeap<(OrderDist, NodeId)> = BinaryHeap::new();

        let d0 = query_dist(entry);
        candidates.push((std::cmp::Reverse(OrderDist(d0)), entry));
        visited.insert(entry, ());
        if !self.is_deleted(entry) {
            results.push((OrderDist(d0), entry));
        }

        // R5：超时检查节流——每 256 次弹出查一次 deadline，免每跳 Instant::now 开销
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
            // 结果集已满且当前候选比最远结果还远 → 停
            if results.len() >= ef {
                if let Some((OrderDist(farthest), _)) = results.peek() {
                    if cd > *farthest {
                        break;
                    }
                }
            }
            // 遍历邻居 —— 直接借 &node.neighbors 迭代，不 clone（每跳省一次堆分配）
            // 安全：循环内仅写局部 candidates/results/visited，不借 self 可变
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
        // 结果按距离升序
        let mut out: Vec<(NodeId, f32)> = results
            .into_iter()
            .map(|(OrderDist(d), id)| (id, d))
            .collect();
        out.sort_by(|a, b| a.1.total_cmp(&b.1));
        out
    }

    /// id 是否软删（compact 重排跳过 deleted，KNN 跳过 deleted）
    pub fn is_deleted(&self, id: NodeId) -> bool {
        self.nodes.get(&id).is_none_or(|n| n.deleted)
    }

    /// KNN 检索 —— 带 timeout（超时返回已遍历最佳子集，A1）。
    /// query_dist: 查询向量到任意节点 id 的距离闭包（caller 零拷贝读向量算）。
    /// ef_search: 候选集宽度（F-ARCH-2 经 VectorSchema 锁定，建库时可配），与 top_k 取大者。
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
        // R5：遍历中已按 deadline 提前终止（每 256 跳查一次）；此处收尾再查一次兜底
        if let Some(dl) = deadline {
            if std::time::Instant::now() > dl {
                tracing::warn!(top_k, "knn exceeded deadline, returning partial best");
            }
        }
        found.truncate(top_k);
        Ok(found)
    }

    /// 软删除 —— 标 deleted=true，KNN 跳过（A3）
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

    /// 快照所有节点（CSR 序列化用，snapshot.rs）
    pub fn snapshot_nodes(&self) -> Vec<(NodeId, Vec<NodeId>)> {
        self.nodes
            .iter()
            .map(|(id, n)| (*id, n.neighbors.clone()))
            .collect()
    }

    /// 快照所有节点含软删标记（L7：snapshot 持久化 deleted 状态用）
    pub fn snapshot_nodes_with_deleted(&self) -> Vec<(NodeId, Vec<NodeId>, bool)> {
        self.nodes
            .iter()
            .map(|(id, n)| (*id, n.neighbors.clone(), n.deleted))
            .collect()
    }

    /// 快照入口点（snapshot.rs header 用）
    pub fn snapshot_entry_point(&self) -> Option<NodeId> {
        self.entry_point
    }

    /// 从 snapshot 恢复单节点（直接置入，不经连边逻辑）
    pub fn restore_node(&mut self, id: NodeId, neighbors: Vec<NodeId>, deleted: bool) {
        self.nodes.insert(id, Node { deleted, neighbors });
    }

    /// 从 snapshot 恢复入口点
    pub fn restore_entry_point(&mut self, id: NodeId) {
        if self.nodes.contains_key(&id) {
            self.entry_point = Some(id);
        }
    }

    // E4：图结构完整性校验 —— snapshot 重载后调用。
    // 1) 每条边的邻居 id 必须在节点集合内（无悬空边，否则检索跳到不存在的节点）
    // 2) 非空图的入口点必须存在且未软删（否则 KNN 无起点）
    // 任一失败 → Err(Corrupt)，caller 据此重建图而非加载坏图。
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

/// 距离包装（BinaryHeap 需 Ord；用 total_cmp 防 NaN 误序）
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

    // 距离闭包工厂：固定向量表 + 查询向量 → |node_id| -> f32
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
        // 10 个 8 维向量，第一维 = id
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
        // 查询接近 id=3
        let q = vec![3.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let df = dist_fn(&v, &q, MetricKind::L2);
        let res = g.search_knn(&df, 3, DEFAULT_EF_SEARCH, None).unwrap();
        assert!(!res.is_empty());
        // 最近应是 id=3
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
        // 软删 id=3
        assert!(g.delete(3));
        let q = vec![3.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let df = dist_fn(&v, &q, MetricKind::L2);
        let res = g.search_knn(&df, 3, DEFAULT_EF_SEARCH, None).unwrap();
        // 结果不含 3
        assert!(!res.iter().any(|(id, _)| *id == 3));
    }
}
