//! 图 snapshot —— CSR 序列化落单 mmap 段 + 重启重载 [v2 H1]
//!
//! HNSW 图常驻 RAM，定期 snapshot 拓扑到单 mmap 段文件（graph_snapshot.mmap）。
//! 重启时从 snapshot 段重载图拓扑到 RAM，无需重建（H1 持久化对策）。
//!
//! CSR 格式（紧凑、顺序访问友好）：
//! ```text
//! [magic u32][node_count u32][edge_count u32][entry_point u64]
//! [node_ids: u64 * node_count]
//! [offsets:  u32 * (node_count+1)]   // 每节点邻居在 edges 区的起止
//! [edges:    u64 * edge_count]       // 邻居 id 扁平
//! [deleted_flags: u8 * node_count]   // 软删标记位
//! ```
//! 重载时按 node_ids/offsets/edges 重建 HnswGraph.nodes。

use std::fs::{File, OpenOptions};
use std::path::Path;

use memmap2::{Mmap, MmapMut};

use crate::error::Result;
use crate::vector::hnsw::HnswGraph;

const MAGIC: u32 = 0xF5_C5_04; // fs-core graph snapshot
const SNAPSHOT_FILE: &str = "graph_snapshot.mmap";

/// 序列化图为 CSR 字节流
pub fn serialize(graph: &HnswGraph) -> Vec<u8> {
    let nodes = graph.snapshot_nodes();
    let node_count = nodes.len() as u32;
    let edge_count: u32 = nodes.iter().map(|(_, nb)| nb.len() as u32).sum();
    let entry_point = graph.snapshot_entry_point().unwrap_or(0);

    // 头 4*4 + 8 = 24 字节
    let header = 24usize;
    let ids_off = header;
    let ids_len = node_count as usize * 8;
    let offsets_off = ids_off + ids_len;
    let offsets_len = (node_count as usize + 1) * 4;
    let edges_off = offsets_off + offsets_len;
    let edges_len = edge_count as usize * 8;
    let flags_off = edges_off + edges_len;
    let flags_len = node_count as usize;
    let total = flags_off + flags_len;

    let mut buf = vec![0u8; total];
    // header
    buf[0..4].copy_from_slice(&MAGIC.to_le_bytes());
    buf[4..8].copy_from_slice(&node_count.to_le_bytes());
    buf[8..12].copy_from_slice(&edge_count.to_le_bytes());
    buf[12..20].copy_from_slice(&entry_point.to_le_bytes());
    // node_ids + offsets + edges + flags
    let mut ids_cursor = ids_off;
    let mut off_cursor = offsets_off;
    let mut edge_cursor = edges_off;
    let mut acc: u32 = 0;
    buf[off_cursor..off_cursor + 4].copy_from_slice(&acc.to_le_bytes());
    off_cursor += 4;
    // deleted 标记区初始化为 0（snapshot_nodes 含 deleted 节点，
    // 但 deleted 标记需由 graph 内部状态补——M2 简化：deleted 不持久化，
    // 重启后全为未删；deleted 节点恢复后需 caller 重新标记。flags 区全 0）
    for (nid, neighbors) in &nodes {
        buf[ids_cursor..ids_cursor + 8].copy_from_slice(&nid.to_le_bytes());
        ids_cursor += 8;
        acc += neighbors.len() as u32;
        buf[off_cursor..off_cursor + 4].copy_from_slice(&acc.to_le_bytes());
        off_cursor += 4;
        for &nb in neighbors {
            buf[edge_cursor..edge_cursor + 8].copy_from_slice(&nb.to_le_bytes());
            edge_cursor += 8;
        }
    }
    tracing::info!(
        node_count,
        edge_count,
        bytes = total,
        "graph snapshot serialized (CSR)"
    );
    buf
}

/// 持久化 snapshot 字节到单 mmap 段文件（覆盖写）
pub fn persist(dir: &Path, bytes: &[u8]) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join(SNAPSHOT_FILE);
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&path)?;
    file.set_len(bytes.len() as u64)?;
    let mut mmap = unsafe { MmapMut::map_mut(&file)? };
    mmap[..bytes.len()].copy_from_slice(bytes);
    mmap.flush()?;
    tracing::info!(path = %path.display(), bytes = bytes.len(), "graph snapshot persisted");
    Ok(())
}

/// 从 mmap 段文件重载图拓扑（重启恢复，H1）
pub fn load(dir: &Path) -> Result<Option<HnswGraph>> {
    let path = dir.join(SNAPSHOT_FILE);
    if !path.exists() {
        tracing::info!("no graph snapshot, fresh graph");
        return Ok(None);
    }
    let file = File::open(&path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let bytes = &mmap[..];
    if bytes.len() < 24 {
        return Err(crate::StoreError::Corrupt(
            "graph snapshot too small".into(),
        ));
    }
    let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    if magic != MAGIC {
        return Err(crate::StoreError::Corrupt(format!(
            "graph snapshot magic mismatch: {:#x}",
            magic
        )));
    }
    let node_count = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let edge_count = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let entry_point = u64::from_le_bytes(bytes[12..20].try_into().unwrap());

    let ids_off = 24;
    let offsets_off = ids_off + node_count * 8;
    let edges_off = offsets_off + (node_count + 1) * 4;
    let flags_off = edges_off + edge_count * 8;

    let mut graph = HnswGraph::new();
    if node_count == 0 {
        tracing::info!("graph snapshot empty, fresh graph");
        return Ok(Some(graph));
    }
    for i in 0..node_count {
        let id = u64::from_le_bytes(
            bytes[ids_off + i * 8..ids_off + i * 8 + 8]
                .try_into()
                .unwrap(),
        );
        let lo = u32::from_le_bytes(
            bytes[offsets_off + i * 4..offsets_off + i * 4 + 4]
                .try_into()
                .unwrap(),
        ) as usize;
        let hi = u32::from_le_bytes(
            bytes[offsets_off + (i + 1) * 4..offsets_off + (i + 1) * 4 + 4]
                .try_into()
                .unwrap(),
        ) as usize;
        let mut neighbors = Vec::with_capacity(hi - lo);
        for j in lo..hi {
            let nb = u64::from_le_bytes(
                bytes[edges_off + j * 8..edges_off + j * 8 + 8]
                    .try_into()
                    .unwrap(),
            );
            neighbors.push(nb);
        }
        let deleted = bytes[flags_off + i] != 0;
        graph.restore_node(id, neighbors, deleted);
    }
    graph.restore_entry_point(entry_point);
    tracing::info!(node_count, edge_count, "graph snapshot loaded into RAM");
    Ok(Some(graph))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn snapshot_roundtrip_preserves_topology() {
        let mut g = HnswGraph::new();
        // 插 3 节点（手工，不需向量——纯拓扑测试）
        g.restore_node(10, vec![11, 12], false);
        g.restore_node(11, vec![10], false);
        g.restore_node(12, vec![10], false);
        g.restore_entry_point(10);
        let bytes = serialize(&g);
        let dir = tempdir().unwrap();
        persist(dir.path(), &bytes).unwrap();
        let loaded = load(dir.path()).unwrap().unwrap();
        // len 计未删节点
        assert_eq!(loaded.len(), 3);
        // 拓扑：节点 10 邻居含 11,12
        let snap = loaded.snapshot_nodes();
        let n10 = snap.iter().find(|(id, _)| *id == 10).unwrap();
        assert!(n10.1.contains(&11) && n10.1.contains(&12));
    }

    #[test]
    fn load_missing_returns_none() {
        let dir = tempdir().unwrap();
        assert!(load(dir.path()).unwrap().is_none());
    }
}
