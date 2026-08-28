//! 图 snapshot —— CSR 序列化落单 mmap 段 + 重启重载 [v2 H1]
//!
//! NSW 图常驻 RAM，定期 snapshot 拓扑到单 mmap 段文件（graph_snapshot.mmap）。
//! 重启时从 snapshot 段重载图拓扑到 RAM，无需重建（H1 持久化对策）。
//!
//! CSR 格式（紧凑、顺序访问友好）：
//! ```text
//! [magic u32][node_count u32][edge_count u32][entry_point u64][crc u32]
//! [node_ids: u64 * node_count]
//! [offsets:  u32 * (node_count+1)]   // 每节点邻居在 edges 区的起止
//! [edges:    u64 * edge_count]       // 邻居 id 扁平
//! [deleted_flags: u8 * node_count]   // 软删标记位
//! ```
//! 重载时按 node_ids/offsets/edges 重建 NswGraph.nodes。

use std::fs::{File, OpenOptions};
use std::path::Path;

use memmap2::{Mmap, MmapMut};

use crate::error::Result;
use crate::vector::nsw::NswGraph;

const MAGIC: u32 = 0xF5_C5_04; // fs-core graph snapshot
const SNAPSHOT_FILE: &str = "graph_snapshot.mmap";
const SNAPSHOT_TMP: &str = "graph_snapshot.mmap.tmp";

// CRC-32 IEEE（table-less）—— snapshot 完整性校验，拒截断/伪造/半写
fn crc32_update(mut crc: u32, data: &[u8]) -> u32 {
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    crc
}

fn crc32_finalize(crc: u32) -> u32 {
    !crc
}

/// 序列化图为 CSR 字节流
pub fn serialize(graph: &NswGraph) -> Vec<u8> {
    // L7：含 deleted 标记，持久化软删状态（重启后不复活已删节点）
    let nodes = graph.snapshot_nodes_with_deleted();
    let node_count = nodes.len() as u32;
    let edge_count: u32 = nodes.iter().map(|(_, nb, _)| nb.len() as u32).sum();
    let entry_point = graph.snapshot_entry_point().unwrap_or(0);

    // 头 4*4 + 8(entry) + 4(crc) = 28 字节
    let header = 28usize;
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
    // L7：逐节点写 deleted flag（1=软删），重启 load 读回，不复活已删节点
    for (i, (nid, neighbors, deleted)) in nodes.iter().enumerate() {
        buf[ids_cursor..ids_cursor + 8].copy_from_slice(&nid.to_le_bytes());
        ids_cursor += 8;
        acc += neighbors.len() as u32;
        buf[off_cursor..off_cursor + 4].copy_from_slice(&acc.to_le_bytes());
        off_cursor += 4;
        for &nb in neighbors {
            buf[edge_cursor..edge_cursor + 8].copy_from_slice(&nb.to_le_bytes());
            edge_cursor += 8;
        }
        buf[flags_off + i] = if *deleted { 1 } else { 0 };
    }
    // CRC 覆盖 ids+offsets+edges+flags（header 不含，crc 字段 [20..24] 自身不含）
    let body = &buf[ids_off..total];
    let crc = crc32_finalize(crc32_update(0xFFFF_FFFF, body));
    buf[20..24].copy_from_slice(&crc.to_le_bytes());
    tracing::info!(
        node_count,
        edge_count,
        bytes = total,
        "graph snapshot serialized (CSR + deleted flags)"
    );
    buf
}

/// 持久化 snapshot 字节到单 mmap 段文件（原子 tmp+fsync+rename，E3）
///
/// 绝不就地覆盖在线文件：先写 `.tmp` → fsync → atomic rename → fsync dir。
/// 崩溃发生在任一点，要么旧文件完整、要么新文件完整，无半写撕裂。
pub fn persist(dir: &Path, bytes: &[u8]) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join(SNAPSHOT_FILE);
    let tmp_path = dir.join(SNAPSHOT_TMP);
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&tmp_path)?;
    file.set_len(bytes.len() as u64)?;
    let mut mmap = unsafe { MmapMut::map_mut(&file)? };
    mmap[..bytes.len()].copy_from_slice(bytes);
    mmap.flush()?;
    drop(mmap);
    file.sync_all()?;
    // 原子切换：rename(tmp -> live)，再 fsync 目录使 rename 落盘
    std::fs::rename(&tmp_path, &path)?;
    if let Ok(dirf) = File::open(dir) {
        let _ = dirf.sync_all();
    }
    tracing::info!(path = %path.display(), bytes = bytes.len(), "graph snapshot persisted (atomic tmp+rename)");
    Ok(())
}

/// 从 mmap 段文件重载图拓扑（重启恢复，H1）
///
/// 全边界检查：任一字段越界 → Err(Corrupt)，绝不 panic。校验 CRC32 —— 拒截断/半写。
pub fn load(dir: &Path) -> Result<Option<NswGraph>> {
    let path = dir.join(SNAPSHOT_FILE);
    if !path.exists() {
        tracing::info!("no graph snapshot, fresh graph");
        return Ok(None);
    }
    let file = File::open(&path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let bytes = &mmap[..];
    if bytes.len() < 28 {
        return Err(crate::StoreError::Corrupt(format!(
            "graph snapshot too small: {} < 28",
            bytes.len()
        )));
    }
    let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if magic != MAGIC {
        return Err(crate::StoreError::Corrupt(format!(
            "graph snapshot magic mismatch: {:#x}",
            magic
        )));
    }
    let node_count = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    let edge_count = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
    let entry_point = u64::from_le_bytes([
        bytes[12], bytes[13], bytes[14], bytes[15], bytes[16], bytes[17], bytes[18], bytes[19],
    ]);
    let stored_crc = u32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);

    let ids_off: usize = 28;
    let offsets_off = ids_off
        .checked_add(node_count * 8)
        .ok_or_else(|| crate::StoreError::Corrupt("node_count overflow".into()))?;
    let edges_off = offsets_off
        .checked_add((node_count + 1) * 4)
        .ok_or_else(|| crate::StoreError::Corrupt("offsets region overflow".into()))?;
    let flags_off = edges_off
        .checked_add(edge_count * 8)
        .ok_or_else(|| crate::StoreError::Corrupt("edges region overflow".into()))?;
    let total = flags_off
        .checked_add(node_count)
        .ok_or_else(|| crate::StoreError::Corrupt("flags region overflow".into()))?;
    if bytes.len() < total {
        return Err(crate::StoreError::Corrupt(format!(
            "graph snapshot truncated: file {} < expected {}",
            bytes.len(),
            total
        )));
    }
    // CRC 覆盖 body（ids+offsets+edges+flags），header[20..24] crc 自身除外
    let body = &bytes[ids_off..total];
    let calc_crc = crc32_finalize(crc32_update(0xFFFF_FFFF, body));
    if calc_crc != stored_crc {
        return Err(crate::StoreError::Corrupt(format!(
            "graph snapshot crc mismatch: stored {:#x} != calc {:#x} (corrupt/half-written)",
            stored_crc, calc_crc
        )));
    }

    let mut graph = NswGraph::new();
    if node_count == 0 {
        tracing::info!("graph snapshot empty, fresh graph");
        return Ok(Some(graph));
    }
    for i in 0..node_count {
        let id = u64::from_le_bytes([
            bytes[ids_off + i * 8],
            bytes[ids_off + i * 8 + 1],
            bytes[ids_off + i * 8 + 2],
            bytes[ids_off + i * 8 + 3],
            bytes[ids_off + i * 8 + 4],
            bytes[ids_off + i * 8 + 5],
            bytes[ids_off + i * 8 + 6],
            bytes[ids_off + i * 8 + 7],
        ]);
        let lo = u32::from_le_bytes([
            bytes[offsets_off + i * 4],
            bytes[offsets_off + i * 4 + 1],
            bytes[offsets_off + i * 4 + 2],
            bytes[offsets_off + i * 4 + 3],
        ]) as usize;
        let hi = u32::from_le_bytes([
            bytes[offsets_off + (i + 1) * 4],
            bytes[offsets_off + (i + 1) * 4 + 1],
            bytes[offsets_off + (i + 1) * 4 + 2],
            bytes[offsets_off + (i + 1) * 4 + 3],
        ]) as usize;
        if hi < lo || hi > edge_count {
            return Err(crate::StoreError::Corrupt(format!(
                "node {} edge range [{}, {}) invalid (edge_count={})",
                id, lo, hi, edge_count
            )));
        }
        let mut neighbors = Vec::with_capacity(hi - lo);
        for j in lo..hi {
            let nb = u64::from_le_bytes([
                bytes[edges_off + j * 8],
                bytes[edges_off + j * 8 + 1],
                bytes[edges_off + j * 8 + 2],
                bytes[edges_off + j * 8 + 3],
                bytes[edges_off + j * 8 + 4],
                bytes[edges_off + j * 8 + 5],
                bytes[edges_off + j * 8 + 6],
                bytes[edges_off + j * 8 + 7],
            ]);
            neighbors.push(nb);
        }
        let deleted = bytes[flags_off + i] != 0;
        graph.restore_node(id, neighbors, deleted);
    }
    graph.restore_entry_point(entry_point);
    // E4：CRC 只防字节损坏；此处校验图结构语义——无悬边、入口点有效。
    // count 碰巧一致但内容损坏（如手工篡改后的合法 CSR）会被此层拦截，不静默加载坏图。
    graph.validate_edges()?;
    tracing::info!(
        node_count,
        edge_count,
        "graph snapshot loaded into RAM (crc ok, edges valid)"
    );
    Ok(Some(graph))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn snapshot_roundtrip_preserves_topology() {
        let mut g = NswGraph::new();
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

    // L7：snapshot 持久化软删标记，重启后 deleted 节点不复活
    #[test]
    fn snapshot_roundtrip_preserves_deleted_flags() {
        let mut g = NswGraph::new();
        g.restore_node(1, vec![2, 3], false);
        g.restore_node(2, vec![1], true); // 软删
        g.restore_node(3, vec![1], false);
        g.restore_entry_point(1);
        let bytes = serialize(&g);
        let dir = tempdir().unwrap();
        persist(dir.path(), &bytes).unwrap();
        let loaded = load(dir.path()).unwrap().unwrap();
        // len 计未删 → 2（1,3 活，2 删）
        assert_eq!(loaded.len(), 2);
        assert!(loaded.is_deleted(2));
        assert!(!loaded.is_deleted(1));
        assert!(!loaded.is_deleted(3));
    }

    // E3：CRC 不匹配拒收（body 被篡改），不 panic 不返回坏图
    #[test]
    fn load_rejects_corrupt_body_via_crc() {
        let mut g = NswGraph::new();
        g.restore_node(10, vec![11], false);
        g.restore_node(11, vec![10], false);
        g.restore_entry_point(10);
        let bytes = serialize(&g);
        let dir = tempdir().unwrap();
        persist(dir.path(), &bytes).unwrap();
        // 篡改 body 第一字节（node_ids 区），不碰 crc 字段
        let path = dir.path().join(SNAPSHOT_FILE);
        let mut corrupted = std::fs::read(&path).unwrap();
        corrupted[28] ^= 0xFF;
        std::fs::write(&path, corrupted).unwrap();
        let res = load(dir.path());
        assert!(res.is_err(), "expected error on corrupt body");
        if let Err(crate::StoreError::Corrupt(m)) = res {
            assert!(m.contains("crc"), "msg: {m}");
        }
    }

    // E3：截断文件拒收（半写场景），不 panic
    #[test]
    fn load_rejects_truncated_file() {
        let mut g = NswGraph::new();
        g.restore_node(10, vec![11], false);
        g.restore_node(11, vec![10], false);
        g.restore_entry_point(10);
        let bytes = serialize(&g);
        let dir = tempdir().unwrap();
        persist(dir.path(), &bytes).unwrap();
        // 截断到一半
        let path = dir.path().join(SNAPSHOT_FILE);
        let full = std::fs::read(&path).unwrap();
        std::fs::write(&path, &full[..full.len() / 2]).unwrap();
        let res = load(dir.path());
        assert!(res.is_err(), "expected error on truncated file");
        if let Err(crate::StoreError::Corrupt(m)) = res {
            assert!(m.contains("truncated"), "msg: {m}");
        }
    }

    // E3：头部过小拒收
    #[test]
    fn load_rejects_tiny_header() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join(SNAPSHOT_FILE), [0u8; 16]).unwrap();
        let res = load(dir.path());
        assert!(res.is_err(), "expected error on tiny header");
        if let Err(crate::StoreError::Corrupt(m)) = res {
            assert!(m.contains("too small"), "msg: {m}");
        }
    }

    // E4：悬边拒收 —— node_count 与 locator 数碰巧一致但边指向不存在节点，
    // CRC 通过（字节合法）却图结构损坏。validate_edges 拦截，不静默加载坏图。
    #[test]
    fn load_rejects_dangling_edge_via_validate() {
        let mut g = NswGraph::new();
        g.restore_node(10, vec![11, 99], false); // 99 不在图 → 悬边
        g.restore_node(11, vec![10], false);
        g.restore_entry_point(10);
        let bytes = serialize(&g);
        let dir = tempdir().unwrap();
        persist(dir.path(), &bytes).unwrap();
        let res = load(dir.path());
        assert!(res.is_err(), "dangling edge must be rejected");
        if let Err(crate::StoreError::Corrupt(m)) = res {
            assert!(m.contains("dangling edge"), "msg: {m}");
        }
    }

    // E4：入口点被软删拒收 —— KNN 无合法起点
    #[test]
    fn load_rejects_deleted_entry_point() {
        let mut g = NswGraph::new();
        g.restore_node(10, vec![11], true); // 入口点软删
        g.restore_node(11, vec![10], false);
        g.restore_entry_point(10);
        let bytes = serialize(&g);
        let dir = tempdir().unwrap();
        persist(dir.path(), &bytes).unwrap();
        let res = load(dir.path());
        assert!(res.is_err(), "deleted entry point must be rejected");
        if let Err(crate::StoreError::Corrupt(m)) = res {
            assert!(m.contains("entry point"), "msg: {m}");
        }
    }

    // E4：非空图无入口点拒收
    #[test]
    fn load_rejects_nonempty_graph_no_entry_point() {
        let mut g = NswGraph::new();
        g.restore_node(10, vec![11], false);
        g.restore_node(11, vec![10], false);
        // 不调 restore_entry_point → entry_point=None
        let bytes = serialize(&g);
        let dir = tempdir().unwrap();
        persist(dir.path(), &bytes).unwrap();
        let res = load(dir.path());
        assert!(
            res.is_err(),
            "non-empty graph without entry point must be rejected"
        );
        if let Err(crate::StoreError::Corrupt(m)) = res {
            assert!(m.contains("entry point"), "msg: {m}");
        }
    }
}
