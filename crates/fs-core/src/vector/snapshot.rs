//! Graph snapshot — CSR serialization into a single mmap segment + reload on restart [v2 H1]
//!
//! The NSW graph is resident in RAM; its topology is periodically snapshotted
//! into a single mmap segment file (graph_snapshot.mmap). On restart the graph
//! topology is reloaded from the snapshot segment into RAM without a rebuild
//! (H1 persistence measure).
//!
//! CSR format (compact, sequential-access friendly):
//! ```text
//! [magic u32][node_count u32][edge_count u32][entry_point u64][crc u32]
//! [node_ids: u64 * node_count]
//! [offsets:  u32 * (node_count+1)]   // start/end of each node's neighbors in the edges region
//! [edges:    u64 * edge_count]       // flattened neighbor ids
//! [deleted_flags: u8 * node_count]   // soft-delete flag bits
//! ```
//! On reload, NswGraph.nodes is rebuilt from node_ids/offsets/edges.

use std::fs::{File, OpenOptions};
use std::path::Path;

use memmap2::{Mmap, MmapMut};

use crate::error::Result;
use crate::vector::nsw::NswGraph;

const MAGIC: u32 = 0xF5_C5_04; // fs-core graph snapshot
const SNAPSHOT_FILE: &str = "graph_snapshot.mmap";
const SNAPSHOT_TMP: &str = "graph_snapshot.mmap.tmp";

// CRC-32 IEEE (table-less) — snapshot integrity check, rejects truncation/forgery/half-writes
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

/// Serialize the graph into a CSR byte stream
pub fn serialize(graph: &NswGraph) -> Vec<u8> {
    // L7: include deleted flags so soft-delete state survives restart (deleted nodes are not revived)
    let nodes = graph.snapshot_nodes_with_deleted();
    let node_count = nodes.len() as u32;
    let edge_count: u32 = nodes.iter().map(|(_, nb, _)| nb.len() as u32).sum();
    let entry_point = graph.snapshot_entry_point().unwrap_or(0);

    // Header 4*4 + 8(entry) + 4(crc) = 28 bytes
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
    // L7: write the deleted flag per node (1 = soft-deleted); load reads it back on restart
    // so deleted nodes are not revived
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
    // CRC covers ids+offsets+edges+flags (header excluded, and the crc field [20..24] itself excluded)
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

/// Persist snapshot bytes to a single mmap segment file (atomic tmp+fsync+rename, E3)
///
/// Never overwrite the live file in place: write `.tmp` → fsync → atomic
/// rename → fsync dir. If a crash happens at any point, either the old file is
/// intact or the new file is complete — no torn half-write.
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
    // Atomic switch: rename(tmp -> live), then fsync the directory to make the rename durable
    std::fs::rename(&tmp_path, &path)?;
    // F-ERR-3: dir fsync makes the rename metadata durable. Failure is not fatal (the file
    // already renamed successfully; the next checkpoint will fsync again), but within the
    // crash window the rename may not have hit disk → rollback to the old snapshot. An
    // explicit error is logged so operators notice.
    if let Ok(dirf) = File::open(dir) {
        if let Err(e) = dirf.sync_all() {
            tracing::error!(
                dir = %dir.display(),
                err = %e,
                "graph snapshot dir fsync failed — rename may not be durable until next checkpoint"
            );
        }
    }
    tracing::info!(path = %path.display(), bytes = bytes.len(), "graph snapshot persisted (atomic tmp+rename)");
    Ok(())
}

/// Reload graph topology from the mmap segment file (restart recovery, H1)
///
/// Full bounds checking: any out-of-range field → Err(Corrupt), never panic.
/// CRC32 is verified — truncation/half-writes are rejected.
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
    // CRC covers the body (ids+offsets+edges+flags), excluding header[20..24] where the crc itself lives
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
    // E4: CRC only guards against byte corruption; this validates graph-structure semantics —
    // no dangling edges, entry point valid. A graph whose counts happen to match but whose
    // content is corrupted (e.g. a hand-tampered yet valid CSR) is caught here instead of
    // silently loading a bad graph.
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
        // Insert 3 nodes manually (no vectors needed — pure topology test)
        g.restore_node(10, vec![11, 12], false);
        g.restore_node(11, vec![10], false);
        g.restore_node(12, vec![10], false);
        g.restore_entry_point(10);
        let bytes = serialize(&g);
        let dir = tempdir().unwrap();
        persist(dir.path(), &bytes).unwrap();
        let loaded = load(dir.path()).unwrap().unwrap();
        // len counts non-deleted nodes
        assert_eq!(loaded.len(), 3);
        // Topology: node 10 neighbors include 11,12
        let snap = loaded.snapshot_nodes();
        let n10 = snap.iter().find(|(id, _)| *id == 10).unwrap();
        assert!(n10.1.contains(&11) && n10.1.contains(&12));
    }

    #[test]
    fn load_missing_returns_none() {
        let dir = tempdir().unwrap();
        assert!(load(dir.path()).unwrap().is_none());
    }

    // L7: snapshot persists soft-delete flags; deleted nodes are not revived after restart
    #[test]
    fn snapshot_roundtrip_preserves_deleted_flags() {
        let mut g = NswGraph::new();
        g.restore_node(1, vec![2, 3], false);
        g.restore_node(2, vec![1], true); // soft-deleted
        g.restore_node(3, vec![1], false);
        g.restore_entry_point(1);
        let bytes = serialize(&g);
        let dir = tempdir().unwrap();
        persist(dir.path(), &bytes).unwrap();
        let loaded = load(dir.path()).unwrap().unwrap();
        // len counts non-deleted → 2 (1,3 alive, 2 deleted)
        assert_eq!(loaded.len(), 2);
        assert!(loaded.is_deleted(2));
        assert!(!loaded.is_deleted(1));
        assert!(!loaded.is_deleted(3));
    }

    // E3: CRC mismatch rejects the file (tampered body), no panic, no bad graph returned
    #[test]
    fn load_rejects_corrupt_body_via_crc() {
        let mut g = NswGraph::new();
        g.restore_node(10, vec![11], false);
        g.restore_node(11, vec![10], false);
        g.restore_entry_point(10);
        let bytes = serialize(&g);
        let dir = tempdir().unwrap();
        persist(dir.path(), &bytes).unwrap();
        // Tamper with the first body byte (node_ids region), leaving the crc field untouched
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

    // E3: truncated file is rejected (half-write scenario), no panic
    #[test]
    fn load_rejects_truncated_file() {
        let mut g = NswGraph::new();
        g.restore_node(10, vec![11], false);
        g.restore_node(11, vec![10], false);
        g.restore_entry_point(10);
        let bytes = serialize(&g);
        let dir = tempdir().unwrap();
        persist(dir.path(), &bytes).unwrap();
        // Truncate to half
        let path = dir.path().join(SNAPSHOT_FILE);
        let full = std::fs::read(&path).unwrap();
        std::fs::write(&path, &full[..full.len() / 2]).unwrap();
        let res = load(dir.path());
        assert!(res.is_err(), "expected error on truncated file");
        if let Err(crate::StoreError::Corrupt(m)) = res {
            assert!(m.contains("truncated"), "msg: {m}");
        }
    }

    // E3: too-small header is rejected
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

    // E4: dangling edge is rejected — node_count happens to match the locator count but an edge
    // points at a nonexistent node; CRC passes (bytes are valid) yet the graph structure is
    // broken. validate_edges catches it instead of silently loading a bad graph.
    #[test]
    fn load_rejects_dangling_edge_via_validate() {
        let mut g = NswGraph::new();
        g.restore_node(10, vec![11, 99], false); // 99 not in graph → dangling edge
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

    // E4: soft-deleted entry point is rejected — KNN has no valid starting point
    #[test]
    fn load_rejects_deleted_entry_point() {
        let mut g = NswGraph::new();
        g.restore_node(10, vec![11], true); // entry point soft-deleted
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

    // E4: non-empty graph without an entry point is rejected
    #[test]
    fn load_rejects_nonempty_graph_no_entry_point() {
        let mut g = NswGraph::new();
        g.restore_node(10, vec![11], false);
        g.restore_node(11, vec![10], false);
        // restore_entry_point never called → entry_point=None
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
