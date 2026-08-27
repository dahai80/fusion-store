//! 向量索引存储 —— vec mmap 段 + id→locator + HNSW 图常驻 RAM [v2 H1/E2]
//!
//! 一个 VectorIndex 持：
//! - VecSegmentPool：append-only mmap 段存原始 f32 向量（fixed dim）
//! - heed id→ValueLocator：向量定位信息（id 唯一）
//! - HnswGraph：纯拓扑常驻 RAM，距离由本模块注入（零拷贝读 mmap 向量算）
//! - schema：dim/metric 建库锁定（E2）
//!
//! insert：append 向量 → 写 locator → 图连边（dist 闭包读 mmap 向量）。
//! search_knn：图检索，dist 闭包按 id 取向量零拷贝算距离。
//! delete：图软删（向量字节不动，M4 compact 回收）。
//! snapshot/restore：图拓扑落单 mmap 段（snapshot.rs）。

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use heed::types::Bytes;
use heed::{Database, Env, EnvOpenOptions, RoTxn, RwTxn};
use memmap2::{Mmap, MmapMut};

use crate::error::Result;
use crate::mem::mmap::MmapHandle;
use crate::mem::segment::ValueLocator;
use crate::vector::hnsw::HnswGraph;
use crate::vector::schema::VectorSchema;
use crate::vector::simd;

/// 向量段默认容量（128MB —— 768 维 f32 单向量 3KB，约 4.5 万向量/段）
const DEFAULT_VEC_SEGMENT_SIZE: u64 = 128 * 1024 * 1024;
/// id→locator 库名
const VEC_LOCATOR_DB: &str = "vec_locators";
/// schema 元信息 key
const SCHEMA_KEY: &[u8] = b"schema";

/// append-only 向量段池 —— 存原始 f32 向量，fixed dim
struct VecSegmentPool {
    data_dir: PathBuf,
    seg_size: u64,
    dim: usize,
    active_id: u32,
    active_file: File,
    active_mmap: MmapMut,
    active_cursor: u64,
    // 封存段只读映射缓存（seg_id -> Arc<MmapHandle>）
    sealed: HashMap<u32, Arc<MmapHandle>>,
    // active 段只读映射缓存 —— 零拷贝读 active 段免每次重新 mmap（性能）。
    // 封存轮转后失效重建（seal_and_rotate 置 None）。MAP_SHARED 与 active_mmap 字节一致。
    active_read: Option<Arc<MmapHandle>>,
}

impl VecSegmentPool {
    fn open(data_dir: &Path, seg_size: u64, dim: usize) -> Result<Self> {
        std::fs::create_dir_all(data_dir)?;
        let seg_size = if seg_size == 0 {
            DEFAULT_VEC_SEGMENT_SIZE
        } else {
            seg_size
        };
        let active_id = next_vec_seg_id(data_dir)?;
        let path = data_dir.join(vec_segment_file(active_id));
        let file = open_or_create_vec_segment(&path, seg_size)?;
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        tracing::info!(seg_id = active_id, dim, seg_size, "vec segment pool opened");
        Ok(Self {
            data_dir: data_dir.to_path_buf(),
            seg_size,
            dim,
            active_id,
            active_file: file,
            active_mmap: mmap,
            active_cursor: 0,
            sealed: HashMap::new(),
            active_read: None,
        })
    }

    fn byte_len(&self) -> usize {
        self.dim * std::mem::size_of::<f32>()
    }

    fn remaining(&self) -> u64 {
        self.seg_size.saturating_sub(self.active_cursor)
    }

    /// 追加一个向量，返回定位信息。写满封存轮转。
    fn append(&mut self, vector: &[f32]) -> Result<ValueLocator> {
        let need = self.byte_len() as u64;
        if need > self.remaining() {
            self.seal_and_rotate()?;
        }
        let seg_id = self.active_id;
        let offset = self.active_cursor as u32;
        let len = self.byte_len() as u32;
        let bytes = bytemuck::cast_slice::<f32, u8>(vector);
        let end = self.active_cursor + need;
        self.active_mmap[self.active_cursor as usize..end as usize].copy_from_slice(bytes);
        self.active_cursor = end;
        tracing::debug!(seg_id, offset, len, "vector appended");
        Ok(ValueLocator {
            seg_id,
            offset,
            len,
        })
    }

    fn seal_and_rotate(&mut self) -> Result<()> {
        let sealed_id = self.active_id;
        tracing::info!(seg_id = sealed_id, "vec segment full, sealing + rotating");
        self.active_mmap.flush()?;
        let sealed_file = self.active_file.try_clone()?;
        let sealed_mmap = unsafe { Mmap::map(&sealed_file)? };
        let mut handle = MmapHandle::from_parts(sealed_file, sealed_mmap);
        handle.seal();
        self.sealed.insert(sealed_id, Arc::new(handle));
        let new_id = sealed_id + 1;
        let path = self.data_dir.join(vec_segment_file(new_id));
        let file = open_or_create_vec_segment(&path, self.seg_size)?;
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        self.active_id = new_id;
        self.active_file = file;
        self.active_mmap = mmap;
        self.active_cursor = 0;
        // 轮转后旧 active_read 缓存指向已封存段，失效重建（下次 vec_slice 按需开）
        self.active_read = None;
        Ok(())
    }

    /// 取某 locator 指向的向量字节切片（零拷贝读 mmap）。
    /// 封存段从缓存/按需开；active 段直接读 active_mmap（MAP_SHARED 跨映射一致）。
    fn read_vec(&mut self, loc: &ValueLocator) -> Result<Vec<f32>> {
        if loc.seg_id == self.active_id {
            // active 段：直接读 active_mmap 切片
            let start = loc.offset as usize;
            let end = start + self.byte_len();
            let bytes = &self.active_mmap[start..end];
            Ok(bytemuck::cast_slice::<u8, f32>(bytes).to_vec())
        } else {
            // 封存段：缓存映射读
            let handle = self.sealed_handle(loc.seg_id)?;
            let bytes = &handle.as_bytes()[loc.offset as usize..(loc.offset + loc.len) as usize];
            Ok(bytemuck::cast_slice::<u8, f32>(bytes).to_vec())
        }
    }

    /// 零拷贝取向量切片 —— 持 Arc<MmapHandle> 保活，返回 ZeroCopyVec（无 Vec 分配）。
    /// 距离热路径用：HNSW 每跳算距离零拷贝读 mmap，不拷贝字节、不分配（H1 零拷贝 + 性能）。
    /// active 段：包装 active_mmap 的只读 Arc 句柄（MAP_SHARED 跨映射一致，读安全）。
    /// 封存段：从缓存/按需开只读映射。
    fn vec_slice(&mut self, loc: &ValueLocator) -> Result<crate::mem::mmap::ZeroCopyVec> {
        let len = (loc.len as usize) / std::mem::size_of::<f32>();
        let offset = loc.offset as usize;
        if loc.seg_id == self.active_id {
            // active 段：构造只读 MmapHandle 包装 active_mmap 映射保活。
            // active 段未 sealed，但 append-only：已写区域永不被改（cursor 单调增），
            // 故本向量区域零拷贝指针在读期间有效（caller 不持跨下次 append）。
            let handle = self.active_handle()?;
            Ok(crate::mem::mmap::ZeroCopyVec::new(handle, offset, len))
        } else {
            let handle = self.sealed_handle(loc.seg_id)?;
            Ok(crate::mem::mmap::ZeroCopyVec::new(handle, offset, len))
        }
    }

    /// active 段只读句柄（供 vec_slice 零拷贝读 active 段）。
    /// 缓存只读映射免每次重新 mmap（性能）；轮转后失效重建。
    /// 双映射同文件 MAP_SHARED 字节一致（active_mmap 写即 active_read 读可见）。
    fn active_handle(&mut self) -> Result<Arc<MmapHandle>> {
        if let Some(h) = &self.active_read {
            return Ok(h.clone());
        }
        let file = self.active_file.try_clone()?;
        let mmap = unsafe { Mmap::map(&file)? };
        let handle = Arc::new(MmapHandle::from_parts(file, mmap));
        self.active_read = Some(handle.clone());
        Ok(handle)
    }

    fn sealed_handle(&mut self, seg_id: u32) -> Result<Arc<MmapHandle>> {
        if let Some(h) = self.sealed.get(&seg_id) {
            return Ok(h.clone());
        }
        let path = self.data_dir.join(vec_segment_file(seg_id));
        if !path.exists() {
            return Err(crate::StoreError::Corrupt(format!(
                "sealed vec segment {} missing",
                seg_id
            )));
        }
        let file = File::open(&path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let mut handle = MmapHandle::from_parts(file, mmap);
        handle.seal();
        let arc = Arc::new(handle);
        self.sealed.insert(seg_id, arc.clone());
        Ok(arc)
    }

    #[allow(dead_code)]
    fn flush(&mut self) -> Result<()> {
        self.active_mmap.flush()?;
        Ok(())
    }

    /// 全部段 id（封存 + active）—— compact 记旧段用
    fn all_seg_ids(&self) -> Vec<u32> {
        let mut ids: Vec<u32> = self.sealed.keys().copied().collect();
        ids.push(self.active_id);
        ids
    }
}

fn vec_segment_file(seg_id: u32) -> String {
    format!("vec_payload_{:04}.mmap", seg_id)
}

/// 从 heed 加载全部 id→locator 进内存缓存（open/reopen 调用，热路径免 heed）。
/// 16B/项元数据（非向量字节，向量仍在 mmap 段，H1）。1M 规模约 16MB，可接受。
fn load_locator_cache(
    env: &Env,
    locator_db: &Database<Bytes, Bytes>,
) -> Result<HashMap<u64, ValueLocator>> {
    let txn = env.read_txn()?;
    let mut cache = HashMap::new();
    for item in locator_db.iter(&txn)? {
        let (key, val) = item?;
        let id = u64::from_le_bytes(key[..8].try_into().unwrap());
        let loc: ValueLocator = serde_json::from_slice(val)?;
        cache.insert(id, loc);
    }
    tracing::info!(count = cache.len(), "locator cache loaded from heed");
    Ok(cache)
}

fn next_vec_seg_id(data_dir: &Path) -> Result<u32> {
    let mut max_id: Option<u32> = None;
    for entry in std::fs::read_dir(data_dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        if let Some(id) = parse_vec_seg_id(&name) {
            max_id = Some(max_id.map_or(id, |m| m.max(id)));
        }
    }
    Ok(max_id.map_or(0, |m| m + 1))
}

fn parse_vec_seg_id(name: &str) -> Option<u32> {
    let name = name.strip_prefix("vec_payload_")?;
    let name = name.strip_suffix(".mmap")?;
    name.parse().ok()
}

fn open_or_create_vec_segment(path: &Path, size: u64) -> Result<File> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)?;
    let cur = file.metadata()?.len();
    if cur < size {
        file.set_len(size)?;
    }
    Ok(file)
}

/// 向量索引 —— vec mmap 段 + heed locator + HNSW 图常驻 RAM
pub struct VectorIndex {
    schema: VectorSchema,
    env: Env,
    locator_db: Database<Bytes, Bytes>,
    pool: RwLock<VecSegmentPool>,
    graph: RwLock<HnswGraph>,
    // locator 内存缓存 —— id→ValueLocator（16B/项，元数据非向量字节）。
    // 热路径距离闭包按 id 取 locator，免 heed read_txn + serde 反序列化（性能）。
    // 向量数据仍存 mmap 段非 RAM（H1）；仅缓存定位元数据（非 PRD 约束的向量字节）。
    locators: RwLock<HashMap<u64, ValueLocator>>,
}

impl VectorIndex {
    /// 新建/打开向量索引。
    /// ns_dir/{data,meta}：data 放 vec 段，meta 放 heed。
    pub fn open(ns_dir: &Path, schema: VectorSchema, seg_size: u64) -> Result<Self> {
        let data_dir = ns_dir.join("vec_data");
        let meta_dir = ns_dir.join("vec_meta");
        std::fs::create_dir_all(&meta_dir)?;
        let env = unsafe {
            EnvOpenOptions::new()
                .map_size(256 * 1024 * 1024)
                .max_dbs(8)
                // NO_SYNC：locator 元数据 commit 不 fsync；WAL 唯一同步点（H5）
                .flags(heed::EnvFlags::NO_SYNC)
                .open(&meta_dir)?
        };
        let mut wtxn = env.write_txn()?;
        let locator_db = env.create_database::<Bytes, Bytes>(&mut wtxn, Some(VEC_LOCATOR_DB))?;
        // 持久化 schema（恢复时读回）
        let schema_bytes = serde_json::to_vec(&schema)?;
        let schema_db = env
            .database_options()
            .types::<Bytes, Bytes>()
            .create(&mut wtxn)?;
        schema_db.put(&mut wtxn, SCHEMA_KEY, &schema_bytes)?;
        wtxn.commit()?;
        let pool = VecSegmentPool::open(&data_dir, seg_size, schema.dim)?;
        // 从 snapshot 段重载图拓扑到 RAM（H1）；无 snapshot 则新图
        let graph = crate::vector::snapshot::load(&data_dir)?.unwrap_or_else(HnswGraph::new);
        // 加载 locator 内存缓存（从 heed 读回全部 id→locator），热路径距离闭包免每跳 heed
        let locators = load_locator_cache(&env, &locator_db)?;
        tracing::info!(
            dim = schema.dim,
            metric = ?schema.metric,
            graph_nodes = graph.len(),
            locators = locators.len(),
            "vector index opened"
        );
        Ok(Self {
            schema,
            env,
            locator_db,
            pool: RwLock::new(pool),
            graph: RwLock::new(graph),
            locators: RwLock::new(locators),
        })
    }

    pub fn schema(&self) -> &VectorSchema {
        &self.schema
    }

    /// 重开已存在的向量索引（从 heed 读回建库时锁定的 schema）[v2 E2]
    /// FFI/CLI 无 schema 入参时用此；schema 未持久化则报 Corrupt。
    pub fn reopen(ns_dir: &Path, seg_size: u64) -> Result<Self> {
        let data_dir = ns_dir.join("vec_data");
        let meta_dir = ns_dir.join("vec_meta");
        let env = unsafe {
            EnvOpenOptions::new()
                .map_size(256 * 1024 * 1024)
                .max_dbs(8)
                // NO_SYNC：locator 元数据 commit 不 fsync；WAL 唯一同步点（H5）
                .flags(heed::EnvFlags::NO_SYNC)
                .open(&meta_dir)?
        };
        // 读回持久化 schema（建库时落 SCHEMA_KEY）
        let schema = {
            let rtxn = env.read_txn()?;
            let schema_db = env
                .database_options()
                .types::<Bytes, Bytes>()
                .open(&rtxn)?
                .ok_or_else(|| crate::StoreError::Corrupt("schema db missing".into()))?;
            let Some(schema_bytes) = schema_db.get(&rtxn, SCHEMA_KEY)? else {
                return Err(crate::StoreError::Corrupt(
                    "vector schema not persisted".into(),
                ));
            };
            serde_json::from_slice::<VectorSchema>(schema_bytes)?
        };
        // locator_db 可能已存在，用 create_database 幂等取句柄
        let mut wtxn = env.write_txn()?;
        let locator_db = env.create_database::<Bytes, Bytes>(&mut wtxn, Some(VEC_LOCATOR_DB))?;
        wtxn.commit()?;
        let pool = VecSegmentPool::open(&data_dir, seg_size, schema.dim)?;
        let graph = crate::vector::snapshot::load(&data_dir)?.unwrap_or_else(HnswGraph::new);
        let locators = load_locator_cache(&env, &locator_db)?;
        tracing::info!(
            dim = schema.dim,
            metric = ?schema.metric,
            graph_nodes = graph.len(),
            locators = locators.len(),
            "vector index reopened from persisted schema"
        );
        Ok(Self {
            schema,
            env,
            locator_db,
            pool: RwLock::new(pool),
            graph: RwLock::new(graph),
            locators: RwLock::new(locators),
        })
    }

    fn put_locator(&self, wtxn: &mut RwTxn, id: u64, loc: &ValueLocator) -> Result<()> {
        let key = id.to_le_bytes();
        let val = serde_json::to_vec(loc)?;
        self.locator_db.put(wtxn, &key, &val)?;
        // 同步内存缓存（热路径免 heed）
        self.locators.write().unwrap().insert(id, *loc);
        Ok(())
    }

    fn get_locator(&self, txn: &RoTxn, id: u64) -> Result<Option<ValueLocator>> {
        let key = id.to_le_bytes();
        let val = self.locator_db.get(txn, &key)?;
        match val {
            Some(bytes) => {
                let loc: ValueLocator = serde_json::from_slice(bytes)?;
                Ok(Some(loc))
            }
            None => Ok(None),
        }
    }

    /// 取某 id 的零拷贝向量切片（公开内部用，read_vec 的零拷贝替代）。
    /// compact 重建图时用 live_map（内存），不走此；此处留供测试/调试。
    #[allow(dead_code)]
    fn vec_of_slice(&self, id: u64) -> Result<crate::mem::mmap::ZeroCopyVec> {
        self.vec_slice(id)
    }

    /// 零拷贝取某 id 的向量切片（读 mmap 段，持 Arc<MmapHandle> 保活）。
    /// 图距离闭包热路径用：免 heed read_txn + serde 反序列化 + Vec 分配（性能 + H1 零拷贝）。
    /// locator 从内存缓存取（未命中回退 heed 读并补缓存）。
    fn vec_slice(&self, id: u64) -> Result<crate::mem::mmap::ZeroCopyVec> {
        let loc = {
            // 先查内存缓存（无锁争用热路径）；未命中回退 heed + 补缓存
            let cache = self.locators.read().unwrap();
            cache.get(&id).copied()
        };
        let loc = match loc {
            Some(l) => l,
            None => {
                // 缓存未命中：回退 heed 读 + 补缓存（罕见：重开后旧 id 未加载？理论全覆盖）
                let txn = self.env.read_txn()?;
                let l = self
                    .get_locator(&txn, id)?
                    .ok_or(crate::StoreError::NotFound)?;
                drop(txn);
                self.locators.write().unwrap().insert(id, l);
                l
            }
        };
        let mut pool = self.pool.write().unwrap();
        pool.vec_slice(&loc)
    }

    /// 距离闭包：查询向量到某 id 节点的距离。
    /// search_knn 用，按 id 零拷贝读向量算距离（ZeroCopyVec 无分配）。
    fn make_query_dist<'a>(&'a self, query: &'a [f32]) -> impl Fn(u64) -> f32 + 'a {
        move |id| {
            let v = match self.vec_slice(id) {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(id, error = %e, "knn dist: read vec failed, treat as max dist");
                    return f32::INFINITY;
                }
            };
            simd::distance(self.schema.metric, query, v.as_f32())
        }
    }

    pub fn insert(
        &self,
        id: u64,
        vector: &[f32],
        _timeout: Option<std::time::Duration>,
    ) -> Result<()> {
        self.schema.validate_dim(vector)?;
        // locator 写入用 wtxn（id 唯一性：重复 put_kv 覆盖，向量 id 重复报错）
        let mut wtxn = self.env.write_txn()?;
        if self.get_locator(&wtxn, id)?.is_some() {
            wtxn.abort();
            return Err(crate::StoreError::DuplicateVector(id));
        }
        let mut pool = self.pool.write().unwrap();
        let loc = pool.append(vector)?;
        self.put_locator(&mut wtxn, id, &loc)?;
        wtxn.commit()?;
        drop(pool);
        // 图连边：dist 闭包零拷贝读现存向量算距离
        let mut graph = self.graph.write().unwrap();
        let query = vector.to_vec();
        let dist = move |nid: u64| {
            // 连边距离 = 新向量到现存节点向量（零拷贝读 mmap）
            let v = match self.vec_slice(nid) {
                Ok(v) => v,
                Err(_) => return f32::INFINITY,
            };
            simd::distance(self.schema.metric, &query, v.as_f32())
        };
        graph.insert(id, &dist)?;
        tracing::info!(id, "vector inserted + graph linked");
        Ok(())
    }

    pub fn insert_batch(
        &self,
        items: &[(u64, &[f32])],
        _timeout: Option<std::time::Duration>,
    ) -> Result<()> {
        for (_, vec) in items {
            self.schema.validate_dim(vec)?;
        }
        let mut wtxn = self.env.write_txn()?;
        // 唯一性预检
        for (id, _) in items {
            if self.get_locator(&wtxn, *id)?.is_some() {
                wtxn.abort();
                return Err(crate::StoreError::DuplicateVector(*id));
            }
        }
        let mut pool = self.pool.write().unwrap();
        let mut locs = Vec::with_capacity(items.len());
        for (id, vec) in items {
            let loc = pool.append(vec)?;
            self.put_locator(&mut wtxn, *id, &loc)?;
            locs.push((*id, loc));
        }
        wtxn.commit()?;
        drop(pool);
        // 批量连边 —— dist 闭包零拷贝读 mmap 算距离（免 O(N) 预取全图入 RAM）。
        // 旧实现每批 rescan 全部 N 旧向量入 live_map → O(N²) 拷贝，1M 规模不可行。
        // 新实现：现存节点经 vec_slice(id) 零拷贝读 mmap（locator 内存缓存免 heed），
        //         本批新向量已 append 落段 + locator 入缓存，同样 vec_slice 可读。
        // 锁顺序 graph.write → vec_slice→pool.write：与 insert（pool 写后释放再 graph 写，
        // 不嵌套）不冲突，与 search_knn（graph.read → vec_slice→pool.write）一致序，无死锁。
        let mut graph = self.graph.write().unwrap();
        for (id, vec) in items {
            let query = vec.to_vec();
            let dist = move |nid: u64| match self.vec_slice(nid) {
                Ok(v) => simd::distance(self.schema.metric, &query, v.as_f32()),
                Err(_) => f32::INFINITY,
            };
            graph.insert(*id, &dist)?;
        }
        tracing::info!(n = items.len(), "vector batch inserted + graph linked");
        Ok(())
    }

    pub fn search_knn(
        &self,
        query: &[f32],
        top_k: usize,
        timeout: Option<std::time::Duration>,
    ) -> Result<Vec<(u64, f32)>> {
        self.schema.validate_dim(query)?;
        let deadline = timeout.map(|t| std::time::Instant::now() + t);
        let graph = self.graph.read().unwrap();
        let dist = self.make_query_dist(query);
        graph.search_knn(&dist, top_k, deadline)
    }

    pub fn delete(&self, id: u64, _timeout: Option<std::time::Duration>) -> Result<bool> {
        let mut graph = self.graph.write().unwrap();
        let removed = graph.delete(id);
        // locator 不删（向量字节 M4 compact 回收）；图软删即生效
        tracing::info!(id, removed, "vector soft-deleted");
        Ok(removed)
    }

    pub fn len(&self) -> usize {
        self.graph.read().unwrap().len()
    }

    // 图常驻 RAM 字节（HNSW 拓扑，向量数据在 mmap 段非 RAM）。
    // §776 #9 >10M 分层加载评估基建：经 /stats 暴露，未来按实际占用决策。
    pub fn graph_memory_usage(&self) -> usize {
        self.graph.read().unwrap().memory_usage()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// checkpoint：图拓扑 snapshot 落单 mmap 段 + 段 flush（H1 持久化）
    pub fn checkpoint(&self) -> Result<()> {
        let graph = self.graph.read().unwrap();
        let bytes = crate::vector::snapshot::serialize(&graph);
        let data_dir = {
            // data_dir = ns_dir/data；pool 持有，但 snapshot 文件存 data_dir
            // 用 pool.data_dir 字段——读锁取路径快照
            let pool = self.pool.read().unwrap();
            pool.data_dir.clone()
        };
        crate::vector::snapshot::persist(&data_dir, &bytes)?;
        let mut pool = self.pool.write().unwrap();
        pool.flush()?;
        tracing::info!("vector index checkpoint done");
        Ok(())
    }

    /// compact COW 原子切换 [v2 A3/H4]：重排有效向量到新段 + 重建图 + 原子切 heed。
    /// 新向量追加到当前 pool 的更高 seg_id（COW：不动旧段字节，旧段不可变 H4）。
    /// 单 heed 写事务原子切 locator 指向新段 → 旧段无新 reader 引用 → 延迟回收。
    /// 期间持写锁，读不阻塞（MVCC：in-flight reader 持旧 heed 快照 + 旧段 Arc）。
    /// 返回 (待回收旧段 id 集, 活向量数)。
    pub fn compact(&self) -> Result<(Vec<u32>, usize)> {
        // 1. 记旧段 id（compact 前全部 seg）—— 切换后这些段待回收
        let old_seg_ids: Vec<u32> = {
            let pool = self.pool.read().unwrap();
            pool.all_seg_ids()
        };
        // 2. 扫全部 locator，读回向量，过滤软删。
        // 锁顺序：先 graph.read 取软删 id 集（释放），再 pool.write 读向量——
        // 不与 insert_batch/search_knn 的 graph→pool 序反向嵌套，免死锁。
        let live: Vec<(u64, Vec<f32>)> = {
            let deleted: std::collections::HashSet<u64> = {
                let graph = self.graph.read().unwrap();
                graph.all_deleted_ids().collect()
            };
            let txn = self.env.read_txn()?;
            let mut pool = self.pool.write().unwrap();
            let mut live = Vec::new();
            for item in self.locator_db.iter(&txn)? {
                let (key, val) = item?;
                let id = u64::from_le_bytes(key[..8].try_into().unwrap());
                if deleted.contains(&id) {
                    continue;
                }
                let loc: ValueLocator = serde_json::from_slice(val)?;
                let vec = pool.read_vec(&loc)?;
                live.push((id, vec));
            }
            live
        };
        let live_count = live.len();
        let metric = self.schema.metric;
        // 3. 追加活向量到当前 pool（更高 seg_id，新段）
        let new_locs: Vec<(u64, ValueLocator)> = {
            let mut pool = self.pool.write().unwrap();
            let mut locs = Vec::with_capacity(live_count);
            for (id, vec) in &live {
                let loc = pool.append(vec)?;
                locs.push((*id, loc));
            }
            pool.flush()?;
            locs
        };
        // 4. 重建 HNSW 图：dist 闭包用内存活向量 map 算距离（不重读段，无并发）
        let live_map: HashMap<u64, Vec<f32>> = live.into_iter().collect();
        let new_graph = {
            let mut g = HnswGraph::new();
            for (id, vec) in &live_map {
                let dist = |nid: u64| match live_map.get(&nid) {
                    Some(v) => simd::distance(metric, vec, v),
                    None => f32::INFINITY,
                };
                g.insert(*id, &dist)?;
            }
            g
        };
        tracing::info!(
            live = live_count,
            "compact: vectors rewritten to new segs + graph rebuilt"
        );
        // 5. 原子切 heed locator：单写事务清旧 + 写新
        {
            let mut wtxn = self.env.write_txn()?;
            let mut stale_keys = Vec::new();
            for item in self.locator_db.iter(&wtxn)? {
                let (key, _) = item?;
                stale_keys.push(key.to_vec());
            }
            for k in &stale_keys {
                self.locator_db.delete(&mut wtxn, k)?;
            }
            for (id, loc) in &new_locs {
                self.put_locator(&mut wtxn, *id, loc)?;
            }
            wtxn.commit()?;
        }
        // 6. 内存图切换为新图
        {
            let mut graph = self.graph.write().unwrap();
            *graph = new_graph;
        }
        // 6.1 重建 locator 内存缓存 —— compact 删了软删 id 的 locator，旧缓存过期。
        // 直接以新活向量 locator 集替换缓存（避免读到已回收旧段）。
        {
            let mut cache = self.locators.write().unwrap();
            cache.clear();
            for (id, loc) in &new_locs {
                cache.insert(*id, *loc);
            }
        }
        // 7. dump 新 snapshot
        let data_dir = {
            let pool = self.pool.read().unwrap();
            pool.data_dir.clone()
        };
        let graph = self.graph.read().unwrap();
        let bytes = crate::vector::snapshot::serialize(&graph);
        crate::vector::snapshot::persist(&data_dir, &bytes)?;
        tracing::info!(
            live = live_count,
            old_segs = old_seg_ids.len(),
            "compact COW done: heed atomically switched, old segs queued for reclaim"
        );
        Ok((old_seg_ids, live_count))
    }

    /// 待回收旧段文件物理删除 [v2 A3 延迟回收]
    /// compact 后 caller 持旧 seg_ids，安全期后调此删文件（in-flight reader 已释放 Arc）。
    /// 删 active 段或仍被引用的段会致错；caller 须确保 seg 不再被新 locator 引用。
    pub fn reclaim_segments(&self, seg_ids: &[u32]) -> Result<()> {
        let data_dir = {
            let pool = self.pool.read().unwrap();
            pool.data_dir.clone()
        };
        for sid in seg_ids {
            let path = data_dir.join(vec_segment_file(*sid));
            if path.exists() {
                std::fs::remove_file(&path)?;
                tracing::info!(seg_id = sid, "reclaimed old segment file");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::schema::MetricKind;
    use tempfile::tempdir;

    fn make_index(dir: &Path, dim: usize, metric: MetricKind) -> VectorIndex {
        let schema = VectorSchema::new(dim, metric);
        VectorIndex::open(dir, schema, 0).unwrap()
    }

    #[test]
    fn insert_then_knn_returns_nearest() {
        let dir = tempdir().unwrap();
        let idx = make_index(dir.path(), 8, MetricKind::L2);
        for i in 0..10u64 {
            let mut v = vec![0.0f32; 8];
            v[0] = i as f32;
            idx.insert(i, &v, None).unwrap();
        }
        let q = vec![3.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let res = idx.search_knn(&q, 3, None).unwrap();
        assert!(!res.is_empty());
        assert_eq!(res[0].0, 3);
    }

    #[test]
    fn insert_duplicate_rejects() {
        let dir = tempdir().unwrap();
        let idx = make_index(dir.path(), 8, MetricKind::L2);
        let v = vec![1.0f32; 8];
        idx.insert(1, &v, None).unwrap();
        let err = idx.insert(1, &v, None).unwrap_err();
        assert!(matches!(err, crate::StoreError::DuplicateVector(1)));
    }

    #[test]
    fn delete_excludes_from_knn() {
        let dir = tempdir().unwrap();
        let idx = make_index(dir.path(), 8, MetricKind::L2);
        for i in 0..10u64 {
            let mut v = vec![0.0f32; 8];
            v[0] = i as f32;
            idx.insert(i, &v, None).unwrap();
        }
        assert!(idx.delete(3, None).unwrap());
        let q = vec![3.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let res = idx.search_knn(&q, 3, None).unwrap();
        assert!(!res.iter().any(|(id, _)| *id == 3));
    }

    #[test]
    fn batch_insert_then_knn() {
        let dir = tempdir().unwrap();
        let idx = make_index(dir.path(), 4, MetricKind::L2);
        let vecs: Vec<(u64, Vec<f32>)> = (0..20u64)
            .map(|i| {
                let mut v = vec![0.0f32; 4];
                v[0] = i as f32;
                (i, v)
            })
            .collect();
        let items: Vec<(u64, &[f32])> = vecs.iter().map(|(id, v)| (*id, v.as_slice())).collect();
        idx.insert_batch(&items, None).unwrap();
        let q = vec![5.0f32, 0.0, 0.0, 0.0];
        let res = idx.search_knn(&q, 3, None).unwrap();
        assert_eq!(res[0].0, 5);
    }

    #[test]
    fn checkpoint_then_reload_restores_graph() {
        let dir = tempdir().unwrap();
        let ns_dir = dir.path().to_path_buf();
        {
            let idx = make_index(&ns_dir, 8, MetricKind::L2);
            for i in 0..10u64 {
                let mut v = vec![0.0f32; 8];
                v[0] = i as f32;
                idx.insert(i, &v, None).unwrap();
            }
            assert_eq!(idx.len(), 10);
            idx.checkpoint().unwrap();
        }
        // 重开：图应从 snapshot 重载
        let idx2 = make_index(&ns_dir, 8, MetricKind::L2);
        assert_eq!(idx2.len(), 10, "graph reloaded from snapshot");
        let q = vec![3.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let res = idx2.search_knn(&q, 1, None).unwrap();
        assert_eq!(res[0].0, 3, "knn works after reload");
    }
}
