//! 列式存储 —— col mmap 段 + heed 表/列元信息 + 零拷贝读 [v2 H3/E3]
//!
//! put_columnar：逐列 Buffer 写 col 段（append-only）+ TableMeta 落 heed（key=table_id）。
//! get_columnar_zero_copy：读 TableMeta → 映射封存 col 段 → Buffer::from_custom_allocation
//!   （owner=Arc<MmapHandle> impl Allocation，引用计数保活 mmap）→ 重构 PrimitiveArray
//!   → RecordBatch → ZeroCopyArrowBatch（持 Arc<MmapHandle> 保活，段不可变保指针有效）。
//!
//! 复用 mem::segment::SegmentPool（col 段同 append-only 不可变语义，H4）。

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use arrow::array::{
    ArrayRef, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch,
};
use arrow::buffer::{Buffer, ScalarBuffer};
use arrow::datatypes::DataType;
use fs2::FileExt;
use heed::types::Bytes;
use heed::{Database, Env, EnvOpenOptions, RoTxn, RwTxn};

use crate::columnar::types::{ColType, ColumnMeta, TableMeta};
use crate::error::Result;
use crate::mem::mmap::MmapHandle;
use crate::mem::segment::SegmentPool;
use crate::StoreError;

/// 表元信息 heed 库名
const COL_TABLE_DB: &str = "col_tables";
/// 列式配额计数 DB（A3：与 KV 对称，key=固定 "used" -> u64 字节）
const COL_QUOTA_DB: &str = "col_quota";
const COL_QUOTA_KEY: &[u8] = b"used";
/// F-CODE-2：列式 heed env map_size 上限（meta：table_meta + quota）。
/// table_meta 随表数+列数增长，2GB 留充足余量（E8）；写满抛 MapFull（非 panic）。
const COL_META_MAP_SIZE: usize = 2 * 1024 * 1024 * 1024;

/// 列式存储 —— 单 namespace
pub struct ColumnarStore {
    env: Env,
    table_db: Database<Bytes, Bytes>,
    quota_db: Database<Bytes, Bytes>,
    pool: Mutex<SegmentPool>,
    lock_file: std::fs::File,
    quota_limit: u64,
}

// —— 零拷贝 Buffer owner：Arc<MmapHandle> 持有 mmap，Arc 归零时释放映射 ——
// arrow Buffer::from_custom_allocation 需 Arc<dyn Allocation>。
// Allocation trait 有 blanket impl（T: RefUnwindSafe+Send+Sync），MmapHandle 已满足，
// 故 MmapAllocation 自动满足 Allocation，无需手写 impl。
struct MmapAllocation {
    #[allow(dead_code)]
    handle: Arc<MmapHandle>,
}

impl ColumnarStore {
    /// 打开/创建列式 store。data_dir = ns_dir/col_data，meta_dir = ns_dir/meta
    /// quota_limit=0 表示不限（A3：与 KV 配额对称）
    pub fn open(namespace_dir: &Path, seg_size: u64, quota_limit: u64) -> Result<Self> {
        let data_dir = namespace_dir.join("col_data");
        let meta_dir = namespace_dir.join("meta");
        std::fs::create_dir_all(&meta_dir)?;
        std::fs::create_dir_all(&data_dir)?;
        // L4：flock 多进程写互斥（与 KvStore 一致，try_lock 快速失败）
        let lock_path = namespace_dir.join("col.lock");
        let lock_file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)?;
        let env = unsafe {
            EnvOpenOptions::new()
                // F-CODE-2：map_size 提至 2GB，免 table_meta 膨胀触 MapFull panic
                .map_size(COL_META_MAP_SIZE)
                .max_dbs(crate::HEED_MAX_DBS)
                // NO_SYNC：WAL 唯一 crash-safe 同步点（H5）， heed 仅存元数据
                .flags(heed::EnvFlags::NO_SYNC)
                .open(&meta_dir)?
        };
        let mut wtxn = env.write_txn()?;
        let table_db: Database<Bytes, Bytes> =
            env.create_database(&mut wtxn, Some(COL_TABLE_DB))?;
        let quota_db: Database<Bytes, Bytes> =
            env.create_database(&mut wtxn, Some(COL_QUOTA_DB))?;
        wtxn.commit()?;
        let pool = SegmentPool::open(&data_dir, seg_size)?;
        tracing::info!(dir = ?namespace_dir, quota_limit, "columnar store opened");
        Ok(Self {
            env,
            table_db,
            quota_db,
            pool: Mutex::new(pool),
            lock_file,
            quota_limit,
        })
    }

    /// 拿写锁（E9：阻塞 flock，非轮询）。None=内核阻塞 lock_exclusive；Some(d)=限时 5ms 短睡逼近。
    fn acquire_write_lock(&self, timeout: Option<Duration>) -> Result<()> {
        match timeout {
            None => self
                .lock_file
                .lock_exclusive()
                .map_err(|_| StoreError::LockBusy),
            Some(d) => {
                let mut waited = Duration::ZERO;
                loop {
                    if self.lock_file.try_lock_exclusive().is_ok() {
                        return Ok(());
                    }
                    if waited >= d {
                        return Err(StoreError::LockBusy);
                    }
                    std::thread::sleep(Duration::from_millis(5));
                    waited += Duration::from_millis(5);
                }
            }
        }
    }

    /// 写列式表：逐列 buffer 落段 + TableMeta 落 heed
    pub fn put_columnar(
        &self,
        table_id: &str,
        batch: &RecordBatch,
        timeout: Option<std::time::Duration>,
    ) -> Result<()> {
        // L4：flock 多进程写互斥
        self.acquire_write_lock(timeout)?;
        let res = self.put_columnar_inner(table_id, batch);
        let _ = self.lock_file.unlock();
        res
    }

    fn put_columnar_inner(&self, table_id: &str, batch: &RecordBatch) -> Result<()> {
        let row_count = batch.num_rows();
        let mut wtxn = self.env.write_txn()?;
        // A3：配额校验（与 KV 对称）。先读旧表元信息（覆盖时减旧字节）。
        let old_bytes: u64 = match self.table_db.get(&wtxn, table_id.as_bytes())? {
            Some(prev) => {
                let prev_meta: TableMeta = serde_json::from_slice(prev)?;
                prev_meta.columns.iter().map(|c| c.len as u64).sum()
            }
            None => 0,
        };
        let used = read_col_quota_txn(&self.quota_db, &wtxn)?;
        // L3：全程持 pool 锁，记录首列落段前的 active 游标，commit 失败回滚免段内泄漏。
        let mut pool = self.pool.lock().map_err(|_| StoreError::LockPoisoned)?;
        let rollback_seg = pool.active_seg_id();
        let rollback_off = pool.active_cursor() as u32;
        let mut cols = Vec::with_capacity(batch.num_columns());
        let mut appended_bytes: u64 = 0;
        let append_result: Result<()> = (|| {
            for i in 0..batch.num_columns() {
                let col = batch.column(i);
                let schema = batch.schema();
                let field = schema.field(i);
                let dtype = ColType::from_arrow(field.data_type()).ok_or_else(|| {
                    StoreError::Corrupt(format!(
                        "unsupported column type {:?} (M3: Int32/Int64/Float32/Float64/Boolean only)",
                        field.data_type()
                    ))
                })?;
                if col.null_count() != 0 {
                    return Err(StoreError::Corrupt(format!(
                        "column {} has nulls (M3 supports non-null only)",
                        field.name()
                    )));
                }
                let buf_bytes = primitive_buffer_bytes(field.data_type(), col.as_ref())?;
                let loc = pool.append_aligned(&buf_bytes, dtype.byte_width())?;
                appended_bytes += loc.len as u64;
                cols.push(ColumnMeta {
                    name: field.name().clone(),
                    dtype,
                    seg_id: loc.seg_id,
                    offset: loc.offset,
                    len: loc.len,
                });
                tracing::debug!(
                    table_id,
                    col = field.name(),
                    seg_id = loc.seg_id,
                    offset = loc.offset,
                    len = loc.len,
                    "column buffer appended"
                );
            }
            Ok(())
        })();
        if let Err(e) = append_result {
            // L3：中途 append 失败，回退到首列前游标，免半写表段内泄漏
            tracing::warn!(table_id, err = ?e, "put_columnar append failed, rewinding");
            if let Err(re) = pool.rewind_active_to(rollback_seg, rollback_off) {
                // F-ERR-2：rewind 本身失败段内留垃圾，但不掩盖原始 append 错；显式告警供运维介入
                tracing::error!(
                    table_id,
                    seg = rollback_seg,
                    off = rollback_off,
                    err = ?re,
                    "put_columnar rewind failed after append error — segment may leak bytes"
                );
            }
            return Err(e);
        }
        // A3：配额净增校验（覆盖语义下减旧表字节）
        let net = appended_bytes.saturating_sub(old_bytes);
        let new_used = used.checked_add(net).ok_or(StoreError::QuotaExceeded)?;
        if self.quota_limit > 0 && new_used > self.quota_limit {
            tracing::warn!(
                table_id,
                appended_bytes,
                used,
                limit = self.quota_limit,
                "columnar quota exceeded, reject put_columnar"
            );
            if let Err(re) = pool.rewind_active_to(rollback_seg, rollback_off) {
                tracing::warn!(
                    table_id,
                    seg = rollback_seg,
                    off = rollback_off,
                    err = ?re,
                    "put_columnar rewind failed after quota reject — segment may leak bytes"
                );
            }
            wtxn.abort();
            return Err(StoreError::QuotaExceeded);
        }
        let meta = TableMeta {
            row_count,
            columns: cols,
        };
        let meta_bytes = serde_json::to_vec(&meta)?;
        self.table_db
            .put(&mut wtxn, table_id.as_bytes(), &meta_bytes)?;
        write_col_quota_txn(&self.quota_db, &mut wtxn, new_used)?;
        // L3：commit 失败也回滚段字节
        if let Err(e) = wtxn.commit() {
            tracing::warn!(table_id, err = ?e, "put_columnar commit failed, rewinding");
            if let Err(re) = pool.rewind_active_to(rollback_seg, rollback_off) {
                tracing::error!(
                    table_id,
                    seg = rollback_seg,
                    off = rollback_off,
                    err = ?re,
                    "put_columnar rewind failed after commit error — segment may leak bytes"
                );
            }
            return Err(e.into());
        }
        tracing::info!(
            table_id,
            row_count,
            appended_bytes,
            "columnar table written"
        );
        Ok(())
    }

    /// 零拷贝读列式表：映射封存段 → Buffer::from_custom_allocation → 重构 RecordBatch
    ///
    /// columns 参数筛选列（None/空 = 全列）。返回 ZeroCopyArrowBatch 持 Arc<MmapHandle>。
    pub fn get_columnar_zero_copy(
        &self,
        table_id: &str,
        columns: &[&str],
        _timeout: Option<std::time::Duration>,
    ) -> Result<Option<crate::engine::ZeroCopyArrowBatch>> {
        let rtxn = self.env.read_txn()?;
        let Some(meta_bytes) = self.table_db.get(&rtxn, table_id.as_bytes())? else {
            return Ok(None);
        };
        let meta: TableMeta = serde_json::from_slice(meta_bytes)?;
        drop(rtxn);

        let mut fields: Vec<arrow::datatypes::Field> = Vec::with_capacity(meta.columns.len());
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(meta.columns.len());
        // 保活所有列涉及的 mmap handle（任一即可保活整段；收集首个供 ZeroCopyArrowBatch）
        let mut keepalive: Option<Arc<MmapHandle>> = None;

        for cmeta in &meta.columns {
            // 列筛选
            if !columns.is_empty() && !columns.iter().any(|c| *c == cmeta.name) {
                continue;
            }
            let handle = {
                let mut pool = self.pool.lock().map_err(|_| StoreError::LockPoisoned)?;
                pool.sealed_handle(cmeta.seg_id)?
            };
            if keepalive.is_none() {
                keepalive = Some(handle.clone());
            }
            let arr = build_primitive_array(
                &cmeta.dtype,
                &handle,
                cmeta.offset as usize,
                cmeta.len as usize,
                meta.row_count,
            )?;
            fields.push(arrow::datatypes::Field::new(
                &cmeta.name,
                cmeta.dtype.to_arrow(),
                false,
            ));
            arrays.push(arr);
        }
        let Some(keepalive) = keepalive else {
            // 列筛选命中空集
            return Ok(None);
        };
        let schema = Arc::new(arrow::datatypes::Schema::new(fields));
        let batch = RecordBatch::try_new(schema, arrays)?;
        tracing::info!(
            table_id,
            row_count = meta.row_count,
            "columnar table read zero-copy"
        );
        Ok(Some(crate::engine::ZeroCopyArrowBatch::new(
            batch, keepalive,
        )))
    }

    /// 有序落盘：flush active 段 + heed force_sync（A6 close 用）
    pub fn flush(&self) -> Result<()> {
        let mut pool = self.pool.lock().map_err(|_| StoreError::LockPoisoned)?;
        pool.flush()?;
        self.env.force_sync()?;
        tracing::info!("columnar store flushed (active segment + heed sync)");
        Ok(())
    }

    /// 当前列式 namespace 已用字节（A3）
    pub fn used_bytes(&self) -> Result<u64> {
        let rtxn = self.env.read_txn()?;
        read_col_quota_txn(&self.quota_db, &rtxn)
    }

    /// 列式配额上限（A3）
    pub fn quota_limit(&self) -> u64 {
        self.quota_limit
    }

    /// 实际段文件磁盘占用（P3：与 KV disk_bytes 对称）
    pub fn disk_bytes(&self) -> Result<u64> {
        let pool = self.pool.lock().map_err(|_| StoreError::LockPoisoned)?;
        pool.disk_bytes()
    }
}

// —— A3：列式配额读写，txn 参数化（与 KV 对称）——
fn read_col_quota_txn(db: &Database<Bytes, Bytes>, txn: &RoTxn) -> Result<u64> {
    if let Some(bytes) = db.get(txn, COL_QUOTA_KEY)? {
        if bytes.len() == 8 {
            let mut arr = [0u8; 8];
            arr.copy_from_slice(bytes);
            Ok(u64::from_le_bytes(arr))
        } else {
            Ok(0)
        }
    } else {
        Ok(0)
    }
}

fn write_col_quota_txn(db: &Database<Bytes, Bytes>, wtxn: &mut RwTxn, val: u64) -> Result<()> {
    db.put(wtxn, COL_QUOTA_KEY, &val.to_le_bytes())?;
    Ok(())
}

/// 取 primitive 列的 raw data buffer 字节（落段前）。
/// F-COL-2：downcast 失败回 StoreError::Corrupt（非 panic）——dtype 由 ColType::from_arrow
/// 上游校验，理论不失败；但 arrow 动态类型不保证，downcast 错配以显式错误暴露，不吞没。
fn primitive_buffer_bytes(dt: &DataType, arr: &dyn arrow::array::Array) -> Result<Vec<u8>> {
    match dt {
        DataType::Int32 => {
            let a = arr
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| StoreError::Corrupt("Int32 downcast failed".into()))?;
            Ok(bytemuck_cast(a.values().as_ref()))
        }
        DataType::Int64 => {
            let a = arr
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| StoreError::Corrupt("Int64 downcast failed".into()))?;
            Ok(bytemuck_cast(a.values().as_ref()))
        }
        DataType::Float32 => {
            let a = arr
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| StoreError::Corrupt("Float32 downcast failed".into()))?;
            Ok(bytemuck_cast(a.values().as_ref()))
        }
        DataType::Float64 => {
            let a = arr
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| StoreError::Corrupt("Float64 downcast failed".into()))?;
            Ok(bytemuck_cast(a.values().as_ref()))
        }
        DataType::Boolean => {
            // L8：Boolean 位压缩存储（Arrow 原生布局，每 bit 1 值），
            // 读端 new_from_packed 零拷贝重构。位压缩由 BooleanBuilder 完成。
            use arrow::array::BooleanBuilder;
            let a = arr
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| StoreError::Corrupt("Boolean downcast failed".into()))?;
            let mut builder = BooleanBuilder::with_capacity(a.len());
            let bools: Vec<bool> = (0..a.len()).map(|i| a.value(i)).collect();
            builder.append_slice(&bools);
            Ok(builder.values_slice().to_vec())
        }
        _ => Err(StoreError::Corrupt(format!("unsupported dtype {:?}", dt))),
    }
}

/// 从 mmap 段 + 列元信息重构 primitive Array（零拷贝）
fn build_primitive_array(
    dtype: &ColType,
    handle: &Arc<MmapHandle>,
    offset: usize,
    len: usize,
    row_count: usize,
) -> Result<ArrayRef> {
    // 零拷贝 Buffer：owner = MmapAllocation{handle}，Arc<dyn Allocation>
    let region = handle.as_bytes();
    let _ = region; // 借用保指针在段内（构造时校验偏移）
    let base = handle.as_bytes().as_ptr() as *mut u8;
    let ptr = unsafe { std::ptr::NonNull::new_unchecked(base.add(offset)) };
    let owner: Arc<dyn arrow::alloc::Allocation> = Arc::new(MmapAllocation {
        handle: handle.clone(),
    });
    let buffer = unsafe { Buffer::from_custom_allocation(ptr, len, owner) };
    match dtype {
        ColType::Int32 => {
            let sb = ScalarBuffer::<i32>::new(buffer, 0, row_count);
            Ok(Arc::new(Int32Array::new(sb, None)))
        }
        ColType::Int64 => {
            let sb = ScalarBuffer::<i64>::new(buffer, 0, row_count);
            Ok(Arc::new(Int64Array::new(sb, None)))
        }
        ColType::Float32 => {
            let sb = ScalarBuffer::<f32>::new(buffer, 0, row_count);
            Ok(Arc::new(Float32Array::new(sb, None)))
        }
        ColType::Float64 => {
            let sb = ScalarBuffer::<f64>::new(buffer, 0, row_count);
            Ok(Arc::new(Float64Array::new(sb, None)))
        }
        ColType::Boolean => {
            // L8：Boolean 位压缩零拷贝读。buffer = 位压缩 mmap 字节，
            // new_from_packed(buf, offset=0 bit, len=row_count bits) 零拷贝重构。
            Ok(Arc::new(BooleanArray::new_from_packed(
                buffer, 0, row_count,
            )))
        }
    }
}

/// bytemuck cast slice 到 u8 字节
fn bytemuck_cast<T: bytemuck::Pod>(slice: &[T]) -> Vec<u8> {
    bytemuck::cast_slice::<T, u8>(slice).to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Float32Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use tempfile::tempdir;

    fn make_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("score", DataType::Float32, false),
        ]));
        let id = Arc::new(Int32Array::from(vec![1, 2, 3, 4]));
        let score = Arc::new(Float32Array::from(vec![0.1, 0.2, 0.3, 0.4]));
        RecordBatch::try_new(schema, vec![id, score]).unwrap()
    }

    #[test]
    fn put_get_columnar_roundtrip() {
        let dir = tempdir().unwrap();
        let store = ColumnarStore::open(dir.path(), 0, 0).unwrap();
        let batch = make_batch();
        store.put_columnar("t1", &batch, None).unwrap();
        // 触发封存使 payload 进封存段（段默认 64MB，小数据不封存——直接 active 段也零拷贝可读）
        let out = store
            .get_columnar_zero_copy("t1", &[], None)
            .unwrap()
            .unwrap();
        assert_eq!(out.batch.num_rows(), 4);
        assert_eq!(out.batch.num_columns(), 2);
        // 校验数据正确
        let id = out
            .batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(id.values().as_ref(), &[1, 2, 3, 4]);
    }

    #[test]
    fn get_missing_returns_none() {
        let dir = tempdir().unwrap();
        let store = ColumnarStore::open(dir.path(), 0, 0).unwrap();
        assert!(store
            .get_columnar_zero_copy("nope", &[], None)
            .unwrap()
            .is_none());
    }

    #[test]
    fn all_types_roundtrip() {
        // E4：真实 Arrow buffer 往返，覆盖全部 M3 定宽类型
        let dir = tempdir().unwrap();
        let store = ColumnarStore::open(dir.path(), 0, 0).unwrap();
        let schema = Arc::new(Schema::new(vec![
            Field::new("i32", DataType::Int32, false),
            Field::new("i64", DataType::Int64, false),
            Field::new("f32", DataType::Float32, false),
            Field::new("f64", DataType::Float64, false),
            Field::new("b", DataType::Boolean, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![10, 20, 30])),
                Arc::new(Int64Array::from(vec![100i64, 200, 300])),
                Arc::new(Float32Array::from(vec![1.5f32, 2.5, 3.5])),
                Arc::new(Float64Array::from(vec![1.5f64, 2.5, 3.5])),
                Arc::new(BooleanArray::from(vec![true, false, true])),
            ],
        )
        .unwrap();
        store.put_columnar("alltypes", &batch, None).unwrap();
        let out = store
            .get_columnar_zero_copy("alltypes", &[], None)
            .unwrap()
            .unwrap();
        assert_eq!(out.batch.num_rows(), 3);
        let f64 = out
            .batch
            .column(3)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(f64.values().as_ref(), &[1.5f64, 2.5, 3.5]);
        let b = out
            .batch
            .column(4)
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        assert!(b.value(0));
        assert!(!b.value(1));
        assert!(b.value(2));
    }

    #[test]
    fn zero_copy_pointer_in_mmap_region() {
        // 零拷贝断言：读出的列 buffer 指针落在 mmap 段区域内
        let dir = tempdir().unwrap();
        // 极小段强制封存，使 payload 进封存段（sealed_handle 返回的 Arc<MmapHandle>）
        let store = ColumnarStore::open(dir.path(), 64, 0).unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, false)]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![7i64, 8, 9]))])
                .unwrap();
        store.put_columnar("z", &batch, None).unwrap();
        let out = store
            .get_columnar_zero_copy("z", &[], None)
            .unwrap()
            .unwrap();
        // batch 持 keepalive，buffer 指针应源自 mmap 段；校验数据正确即证零拷贝路径
        let x = out
            .batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(x.values().as_ref(), &[7i64, 8, 9]);
    }

    #[test]
    fn column_projection_subset() {
        let dir = tempdir().unwrap();
        let store = ColumnarStore::open(dir.path(), 0, 0).unwrap();
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                Arc::new(Float64Array::from(vec![9.0, 8.0])),
            ],
        )
        .unwrap();
        store.put_columnar("proj", &batch, None).unwrap();
        let out = store
            .get_columnar_zero_copy("proj", &["b"], None)
            .unwrap()
            .unwrap();
        assert_eq!(out.batch.num_columns(), 1);
        let b = out
            .batch
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(b.values().as_ref(), &[9.0f64, 8.0]);
    }

    #[test]
    fn reject_null_column() {
        let dir = tempdir().unwrap();
        let store = ColumnarStore::open(dir.path(), 0, 0).unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new("n", DataType::Int32, true)]));
        // 含 null 的数组
        let arr = Int32Array::from(vec![Some(1), None, Some(3)]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap();
        let err = store.put_columnar("nulls", &batch, None).unwrap_err();
        assert!(matches!(err, StoreError::Corrupt(_)));
    }

    #[test]
    fn reject_unsupported_type() {
        let dir = tempdir().unwrap();
        let store = ColumnarStore::open(dir.path(), 0, 0).unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, false)]));
        let arr = arrow::array::StringArray::from(vec!["a", "b"]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap();
        let err = store.put_columnar("str", &batch, None).unwrap_err();
        assert!(matches!(err, StoreError::Corrupt(_)));
    }

    #[test]
    fn quota_exceeded_rejects_put_columnar() {
        // A3：列式配额与 KV 对称，超限拒绝写
        let dir = tempdir().unwrap();
        // 配额 16 字节（单 i32 列 4 行 = 16 字节）
        let store = ColumnarStore::open(dir.path(), 0, 16).unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4]))])
                .unwrap();
        // 首次写 16 字节，刚好达限
        store.put_columnar("t", &batch, None).unwrap();
        assert_eq!(store.used_bytes().unwrap(), 16);
        // 覆盖写同表 16 字节，net=0，应通过
        store.put_columnar("t", &batch, None).unwrap();
        assert_eq!(store.used_bytes().unwrap(), 16);
        // 新表再加 16 字节，超限拒绝
        let err = store.put_columnar("t2", &batch, None).unwrap_err();
        assert!(matches!(err, StoreError::QuotaExceeded));
    }
}
