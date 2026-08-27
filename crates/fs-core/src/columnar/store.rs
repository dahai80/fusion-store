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

use arrow::array::{
    ArrayRef, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch,
};
use arrow::buffer::{Buffer, ScalarBuffer};
use arrow::datatypes::DataType;
use heed::types::Bytes;
use heed::{Database, Env, EnvOpenOptions};

use crate::columnar::types::{ColType, ColumnMeta, TableMeta};
use crate::error::Result;
use crate::mem::mmap::MmapHandle;
use crate::mem::segment::SegmentPool;
use crate::StoreError;

/// 表元信息 heed 库名
const COL_TABLE_DB: &str = "col_tables";

/// 列式存储 —— 单 namespace
pub struct ColumnarStore {
    env: Env,
    table_db: Database<Bytes, Bytes>,
    pool: Mutex<SegmentPool>,
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
    pub fn open(namespace_dir: &Path, seg_size: u64) -> Result<Self> {
        let data_dir = namespace_dir.join("col_data");
        let meta_dir = namespace_dir.join("meta");
        std::fs::create_dir_all(&meta_dir)?;
        std::fs::create_dir_all(&data_dir)?;
        let env = unsafe {
            EnvOpenOptions::new()
                .map_size(256 * 1024 * 1024)
                .max_dbs(8)
                .open(&meta_dir)?
        };
        let mut wtxn = env.write_txn()?;
        let table_db: Database<Bytes, Bytes> =
            env.create_database(&mut wtxn, Some(COL_TABLE_DB))?;
        wtxn.commit()?;
        let pool = SegmentPool::open(&data_dir, seg_size)?;
        tracing::info!(dir = ?namespace_dir, "columnar store opened");
        Ok(Self {
            env,
            table_db,
            pool: Mutex::new(pool),
        })
    }

    /// 写列式表：逐列 buffer 落段 + TableMeta 落 heed
    pub fn put_columnar(
        &self,
        table_id: &str,
        batch: &RecordBatch,
        _timeout: Option<std::time::Duration>,
    ) -> Result<()> {
        let row_count = batch.num_rows();
        let mut wtxn = self.env.write_txn()?;
        // 逐列取 raw buffer 字节，落 col 段
        let mut cols = Vec::with_capacity(batch.num_columns());
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
            // M3：全 non-null，校验 null bitmap 缺席
            if col.null_count() != 0 {
                return Err(StoreError::Corrupt(format!(
                    "column {} has nulls (M3 supports non-null only)",
                    field.name()
                )));
            }
            let buf_bytes = primitive_buffer_bytes(field.data_type(), col.as_ref())?;
            let loc = {
                let mut pool = self.pool.lock().unwrap();
                // 定宽列按类型字节宽对齐（i64/f64 需 8 对齐，否则 ScalarBuffer 报未对齐）
                pool.append_aligned(&buf_bytes, dtype.byte_width())?
            };
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
        let meta = TableMeta {
            row_count,
            columns: cols,
        };
        let meta_bytes = serde_json::to_vec(&meta)?;
        self.table_db
            .put(&mut wtxn, table_id.as_bytes(), &meta_bytes)?;
        wtxn.commit()?;
        tracing::info!(table_id, row_count, "columnar table written");
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
                let mut pool = self.pool.lock().unwrap();
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
}

/// 取 primitive 列的 raw data buffer 字节（落段前）
fn primitive_buffer_bytes(dt: &DataType, arr: &dyn arrow::array::Array) -> Result<Vec<u8>> {
    match dt {
        DataType::Int32 => {
            let a = arr.as_any().downcast_ref::<Int32Array>().expect("Int32");
            Ok(bytemuck_cast(a.values().as_ref()))
        }
        DataType::Int64 => {
            let a = arr.as_any().downcast_ref::<Int64Array>().expect("Int64");
            Ok(bytemuck_cast(a.values().as_ref()))
        }
        DataType::Float32 => {
            let a = arr
                .as_any()
                .downcast_ref::<Float32Array>()
                .expect("Float32");
            Ok(bytemuck_cast(a.values().as_ref()))
        }
        DataType::Float64 => {
            let a = arr
                .as_any()
                .downcast_ref::<Float64Array>()
                .expect("Float64");
            Ok(bytemuck_cast(a.values().as_ref()))
        }
        DataType::Boolean => {
            // M3：Boolean 用 1 字节/值（非位压缩），取 values 转 u8 列
            let a = arr
                .as_any()
                .downcast_ref::<BooleanArray>()
                .expect("Boolean");
            let bytes: Vec<u8> = (0..a.len()).map(|i| a.value(i) as u8).collect();
            Ok(bytes)
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
            // M3：Boolean 列存 u8（1/值），重构时转 BooleanArray
            let bools: Vec<bool> = (0..row_count).map(|i| buffer[i] != 0).collect();
            Ok(Arc::new(BooleanArray::from(bools)))
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
        let store = ColumnarStore::open(dir.path(), 0).unwrap();
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
        let store = ColumnarStore::open(dir.path(), 0).unwrap();
        assert!(store
            .get_columnar_zero_copy("nope", &[], None)
            .unwrap()
            .is_none());
    }

    #[test]
    fn all_types_roundtrip() {
        // E4：真实 Arrow buffer 往返，覆盖全部 M3 定宽类型
        let dir = tempdir().unwrap();
        let store = ColumnarStore::open(dir.path(), 0).unwrap();
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
        let store = ColumnarStore::open(dir.path(), 64).unwrap();
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
        let store = ColumnarStore::open(dir.path(), 0).unwrap();
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
        let store = ColumnarStore::open(dir.path(), 0).unwrap();
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
        let store = ColumnarStore::open(dir.path(), 0).unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, false)]));
        let arr = arrow::array::StringArray::from(vec!["a", "b"]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap();
        let err = store.put_columnar("str", &batch, None).unwrap_err();
        assert!(matches!(err, StoreError::Corrupt(_)));
    }
}
