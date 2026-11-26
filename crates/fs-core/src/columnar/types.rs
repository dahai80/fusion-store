//! Columnar type mapping — self-maintained lightweight dtype (no arrow serde feature) [v2 H3]

use arrow::datatypes::DataType;

/// Fixed-width primitive column types supported in M3
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ColType {
    Int32,
    Int64,
    Float32,
    Float64,
    Boolean,
}

impl ColType {
    /// Map to the Arrow DataType
    pub fn to_arrow(&self) -> DataType {
        match self {
            ColType::Int32 => DataType::Int32,
            ColType::Int64 => DataType::Int64,
            ColType::Float32 => DataType::Float32,
            ColType::Float64 => DataType::Float64,
            ColType::Boolean => DataType::Boolean,
        }
    }

    /// Per-element byte width (L8: Boolean is bit-packed, stored per bit;
    /// append_aligned uses 1 = no alignment padding)
    pub fn byte_width(&self) -> usize {
        match self {
            ColType::Int32 | ColType::Float32 => 4,
            ColType::Int64 | ColType::Float64 => 8,
            // Boolean: bit-packed (1 value per bit), byte buffer alignment 1, no padding
            ColType::Boolean => 1,
        }
    }

    /// Infer from an Arrow DataType (M3-supported range only)
    pub fn from_arrow(dt: &DataType) -> Option<Self> {
        match dt {
            DataType::Int32 => Some(ColType::Int32),
            DataType::Int64 => Some(ColType::Int64),
            DataType::Float32 => Some(ColType::Float32),
            DataType::Float64 => Some(ColType::Float64),
            DataType::Boolean => Some(ColType::Boolean),
            _ => None,
        }
    }
}

/// Per-column persisted metadata (stored in heed)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ColumnMeta {
    pub name: String,
    pub dtype: ColType,
    pub seg_id: u32,
    pub offset: u32,
    pub len: u32, // column buffer length in bytes
}

/// Per-table persisted metadata (stored in heed, key = table_id)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TableMeta {
    pub row_count: usize,
    pub columns: Vec<ColumnMeta>,
}
