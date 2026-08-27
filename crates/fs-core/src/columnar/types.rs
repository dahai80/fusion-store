//! 列式类型映射 —— 自维护轻量 dtype（不依赖 arrow serde feature）[v2 H3]

use arrow::datatypes::DataType;

/// M3 支持的定宽 primitive 列类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ColType {
    Int32,
    Int64,
    Float32,
    Float64,
    Boolean,
}

impl ColType {
    /// 映射到 Arrow DataType
    pub fn to_arrow(&self) -> DataType {
        match self {
            ColType::Int32 => DataType::Int32,
            ColType::Int64 => DataType::Int64,
            ColType::Float32 => DataType::Float32,
            ColType::Float64 => DataType::Float64,
            ColType::Boolean => DataType::Boolean,
        }
    }

    /// 单元素字节宽（Boolean 按位但 buffer 存 u8 bitmap，M3 全 non-null 一字节/值占位）
    pub fn byte_width(&self) -> usize {
        match self {
            ColType::Int32 | ColType::Float32 => 4,
            ColType::Int64 | ColType::Float64 => 8,
            // Boolean：M3 用 1 字节/值简化（非位压缩），避免 bitmap 偏移复杂度
            ColType::Boolean => 1,
        }
    }

    /// 从 Arrow DataType 反推（仅 M3 支持范围）
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

/// 单列持久化元信息（落 heed）
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ColumnMeta {
    pub name: String,
    pub dtype: ColType,
    pub seg_id: u32,
    pub offset: u32,
    pub len: u32, // 列 buffer 字节长度
}

/// 表持久化元信息（落 heed，key = table_id）
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TableMeta {
    pub row_count: usize,
    pub columns: Vec<ColumnMeta>,
}
