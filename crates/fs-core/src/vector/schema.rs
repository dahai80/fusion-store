//! 向量库 schema —— 建库时锁定 dim/metric 强不变量 [v2 E2]

use serde::{Deserialize, Serialize};

/// 距离度量类型（建库时固定一种）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetricKind {
    Cosine,
    L2,
    Dot,
}

/// 向量库 schema —— insert/search 校验维度一致，不符报 DimensionMismatch [v2 E2]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VectorSchema {
    pub dim: usize,
    pub metric: MetricKind,
}

impl VectorSchema {
    pub fn new(dim: usize, metric: MetricKind) -> Self {
        Self { dim, metric }
    }

    /// 校验向量维度与 schema 一致，不符返回 DimensionMismatch [v2 E2]
    pub fn validate_dim(&self, vector: &[f32]) -> crate::Result<()> {
        if vector.len() == self.dim {
            Ok(())
        } else {
            Err(crate::StoreError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_dim_passes_on_match() {
        let schema = VectorSchema::new(768, MetricKind::Cosine);
        let v = vec![0.0f32; 768];
        assert!(schema.validate_dim(&v).is_ok());
    }

    #[test]
    fn validate_dim_rejects_mismatch() {
        let schema = VectorSchema::new(768, MetricKind::Cosine);
        let v = vec![0.0f32; 1024];
        let err = schema.validate_dim(&v).unwrap_err();
        match err {
            crate::StoreError::DimensionMismatch { expected, got } => {
                assert_eq!(expected, 768);
                assert_eq!(got, 1024);
            }
            _ => panic!("expected DimensionMismatch"),
        }
    }
}
