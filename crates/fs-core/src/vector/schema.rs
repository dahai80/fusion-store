//! 向量库 schema —— 建库时锁定 dim/metric/ef_search 强不变量 [v2 E2]

use serde::{Deserialize, Serialize};

use crate::vector::nsw::DEFAULT_EF_SEARCH;

/// 距离度量类型（建库时固定一种）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetricKind {
    Cosine,
    L2,
    Dot,
}

/// 向量库 schema —— insert/search 校验维度一致，不符报 DimensionMismatch [v2 E2]
///
/// F-ARCH-2：`ef_search` 建库时锁定，检索经此值定候选集宽度（与 top_k 取大者）。
/// 默认 200（达 PRD §2.5 ≥0.95 召回 SLA）；建库时可调（高召回场景调高、低延迟场景调低）。
/// serde default：旧 snapshot/schema 未持久化此字段时回落 DEFAULT_EF_SEARCH（向后兼容）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VectorSchema {
    pub dim: usize,
    pub metric: MetricKind,
    #[serde(default = "default_ef_search")]
    pub ef_search: usize,
}

fn default_ef_search() -> usize {
    DEFAULT_EF_SEARCH
}

impl VectorSchema {
    pub fn new(dim: usize, metric: MetricKind) -> Self {
        Self {
            dim,
            metric,
            ef_search: DEFAULT_EF_SEARCH,
        }
    }

    /// 显式指定 ef_search 建库（F-ARCH-2）。0 或缺失回落默认。
    pub fn with_ef_search(dim: usize, metric: MetricKind, ef_search: usize) -> Self {
        Self {
            dim,
            metric,
            ef_search: if ef_search == 0 {
                DEFAULT_EF_SEARCH
            } else {
                ef_search
            },
        }
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
