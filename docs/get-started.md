---
doc_id: fusion-store/get-started
product: fusion-store
class: core
entry: service
doc_type: get-started
title: { zh: "fusion-store 快速上手", en: "fusion-store quick start" }
audience: engineer
status: draft
version: 0.2.0-rc.4
updated: 2026-11-07
owner: "@dahai"
source_of_truth:
  - README.md
  - crates/fs-cli/src/main.rs
  - crates/fs-serve/src/main.rs
---

# fusion-store 快速上手（Get Started）

> 目标：10 分钟内完成构建，并用 CLI 写入/读取第一批 KV 与向量数据。

## 1. 前置条件

- macOS（Apple Silicon），Rust toolchain 1.94（`rust-toolchain.toml` 已锁定）
- 端口 11463（已在 `~/fusion/architecture/port-registry.yaml` 注册，HTTP daemon 用）

## 2. 构建与测试

```bash
cd ~/fusion/fusion-store
cargo check                      # 类型检查
cargo build --release            # 产物：target/release/fs-cli + fs-serve
cargo test --workspace --exclude fs-ffi-py   # 全量测试（fs-ffi-py 单列 Python job）
```

## 3. CLI 首次体验（KV + 向量）

CLI 二进制为 `fs-cli`，数据目录 `--home` 默认 `~/.fusion-store`（可用 `FS_HOME` 环境变量覆盖）。日志走 stderr，stdout 只输出结果（便于脚本解析）。

```bash
# 初始化 namespace（--dim 锁定向量维度，创建后不可改）
target/release/fs-cli --home /tmp/fs-demo init --namespace demo --dim 768

# 写 KV
target/release/fs-cli --home /tmp/fs-demo put --namespace demo greeting hello

# 读 KV（不存在时退出码 2）
target/release/fs-cli --home /tmp/fs-demo get --namespace demo greeting

# 查看统计（segment 数 / 向量数 / 配额用量）
target/release/fs-cli --home /tmp/fs-demo stats --namespace demo
```

## 4. 启动 HTTP daemon

```bash
# 默认 127.0.0.1:11463；写端点需 token（见 config-reference）
FS_AUTH_TOKEN=dev-token target/release/fs-serve --home /tmp/fs-demo --namespace demo

# 另一个终端验证
curl -s http://127.0.0.1:11463/health
curl -s http://127.0.0.1:11463/stats
curl -s -X POST http://127.0.0.1:11463/kv -H "Authorization: Bearer dev-token" \
  -H "Content-Type: application/json" -d '{"key":"k1","value":"v1"}'
curl -s -X POST http://127.0.0.1:11463/knn -H "Content-Type: application/json" \
  -d '{"query":[0,0,0],"top_k":5}'   # 维度须与建库 --dim 一致
curl -s http://127.0.0.1:11463/metrics | head
```

写端点（`/kv` `/vector` `/columnar` `/admin/compact`）带 Bearer Token；只读端点（`/health` `/stats` `/metrics`）环回绑定下开放。

## 5. 生产启动方式

```bash
deploy/start.sh start|stop|status|restart   # 读 FS_HOME / FUSION_STORE_PORT / FS_BIND / FS_AUTH_TOKEN
```

## 6. 下一步

- 完整 API（Engine trait / HTTP / C-ABI / Python）：`docs/api-reference.md`
- 全部配置项与环境变量：`docs/config-reference.md`
- 内部结构（mmap 段 / NSW / WAL / compact）：`docs/architecture.md`
- 故障定位：`docs/troubleshooting.md`

## 7. Python 绑定（可选）

```bash
cd crates/fs-ffi-py && maturin develop    # 装入当前 venv（arm64 CPython）
python tests/test_binding.py              # 7 个绑定测试
```

```python
import numpy as np
import fusion_store

store = fusion_store.Store.open("/tmp/fs-demo-py", dim=768)
store.put_kv(b"k", b"v")
store.insert_vector(0, np.zeros(768, dtype=np.float32))
ids, dists = store.search_knn(np.zeros(768, dtype=np.float32), top_k=10, timeout_ms=100)
store.checkpoint()
```

约束提醒：Python 侧输入向量用 numpy f32 连续数组（零拷贝读入）；输出 ids/dists/get_kv 均为强制拷贝的 owned 对象；同进程同目录不能同时开两个 Store（LMDB 单写者 env），先 `del` 旧句柄再重开。
