---
doc_id: fusion-store/troubleshooting
product: fusion-store
class: core
entry: service
doc_type: troubleshooting
title: { zh: "fusion-store 故障排查", en: "fusion-store troubleshooting" }
audience: engineer
status: draft
version: 0.2.0-rc.4
updated: 2026-11-07
owner: "@dahai"
source_of_truth:
  - README.md
  - SECURITY.md
  - ROADMAP.md
  - crates/fs-core/src/error.rs
  - crates/fs-serve/src/lib.rs
---

# fusion-store 故障排查（Troubleshooting）

## 症状速查表

| 症状 | 可能原因 | 处置 |
|------|----------|------|
| `get` 退出码 2 | key 不存在 | 属正常语义（未命中即退出码 2），非故障 |
| 写入返 `QuotaExceeded` | namespace 配额用尽（净 payload，不含段 padding） | `stats` 确认用量；调大 `--quota` 或 compact 回收空间 |
| 写入返 `MapFull` | LMDB map_size 固定上限（KV 2GB / 列式 2GB / vec_meta 256MB） | 改 `crates/fs-core/src/` 内常量重编（F-PERF-4，无自动 grow） |
| HTTP 写请求 503 | 有界写队列满（背压，A2） | 上游按 `/health`（degraded=80% 水位）熔断退避；调大 `--queue-cap` |
| HTTP 请求 413 | 请求体超 16MB | 拆小批次（向量分批、列式分表） |
| HTTP 401 / 403 | 无 token / 角色不足 | 写端点 ≥Readwrite，`/admin/compact` 需 Admin；核对 `FS_AUTH_TOKEN*` 三对变量 |
| 非环回绑定启动失败 | 非环回 + 无 token（hard fail，F-SEC-1） | 配 token 再启动；这是设计行为不是缺陷 |
| `/stats` 401 | 非环回绑定需 ≥Readonly（F-OPS-8） | 带 RO token 访问，或回到环回绑定 |
| 两进程同时写卡住 / `LockBusy` | flock 写锁争用 | put_kv/delete_kv 支持 flock 超时快速失败；确认没有绕过 Engine 直写段文件的进程（advisory 锁不拦，F-SEC-4） |
| 崩溃后数据"丢失" | WAL 未落 checkpoint 的写入会在 `Engine::open` 自动重放 | 先 `recover --dry-run` 查看；不要手工删 WAL/段文件 |
| 重开向量库后召回/检索异常 | close 前未 checkpoint，图快照缺失 | C-ABI/Python 侧 close 返 void（F-SEC-5），务必 close 前显式 `fs_store_checkpoint` / `store.checkpoint()` |
| Python `EnvAlreadyOpened` | 同进程同目录开了第二个 Store | 先 `del` 旧句柄再重开；跨进程不受影响 |
| Python 数组被拒 | 非 f32 或非连续 numpy 数组 | `np.ascontiguousarray(v, dtype=np.float32)` |
| `/health` 返回 `unavailable` | compact 进行中 | 等待完成；上游按 E11 熔断；磁盘峰值约 2× 在库数据（F-PERF-5），确认磁盘余量 |
| 检索延迟随数据量上涨 | 单层 NSW 复杂度 O(N^log M)，N 线性劣化 | 监控 `/stats` `vector_count`；逼近 1-2M 软上限时用 `ShardedEngine` 分片（ROADMAP Path A） |
| 批量写入吞吐低于 10K vecs/s | batch 过大（NSW 构建超线性成本） | batch ~100 实测 ~16K vecs/s；batch 500 → 5.9K、1000 → 4K，属固有成本非缺陷（R1） |
| 召回率不达标 | ef_s 过小 | ef_s=50 实测 0.925 不达标；ef_s=200 实测 0.995（参数为编译期常量） |

## 诊断命令

```bash
# CLI 视角：段/向量/配额水位
target/release/fs-cli --home ~/.fusion-store stats --namespace default

# 幂等崩溃恢复（先干跑）
target/release/fs-cli --home ~/.fusion-store recover --namespace default --dry-run
target/release/fs-cli --home ~/.fusion-store recover --namespace default

# 服务视角：健康 / 规模 / 指标
curl -s http://127.0.0.1:11463/health
curl -s http://127.0.0.1:11463/stats
curl -s http://127.0.0.1:11463/metrics | grep -E "fusion_store_(ops|backpressure|poison)"

# 日志
tail -f ~/.fusion-store/fs-serve.log          # deploy/start.sh 启动时
RUST_LOG=debug target/release/fs-serve --home ~/.fusion-store   # 前台调参
```

关键指标：`fusion_store_ops_latency_us`（p99 对照 5ms SLA）、`fusion_store_ops_total`（写入速率）、`poison_lock_recover`（写锁中毒恢复次数）、`kv_disk_bytes` / `vec_disk_bytes`（真实段占用，含 padding，大于净 payload）、`vector_count`（扩容水位）。

## 已知边界（非缺陷，按设计）

- **timeout 语义分层**（F-ERR-5）：写路径 flock 超时有效；insert_vector/search_knn/get_kv_zero_copy 传 None 无限等待；get_vector/list_vector_ids/delete_vector 忽略 timeout。卡住调用先查是否误信了读路径 timeout。
- **advisory flock**（F-SEC-4）：外部进程绕过 Engine 直写 mmap 段文件不受锁保护；恢复此类损坏需 `recover`，且不保证完整。
- **close 无错误码**（F-SEC-5）：`fs_store_close` 返 void，落盘失败仅 `tracing::error` 日志；检查日志确认无 `fs_store_close: engine close FAILED`。
- **单层 NSW 容量上限**（F-ARCH-3）：无数据正确性风险，仅性能劣化；分片决策见 `ROADMAP.md`。
- **零拷贝仅 Rust 进程内**（A7）：跨语言读强制拷贝是公开定位，不是性能回归。

## 上报问题

先跑上面诊断命令收集 `/stats` + `/metrics` + 日志片段；确认非已知边界后，在 GitHub 开 issue（英文），附复现步骤与 `fs-cli stats` 输出。
