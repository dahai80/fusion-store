---
doc_id: fusion-store/config-reference
product: fusion-store
class: core
entry: service
doc_type: config-reference
title: { zh: "fusion-store 配置参考", en: "fusion-store config reference" }
audience: engineer
status: draft
version: 0.2.0-rc.4
updated: 2026-11-07
owner: @dahai
source_of_truth:
  - crates/fs-cli/src/main.rs
  - crates/fs-serve/src/main.rs
  - crates/fs-serve/src/lib.rs
  - deploy/start.sh
  - SECURITY.md
---

# fusion-store 配置参考（Config Reference）

fusion-store 无配置文件；全部经 CLI 参数 + 环境变量。CLI 全局参数 `--home` 可被 `FS_HOME` 覆盖，默认 `~/.fusion-store`。

## 1. 数据目录与 namespace

| 参数 | 作用域 | 默认 | 说明 |
|------|--------|------|------|
| `--home` / `FS_HOME` | fs-cli, fs-serve | `~/.fusion-store` | 数据根目录；每个 namespace 一个独立子目录（含独立 WAL + heed env + flock） |
| `--namespace` | fs-cli 子命令, fs-serve | `default` | namespace 名；单 namespace = 单 Engine（A4） |
| `--seg-size` | fs-cli init/put | 内置默认 | mmap 段大小；段满即 seal（只读）并开新段 |
| `--quota` | fs-cli init/put, fs-serve | 0（不限） | 单 namespace payload 净增配额（`QuotaExceeded` 拒写，不计段 padding/尾空洞） |
| `--dim` | fs-cli init/serve, fs-serve | 768 | 向量维度；**建库后锁定 schema，不可更改** |

## 2. fs-serve daemon

| 参数 / 环境变量 | 默认 | 说明 |
|-----------------|------|------|
| `--port` / `FUSION_STORE_PORT` | `11463` | 监听端口（port-registry 已注册） |
| `--bind` / `FS_BIND` | `127.0.0.1` | 监听地址；非环回绑定时强警告，且必须配 token（F-SEC-1） |
| `--queue-cap` / fs-cli `serve` 同名 | 0（自动） | 有界写队列容量；满则写请求 503 背压（A2） |
| `RUST_LOG` | `fs_serve=info,fs_core=info` | tracing `EnvFilter` 日志过滤（fs-serve；未设置时用内置默认） |

## 3. 认证与 RBAC（F-SEC-2 / R10）

三对环境变量，每对优先 env 直传、回退 token 文件：

| 环境变量 | 角色 | 权限 |
|----------|------|------|
| `FS_AUTH_TOKEN` / `FS_AUTH_TOKEN_FILE` | Admin（2） | 写 + compact + 读 |
| `FS_AUTH_TOKEN_RW` / `FS_AUTH_TOKEN_RW_FILE` | Readwrite（1） | 写 + 读，禁 compact |
| `FS_AUTH_TOKEN_RO` / `FS_AUTH_TOKEN_RO_FILE` | Readonly（0） | 仅读 |

- 角色比较用 `Ord`：admin ≥ readwrite ≥ readonly；token 常量时间比较（`constant_eq`）。
- 写端点（`/kv` `/vector` `/columnar`）≥Readwrite；`/admin/compact` 需 Admin；无 token 401，角色不足 403。
- 只读端点（`/health` `/metrics` `/knn`）开放；`/stats` 环回开放、非环回需 ≥Readonly（F-OPS-8）。
- **完全未配置任何 token** 时：环回绑定下匿名放行（内部按 Admin）；非环回绑定 + 无 token 拒绝启动（hard fail，F-SEC-1）。
- 请求体上限 16MB（`DefaultBodyLimit`），超限 413。

## 4. 基准环境变量

| 环境变量 | 说明 |
|----------|------|
| `FS_BENCH_N` / `FS_BENCH_DIM` / `FS_BENCH_TOP_K` | `crates/fs-core/examples/full_scale_bench.rs` 规模配置（例：`FS_BENCH_N=1000000 FS_BENCH_DIM=768`，需 release + ~4GB RAM） |

## 5. 部署（deploy/）

- `deploy/start.sh start|stop|status|restart`：读 `FS_HOME` / `FUSION_STORE_PORT` / `FS_BIND` / `FS_AUTH_TOKEN`；PID 与日志写 `$HOME_DIR/fs-serve.pid` / `fs-serve.log`；二进制路径 `target/release/fs-serve`。
- `deploy/Dockerfile`：容器化；容器内绑 `0.0.0.0` 时务必同时配 `FS_AUTH_TOKEN` + 网络层 ACL。
- `deploy/fusion-store.plist`：launchd 自启。
- `ops/grafana-dashboard.json`：Grafana 面板（吞吐 / p99 / 段用量 / 背压队列深度，8000 黄 / 10000 红阈值）。

## 6. 内置常量（改动需重编译，F-PERF-4）

| 常量 | 值 | 说明 |
|------|-----|------|
| KV LMDB map_size | 2GB | `MapFull` 返错拒写，无自动 grow |
| 列式 LMDB map_size | 2GB | 同上 |
| vec_meta LMDB map_size | 256MB | 同上 |
| compact 旧段回收延迟 | 60s（`reclaim_safe`） | 期间磁盘峰值约 2× 在库数据（F-PERF-5） |
| NSW 参数 | M=16 / ef_c=200 / ef_s=200 | 单层 NSW（非 HNSW，A2）；ef_s=200 为召回 0.995 的实测值 |
| 背压水位 | 80% degraded；8000 黄 / 10000 红 | `/health` 供上游熔断；Grafana 阈值一致 |
