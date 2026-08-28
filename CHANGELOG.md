# Changelog

fusion-store 版本变更记录。格式遵循 [Keep a Changelog](https://keepachangelog.com/)，版本号语义化（SemVer）。

## [Unreleased]

- NSW 规模演进 Path A：`ShardedEngine` 分片薄层（fan-out 检索 + 路由写 + 归并 top-k，详见 ROADMAP）。排入 0.2.0-rc.2 → 0.2.0。

## [0.2.0-rc.1] — 2026-08-28 (Pre-release)

### 定位

**0.2 线首个 Release Candidate 基线**，非 feature-complete 候选。

- 当前态 = 受控商用 / 技术预览（审计 P0/P1 闭环，与 0.1.1 一致）。
- 本 RC 在 `ShardedEngine` 落地前作为基线锚点，便于后续演进以 rc.2 → 0.2.0 稳定迭代。
- 未含新功能：代码面与 0.1.1 相同。版本号上开 0.2 线，宣告进入 NSW 规模演进周期（ROADMAP Path A）。
- **Pre-release**：不建议直接用于无条件企业级生产商用。差距 = NSW 演进（F-ARCH-3，单层 NSW ≤1-2M 仅文档化）。

### 已知限制（沿用 0.1.1）

- 单层 NSW，无分层/分片，单 namespace 向量数软上限 ≤1-2M（`/stats` `vector_count` 暴露水位）。
- 零拷贝仅 Rust 进程内；C/Python 经 C-ABI 读强制拷贝（A7）。
- 单 namespace = 单 `Engine` 实例；多 namespace 是消费方职责（A4）。

### CI / 产物

- GitHub Actions runner 仍被账户 billing 阻断（非代码问题），wheel 产物暂不可自动构建。
- tag `v0.2.0-rc.1` + GitHub Pre-release 已落地。billing 解后 `gh run rerun` 触发 wheel。

## [0.1.1] — 2026-08-28

### 生产就绪审计修复（P0-P3，`audit/fusion-store-audit-result-product-0827.md`）

#### P0
- LICENSE 文件补齐（MIT）+ `rust-toolchain.toml` 锁 toolchain 1.94。

#### P1
- CI 矩阵（fmt + clippy + check + test debug/release）+ wheel pytest job + `Cargo.lock` 入仓 + release profile（LTO + strip）+ workspace deps 统一。
- 部署：Dockerfile + launchd plist + `start.sh` start/stop。
- 代码：`--bind` 可配 + `EF_SEARCH` 可配 + `KV_MAP_SIZE` 常量化 + metrics 注册降级（重名不 panic）+ rewind/snapshot 日志补全。
- 安全：非环回 + 无 token 拒启动 hard fail（F-SEC-1 / Rule 12，`enforce_bind_auth_policy`，fs-serve + fs-cli serve 共用，可测）。

#### P2
- 代码：RBAC 三角色（Admin/Readwrite/Readonly）+ audit log + token 文件回退 + body limit 16MB + `/stats` 非环回认证门禁 + poison lock 指标。
- 测试：proptest fuzz harness（F-TEST-2）—— WAL torn-frame 容错、并发 compact+insert+search、边界 dim/value/KNN top_k，9 属性测试。
- 文档：compact 2× 磁盘峰值（F-PERF-5）+ flock advisory（F-SEC-4）+ timeout 语义分层（F-ERR-5）+ 列式定长原语限定（F-COL-1）+ 零拷贝进程内限定（F-ARCH-5）+ NSW 容量边界 ≤1-2M（F-ARCH-3）。

#### P3
- 代码：columnar `expect` → 语义化（F-COL-2，前会话完成）；CI 显式 `--exclude fs-ffi-py` + 单列 Python job（F-OPS-3）。
- 文档：C-ABI 12 符号计数同步（F-API-1，9→12）；C-ABI 列式不暴露边界（F-API-2，设计取舍）；close 返 void 文档化（F-SEC-5，头文件 + README）。
- 运维：`ops/grafana-dashboard.json` 模板（F-OPS-7）；CHANGELOG.md + SECURITY.md + RELEASING.md（F-OPS-4）。

## [0.1.0] — 审计前基线（commit 47d5b83）

### 里程碑（PRD v2.0 roadmap 全落地）
- **M0** — workspace 骨架 + fs-core trait。
- **M1** — mmap 段 + 块分配器 + heed KV + namespace 配额。
- **M2** — 单层 NSW 图常驻 RAM + snapshot + NEON SIMD 位等 + batch。全规模 1M×768 基准：召回 0.995 / p99 1544us / 图 RAM 415MB。
- **M3** — Arrow 定长原语列式 + C-ABI（12 符号）+ Python 绑定 fs-ffi-py（PyO3，读强制拷贝）。
- **M4** — WAL 幂等 + recover + compact COW + fs-serve (11463) + prometheus + 背压 + SLA。单条 put_kv ~183K ops/s。

### 审计纠偏（A2/A4/A7/E6，诚实性修正）
- A2：单层 NSW 误称 HNSW → 全量改名 `NswGraph`，明确 O(N^log M) 复杂度与扩展上限。
- A4：单 namespace 单 Engine 文档化为有意设计。
- A7：FFI 零拷贝诚实文档化为仅 Rust 进程内，C/Python 强制拷贝。
- E6：锁顺序不变量文档化（insert/search/compact 三路径锁获取顺序固定）。

### 缺陷修复
- 审计 30 缺陷全修复（A1-A7 / R1-R10 / E1-E13）。
- issue #2/#3/#4：delete_vector / get_vector / list_vector_ids 三层暴露（VectorIndex → trait → Engine → C-ABI → Python）。
