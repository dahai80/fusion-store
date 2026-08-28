# Releasing

fusion-store 发布流程。版本号语义化（SemVer），当前 0.1.0（M0-M4 + 审计修复完成，预 1.0）。

## 前置检查（发布门禁）

全部须绿，任何一项红即阻断：

```bash
# 1. 格式 + lint（fs-ffi-py 排除：PyO3 cdylib 无 Rust 测试，单列 Python job）
cargo fmt --check
cargo clippy --workspace --all-targets --exclude fs-ffi-py -- -D warnings

# 2. 测试（debug + release 双绿，131 Rust 测试 + 9 proptest fuzz）
cargo test --workspace --exclude fs-ffi-py
cargo test --workspace --exclude fs-ffi-py --release

# 3. Python 绑定（需 maturin + numpy 环境，7 测试）
cd crates/fs-ffi-py && maturin develop && cd ../..
python crates/fs-ffi-py/tests/test_binding.py

# 4. C-ABI 真实链接验证（fs-ffi-c examples，cc 链接 libfusion_store.a 运行）
cargo build -p fs-ffi-c --release
# （见 crates/fs-ffi-c 下的 C 示例，cc app.c ... 链接运行）
```

CI 自动跑 1-2（`.github/workflows/ci.yml`），wheel 构建 3（`.github/workflows/wheel.yml`，push tag `v*` 触发）。本地发布前手动全跑确认。

## 版本号 bump

workspace 单版本（`Cargo.toml` `[workspace.package] version`），各 crate `version.workspace = true` 同步：

```bash
# bump patch/minor/major（手动改 Cargo.toml version 字段，或 cargo-release）
# 当前 0.1.0 → 0.1.1（审计修复）/ 0.2.0（新功能）/ 1.0.0（API 稳定）
```

## 发版步骤

1. **门禁全绿**（上节 1-4）。
2. **更新 CHANGELOG.md**：本次版本条目，从 `Unreleased` 移至版本号 + 日期。
3. **更新 README.md 状态行**：测试数 / 里程碑 / 缺陷修复计数同步。
4. **commit**：`chore(release): vX.Y.Z`（中文 commit message 用 `发布 vX.Y.Z`）。
5. **tag**：`git tag vX.Y.Z && git push origin vX.Y-Z`——触发 wheel.yml 构建 arm64 wheel 并上传 artifact。
6. **GitHub Release**：基于 tag 建 Release，贴 CHANGELOG 段 + wheel artifact 下载链接。

## 产物

| 产物 | 来源 | 用途 |
|------|------|------|
| `libfusion_store.a` / `.dylib` | `cargo build -p fs-ffi-c --release` | C/Swift 消费方链接 |
| `fusion_store` wheel (arm64) | wheel.yml（tag 触发） | Python `pip install` |
| `fusion-store` 二进制 | `cargo build -p fs-cli --release` | CLI 管理 |

## 回滚

无在线迁移（单机本地存储，schema 稳定）。回滚 = 装旧版本二进制/wheel。WAL 幂等重放保证旧版重开新版写的目录不损坏（同 WAL 帧格式）。若 schema 不兼容（未来 dim 变更）走 namespace 目录隔离，新目录不污染旧。
