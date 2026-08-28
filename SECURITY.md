# Security Policy

## Scope

fusion-store 是单机零拷贝存储引擎，设计为**本地优先**（local-first）：所有 I/O 限 127.0.0.1，无云依赖，无出站网络。本文档记录已知安全边界与漏洞上报流程。

## 认证与授权

fs-serve daemon（端口 11463）的访问控制：

- **RBAC 三角色**（`AuthRole`）：`Admin`（2，写 + compact + 读）/ `Readwrite`（1，写 + 读）/ `Readonly`（0，仅读）。角色比较用 `Ord` derive，admin ≥ readwrite ≥ readonly。
- **Token 来源**：三对环境变量，每对优先 env 直传，回退 token 文件：
  - `FS_AUTH_TOKEN` / `FS_AUTH_TOKEN_FILE` → Admin
  - `FS_AUTH_TOKEN_RW` / `FS_AUTH_TOKEN_RW_FILE` → Readwrite
  - `FS_AUTH_TOKEN_RO` / `FS_AUTH_TOKEN_RO_FILE` → Readonly
- **Token 比较**：常量时间比较（`constant_eq`），抗时序侧信道。
- **端点门禁**：
  - 写端点（`/kv` `/vector` `/columnar` `/admin/compact`）需 ≥ Readwrite（compact 需 Admin）；无 token → 401，角色不足 → 403。
  - 只读端点（`/health` `/metrics`）环回绑定（`bind_is_loopback=true`）放行；非环回绑定时 `/stats` 需 ≥ Readonly。
- **默认绑定**：127.0.0.1。`--bind` 覆盖至非环回时强警告——非环回部署必须配 token，否则 `/stats` 等暴露内部状态。

## 请求体限制

`DefaultBodyLimit::max(MAX_BODY_BYTES)`（16 MB）在反序列化前拦截超限请求体 → 413 Payload Too Large，防 OOM / 资源耗尽。

## 并发与崩溃安全

- **多进程写互斥**：`fs2` flock（advisory，`lock_exclusive` 阻塞或 `try_lock` 快速失败 `LockBusy`）。**advisory 锁**依赖所有写者遵守协议；外部进程绕过 Engine 直写 mmap 段文件锁不阻止（消费方不应绕过 API，见 README F-SEC-4）。
- **崩溃恢复**：WAL 唯一 crash-safe 同步点（fsync 落 WAL），mmap 段 + heed 元数据延迟刷。`Engine::open` 自动重放 WAL（幂等 seq/applied_seq），torn-frame 容错（截断不完整尾帧）。`poison_lock_recover` 指标暴露写锁中毒恢复次数。
- **配额**：单 namespace payload 净增配额（`QuotaExceeded`），不计段 padding/尾空洞。

## 已知边界（非缺陷，按设计取舍）

- **flock advisory**：非内核强制锁，单 namespace 单 Engine 设计下风险低（F-SEC-4）。
- **C-ABI close 返 void**：close 落盘失败仅记日志，caller 无错误码感知（F-SEC-5）；关键数据先调 `fs_store_checkpoint`（返错误码）再 close。
- **timeout 语义分层**：写路径（put_kv/delete_kv）实现 flock 超时；读/枚举路径部分忽略 timeout（见 README F-ERR-5）。
- **无加密静态数据**：本地磁盘明文，依赖 OS 文件权限保护。多租户隔离是消费方职责（单 namespace 单 Engine，A4）。

## 报告漏洞

私有披露：在 GitHub 仓库开 Security Advisory（Private vulnerability reporting），或邮件联系维护者。勿在公开 issue 讨论未修复漏洞。响应目标：72 小时内确认，7 日内初评。
