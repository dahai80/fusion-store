//! fs-cli 集成测试 —— 真实子进程跑二进制 [v2 M4d/E4]
//!
//! 不 mock：cargo 编译 fs-cli，测试用 std::process::Command 起子进程，
//! 各子命令真实落盘 + 校验输出。CARGO_BIN_EXE_fs-cli 给出二进制路径。

use std::path::PathBuf;
use std::process::Command;

use tempfile::tempdir;

fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_fs-cli"))
}

fn run(home: &PathBuf, args: &[&str]) -> (bool, String, String) {
    let out = Command::new(bin())
        .arg("--home")
        .arg(home)
        .args(args)
        .output()
        .expect("run fs-cli");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).to_string(),
        String::from_utf8_lossy(&out.stderr).to_string(),
    )
}

#[test]
fn init_put_get_roundtrips() {
    let dir = tempdir().unwrap();
    let home = dir.path().to_path_buf();
    let (ok, out, _) = run(&home, &["init", "--namespace", "ns1", "--dim", "8"]);
    assert!(ok, "init success");
    assert!(out.contains("initialized"), "init output: {out}");

    let (ok, out, _) = run(&home, &["put", "--namespace", "ns1", "k1", "v1-fusion"]);
    assert!(ok, "put success");
    assert!(out.contains("put ok"), "put output: {out}");

    let (ok, out, _) = run(&home, &["get", "--namespace", "ns1", "k1"]);
    assert!(ok, "get success");
    assert_eq!(out.trim(), "v1-fusion", "get returns value");
}

#[test]
fn get_missing_exits_nonzero() {
    let dir = tempdir().unwrap();
    let home = dir.path().to_path_buf();
    run(&home, &["init", "--namespace", "ns2", "--dim", "4"]);
    let (ok, out, _) = run(&home, &["get", "--namespace", "ns2", "nope"]);
    assert!(!ok, "missing key exits nonzero");
    assert!(out.contains("not found"), "missing key output: {out}");
}

#[test]
fn stats_reports_namespace_state() {
    let dir = tempdir().unwrap();
    let home = dir.path().to_path_buf();
    run(&home, &["init", "--namespace", "ns3", "--dim", "16"]);
    run(&home, &["put", "--namespace", "ns3", "k", "val"]);
    let (ok, out, _) = run(&home, &["stats", "--namespace", "ns3"]);
    assert!(ok, "stats success");
    assert!(out.contains("namespace: ns3"), "stats namespace: {out}");
    assert!(out.contains("kv_used_bytes:"), "stats kv_used: {out}");
    assert!(out.contains("vector_dim: 16"), "stats dim: {out}");
}

#[test]
fn compact_runs_on_empty_namespace() {
    // compact 空索引 noop（不 panic）
    let dir = tempdir().unwrap();
    let home = dir.path().to_path_buf();
    run(&home, &["init", "--namespace", "ns4", "--dim", "4"]);
    let (ok, out, _) = run(&home, &["compact", "--namespace", "ns4"]);
    assert!(ok, "compact success");
    assert!(out.contains("compact ok"), "compact output: {out}");
    assert!(
        out.contains("live_vectors=0"),
        "empty compact live 0: {out}"
    );
}

#[test]
fn recover_dry_run_on_fresh_namespace() {
    // 新 ns 无 WAL 待重放，dry_run 报 nothing
    let dir = tempdir().unwrap();
    let home = dir.path().to_path_buf();
    run(&home, &["init", "--namespace", "ns5", "--dim", "4"]);
    let (ok, out, _) = run(&home, &["recover", "--namespace", "ns5", "--dry-run"]);
    assert!(ok, "recover dry_run success");
    assert!(
        out.contains("to_replay=0"),
        "fresh recover no entries: {out}"
    );
}
