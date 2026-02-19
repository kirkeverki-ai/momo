use std::process::Command;

#[test]
fn startup_fails_closed_without_api_keys() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");

    let output = Command::new(env!("CARGO_BIN_EXE_momo"))
        .current_dir(temp_dir.path())
        .arg("--mode")
        .arg("api")
        .env_remove("MOMO_API_KEYS")
        .env("MOMO_ALLOW_NO_AUTH", "0")
        .env("MOMO_HOST", "127.0.0.1")
        .env("MOMO_MCP_ENABLED", "false")
        .output()
        .expect("run momo binary");

    assert!(
        !output.status.success(),
        "expected startup to fail without API keys"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let combined = format!("{stderr}\n{stdout}");

    assert!(
        combined.contains("Refusing to start without API keys"),
        "expected fail-closed error message, got: {combined}"
    );
}
