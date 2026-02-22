#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

#[cfg(feature = "runtime-download")]
use flate2::read::GzDecoder;
#[cfg(feature = "runtime-download")]
use serde::Deserialize;
use serde::Serialize;
#[cfg(feature = "runtime-download")]
use sha2::{Digest, Sha256};
use std::env;
#[cfg(feature = "runtime-download")]
use std::fs::File;
use std::fs::{self, OpenOptions};
#[cfg(feature = "runtime-download")]
use std::io::{copy, Read};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::thread::sleep;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tauri::Manager;
use wait_timeout::ChildExt;

const STARTUP_TIMEOUT: Duration = Duration::from_secs(45);
const HEALTH_CHECK_INTERVAL: Duration = Duration::from_millis(350);
const SHUTDOWN_GRACE: Duration = Duration::from_secs(5);
const MAX_LOG_FILES: usize = 10;

type SharedSupervisor = Mutex<SupervisorState>;

#[derive(Default)]
struct SupervisorState {
    backend: Option<BackendProcess>,
}

struct BackendProcess {
    child: Child,
    base_url: String,
    health_url: String,
    runtime_dir: PathBuf,
    log_file: PathBuf,
}

#[derive(Clone, Serialize)]
struct StartupResponse {
    base_url: String,
    health_url: String,
    runtime_dir: String,
    log_file: String,
}

impl BackendProcess {
    fn as_response(&self) -> StartupResponse {
        StartupResponse {
            base_url: self.base_url.clone(),
            health_url: self.health_url.clone(),
            runtime_dir: self.runtime_dir.display().to_string(),
            log_file: self.log_file.display().to_string(),
        }
    }
}

#[tauri::command]
fn start_backend(
    app: tauri::AppHandle,
    state: tauri::State<SharedSupervisor>,
) -> Result<StartupResponse, String> {
    let mut guard = state
        .lock()
        .map_err(|_| "Failed to acquire backend state lock".to_string())?;

    if let Some(existing) = guard.backend.as_mut() {
        if is_healthy(&existing.health_url) {
            return Ok(existing.as_response());
        }
        terminate_backend(existing);
        guard.backend = None;
    }

    let app_data_dir = resolve_app_data_dir(&app)?;
    fs::create_dir_all(&app_data_dir)
        .map_err(|err| format!("Failed to create app data directory: {err}"))?;

    let runtime_dir = resolve_runtime_dir(&app, &app_data_dir)?;
    let runtime = validate_runtime_layout(&runtime_dir)?;
    let log_file = create_runtime_log_file(&app_data_dir)?;
    let port = reserve_local_port()?;
    let base_url = format!("http://127.0.0.1:{port}");
    let health_url = format!("{base_url}/health");

    let mut child = spawn_backend(
        &runtime.python_executable,
        &runtime.backend_app_dir,
        &runtime.web_dir,
        &app_data_dir,
        &log_file,
        port,
    )?;

    if let Err(err) = wait_for_health(&mut child, &health_url) {
        terminate_child(&mut child);
        return Err(format!(
            "{err}\nRuntime: {}\nLogs: {}",
            runtime_dir.display(),
            log_file.display()
        ));
    }

    let process = BackendProcess {
        child,
        base_url: base_url.clone(),
        health_url: health_url.clone(),
        runtime_dir: runtime_dir.clone(),
        log_file: log_file.clone(),
    };
    let response = process.as_response();
    guard.backend = Some(process);
    Ok(response)
}

#[tauri::command]
fn backend_status(state: tauri::State<SharedSupervisor>) -> Option<StartupResponse> {
    let mut guard = state.lock().ok()?;
    let existing = guard.backend.as_mut()?;
    if is_healthy(&existing.health_url) {
        return Some(existing.as_response());
    }
    terminate_backend(existing);
    guard.backend = None;
    None
}

#[tauri::command]
fn stop_backend(state: tauri::State<SharedSupervisor>) -> Result<(), String> {
    let mut guard = state
        .lock()
        .map_err(|_| "Failed to acquire backend state lock".to_string())?;
    if let Some(existing) = guard.backend.as_mut() {
        terminate_backend(existing);
    }
    guard.backend = None;
    Ok(())
}

#[tauri::command]
fn save_tokenizer_artifact(file_name: String, bytes: Vec<u8>) -> Result<Option<String>, String> {
    let saved_path = save_artifact_with_native_dialog(&file_name, &bytes)?;
    Ok(saved_path.map(|path| path.display().to_string()))
}

fn sanitize_suggested_file_name(file_name: &str) -> String {
    let trimmed = file_name.trim();
    let candidate = if trimmed.is_empty() {
        "tokenizer-artifact.json"
    } else {
        trimmed
    };

    let sanitized = candidate
        .chars()
        .map(|ch| match ch {
            '/' | '\\' | ':' => '_',
            _ => ch,
        })
        .collect::<String>();

    if sanitized.trim().is_empty() {
        "tokenizer-artifact.json".to_string()
    } else {
        sanitized
    }
}

#[cfg(target_os = "macos")]
fn apple_script_string_literal(value: &str) -> String {
    let escaped = value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n");
    format!("\"{escaped}\"")
}

#[cfg(target_os = "macos")]
fn save_artifact_with_native_dialog(file_name: &str, bytes: &[u8]) -> Result<Option<PathBuf>, String> {
    let suggested_name = sanitize_suggested_file_name(file_name);
    let script = format!(
        "set chosenFile to choose file name with prompt {} default name {}\nPOSIX path of chosenFile",
        apple_script_string_literal("Save tokenizer artifact"),
        apple_script_string_literal(&suggested_name)
    );

    let output = Command::new("osascript")
        .arg("-e")
        .arg(script)
        .output()
        .map_err(|err| format!("Failed to open macOS Save dialog: {err}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let message = stderr.trim();
        let lower = message.to_ascii_lowercase();
        if lower.contains("user canceled") || lower.contains("user cancelled") {
            return Ok(None);
        }
        return Err(format!("Failed to open macOS Save dialog: {message}"));
    }

    let path_text = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if path_text.is_empty() {
        return Ok(None);
    }

    let path = PathBuf::from(path_text);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            format!(
                "Failed to prepare destination directory {}: {err}",
                parent.display()
            )
        })?;
    }
    fs::write(&path, bytes)
        .map_err(|err| format!("Failed to write artifact to {}: {err}", path.display()))?;

    Ok(Some(path))
}

#[cfg(not(target_os = "macos"))]
fn save_artifact_with_native_dialog(
    _file_name: &str,
    _bytes: &[u8],
) -> Result<Option<PathBuf>, String> {
    Err("Native Save As is currently implemented only for macOS builds".to_string())
}

fn main() {
    let app = tauri::Builder::default()
        .manage(Mutex::new(SupervisorState::default()))
        .invoke_handler(tauri::generate_handler![
            start_backend,
            backend_status,
            stop_backend,
            save_tokenizer_artifact
        ])
        .build(tauri::generate_context!())
        .expect("failed to build Tokenizer Studio desktop app");

    app.run(|app_handle, event| {
        if matches!(
            event,
            tauri::RunEvent::Exit | tauri::RunEvent::ExitRequested { .. }
        ) {
            let state: tauri::State<SharedSupervisor> = app_handle.state();
            if let Ok(mut guard) = state.lock() {
                if let Some(existing) = guard.backend.as_mut() {
                    terminate_backend(existing);
                }
                guard.backend = None;
            };
        }
    });
}

fn resolve_app_data_dir(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    app.path()
        .app_data_dir()
        .map_err(|err| format!("Failed to resolve app-data path: {err}"))
}

fn resolve_runtime_dir(app: &tauri::AppHandle, app_data_dir: &Path) -> Result<PathBuf, String> {
    if let Ok(explicit) = env::var("TOKENIZER_STUDIO_RUNTIME_DIR") {
        let path = PathBuf::from(explicit.trim());
        if path.exists() {
            return Ok(path);
        }
        return Err(format!(
            "TOKENIZER_STUDIO_RUNTIME_DIR is set but missing: {}",
            path.display()
        ));
    }

    let candidates = runtime_candidates(app, app_data_dir);
    for candidate in candidates {
        if candidate.exists() {
            return Ok(candidate);
        }
    }

    #[cfg(feature = "runtime-download")]
    if let Some(manifest_url) = read_optional_env("TOKENIZER_STUDIO_RUNTIME_MANIFEST_URL") {
        return install_runtime_from_manifest(app_data_dir, &manifest_url);
    }

    #[cfg(not(feature = "runtime-download"))]
    if read_optional_env("TOKENIZER_STUDIO_RUNTIME_MANIFEST_URL").is_some() {
        return Err(
            "Runtime manifest URL is set, but this desktop binary was built without the runtime-download feature"
                .to_string(),
        );
    }

    Err(format!(
        "No runtime bundle found. Set TOKENIZER_STUDIO_RUNTIME_DIR, configure TOKENIZER_STUDIO_RUNTIME_MANIFEST_URL, or install runtime under {}",
        app_data_dir.join("runtime").display()
    ))
}

fn runtime_candidates(app: &tauri::AppHandle, app_data_dir: &Path) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    candidates.push(app_data_dir.join("runtime").join("current"));
    candidates.push(app_data_dir.join("runtime").join("default"));

    if let Ok(resource_dir) = app.path().resource_dir() {
        candidates.push(resource_dir.join("runtime"));
    }

    if let Ok(cwd) = env::current_dir() {
        candidates.push(cwd.join("runtime"));
        candidates.push(cwd.join("..").join("runtime"));
        candidates.push(cwd.join("..").join("api-runtime"));
    }

    candidates
}

fn read_optional_env(key: &str) -> Option<String> {
    let value = env::var(key).ok()?;
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(trimmed.to_string())
}

#[cfg(feature = "runtime-download")]
fn install_runtime_from_manifest(
    app_data_dir: &Path,
    manifest_url: &str,
) -> Result<PathBuf, String> {
    let manifest = fetch_runtime_manifest(manifest_url)?;
    let archive_url = resolve_archive_url(manifest_url, &manifest.runtime_archive)?;

    let runtime_root = app_data_dir.join("runtime");
    let downloads_dir = runtime_root.join("downloads");
    let versions_dir = runtime_root.join("versions");
    let staging_dir = runtime_root.join("staging");
    let current_dir = runtime_root.join("current");

    fs::create_dir_all(&downloads_dir).map_err(|err| {
        format!(
            "Failed to create runtime download directory {}: {err}",
            downloads_dir.display()
        )
    })?;
    fs::create_dir_all(&versions_dir).map_err(|err| {
        format!(
            "Failed to create runtime versions directory {}: {err}",
            versions_dir.display()
        )
    })?;

    let archive_file_name = manifest
        .runtime_archive
        .split('/')
        .next_back()
        .unwrap_or("runtime.tar.gz");
    let archive_path = downloads_dir.join(archive_file_name);
    download_file(&archive_url, &archive_path)?;
    verify_sha256(&archive_path, &manifest.sha256)?;

    if staging_dir.exists() {
        fs::remove_dir_all(&staging_dir).map_err(|err| {
            format!(
                "Failed to clear runtime staging directory {}: {err}",
                staging_dir.display()
            )
        })?;
    }
    fs::create_dir_all(&staging_dir).map_err(|err| {
        format!(
            "Failed to create runtime staging directory {}: {err}",
            staging_dir.display()
        )
    })?;

    extract_runtime_archive(&archive_path, &staging_dir)?;

    let version_dir_name = format!("{}-{}", manifest.runtime_version, manifest.platform);
    let version_dir = versions_dir.join(version_dir_name);
    if version_dir.exists() {
        fs::remove_dir_all(&version_dir).map_err(|err| {
            format!(
                "Failed to replace existing runtime version {}: {err}",
                version_dir.display()
            )
        })?;
    }
    fs::rename(&staging_dir, &version_dir).map_err(|err| {
        format!(
            "Failed to promote runtime staging directory {} to {}: {err}",
            staging_dir.display(),
            version_dir.display()
        )
    })?;

    if current_dir.exists() {
        fs::remove_dir_all(&current_dir).map_err(|err| {
            format!(
                "Failed to clear existing runtime current directory {}: {err}",
                current_dir.display()
            )
        })?;
    }
    copy_tree(&version_dir, &current_dir)?;

    Ok(current_dir)
}

#[cfg(feature = "runtime-download")]
fn fetch_runtime_manifest(manifest_url: &str) -> Result<RuntimeManifest, String> {
    let response = ureq::get(manifest_url)
        .call()
        .map_err(|err| format!("Failed to download runtime manifest {manifest_url}: {err}"))?;

    response
        .into_json::<RuntimeManifest>()
        .map_err(|err| format!("Failed to parse runtime manifest JSON: {err}"))
}

#[cfg(feature = "runtime-download")]
fn resolve_archive_url(manifest_url: &str, runtime_archive: &str) -> Result<String, String> {
    if runtime_archive.starts_with("http://") || runtime_archive.starts_with("https://") {
        return Ok(runtime_archive.to_string());
    }

    let Some((base, _)) = manifest_url.rsplit_once('/') else {
        return Err(format!(
            "Invalid manifest URL (cannot resolve archive path): {manifest_url}"
        ));
    };
    Ok(format!("{base}/{runtime_archive}"))
}

#[cfg(feature = "runtime-download")]
fn download_file(url: &str, destination: &Path) -> Result<(), String> {
    let response = ureq::get(url)
        .call()
        .map_err(|err| format!("Failed to download runtime archive {url}: {err}"))?;

    let mut reader = response.into_reader();
    let mut output = File::create(destination).map_err(|err| {
        format!(
            "Failed to create runtime archive file {}: {err}",
            destination.display()
        )
    })?;
    copy(&mut reader, &mut output).map_err(|err| {
        format!(
            "Failed to write runtime archive file {}: {err}",
            destination.display()
        )
    })?;
    Ok(())
}

#[cfg(feature = "runtime-download")]
fn verify_sha256(path: &Path, expected_sha256: &str) -> Result<(), String> {
    let mut file = File::open(path).map_err(|err| {
        format!(
            "Failed to read downloaded runtime archive {}: {err}",
            path.display()
        )
    })?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 64];

    loop {
        let count = file.read(&mut buffer).map_err(|err| {
            format!(
                "Failed while hashing runtime archive {}: {err}",
                path.display()
            )
        })?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }

    let actual = format!("{:x}", hasher.finalize());
    if actual.eq_ignore_ascii_case(expected_sha256) {
        return Ok(());
    }

    Err(format!(
        "Runtime archive checksum mismatch for {}. Expected {}, got {}",
        path.display(),
        expected_sha256,
        actual
    ))
}

#[cfg(feature = "runtime-download")]
fn extract_runtime_archive(archive_path: &Path, destination: &Path) -> Result<(), String> {
    let file = File::open(archive_path).map_err(|err| {
        format!(
            "Failed to open runtime archive for extraction {}: {err}",
            archive_path.display()
        )
    })?;
    let decoder = GzDecoder::new(file);
    let mut archive = tar::Archive::new(decoder);
    archive.unpack(destination).map_err(|err| {
        format!(
            "Failed to extract runtime archive {}: {err}",
            archive_path.display()
        )
    })
}

#[cfg(feature = "runtime-download")]
fn copy_tree(source: &Path, destination: &Path) -> Result<(), String> {
    fs::create_dir_all(destination).map_err(|err| {
        format!(
            "Failed to create destination directory {}: {err}",
            destination.display()
        )
    })?;

    let entries = fs::read_dir(source)
        .map_err(|err| format!("Failed to read directory {}: {err}", source.display()))?;

    for entry_result in entries {
        let entry = entry_result
            .map_err(|err| format!("Failed to iterate directory {}: {err}", source.display()))?;
        let source_path = entry.path();
        let destination_path = destination.join(entry.file_name());
        let metadata = entry.metadata().map_err(|err| {
            format!(
                "Failed to read file metadata for {}: {err}",
                source_path.display()
            )
        })?;

        if metadata.is_dir() {
            copy_tree(&source_path, &destination_path)?;
        } else if metadata.is_file() {
            fs::copy(&source_path, &destination_path).map_err(|err| {
                format!(
                    "Failed to copy runtime file {} to {}: {err}",
                    source_path.display(),
                    destination_path.display()
                )
            })?;
        }
    }
    Ok(())
}

struct RuntimeLayout {
    python_executable: PathBuf,
    backend_app_dir: PathBuf,
    web_dir: PathBuf,
}

#[cfg(feature = "runtime-download")]
#[derive(Debug, Deserialize)]
struct RuntimeManifest {
    runtime_version: String,
    runtime_archive: String,
    sha256: String,
    platform: String,
}

fn validate_runtime_layout(runtime_dir: &Path) -> Result<RuntimeLayout, String> {
    let python_executable = python_executable_path(runtime_dir);
    if !python_executable.exists() {
        return Err(format!(
            "Runtime is missing Python executable: {}",
            python_executable.display()
        ));
    }

    let backend_app_dir = runtime_dir.join("app");
    if !backend_app_dir.exists() {
        return Err(format!(
            "Runtime is missing backend app directory: {}",
            backend_app_dir.display()
        ));
    }

    let web_dir = runtime_dir.join("web");
    if !web_dir.exists() {
        return Err(format!(
            "Runtime is missing built web assets directory: {}",
            web_dir.display()
        ));
    }

    Ok(RuntimeLayout {
        python_executable,
        backend_app_dir,
        web_dir,
    })
}

fn python_executable_path(runtime_dir: &Path) -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        runtime_dir.join("python").join("python.exe")
    }
    #[cfg(not(target_os = "windows"))]
    {
        let preferred = runtime_dir.join("python").join("bin").join("python3");
        if preferred.exists() {
            return preferred;
        }
        runtime_dir.join("python").join("bin").join("python")
    }
}

fn create_runtime_log_file(app_data_dir: &Path) -> Result<PathBuf, String> {
    let log_dir = app_data_dir.join("logs");
    fs::create_dir_all(&log_dir).map_err(|err| {
        format!(
            "Failed to create backend log directory {}: {err}",
            log_dir.display()
        )
    })?;
    rotate_backend_logs(&log_dir, MAX_LOG_FILES)?;

    let unix_seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    Ok(log_dir.join(format!("backend-{unix_seconds}.log")))
}

fn rotate_backend_logs(log_dir: &Path, keep: usize) -> Result<(), String> {
    let mut candidates: Vec<(PathBuf, SystemTime)> = Vec::new();

    let entries = fs::read_dir(log_dir).map_err(|err| {
        format!(
            "Failed to read backend log directory {}: {err}",
            log_dir.display()
        )
    })?;

    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        if !name.starts_with("backend-") || !name.ends_with(".log") {
            continue;
        }
        let modified = entry
            .metadata()
            .ok()
            .and_then(|meta| meta.modified().ok())
            .unwrap_or(UNIX_EPOCH);
        candidates.push((path, modified));
    }

    candidates.sort_by(|(_, left), (_, right)| right.cmp(left));
    for (path, _) in candidates.into_iter().skip(keep) {
        let _ = fs::remove_file(path);
    }
    Ok(())
}

fn reserve_local_port() -> Result<u16, String> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .map_err(|err| format!("Failed to reserve local port: {err}"))?;
    listener
        .local_addr()
        .map(|value| value.port())
        .map_err(|err| format!("Failed to read reserved local port: {err}"))
}

fn spawn_backend(
    python_executable: &Path,
    backend_app_dir: &Path,
    web_dir: &Path,
    app_data_dir: &Path,
    log_file: &Path,
    port: u16,
) -> Result<Child, String> {
    let stdout_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_file)
        .map_err(|err| {
            format!(
                "Failed to open backend log file {}: {err}",
                log_file.display()
            )
        })?;
    let stderr_file = stdout_file
        .try_clone()
        .map_err(|err| format!("Failed to duplicate backend log output handle: {err}"))?;

    let mut command = Command::new(python_executable);
    command
        .arg("-m")
        .arg("app.serve")
        .current_dir(backend_app_dir)
        .env("TOKENIZER_STUDIO_HOST", "127.0.0.1")
        .env("TOKENIZER_STUDIO_PORT", port.to_string())
        .env("TOKENIZER_STUDIO_DATA_DIR", app_data_dir)
        .env("TOKENIZER_STUDIO_WEB_DIST_DIR", web_dir)
        .env("TOKENIZER_STUDIO_SERVE_WEB", "1")
        .env("PYTHONUNBUFFERED", "1")
        .stdout(Stdio::from(stdout_file))
        .stderr(Stdio::from(stderr_file));

    command
        .spawn()
        .map_err(|err| format!("Failed to spawn backend sidecar process: {err}"))
}

fn wait_for_health(child: &mut Child, health_url: &str) -> Result<(), String> {
    let started = Instant::now();
    loop {
        if is_healthy(health_url) {
            return Ok(());
        }

        if let Some(status) = child
            .try_wait()
            .map_err(|err| format!("Failed while waiting on backend sidecar process: {err}"))?
        {
            return Err(format!(
                "Backend exited before becoming ready. Exit status: {status}"
            ));
        }

        if started.elapsed() >= STARTUP_TIMEOUT {
            return Err(format!(
                "Timed out waiting for backend health after {} seconds",
                STARTUP_TIMEOUT.as_secs()
            ));
        }
        sleep(HEALTH_CHECK_INTERVAL);
    }
}

fn is_healthy(health_url: &str) -> bool {
    match ureq::get(health_url).call() {
        Ok(response) => response.status() == 200,
        Err(_) => false,
    }
}

fn terminate_backend(process: &mut BackendProcess) {
    terminate_child(&mut process.child);
}

fn terminate_child(child: &mut Child) {
    #[cfg(unix)]
    {
        unsafe {
            libc::kill(child.id() as i32, libc::SIGTERM);
        }
    }

    match child.wait_timeout(SHUTDOWN_GRACE) {
        Ok(Some(_)) => return,
        Ok(None) => {}
        Err(_) => {}
    }

    let _ = child.kill();
    let _ = child.wait();
}
