#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use rand::{rngs::OsRng, RngCore};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::sleep;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tauri::menu::{MenuBuilder, SubmenuBuilder};
use tauri::{Emitter, Manager};
#[cfg(windows)]
use wait_timeout::ChildExt;

const SHELL_VERSION: &str = env!("CARGO_PKG_VERSION");
const SUPPORTED_MANIFEST_SCHEMA: u32 = 1;
const SUPPORTED_API_CONTRACT: &str = "1";
const SUPPORTED_DATA_SCHEMA: &str = "3";
const STARTUP_TIMEOUT: Duration = Duration::from_secs(180);
const HEALTH_INTERVAL: Duration = Duration::from_millis(300);
const SHUTDOWN_GRACE: Duration = Duration::from_secs(8);
const MAX_LOG_FILES: usize = 10;
const MAX_SHELL_LOG_BYTES: u64 = 5 * 1024 * 1024;
const MAX_SHELL_LOG_BACKUPS: usize = 5;
const MAX_DIAGNOSTIC_STORAGE_ENTRIES: usize = 100_000;
const MAX_DIAGNOSTIC_STORAGE_SCAN: Duration = Duration::from_secs(3);
const MAX_START_ATTEMPTS: u32 = 5;
const RETRY_COOLDOWN: Duration = Duration::from_secs(2);
const RUNTIME_REQUEST_TIMEOUT: Duration = Duration::from_secs(300);

type SharedSupervisor = Arc<Mutex<Supervisor>>;
type StartupCancellation = Arc<AtomicBool>;
#[cfg(windows)]
type WindowsJob = std::os::windows::io::OwnedHandle;

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Lifecycle {
    Starting,
    Ready,
    Failed,
    Stopping,
    Stopped,
}

impl Default for Lifecycle {
    fn default() -> Self {
        Self::Stopped
    }
}

#[derive(Default)]
struct Supervisor {
    lifecycle: Lifecycle,
    backend: Option<BackendProcess>,
    last_error: Option<String>,
    attempts: u32,
    last_attempt_at: Option<Instant>,
}

struct BackendProcess {
    child: Child,
    #[cfg(windows)]
    // The option permits consuming the owned handle exactly once during shutdown.
    job: Option<WindowsJob>,
    bootstrap: RuntimeBootstrap,
    manifest: RuntimeManifest,
}

#[derive(Clone, Debug, Serialize)]
struct RuntimeBootstrap {
    environment: &'static str,
    api_base_url: String,
    runtime_token: String,
    capabilities: Capabilities,
    versions: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Serialize)]
struct Capabilities {
    native_save: bool,
    open_logs: bool,
    open_data: bool,
    reveal_artifact: bool,
    diagnostics_export: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct RuntimeManifest {
    schema_version: u32,
    runtime_version: String,
    shell_compatibility: ShellCompatibility,
    api_contract_version: String,
    data_schema_version: String,
    platform: String,
    architecture: String,
    python_version: String,
    source_root: String,
    python_executable: String,
    required_files: Vec<String>,
    file_hashes: BTreeMap<String, String>,
    #[serde(default)]
    dependency_versions: BTreeMap<String, String>,
    #[serde(default)]
    provenance: BTreeMap<String, String>,
    #[serde(default)]
    size: Option<RuntimeSize>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ShellCompatibility {
    minimum: String,
    maximum_exclusive: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
struct RuntimeSize {
    threshold_kind: String,
    target: String,
    max_payload_bytes: u64,
    max_payload_files: u64,
    payload_total_bytes: u64,
    payload_file_count: u64,
    policy_file: String,
}

#[derive(Debug, Deserialize)]
struct StartupHandshake {
    schema_version: u32,
    host: String,
    port: u16,
    base_url: String,
    pid: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ActiveJobs {
    active: bool,
    #[serde(default)]
    tokenizer_jobs: Vec<serde_json::Value>,
    #[serde(default)]
    training_jobs: Vec<serde_json::Value>,
    #[serde(default)]
    has_active_local_training: bool,
    #[serde(default)]
    has_active_runpod_training: bool,
}

#[derive(Clone, Debug, Serialize)]
struct CloseRequest {
    active_jobs: ActiveJobs,
    message: String,
}

#[derive(Clone, Debug, Serialize)]
struct StartupProgress {
    stage: &'static str,
    message: String,
}

#[derive(Clone, Debug, Serialize)]
struct RuntimeStatus {
    lifecycle: Lifecycle,
    last_error: Option<String>,
    start_attempts: u32,
}

#[derive(Debug, Deserialize)]
struct RuntimeHttpRequest {
    method: String,
    path: String,
    #[serde(default)]
    headers: BTreeMap<String, String>,
    #[serde(default)]
    body: Vec<u8>,
}

#[derive(Debug, Serialize)]
struct RuntimeHttpResponse {
    status: u16,
    headers: BTreeMap<String, String>,
    body: Vec<u8>,
}

#[derive(Serialize)]
struct Diagnostics {
    schema_version: u32,
    generated_at_unix: u64,
    shell_version: &'static str,
    lifecycle: Lifecycle,
    last_error: Option<String>,
    start_attempts: u32,
    health: Option<serde_json::Value>,
    capabilities: Capabilities,
    log_inventory: LogInventory,
    storage_inventory: StorageInventory,
    recent_shell_events: Vec<serde_json::Value>,
    runtime: Option<RuntimeDiagnostics>,
}

#[derive(Serialize)]
struct LogInventory {
    backend_log_files: usize,
    shell_log_files: usize,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
struct StorageInventory {
    data: StorageRootInventory,
    cache: StorageRootInventory,
    logs: StorageRootInventory,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
struct StorageRootInventory {
    root_exists: bool,
    file_count: u64,
    directory_count: u64,
    symlink_count: u64,
    other_entry_count: u64,
    total_file_bytes: u64,
    scanned_entries: u64,
    max_entries: u64,
    max_scan_milliseconds: u64,
    scan_complete: bool,
    time_limit_reached: bool,
    error_count: u64,
}

impl StorageRootInventory {
    fn new(max_entries: usize, max_duration: Duration) -> Self {
        Self {
            root_exists: false,
            file_count: 0,
            directory_count: 0,
            symlink_count: 0,
            other_entry_count: 0,
            total_file_bytes: 0,
            scanned_entries: 0,
            max_entries: max_entries as u64,
            max_scan_milliseconds: max_duration.as_millis().try_into().unwrap_or(u64::MAX),
            scan_complete: true,
            time_limit_reached: false,
            error_count: 0,
        }
    }

    fn resolution_error(max_entries: usize, max_duration: Duration) -> Self {
        let mut inventory = Self::new(max_entries, max_duration);
        inventory.scan_complete = false;
        inventory.error_count = 1;
        inventory
    }
}

#[derive(Serialize)]
struct RuntimeDiagnostics {
    manifest_schema_version: u32,
    runtime_version: String,
    shell_compatibility: ShellCompatibility,
    api_contract_version: String,
    data_schema_version: String,
    platform: String,
    architecture: String,
    python_version: String,
    required_file_count: usize,
    hashed_file_count: usize,
    dependency_count: usize,
    build_mode: Option<String>,
    size: Option<RuntimeSize>,
    backend_running: bool,
}

#[tauri::command]
async fn runtime_bootstrap(
    app: tauri::AppHandle,
    state: tauri::State<'_, SharedSupervisor>,
    cancellation: tauri::State<'_, StartupCancellation>,
) -> Result<RuntimeBootstrap, String> {
    let supervisor = Arc::clone(state.inner());
    let cancellation = Arc::clone(cancellation.inner());
    cancellation.store(false, Ordering::SeqCst);
    tauri::async_runtime::spawn_blocking(move || {
        bootstrap_runtime(&app, &supervisor, &cancellation, false)
    })
    .await
    .map_err(|error| format!("Runtime bootstrap worker failed: {error}"))?
}

#[tauri::command]
async fn runtime_request(
    request: RuntimeHttpRequest,
    state: tauri::State<'_, SharedSupervisor>,
) -> Result<RuntimeHttpResponse, String> {
    let (api_base_url, token) = {
        let mut supervisor = lock_supervisor(&state)?;
        let Some(backend) = supervisor.backend.as_mut() else {
            return Err("The local runtime is unavailable.".to_string());
        };
        if !backend_is_ready(backend) {
            return Err("The local runtime is not ready.".to_string());
        }
        (
            backend.bootstrap.api_base_url.clone(),
            backend.bootstrap.runtime_token.clone(),
        )
    };
    tauri::async_runtime::spawn_blocking(move || {
        forward_runtime_request(&api_base_url, &token, request)
    })
    .await
    .map_err(|error| format!("Runtime request worker failed: {error}"))?
}

#[tauri::command]
async fn retry_runtime(
    app: tauri::AppHandle,
    state: tauri::State<'_, SharedSupervisor>,
    cancellation: tauri::State<'_, StartupCancellation>,
) -> Result<RuntimeBootstrap, String> {
    let supervisor = Arc::clone(state.inner());
    let cancellation = Arc::clone(cancellation.inner());
    cancellation.store(false, Ordering::SeqCst);
    tauri::async_runtime::spawn_blocking(move || {
        bootstrap_runtime(&app, &supervisor, &cancellation, true)
    })
    .await
    .map_err(|error| format!("Runtime retry worker failed: {error}"))?
}

#[tauri::command]
fn cancel_runtime_start(cancellation: tauri::State<StartupCancellation>) {
    cancellation.store(true, Ordering::SeqCst);
}

fn bootstrap_runtime(
    app: &tauri::AppHandle,
    state: &SharedSupervisor,
    cancellation: &StartupCancellation,
    retry: bool,
) -> Result<RuntimeBootstrap, String> {
    let mut supervisor = lock_supervisor(state)?;
    if retry {
        if let Some(backend) = supervisor.backend.as_mut() {
            terminate_backend(backend);
        }
        supervisor.backend = None;
        supervisor.lifecycle = Lifecycle::Stopped;
    } else if let Some(backend) = supervisor.backend.as_mut() {
        let ready_bootstrap = backend_is_ready(backend).then(|| backend.bootstrap.clone());
        if let Some(bootstrap) = ready_bootstrap {
            supervisor.lifecycle = Lifecycle::Ready;
            return Ok(bootstrap);
        }
        terminate_backend(backend);
        supervisor.backend = None;
    }
    start_backend(app, &mut supervisor, cancellation)
}

#[tauri::command]
fn runtime_status(state: tauri::State<SharedSupervisor>) -> Result<RuntimeStatus, String> {
    let supervisor = lock_supervisor(&state)?;
    Ok(RuntimeStatus {
        lifecycle: supervisor.lifecycle.clone(),
        last_error: supervisor.last_error.clone(),
        start_attempts: supervisor.attempts,
    })
}

#[tauri::command]
fn active_jobs(state: tauri::State<SharedSupervisor>) -> Result<ActiveJobs, String> {
    let mut supervisor = lock_supervisor(&state)?;
    active_jobs_for_supervisor(&mut supervisor)
}

#[tauri::command]
fn stop_and_exit(
    app: tauri::AppHandle,
    state: tauri::State<SharedSupervisor>,
) -> Result<(), String> {
    stop_supervisor(&state)?;
    app.exit(0);
    Ok(())
}

#[tauri::command]
fn quit_app(app: tauri::AppHandle, state: tauri::State<SharedSupervisor>) -> Result<(), String> {
    {
        let mut supervisor = lock_supervisor(&state)?;
        let jobs = active_jobs_for_supervisor(&mut supervisor)?;
        if jobs.active {
            return Err(
                "Active jobs require an explicit close decision before LLM Studio can quit."
                    .to_string(),
            );
        }
    }
    stop_supervisor(&state)?;
    app.exit(0);
    Ok(())
}

#[tauri::command]
fn save_file(suggested_name: String, bytes: Vec<u8>) -> Result<Option<String>, String> {
    let name = sanitize_suggested_file_name(&suggested_name);
    let Some(path) = rfd::FileDialog::new().set_file_name(&name).save_file() else {
        return Ok(None);
    };
    atomic_write(&path, &bytes)?;
    Ok(Some(path.display().to_string()))
}

#[tauri::command]
async fn save_api_artifact(
    api_path: String,
    suggested_name: String,
    state: tauri::State<'_, SharedSupervisor>,
) -> Result<Option<String>, String> {
    validate_api_artifact_path(&api_path)?;
    let (url, token) = {
        let mut supervisor = lock_supervisor(&state)?;
        let Some(backend) = supervisor.backend.as_mut() else {
            return Err("The local runtime is unavailable.".to_string());
        };
        if !backend_is_ready(backend) {
            return Err("The local runtime is not ready.".to_string());
        }
        (
            format!("{}{}", backend.bootstrap.api_base_url, api_path),
            backend.bootstrap.runtime_token.clone(),
        )
    };
    let name = sanitize_suggested_file_name(&suggested_name);
    let Some(path) = rfd::FileDialog::new().set_file_name(&name).save_file() else {
        return Ok(None);
    };
    tauri::async_runtime::spawn_blocking(move || {
        download_authenticated_to_path(&url, &token, &path)?;
        Ok(Some(path.display().to_string()))
    })
    .await
    .map_err(|error| format!("Artifact download worker failed: {error}"))?
}

#[tauri::command]
async fn reveal_api_artifact(
    api_path: String,
    app: tauri::AppHandle,
    state: tauri::State<'_, SharedSupervisor>,
) -> Result<(), String> {
    let (metadata_path, field) = artifact_metadata_request(&api_path)?;
    let (url, token) = {
        let mut supervisor = lock_supervisor(&state)?;
        let Some(backend) = supervisor.backend.as_mut() else {
            return Err("The local runtime is unavailable.".to_string());
        };
        if !backend_is_ready(backend) {
            return Err("The local runtime is not ready.".to_string());
        }
        (
            format!("{}{}", backend.bootstrap.api_base_url, metadata_path),
            backend.bootstrap.runtime_token.clone(),
        )
    };
    let data_root = app
        .path()
        .app_data_dir()
        .map_err(|error| format!("Failed to resolve app data: {error}"))?;

    tauri::async_runtime::spawn_blocking(move || {
        let metadata = authenticated_json(&url, &token)?;
        let artifact_path = managed_artifact_path_from_metadata(&metadata, field)?;
        let folder = validate_managed_reveal_path(&data_root, &artifact_path)?;
        open_managed_folder(&folder)
    })
    .await
    .map_err(|error| format!("Artifact reveal worker failed: {error}"))?
}

#[tauri::command]
fn open_logs_folder(app: tauri::AppHandle) -> Result<(), String> {
    let path = app
        .path()
        .app_log_dir()
        .map_err(|error| format!("Failed to resolve logs folder: {error}"))?;
    open_managed_folder(&path)
}

#[tauri::command]
fn open_data_folder(app: tauri::AppHandle) -> Result<(), String> {
    let path = app
        .path()
        .app_data_dir()
        .map_err(|error| format!("Failed to resolve data folder: {error}"))?;
    open_managed_folder(&path)
}

#[tauri::command]
async fn export_diagnostics(
    app: tauri::AppHandle,
    state: tauri::State<'_, SharedSupervisor>,
) -> Result<Option<String>, String> {
    let (lifecycle, last_error, start_attempts, health_request, runtime) = {
        let supervisor = lock_supervisor(&state)?;
        let backend = supervisor.backend.as_ref();
        (
            supervisor.lifecycle.clone(),
            supervisor.last_error.as_deref().map(redact_error),
            supervisor.attempts,
            backend.map(|backend| {
                (
                    format!("{}/health", backend.bootstrap.api_base_url),
                    backend.bootstrap.runtime_token.clone(),
                )
            }),
            backend.map(|backend| RuntimeDiagnostics {
                manifest_schema_version: backend.manifest.schema_version,
                runtime_version: backend.manifest.runtime_version.clone(),
                shell_compatibility: backend.manifest.shell_compatibility.clone(),
                api_contract_version: backend.manifest.api_contract_version.clone(),
                data_schema_version: backend.manifest.data_schema_version.clone(),
                platform: backend.manifest.platform.clone(),
                architecture: backend.manifest.architecture.clone(),
                python_version: backend.manifest.python_version.clone(),
                required_file_count: backend.manifest.required_files.len(),
                hashed_file_count: backend.manifest.file_hashes.len(),
                dependency_count: backend.manifest.dependency_versions.len(),
                build_mode: backend.manifest.provenance.get("build_mode").cloned(),
                size: backend.manifest.size.clone(),
                backend_running: true,
            }),
        )
    };
    let app_for_collection = app.clone();
    let (health, log_inventory, storage_inventory, recent_shell_events) =
        tauri::async_runtime::spawn_blocking(move || {
            let health =
                health_request.and_then(|(url, token)| authenticated_json(&url, &token).ok());
            (
                health,
                log_inventory(&app_for_collection),
                storage_inventory(&app_for_collection),
                recent_shell_events(&app_for_collection, 50),
            )
        })
        .await
        .map_err(|error| format!("Diagnostics collection worker failed: {error}"))?;
    let diagnostics = Diagnostics {
        schema_version: 2,
        generated_at_unix: unix_timestamp(),
        shell_version: SHELL_VERSION,
        lifecycle,
        last_error,
        start_attempts,
        health,
        capabilities: Capabilities {
            native_save: true,
            open_logs: true,
            open_data: true,
            reveal_artifact: true,
            diagnostics_export: true,
        },
        log_inventory,
        storage_inventory,
        recent_shell_events,
        runtime,
    };
    let bytes = serde_json::to_vec_pretty(&diagnostics)
        .map_err(|error| format!("Failed to create diagnostics: {error}"))?;

    let Some(path) = rfd::FileDialog::new()
        .set_file_name("llm-studio-diagnostics.json")
        .save_file()
    else {
        return Ok(None);
    };
    let display = path.display().to_string();
    tauri::async_runtime::spawn_blocking(move || atomic_write(&path, &bytes))
        .await
        .map_err(|error| format!("Diagnostics export worker failed: {error}"))??;
    Ok(Some(display))
}

fn main() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            focus_main_window(app);
        }))
        .menu(|app| {
            let app_menu = SubmenuBuilder::new(app, "LLM Studio")
                .about(None)
                .separator()
                .text("menu.quit", "Quit LLM Studio")
                .build()?;
            let runtime_menu = SubmenuBuilder::new(app, "Runtime")
                .text("menu.retry_runtime", "Retry Local Runtime")
                .text("menu.reload_ui", "Reload Interface")
                .separator()
                .text("menu.open_data", "Open Data Folder")
                .text("menu.open_logs", "Open Logs Folder")
                .build()?;
            let edit_menu = SubmenuBuilder::new(app, "Edit")
                .undo()
                .redo()
                .separator()
                .cut()
                .copy()
                .paste()
                .select_all()
                .build()?;
            let window_menu = SubmenuBuilder::new(app, "Window")
                .minimize()
                .maximize()
                .close_window()
                .build()?;
            MenuBuilder::new(app)
                .items(&[&app_menu, &runtime_menu, &edit_menu, &window_menu])
                .build()
        })
        .on_menu_event(handle_menu_event)
        .manage(Arc::new(Mutex::new(Supervisor::default())))
        .manage(Arc::new(AtomicBool::new(false)))
        .invoke_handler(tauri::generate_handler![
            runtime_bootstrap,
            runtime_request,
            retry_runtime,
            cancel_runtime_start,
            runtime_status,
            active_jobs,
            stop_and_exit,
            quit_app,
            save_file,
            save_api_artifact,
            reveal_api_artifact,
            open_logs_folder,
            open_data_folder,
            export_diagnostics
        ])
        .build(tauri::generate_context!())
        .expect("failed to build LLM Studio desktop app");

    app.run(|app_handle, event| match event {
        tauri::RunEvent::WindowEvent {
            label,
            event: tauri::WindowEvent::CloseRequested { api, .. },
            ..
        } if label == "main" => {
            let cancellation: tauri::State<StartupCancellation> = app_handle.state();
            cancellation.store(true, Ordering::SeqCst);
            let state: tauri::State<SharedSupervisor> = app_handle.state();
            let should_prevent = match lock_supervisor(&state)
                .and_then(|mut supervisor| active_jobs_for_supervisor(&mut supervisor))
            {
                Ok(jobs) if jobs.active => {
                    let message = close_warning(&jobs);
                    let _ = app_handle.emit(
                        "desktop-close-requested",
                        CloseRequest {
                            active_jobs: jobs,
                            message,
                        },
                    );
                    true
                }
                Ok(_) => {
                    let _ = stop_supervisor(&state);
                    false
                }
                Err(error) => {
                    let _ = app_handle.emit(
                        "desktop-close-requested",
                        CloseRequest {
                            active_jobs: empty_active_jobs(),
                            message: format!(
                                "LLM Studio could not confirm active job status: {error}"
                            ),
                        },
                    );
                    true
                }
            };
            if should_prevent {
                api.prevent_close();
            }
        }
        tauri::RunEvent::Exit => {
            let _ = write_shell_log(
                app_handle,
                "shell.shutdown.begin",
                "Native shell shutdown began.",
            );
            let state: tauri::State<SharedSupervisor> = app_handle.state();
            let _ = stop_supervisor(&state);
            let _ = write_shell_log(
                app_handle,
                "shell.shutdown.complete",
                "Native shell shutdown completed.",
            );
        }
        _ => {}
    });
}

fn handle_menu_event(app: &tauri::AppHandle, event: tauri::menu::MenuEvent) {
    let result = match event.id().0.as_str() {
        "menu.retry_runtime" => app
            .emit("desktop-retry-runtime-requested", ())
            .map_err(|error| format!("Failed to request runtime retry: {error}")),
        "menu.reload_ui" => app
            .get_webview_window("main")
            .ok_or_else(|| "Main window is unavailable.".to_string())
            .and_then(|window| {
                window
                    .eval("window.location.reload()")
                    .map_err(|error| format!("Failed to reload interface: {error}"))
            }),
        "menu.open_data" => app
            .path()
            .app_data_dir()
            .map_err(|error| format!("Failed to resolve data folder: {error}"))
            .and_then(|path| open_managed_folder(&path)),
        "menu.open_logs" => app
            .path()
            .app_log_dir()
            .map_err(|error| format!("Failed to resolve logs folder: {error}"))
            .and_then(|path| open_managed_folder(&path)),
        "menu.quit" => app
            .get_webview_window("main")
            .ok_or_else(|| "Main window is unavailable.".to_string())
            .and_then(|window| {
                window
                    .close()
                    .map_err(|error| format!("Failed to request application close: {error}"))
            }),
        _ => Ok(()),
    };
    if let Err(error) = result {
        let _ = write_shell_log(app, "shell.menu.action_failed", &error);
    }
}

fn focus_main_window(app: &tauri::AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.unminimize();
        let _ = window.show();
        let _ = window.set_focus();
    }
}

fn start_backend(
    app: &tauri::AppHandle,
    supervisor: &mut Supervisor,
    cancellation: &StartupCancellation,
) -> Result<RuntimeBootstrap, String> {
    validate_start_attempt(supervisor)?;
    let _ = write_shell_log(app, "shell.runtime.start", "Local runtime startup began.");
    supervisor.lifecycle = Lifecycle::Starting;
    supervisor.attempts = supervisor.attempts.saturating_add(1);
    supervisor.last_attempt_at = Some(Instant::now());
    supervisor.last_error = None;

    let result = start_backend_inner(app, cancellation);
    match result {
        Ok(backend) => {
            let bootstrap = backend.bootstrap.clone();
            supervisor.backend = Some(backend);
            supervisor.lifecycle = Lifecycle::Ready;
            emit_startup_progress(app, "ready", "Local runtime is ready.");
            let _ = write_shell_log(app, "shell.runtime.ready", "Local runtime is ready.");
            Ok(bootstrap)
        }
        Err(error) => {
            supervisor.lifecycle = Lifecycle::Failed;
            supervisor.last_error = Some(redact_error(&error));
            emit_startup_progress(app, "failed", "Local runtime startup failed.");
            let _ = write_shell_log(app, "shell.runtime.failed", "Local runtime startup failed.");
            Err(error)
        }
    }
}

fn validate_start_attempt(supervisor: &Supervisor) -> Result<(), String> {
    if supervisor.attempts >= MAX_START_ATTEMPTS {
        return Err(format!(
            "The local runtime failed to start {MAX_START_ATTEMPTS} times. Quit and relaunch LLM Studio after reviewing diagnostics."
        ));
    }
    if let Some(last_attempt) = supervisor.last_attempt_at {
        let elapsed = last_attempt.elapsed();
        if elapsed < RETRY_COOLDOWN {
            let wait = RETRY_COOLDOWN.saturating_sub(elapsed).as_secs_f32();
            return Err(format!(
                "Retry is temporarily limited to prevent a rapid crash loop. Try again in {wait:.1} seconds."
            ));
        }
    }
    Ok(())
}

fn start_backend_inner(
    app: &tauri::AppHandle,
    cancellation: &StartupCancellation,
) -> Result<BackendProcess, String> {
    emit_startup_progress(
        app,
        "data_paths",
        "Preparing managed data, cache, and log folders.",
    );
    let app_data = app
        .path()
        .app_data_dir()
        .map_err(|error| format!("Failed to resolve app data: {error}"))?;
    let app_cache = app
        .path()
        .app_cache_dir()
        .map_err(|error| format!("Failed to resolve app cache: {error}"))?;
    let app_logs = app
        .path()
        .app_log_dir()
        .map_err(|error| format!("Failed to resolve app logs: {error}"))?;
    for directory in [&app_data, &app_cache, &app_logs] {
        fs::create_dir_all(directory)
            .map_err(|error| format!("Failed to create {}: {error}", directory.display()))?;
    }
    ensure_startup_not_cancelled(cancellation)?;

    emit_startup_progress(
        app,
        "runtime_validation",
        "Validating packaged runtime and checksums.",
    );
    let (runtime_dir, development_override) = resolve_runtime_dir(app)?;
    let manifest = validate_runtime(&runtime_dir, cfg!(debug_assertions) && development_override)?;
    ensure_startup_not_cancelled(cancellation)?;
    let python = runtime_path(&runtime_dir, &manifest.python_executable)?;
    let source_root = runtime_path(&runtime_dir, &manifest.source_root)?;
    let api_root = source_root.join("apps").join("llm-studio").join("api");
    if !api_root.is_dir() {
        return Err(format!(
            "Runtime API root is missing: {}",
            api_root.display()
        ));
    }

    rotate_logs(&app_logs)?;
    let log_file = app_logs.join(format!("backend-{}.log", unix_timestamp()));
    let stdout = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_file)
        .map_err(|error| format!("Failed to create backend log: {error}"))?;
    let stderr = stdout
        .try_clone()
        .map_err(|error| format!("Failed to clone backend log handle: {error}"))?;

    let token = generate_runtime_token();
    let handshake = app_cache.join(format!(
        "startup-handshake-{}-{}.json",
        std::process::id(),
        unix_timestamp()
    ));
    let _ = fs::remove_file(&handshake);

    let mut command = Command::new(&python);
    configure_clean_environment(&mut command);
    command
        .arg("-m")
        .arg("app.serve")
        .current_dir(&api_root)
        .env("LLM_STUDIO_DESKTOP", "1")
        .env("LLM_STUDIO_HOST", "127.0.0.1")
        .env("LLM_STUDIO_PORT", "0")
        .env("LLM_STUDIO_RUNTIME_TOKEN", &token)
        .env("LLM_STUDIO_RUNTIME_VERSION", &manifest.runtime_version)
        .env("LLM_STUDIO_SOURCE_ROOT", &source_root)
        .env("LLM_STUDIO_DATA_DIR", &app_data)
        .env("LLM_STUDIO_CACHE_DIR", &app_cache)
        .env("LLM_STUDIO_LOG_DIR", &app_logs)
        .env("LLM_STUDIO_STARTUP_HANDSHAKE_PATH", &handshake)
        .env("LLM_STUDIO_PARENT_PID", std::process::id().to_string())
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .stdin(Stdio::null());
    configure_child_process_group(&mut command);

    ensure_startup_not_cancelled(cancellation)?;
    emit_startup_progress(
        app,
        "python_start",
        "Starting Python and loading local compute dependencies.",
    );
    let mut child = command
        .spawn()
        .map_err(|error| format!("Failed to spawn packaged backend: {error}"))?;
    #[cfg(windows)]
    let job = match assign_windows_job_or_terminate(&mut child) {
        Ok(job) => job,
        Err(error) => {
            let _ = fs::remove_file(&handshake);
            return Err(error);
        }
    };

    let startup =
        match wait_for_handshake_and_readiness(app, &mut child, &handshake, &token, cancellation) {
            Ok(value) => value,
            Err(error) => {
                terminate_child(
                    &mut child,
                    #[cfg(windows)]
                    job,
                );
                return Err(format!(
                    "{error}\nReview the local backend log for non-secret details."
                ));
            }
        };
    let _ = fs::remove_file(&handshake);

    let mut versions = BTreeMap::new();
    versions.insert("shell".to_string(), SHELL_VERSION.to_string());
    versions.insert("runtime".to_string(), manifest.runtime_version.clone());
    versions.insert(
        "api_contract".to_string(),
        manifest.api_contract_version.clone(),
    );
    versions.insert(
        "data_schema".to_string(),
        manifest.data_schema_version.clone(),
    );
    versions.insert("python".to_string(), manifest.python_version.clone());

    Ok(BackendProcess {
        child,
        #[cfg(windows)]
        job: Some(job),
        bootstrap: RuntimeBootstrap {
            environment: "desktop",
            api_base_url: format!("{}/api/v1", startup.base_url),
            runtime_token: token,
            capabilities: Capabilities {
                native_save: true,
                open_logs: true,
                open_data: true,
                reveal_artifact: true,
                diagnostics_export: true,
            },
            versions,
        },
        manifest,
    })
}

fn resolve_runtime_dir(app: &tauri::AppHandle) -> Result<(PathBuf, bool), String> {
    if let Some(explicit) = non_empty_env("LLM_STUDIO_RUNTIME_DIR") {
        let path = PathBuf::from(explicit);
        if path.is_dir() {
            return Ok((path, true));
        }
        return Err(format!(
            "LLM_STUDIO_RUNTIME_DIR points to a missing directory: {}",
            path.display()
        ));
    }

    let resource_dir = app
        .path()
        .resource_dir()
        .map_err(|error| format!("Failed to resolve application resources: {error}"))?;
    let candidates = [
        resource_dir.join("resources").join("runtime"),
        resource_dir.join("runtime"),
    ];
    candidates
        .into_iter()
        .find(|candidate| candidate.join("manifest.json").is_file())
        .map(|candidate| (candidate, false))
        .ok_or_else(|| {
            "No packaged runtime was found. Reinstall LLM Studio or set LLM_STUDIO_RUNTIME_DIR for development."
                .to_string()
        })
}

fn validate_runtime(
    runtime_dir: &Path,
    allow_development_runtime: bool,
) -> Result<RuntimeManifest, String> {
    let manifest_path = runtime_dir.join("manifest.json");
    let manifest_file = File::open(&manifest_path)
        .map_err(|error| format!("Failed to open runtime manifest: {error}"))?;
    let manifest: RuntimeManifest = serde_json::from_reader(manifest_file)
        .map_err(|error| format!("Failed to parse runtime manifest: {error}"))?;

    validate_manifest_compatibility(&manifest, allow_development_runtime)?;

    let mut failures = Vec::new();
    for relative in &manifest.required_files {
        match runtime_path(runtime_dir, relative) {
            Ok(path) if path.exists() => {
                if !allow_development_runtime {
                    if let Err(error) = ensure_runtime_path_is_contained(runtime_dir, &path) {
                        failures.push(error);
                    }
                }
            }
            Ok(path) => failures.push(format!("missing {}", path.display())),
            Err(error) => failures.push(error),
        }
    }
    for (relative, expected) in &manifest.file_hashes {
        match runtime_path(runtime_dir, relative).and_then(|path| {
            if !allow_development_runtime {
                ensure_runtime_path_is_contained(runtime_dir, &path)?;
            }
            sha256_file(&path)
        }) {
            Ok(actual) if actual.eq_ignore_ascii_case(expected) => {}
            Ok(actual) => failures.push(format!(
                "checksum mismatch for {relative}: expected {expected}, got {actual}"
            )),
            Err(error) => failures.push(error),
        }
    }
    if !failures.is_empty() {
        return Err(format!(
            "Runtime validation failed:\n- {}",
            failures.join("\n- ")
        ));
    }
    Ok(manifest)
}

fn validate_manifest_compatibility(
    manifest: &RuntimeManifest,
    allow_development_runtime: bool,
) -> Result<(), String> {
    if manifest.schema_version != SUPPORTED_MANIFEST_SCHEMA {
        return Err(format!(
            "Unsupported runtime manifest schema {} (expected {}).",
            manifest.schema_version, SUPPORTED_MANIFEST_SCHEMA
        ));
    }
    if manifest.api_contract_version != SUPPORTED_API_CONTRACT {
        return Err(format!(
            "Runtime API contract {} is incompatible with shell contract {}.",
            manifest.api_contract_version, SUPPORTED_API_CONTRACT
        ));
    }
    if manifest.data_schema_version != SUPPORTED_DATA_SCHEMA {
        return Err(format!(
            "Runtime data schema {} is incompatible with shell schema {}.",
            manifest.data_schema_version, SUPPORTED_DATA_SCHEMA
        ));
    }
    let build_mode = manifest
        .provenance
        .get("build_mode")
        .map(String::as_str)
        .unwrap_or("unknown");
    if build_mode != "portable" && !allow_development_runtime {
        return Err(format!(
            "Bundled runtime build mode {build_mode:?} is not release-portable."
        ));
    }
    let shell_version = semver::Version::parse(SHELL_VERSION)
        .map_err(|error| format!("Shell version is invalid: {error}"))?;
    let minimum = semver::Version::parse(&manifest.shell_compatibility.minimum)
        .map_err(|error| format!("Runtime minimum shell version is invalid: {error}"))?;
    let maximum_exclusive = semver::Version::parse(&manifest.shell_compatibility.maximum_exclusive)
        .map_err(|error| format!("Runtime maximum shell version is invalid: {error}"))?;
    if shell_version < minimum || shell_version >= maximum_exclusive {
        return Err(format!(
            "Runtime supports shell versions from {minimum} up to but excluding {maximum_exclusive}; current shell is {shell_version}."
        ));
    }
    if normalize_platform(&manifest.platform) != normalize_platform(env::consts::OS) {
        return Err(format!(
            "Runtime platform {} does not match host {}.",
            manifest.platform,
            env::consts::OS
        ));
    }
    if normalize_architecture(&manifest.architecture) != normalize_architecture(env::consts::ARCH) {
        return Err(format!(
            "Runtime architecture {} does not match host {}.",
            manifest.architecture,
            env::consts::ARCH
        ));
    }
    Ok(())
}

fn wait_for_handshake_and_readiness(
    app: &tauri::AppHandle,
    child: &mut Child,
    handshake_path: &Path,
    token: &str,
    cancellation: &StartupCancellation,
) -> Result<StartupHandshake, String> {
    wait_for_handshake_and_readiness_with_progress(
        child,
        handshake_path,
        token,
        cancellation,
        STARTUP_TIMEOUT,
        |stage, message| emit_startup_progress(app, stage, message),
    )
}

fn wait_for_handshake_and_readiness_with_progress(
    child: &mut Child,
    handshake_path: &Path,
    token: &str,
    cancellation: &StartupCancellation,
    timeout: Duration,
    mut progress: impl FnMut(&'static str, &str),
) -> Result<StartupHandshake, String> {
    let deadline = Instant::now() + timeout;
    let mut handshake: Option<StartupHandshake> = None;
    let mut last_readiness = "Waiting for backend bind handshake.".to_string();
    let mut last_emitted = String::new();

    while Instant::now() < deadline {
        ensure_startup_not_cancelled(cancellation)?;
        if let Some(status) = child
            .try_wait()
            .map_err(|error| format!("Failed to inspect backend process: {error}"))?
        {
            return Err(format!(
                "Backend exited during startup with status {status}."
            ));
        }
        if handshake.is_none() && handshake_path.is_file() {
            let file = File::open(handshake_path)
                .map_err(|error| format!("Failed to open startup handshake: {error}"))?;
            let parsed: StartupHandshake = serde_json::from_reader(file)
                .map_err(|error| format!("Failed to parse startup handshake: {error}"))?;
            validate_handshake(&parsed, child.id())?;
            handshake = Some(parsed);
            progress(
                "backend_bind",
                "Backend bound to an authenticated loopback port.",
            );
        }
        if let Some(value) = &handshake {
            match authenticated_json(&format!("{}/api/v1/health", value.base_url), token) {
                Ok(body)
                    if body.get("ready").and_then(serde_json::Value::as_bool) == Some(true) =>
                {
                    return handshake.ok_or_else(|| "Startup handshake disappeared.".to_string());
                }
                Ok(body) => {
                    last_readiness = body
                        .get("startup_detail")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("Backend is alive but not ready.")
                        .to_string();
                    if last_readiness != last_emitted {
                        progress("api_readiness", &last_readiness);
                        last_emitted.clone_from(&last_readiness);
                    }
                }
                Err(error) => last_readiness = error,
            }
        }
        sleep(HEALTH_INTERVAL);
    }
    Err(format!(
        "Backend startup timed out after {} seconds. Last status: {last_readiness}",
        timeout.as_secs_f32()
    ))
}

fn ensure_startup_not_cancelled(cancellation: &StartupCancellation) -> Result<(), String> {
    if cancellation.load(Ordering::SeqCst) {
        Err("Local runtime startup was cancelled.".to_string())
    } else {
        Ok(())
    }
}

fn emit_startup_progress(app: &tauri::AppHandle, stage: &'static str, message: impl Into<String>) {
    let _ = app.emit(
        "desktop-startup-progress",
        StartupProgress {
            stage,
            message: message.into(),
        },
    );
}

fn validate_handshake(handshake: &StartupHandshake, child_pid: u32) -> Result<(), String> {
    if handshake.schema_version != 1 {
        return Err(format!(
            "Unsupported startup handshake schema {}.",
            handshake.schema_version
        ));
    }
    if handshake.host != "127.0.0.1" {
        return Err(format!(
            "Backend reported a non-loopback host: {}",
            handshake.host
        ));
    }
    if handshake.port == 0 || handshake.base_url != format!("http://127.0.0.1:{}", handshake.port) {
        return Err("Backend reported an invalid startup URL.".to_string());
    }
    if handshake.pid != child_pid {
        return Err("Backend startup handshake PID does not match the owned process.".to_string());
    }
    Ok(())
}

fn backend_is_ready(backend: &mut BackendProcess) -> bool {
    if backend.child.try_wait().ok().flatten().is_some() {
        return false;
    }
    authenticated_json(
        &format!("{}/health", backend.bootstrap.api_base_url),
        &backend.bootstrap.runtime_token,
    )
    .ok()
    .and_then(|body| body.get("ready").and_then(serde_json::Value::as_bool))
        == Some(true)
}

fn active_jobs_for_supervisor(supervisor: &mut Supervisor) -> Result<ActiveJobs, String> {
    let Some(backend) = supervisor.backend.as_mut() else {
        return Ok(empty_active_jobs());
    };
    if backend.child.try_wait().ok().flatten().is_some() {
        supervisor.backend = None;
        supervisor.lifecycle = Lifecycle::Failed;
        return Ok(empty_active_jobs());
    }
    let value = authenticated_json(
        &format!("{}/desktop/active-jobs", backend.bootstrap.api_base_url),
        &backend.bootstrap.runtime_token,
    )?;
    serde_json::from_value(value)
        .map_err(|error| format!("Failed to parse active-job status: {error}"))
}

fn authenticated_json(url: &str, token: &str) -> Result<serde_json::Value, String> {
    let response = ureq::get(url)
        .set("X-LLM-Studio-Token", token)
        .timeout(Duration::from_secs(3))
        .call()
        .map_err(|error| format!("Runtime request failed: {error}"))?;
    response
        .into_json()
        .map_err(|error| format!("Runtime returned invalid JSON: {error}"))
}

fn forward_runtime_request(
    api_base_url: &str,
    token: &str,
    request: RuntimeHttpRequest,
) -> Result<RuntimeHttpResponse, String> {
    let method = validate_runtime_request_method(&request.method)?;
    let path = validate_runtime_request_path(&request.path)?;
    let mut forwarded = ureq::request(method, &format!("{api_base_url}{path}"))
        .set("X-LLM-Studio-Token", token)
        .timeout(RUNTIME_REQUEST_TIMEOUT);
    for (name, value) in request.headers {
        let normalized = name.to_ascii_lowercase();
        if matches!(normalized.as_str(), "accept" | "content-type") {
            forwarded = forwarded.set(&name, &value);
        }
    }
    let result = if request.body.is_empty() {
        forwarded.call()
    } else {
        forwarded.send_bytes(&request.body)
    };
    let response = match result {
        Ok(response) => response,
        Err(ureq::Error::Status(_, response)) => response,
        Err(error) => return Err(format!("Runtime request failed: {error}")),
    };
    let status = response.status();
    let headers = response
        .headers_names()
        .into_iter()
        .filter_map(|name| {
            response
                .header(&name)
                .map(|value| (name.to_ascii_lowercase(), value.to_string()))
        })
        .collect();
    let mut body = Vec::new();
    response
        .into_reader()
        .read_to_end(&mut body)
        .map_err(|error| format!("Failed to read runtime response: {error}"))?;
    Ok(RuntimeHttpResponse {
        status,
        headers,
        body,
    })
}

fn validate_runtime_request_method(method: &str) -> Result<&str, String> {
    match method {
        "GET" | "POST" | "PUT" | "DELETE" => Ok(method),
        _ => Err("Runtime request method is not allowed.".to_string()),
    }
}

fn validate_runtime_request_path(path: &str) -> Result<&str, String> {
    if !path.starts_with('/')
        || path.starts_with("//")
        || path.contains(['\\', '#', '\r', '\n'])
        || path
            .split('?')
            .next()
            .unwrap_or(path)
            .split('/')
            .any(|segment| matches!(segment, "." | ".."))
    {
        return Err("Runtime request path is not allowed.".to_string());
    }
    Ok(path)
}

fn stop_supervisor(state: &SharedSupervisor) -> Result<(), String> {
    let mut supervisor = lock_supervisor(state)?;
    supervisor.lifecycle = Lifecycle::Stopping;
    if let Some(backend) = supervisor.backend.as_mut() {
        terminate_backend(backend);
    }
    supervisor.backend = None;
    supervisor.lifecycle = Lifecycle::Stopped;
    Ok(())
}

fn terminate_backend(backend: &mut BackendProcess) {
    #[cfg(unix)]
    terminate_child(&mut backend.child);
    #[cfg(windows)]
    match backend.job.take() {
        Some(job) => terminate_child(&mut backend.child, job),
        None => {
            let _ = backend.child.kill();
            let _ = backend.child.wait_timeout(SHUTDOWN_GRACE);
            let _ = backend.child.wait();
        }
    }
}

#[cfg(unix)]
fn configure_child_process_group(command: &mut Command) {
    use std::os::unix::process::CommandExt;
    command.process_group(0);
}

#[cfg(windows)]
fn configure_child_process_group(command: &mut Command) {
    use std::os::windows::process::CommandExt;
    command.creation_flags(0x0000_0200);
}

#[cfg(not(any(unix, windows)))]
fn configure_child_process_group(_command: &mut Command) {}

#[cfg(unix)]
fn terminate_child(child: &mut Child) {
    terminate_child_with_grace(child, SHUTDOWN_GRACE);
}

#[cfg(unix)]
fn terminate_child_with_grace(child: &mut Child, grace: Duration) {
    let group = child.id() as i32;
    unsafe {
        libc::kill(-group, libc::SIGTERM);
    }

    let graceful_deadline = Instant::now() + grace;
    while Instant::now() < graceful_deadline {
        let leader_exited = child.try_wait().ok().flatten().is_some();
        if leader_exited && !unix_process_group_exists(group) {
            return;
        }
        sleep(Duration::from_millis(25));
    }

    if unix_process_group_exists(group) {
        unsafe {
            libc::kill(-group, libc::SIGKILL);
        }
    }

    let _ = child.wait();
    let forced_deadline = Instant::now() + grace;
    while unix_process_group_exists(group) && Instant::now() < forced_deadline {
        sleep(Duration::from_millis(25));
    }
}

#[cfg(unix)]
fn unix_process_group_exists(group: i32) -> bool {
    unsafe { libc::kill(-group, 0) == 0 }
}

#[cfg(windows)]
fn terminate_child(child: &mut Child, job: WindowsJob) {
    use std::os::windows::io::AsRawHandle;

    unsafe {
        windows_sys::Win32::System::JobObjects::TerminateJobObject(job.as_raw_handle() as _, 1);
    }
    drop(job);
    let _ = child.wait_timeout(SHUTDOWN_GRACE);
    let _ = child.wait();
}

#[cfg(windows)]
fn assign_windows_job(child: &Child) -> Result<WindowsJob, String> {
    use std::os::windows::io::{AsRawHandle, FromRawHandle};
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
        SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };
    unsafe {
        let raw_job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        if raw_job.is_null() {
            return Err("Failed to create Windows Job Object.".to_string());
        }
        let job = WindowsJob::from_raw_handle(raw_job as _);
        let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if SetInformationJobObject(
            job.as_raw_handle() as _,
            JobObjectExtendedLimitInformation,
            &limits as *const _ as *const _,
            std::mem::size_of_val(&limits) as u32,
        ) == 0
            || AssignProcessToJobObject(job.as_raw_handle() as _, child.as_raw_handle() as _) == 0
        {
            return Err("Failed to assign backend to Windows Job Object.".to_string());
        }
        Ok(job)
    }
}

#[cfg(windows)]
fn assign_windows_job_or_terminate(child: &mut Child) -> Result<WindowsJob, String> {
    match assign_windows_job(child) {
        Ok(job) => Ok(job),
        Err(error) => {
            // The process is not safely owned until it belongs to the kill-on-close job.
            let _ = child.kill();
            let _ = child.wait();
            Err(error)
        }
    }
}

fn configure_clean_environment(command: &mut Command) {
    const PRESERVED: &[&str] = &[
        "PATH",
        "HOME",
        "USERPROFILE",
        "TMPDIR",
        "TEMP",
        "TMP",
        "SystemRoot",
        "WINDIR",
        "COMSPEC",
        "LANG",
        "LC_ALL",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    ];
    let values: Vec<(String, String)> = PRESERVED
        .iter()
        .filter_map(|key| env::var(key).ok().map(|value| ((*key).to_string(), value)))
        .collect();
    command.env_clear();
    command.envs(values);
}

fn runtime_path(root: &Path, relative: &str) -> Result<PathBuf, String> {
    let segments = relative.split('/').collect::<Vec<_>>();
    if segments.iter().any(|segment| {
        segment.is_empty() || matches!(*segment, "." | "..") || segment.contains(['\\', ':'])
    }) {
        return Err(format!(
            "Runtime manifest contains an unsafe path: {relative}"
        ));
    }
    Ok(segments
        .iter()
        .fold(root.to_path_buf(), |path, segment| path.join(segment)))
}

fn ensure_runtime_path_is_contained(root: &Path, candidate: &Path) -> Result<(), String> {
    let canonical_root = root
        .canonicalize()
        .map_err(|error| format!("Failed to resolve runtime root: {error}"))?;
    let canonical_candidate = candidate.canonicalize().map_err(|error| {
        format!(
            "Failed to resolve runtime file {}: {error}",
            candidate.display()
        )
    })?;
    if !canonical_candidate.starts_with(&canonical_root) {
        return Err(format!(
            "Runtime file resolves outside packaged resources: {}",
            candidate.display()
        ));
    }
    let relative = candidate
        .strip_prefix(root)
        .map_err(|_| format!("Runtime file is outside its root: {}", candidate.display()))?;
    let mut current = root.to_path_buf();
    for component in relative.components() {
        current.push(component);
        if current
            .symlink_metadata()
            .map(|metadata| metadata.file_type().is_symlink())
            .unwrap_or(false)
        {
            return Err(format!(
                "Release runtime contains a symlink: {}",
                current.display()
            ));
        }
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = File::open(path)
        .map_err(|error| format!("Failed to open {} for hashing: {error}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| format!("Failed to hash {}: {error}", path.display()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(hex::encode(hasher.finalize()))
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    atomic_write_from_reader(path, &mut std::io::Cursor::new(bytes))
}

fn atomic_write_from_reader(path: &Path, reader: &mut impl Read) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| "Selected destination has no parent directory.".to_string())?;
    if !parent.is_dir() {
        return Err("Selected destination directory does not exist.".to_string());
    }
    let temporary = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("llm-studio-export"),
        unique_suffix()
    ));
    let result = (|| {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)
            .map_err(|error| format!("Failed to create temporary export: {error}"))?;
        std::io::copy(reader, &mut file)
            .map_err(|error| format!("Failed to write export: {error}"))?;
        file.sync_all()
            .map_err(|error| format!("Failed to flush export: {error}"))?;
        replace_file(&temporary, path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn validate_api_artifact_path(path: &str) -> Result<(), String> {
    let segments = path
        .strip_prefix('/')
        .ok_or_else(|| "Artifact API path must begin with a slash.".to_string())?
        .split('/')
        .collect::<Vec<_>>();
    let identifier_is_safe = |value: &str| {
        !value.is_empty()
            && value.chars().all(|character| {
                character.is_ascii_alphanumeric() || matches!(character, '-' | '_')
            })
    };
    let allowed = match segments.as_slice() {
        ["projects", identifier, "artifact"] => identifier_is_safe(identifier),
        ["tokenizer", "jobs", identifier, "artifact"] => identifier_is_safe(identifier),
        ["training", "jobs", identifier, "artifact"] => identifier_is_safe(identifier),
        _ => false,
    };
    if allowed {
        Ok(())
    } else {
        Err("Artifact API path is not an allowed download endpoint.".to_string())
    }
}

fn artifact_metadata_request(path: &str) -> Result<(String, &'static str), String> {
    validate_api_artifact_path(path)?;
    let segments = path
        .strip_prefix('/')
        .expect("validated artifact paths always begin with a slash")
        .split('/')
        .collect::<Vec<_>>();
    match segments.as_slice() {
        ["projects", identifier, "artifact"] => {
            Ok((format!("/projects/{identifier}"), "artifact_path"))
        }
        ["tokenizer", "jobs", identifier, "artifact"] => Ok((
            format!("/tokenizer/jobs/{identifier}/artifact/meta"),
            "artifact_path",
        )),
        ["training", "jobs", identifier, "artifact"] => {
            Ok((format!("/training/jobs/{identifier}"), "artifact_dir"))
        }
        _ => Err("Artifact API path is not an allowed reveal endpoint.".to_string()),
    }
}

fn managed_artifact_path_from_metadata(
    metadata: &serde_json::Value,
    field: &str,
) -> Result<PathBuf, String> {
    let raw = metadata
        .get(field)
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| "Artifact metadata does not contain a managed path.".to_string())?;
    let path = PathBuf::from(raw);
    if !path.is_absolute() {
        return Err("Artifact metadata path is not absolute.".to_string());
    }
    Ok(path)
}

fn validate_managed_reveal_path(data_root: &Path, artifact_path: &Path) -> Result<PathBuf, String> {
    let canonical_root = data_root
        .canonicalize()
        .map_err(|error| format!("Failed to resolve app data root: {error}"))?;
    let canonical_artifact = artifact_path
        .canonicalize()
        .map_err(|error| format!("Failed to resolve managed artifact: {error}"))?;
    if !canonical_artifact.starts_with(&canonical_root) {
        return Err("Artifact resolves outside managed app data.".to_string());
    }
    if canonical_artifact.is_dir() {
        return Ok(canonical_artifact);
    }
    if canonical_artifact.is_file() {
        return canonical_artifact
            .parent()
            .map(Path::to_path_buf)
            .ok_or_else(|| "Managed artifact has no containing folder.".to_string());
    }
    Err("Managed artifact is not a file or directory.".to_string())
}

fn download_authenticated_to_path(url: &str, token: &str, path: &Path) -> Result<(), String> {
    let response = ureq::get(url)
        .set("X-LLM-Studio-Token", token)
        .timeout(Duration::from_secs(30 * 60))
        .call()
        .map_err(|error| format!("Artifact download failed: {error}"))?;
    atomic_write_from_reader(path, &mut response.into_reader())
}

#[cfg(unix)]
fn replace_file(temporary: &Path, destination: &Path) -> Result<(), String> {
    fs::rename(temporary, destination)
        .map_err(|error| format!("Failed to atomically finalize export: {error}"))
}

#[cfg(not(unix))]
fn replace_file(temporary: &Path, destination: &Path) -> Result<(), String> {
    if !destination.exists() {
        return fs::rename(temporary, destination)
            .map_err(|error| format!("Failed to finalize export: {error}"));
    }

    let backup = destination.with_file_name(format!(
        ".{}.{}.backup",
        destination
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("llm-studio-export"),
        unique_suffix()
    ));
    fs::rename(destination, &backup)
        .map_err(|error| format!("Failed to prepare existing export for replacement: {error}"))?;
    match fs::rename(temporary, destination) {
        Ok(()) => {
            let _ = fs::remove_file(backup);
            Ok(())
        }
        Err(error) => {
            let restore = fs::rename(&backup, destination);
            if let Err(restore_error) = restore {
                Err(format!(
                    "Failed to finalize export ({error}) and restore the original ({restore_error})."
                ))
            } else {
                Err(format!(
                    "Failed to finalize export; the original file was restored: {error}"
                ))
            }
        }
    }
}

fn sanitize_suggested_file_name(value: &str) -> String {
    let sanitized: String = value
        .trim()
        .chars()
        .filter_map(|character| match character {
            '/' | '\\' | ':' | '\0' => Some('_'),
            character if character.is_control() => None,
            character => Some(character),
        })
        .take(180)
        .collect();
    let sanitized = sanitized.trim_matches(['.', ' ']);
    if sanitized.is_empty() {
        "llm-studio-export.json".to_string()
    } else {
        sanitized.to_string()
    }
}

fn open_managed_folder(path: &Path) -> Result<(), String> {
    fs::create_dir_all(path)
        .map_err(|error| format!("Failed to prepare {}: {error}", path.display()))?;
    open::that(path).map_err(|error| format!("Failed to open managed folder: {error}"))
}

fn rotate_logs(log_dir: &Path) -> Result<(), String> {
    let mut logs: Vec<_> = fs::read_dir(log_dir)
        .map_err(|error| format!("Failed to inspect log directory: {error}"))?
        .filter_map(Result::ok)
        .filter(|entry| entry.file_name().to_string_lossy().starts_with("backend-"))
        .collect();
    logs.sort_by_key(|entry| entry.metadata().and_then(|value| value.modified()).ok());
    let remove_count = logs.len().saturating_sub(MAX_LOG_FILES.saturating_sub(1));
    for entry in logs.into_iter().take(remove_count) {
        let _ = fs::remove_file(entry.path());
    }
    Ok(())
}

fn write_shell_log(app: &tauri::AppHandle, event_id: &str, message: &str) -> Result<(), String> {
    let log_dir = app
        .path()
        .app_log_dir()
        .map_err(|error| format!("Failed to resolve shell log directory: {error}"))?;
    fs::create_dir_all(&log_dir)
        .map_err(|error| format!("Failed to create shell log directory: {error}"))?;
    let path = log_dir.join("shell.jsonl");
    rotate_sized_log(&path, MAX_SHELL_LOG_BYTES, MAX_SHELL_LOG_BACKUPS)?;
    let payload = serde_json::json!({
        "timestamp_unix": unix_timestamp(),
        "event_id": event_id,
        "message": redact_error(message),
    });
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(|error| format!("Failed to open shell log: {error}"))?;
    serde_json::to_writer(&mut file, &payload)
        .map_err(|error| format!("Failed to serialize shell log event: {error}"))?;
    file.write_all(b"\n")
        .map_err(|error| format!("Failed to append shell log event: {error}"))
}

fn rotate_sized_log(path: &Path, max_bytes: u64, backups: usize) -> Result<(), String> {
    if path.metadata().map(|metadata| metadata.len()).unwrap_or(0) < max_bytes {
        return Ok(());
    }
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| "Shell log path has no valid file name.".to_string())?;
    for index in (1..=backups).rev() {
        let source = if index == 1 {
            path.to_path_buf()
        } else {
            path.with_file_name(format!("{file_name}.{}", index - 1))
        };
        let destination = path.with_file_name(format!("{file_name}.{index}"));
        if destination.exists() {
            fs::remove_file(&destination)
                .map_err(|error| format!("Failed to remove old shell log backup: {error}"))?;
        }
        if source.exists() {
            fs::rename(&source, &destination)
                .map_err(|error| format!("Failed to rotate shell log: {error}"))?;
        }
    }
    Ok(())
}

fn log_inventory(app: &tauri::AppHandle) -> LogInventory {
    let entries = app
        .path()
        .app_log_dir()
        .ok()
        .and_then(|directory| fs::read_dir(directory).ok())
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .filter_map(|entry| entry.file_name().into_string().ok())
        .collect::<Vec<_>>();
    LogInventory {
        backend_log_files: entries
            .iter()
            .filter(|name| name.starts_with("backend"))
            .count(),
        shell_log_files: entries
            .iter()
            .filter(|name| name.starts_with("shell.jsonl"))
            .count(),
    }
}

fn storage_inventory(app: &tauri::AppHandle) -> StorageInventory {
    let path = app.path();
    StorageInventory {
        data: path
            .app_data_dir()
            .map(|root| {
                inventory_storage_root(
                    &root,
                    MAX_DIAGNOSTIC_STORAGE_ENTRIES,
                    MAX_DIAGNOSTIC_STORAGE_SCAN,
                )
            })
            .unwrap_or_else(|_| {
                StorageRootInventory::resolution_error(
                    MAX_DIAGNOSTIC_STORAGE_ENTRIES,
                    MAX_DIAGNOSTIC_STORAGE_SCAN,
                )
            }),
        cache: path
            .app_cache_dir()
            .map(|root| {
                inventory_storage_root(
                    &root,
                    MAX_DIAGNOSTIC_STORAGE_ENTRIES,
                    MAX_DIAGNOSTIC_STORAGE_SCAN,
                )
            })
            .unwrap_or_else(|_| {
                StorageRootInventory::resolution_error(
                    MAX_DIAGNOSTIC_STORAGE_ENTRIES,
                    MAX_DIAGNOSTIC_STORAGE_SCAN,
                )
            }),
        logs: path
            .app_log_dir()
            .map(|root| {
                inventory_storage_root(
                    &root,
                    MAX_DIAGNOSTIC_STORAGE_ENTRIES,
                    MAX_DIAGNOSTIC_STORAGE_SCAN,
                )
            })
            .unwrap_or_else(|_| {
                StorageRootInventory::resolution_error(
                    MAX_DIAGNOSTIC_STORAGE_ENTRIES,
                    MAX_DIAGNOSTIC_STORAGE_SCAN,
                )
            }),
    }
}

fn inventory_storage_root(
    root: &Path,
    max_entries: usize,
    max_duration: Duration,
) -> StorageRootInventory {
    let started = Instant::now();
    let mut inventory = StorageRootInventory::new(max_entries, max_duration);
    let metadata = match fs::symlink_metadata(root) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return inventory,
        Err(_) => {
            inventory.scan_complete = false;
            inventory.error_count = 1;
            return inventory;
        }
    };
    inventory.root_exists = true;

    let mut directories = Vec::new();
    if !record_storage_entry(
        &metadata,
        &mut inventory,
        &mut directories,
        root.to_path_buf(),
    ) {
        return inventory;
    }

    while let Some(directory) = directories.pop() {
        if storage_scan_time_limit_reached(&mut inventory, started, max_duration) {
            return inventory;
        }
        let entries = match fs::read_dir(directory) {
            Ok(entries) => entries,
            Err(_) => {
                inventory.scan_complete = false;
                inventory.error_count = inventory.error_count.saturating_add(1);
                continue;
            }
        };
        for entry in entries {
            if storage_scan_time_limit_reached(&mut inventory, started, max_duration) {
                return inventory;
            }
            if inventory.scanned_entries >= inventory.max_entries {
                inventory.scan_complete = false;
                return inventory;
            }
            let entry = match entry {
                Ok(entry) => entry,
                Err(_) => {
                    inventory.scan_complete = false;
                    inventory.error_count = inventory.error_count.saturating_add(1);
                    continue;
                }
            };
            let metadata = match fs::symlink_metadata(entry.path()) {
                Ok(metadata) => metadata,
                Err(_) => {
                    inventory.scan_complete = false;
                    inventory.error_count = inventory.error_count.saturating_add(1);
                    continue;
                }
            };
            if !record_storage_entry(&metadata, &mut inventory, &mut directories, entry.path()) {
                return inventory;
            }
        }
    }
    inventory
}

fn storage_scan_time_limit_reached(
    inventory: &mut StorageRootInventory,
    started: Instant,
    max_duration: Duration,
) -> bool {
    if started.elapsed() < max_duration {
        return false;
    }
    inventory.scan_complete = false;
    inventory.time_limit_reached = true;
    true
}

fn record_storage_entry(
    metadata: &fs::Metadata,
    inventory: &mut StorageRootInventory,
    directories: &mut Vec<PathBuf>,
    path: PathBuf,
) -> bool {
    if inventory.scanned_entries >= inventory.max_entries {
        inventory.scan_complete = false;
        return false;
    }
    inventory.scanned_entries = inventory.scanned_entries.saturating_add(1);
    let file_type = metadata.file_type();
    if file_type.is_symlink() {
        inventory.symlink_count = inventory.symlink_count.saturating_add(1);
    } else if file_type.is_dir() {
        inventory.directory_count = inventory.directory_count.saturating_add(1);
        directories.push(path);
    } else if file_type.is_file() {
        inventory.file_count = inventory.file_count.saturating_add(1);
        inventory.total_file_bytes = inventory.total_file_bytes.saturating_add(metadata.len());
    } else {
        inventory.other_entry_count = inventory.other_entry_count.saturating_add(1);
    }
    true
}

fn recent_shell_events(app: &tauri::AppHandle, limit: usize) -> Vec<serde_json::Value> {
    let Ok(log_dir) = app.path().app_log_dir() else {
        return Vec::new();
    };
    let Ok(content) = fs::read_to_string(log_dir.join("shell.jsonl")) else {
        return Vec::new();
    };
    let mut events: Vec<_> = content
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .collect();
    if events.len() > limit {
        events.drain(..events.len() - limit);
    }
    events
}

fn lock_supervisor(
    state: &SharedSupervisor,
) -> Result<std::sync::MutexGuard<'_, Supervisor>, String> {
    state
        .lock()
        .map_err(|_| "Desktop supervisor lock is poisoned.".to_string())
}

fn generate_runtime_token() -> String {
    let mut bytes = [0_u8; 32];
    OsRng.fill_bytes(&mut bytes);
    hex::encode(bytes)
}

fn non_empty_env(key: &str) -> Option<String> {
    env::var(key)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn normalize_platform(value: &str) -> &str {
    match value {
        "darwin" => "macos",
        "win32" => "windows",
        other => other,
    }
}

fn normalize_architecture(value: &str) -> &str {
    match value {
        "amd64" | "x86_64" => "x86_64",
        "aarch64" | "arm64" => "aarch64",
        other => other,
    }
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn unique_suffix() -> u128 {
    let mut bytes = [0_u8; 16];
    OsRng.fill_bytes(&mut bytes);
    u128::from_le_bytes(bytes)
}

fn redact_error(value: &str) -> String {
    let redacted = value
        .lines()
        .map(|line| {
            let lower = line.to_ascii_lowercase();
            if ["token", "api_key", "api key", "authorization", "secret"]
                .iter()
                .any(|marker| lower.contains(marker))
            {
                "[redacted secret-related error]"
            } else {
                line
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    redact_sensitive_paths(&redacted)
}

fn redact_sensitive_paths(value: &str) -> String {
    let mut redacted = value.to_string();
    for path in [
        env::var("HOME").ok(),
        env::var("USERPROFILE").ok(),
        env::current_dir()
            .ok()
            .map(|path| path.display().to_string()),
    ]
    .into_iter()
    .flatten()
    .filter(|path| !path.is_empty())
    {
        redacted = redacted.replace(&path, "[REDACTED_PATH]");
    }
    redacted
}

fn close_warning(jobs: &ActiveJobs) -> String {
    if jobs.has_active_runpod_training {
        "A RunPod job may continue running and billing after LLM Studio exits. Return to Training and stop or clean up the RunPod job, or explicitly exit knowing billing may continue.".to_string()
    } else if jobs.has_active_local_training {
        "Local training is active. Choose stop and exit to terminate the owned process tree, or continue running in the app.".to_string()
    } else {
        "A tokenizer or training job is active. Choose stop and exit or continue running in the app.".to_string()
    }
}

fn empty_active_jobs() -> ActiveJobs {
    ActiveJobs {
        active: false,
        tokenizer_jobs: Vec::new(),
        training_jobs: Vec::new(),
        has_active_local_training: false,
        has_active_runpod_training: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::process::Stdio;
    use std::thread;

    #[test]
    fn generated_tokens_are_random_and_high_entropy() {
        let first = generate_runtime_token();
        let second = generate_runtime_token();
        assert_eq!(first.len(), 64);
        assert_eq!(second.len(), 64);
        assert_ne!(first, second);
        assert!(first.chars().all(|character| character.is_ascii_hexdigit()));
    }

    #[test]
    fn temporary_suffixes_are_unique_across_a_parallel_burst() {
        let suffixes = (0..1_000)
            .map(|_| unique_suffix())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(suffixes.len(), 1_000);
    }

    #[test]
    fn suggested_names_remove_path_traversal_and_platform_separators() {
        assert_eq!(
            sanitize_suggested_file_name("../../unsafe\\name:artifact.json"),
            "_.._unsafe_name_artifact.json"
        );
        assert_eq!(
            sanitize_suggested_file_name(" \n "),
            "llm-studio-export.json"
        );
    }

    #[test]
    fn artifact_download_paths_are_narrowly_scoped() {
        for path in [
            "/projects/project_123/artifact",
            "/tokenizer/jobs/tokenizer-123/artifact",
            "/training/jobs/training_123/artifact",
        ] {
            assert!(validate_api_artifact_path(path).is_ok(), "{path}");
        }
        for path in [
            "projects/project_123/artifact",
            "/projects/../artifact",
            "/projects/project_123/artifact?token=secret",
            "/training/providers/runpod/pods",
            "https://example.com/artifact",
        ] {
            assert!(validate_api_artifact_path(path).is_err(), "{path}");
        }
    }

    #[test]
    fn runtime_proxy_paths_and_methods_are_narrowly_scoped() {
        for method in ["GET", "POST", "PUT", "DELETE"] {
            assert_eq!(validate_runtime_request_method(method).unwrap(), method);
        }
        for method in ["", "PATCH", "OPTIONS", "get"] {
            assert!(validate_runtime_request_method(method).is_err(), "{method}");
        }
        for path in [
            "/projects",
            "/training/jobs/job_123?limit=10",
            "/tokenizer/config/templates",
        ] {
            assert_eq!(validate_runtime_request_path(path).unwrap(), path);
        }
        for path in [
            "",
            "projects",
            "//example.com/projects",
            "/projects/../outside",
            "/projects#fragment",
            "/projects\r\nX-Evil: value",
        ] {
            assert!(validate_runtime_request_path(path).is_err(), "{path:?}");
        }
    }

    #[test]
    fn runtime_proxy_forwards_only_safe_headers_and_preserves_error_responses() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 4096];
            let count = stream.read(&mut request).unwrap();
            let request = String::from_utf8_lossy(&request[..count]).to_ascii_lowercase();
            assert!(request.starts_with("post /api/v1/projects http/1.1"));
            assert!(request.contains("x-llm-studio-token: proxy-token"));
            assert!(request.contains("content-type: application/json"));
            assert!(!request.contains("x-unsafe-header"));
            assert!(request.ends_with(r#"{"name":"test"}"#));
            let body = br#"{"detail":"invalid project"}"#;
            let response = format!(
                "HTTP/1.1 422 Unprocessable Entity\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            stream.write_all(response.as_bytes()).unwrap();
            stream.write_all(body).unwrap();
        });
        let response = forward_runtime_request(
            &format!("http://{address}/api/v1"),
            "proxy-token",
            RuntimeHttpRequest {
                method: "POST".to_string(),
                path: "/projects".to_string(),
                headers: BTreeMap::from([
                    ("Content-Type".to_string(), "application/json".to_string()),
                    ("X-Unsafe-Header".to_string(), "blocked".to_string()),
                ]),
                body: br#"{"name":"test"}"#.to_vec(),
            },
        )
        .unwrap();
        server.join().unwrap();

        assert_eq!(response.status, 422);
        assert_eq!(
            response.headers.get("content-type").map(String::as_str),
            Some("application/json")
        );
        assert_eq!(response.body, br#"{"detail":"invalid project"}"#);
    }

    #[test]
    fn artifact_reveal_uses_only_typed_metadata_endpoints() {
        assert_eq!(
            artifact_metadata_request("/projects/project_123/artifact").unwrap(),
            ("/projects/project_123".to_string(), "artifact_path")
        );
        assert_eq!(
            artifact_metadata_request("/tokenizer/jobs/tokenizer-123/artifact").unwrap(),
            (
                "/tokenizer/jobs/tokenizer-123/artifact/meta".to_string(),
                "artifact_path"
            )
        );
        assert_eq!(
            artifact_metadata_request("/training/jobs/training_123/artifact").unwrap(),
            ("/training/jobs/training_123".to_string(), "artifact_dir")
        );
        assert!(artifact_metadata_request("/training/jobs/training_123/logs").is_err());
        assert!(artifact_metadata_request("/projects/../artifact").is_err());
    }

    #[test]
    fn artifact_reveal_is_confined_to_managed_app_data() {
        let root =
            std::env::temp_dir().join(format!("llm-studio-artifact-reveal-{}", unique_suffix()));
        let managed = root.join("data");
        let project = managed.join("projects").join("project_123");
        fs::create_dir_all(&project).unwrap();
        let artifact = project.join("model_config.json");
        fs::write(&artifact, b"{}").unwrap();
        let outside = root.join("outside.json");
        fs::write(&outside, b"{}").unwrap();

        assert_eq!(
            validate_managed_reveal_path(&managed, &artifact).unwrap(),
            project.canonicalize().unwrap()
        );
        assert_eq!(
            validate_managed_reveal_path(&managed, &project).unwrap(),
            project.canonicalize().unwrap()
        );
        assert!(validate_managed_reveal_path(&managed, &outside).is_err());
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&outside, managed.join("escaped-link")).unwrap();
            assert!(validate_managed_reveal_path(&managed, &managed.join("escaped-link")).is_err());
        }
        assert!(managed_artifact_path_from_metadata(
            &serde_json::json!({"artifact_path": "relative/artifact.json"}),
            "artifact_path"
        )
        .is_err());

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn authenticated_artifact_download_streams_to_an_atomic_file() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 4096];
            let count = stream.read(&mut request).unwrap();
            let request = String::from_utf8_lossy(&request[..count]).to_ascii_lowercase();
            assert!(request.contains("x-llm-studio-token: stream-token"));
            let body = b"streamed-artifact";
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            stream.write_all(response.as_bytes()).unwrap();
            stream.write_all(body).unwrap();
        });
        let root =
            std::env::temp_dir().join(format!("llm-studio-download-test-{}", unique_suffix()));
        fs::create_dir_all(&root).unwrap();
        let destination = root.join("artifact.zip");

        download_authenticated_to_path(
            &format!("http://{address}/artifact"),
            "stream-token",
            &destination,
        )
        .unwrap();
        server.join().unwrap();

        assert_eq!(fs::read(&destination).unwrap(), b"streamed-artifact");
        assert_eq!(fs::read_dir(&root).unwrap().count(), 1);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn atomic_write_creates_and_replaces_without_leaving_temporary_files() {
        let root = std::env::temp_dir().join(format!("llm-studio-write-test-{}", unique_suffix()));
        fs::create_dir_all(&root).unwrap();
        let destination = root.join("artifact.json");

        atomic_write(&destination, b"first").unwrap();
        atomic_write(&destination, b"second").unwrap();

        assert_eq!(fs::read(&destination).unwrap(), b"second");
        assert_eq!(fs::read_dir(&root).unwrap().count(), 1);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn atomic_write_rejects_missing_destination_directory() {
        let root = std::env::temp_dir().join(format!("llm-studio-write-test-{}", unique_suffix()));
        let error =
            atomic_write(&root.join("missing").join("artifact.json"), b"value").unwrap_err();
        assert!(error.contains("does not exist"));
        assert!(!root.exists());
    }

    #[test]
    fn sized_log_rotation_preserves_bounded_backups() {
        let root = std::env::temp_dir().join(format!("llm-studio-log-test-{}", unique_suffix()));
        fs::create_dir_all(&root).unwrap();
        let log = root.join("shell.jsonl");
        fs::write(&log, b"oversized").unwrap();

        rotate_sized_log(&log, 1, 2).unwrap();
        assert!(!log.exists());
        assert_eq!(fs::read(root.join("shell.jsonl.1")).unwrap(), b"oversized");

        fs::write(&log, b"new").unwrap();
        rotate_sized_log(&log, 1, 2).unwrap();
        assert_eq!(fs::read(root.join("shell.jsonl.1")).unwrap(), b"new");
        assert_eq!(fs::read(root.join("shell.jsonl.2")).unwrap(), b"oversized");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn storage_inventory_reports_large_files_without_sensitive_paths() {
        let root =
            std::env::temp_dir().join(format!("llm-studio-storage-test-{}", unique_suffix()));
        let nested = root.join("nested");
        fs::create_dir_all(&nested).unwrap();
        fs::write(root.join("small.bin"), b"abc").unwrap();
        File::create(nested.join("large.bin"))
            .unwrap()
            .set_len(128 * 1024 * 1024)
            .unwrap();

        let inventory = inventory_storage_root(&root, 100, Duration::from_secs(10));
        let serialized = serde_json::to_string(&inventory).unwrap();

        assert_eq!(
            inventory,
            StorageRootInventory {
                root_exists: true,
                file_count: 2,
                directory_count: 2,
                symlink_count: 0,
                other_entry_count: 0,
                total_file_bytes: 128 * 1024 * 1024 + 3,
                scanned_entries: 4,
                max_entries: 100,
                max_scan_milliseconds: 10_000,
                scan_complete: true,
                time_limit_reached: false,
                error_count: 0,
            }
        );
        assert!(!serialized.contains(&root.display().to_string()));
        assert!(!serialized.contains("large.bin"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn storage_inventory_is_bounded_and_reports_missing_roots() {
        let root =
            std::env::temp_dir().join(format!("llm-studio-storage-test-{}", unique_suffix()));
        fs::create_dir_all(&root).unwrap();
        for index in 0..10 {
            fs::write(root.join(format!("{index}.bin")), b"x").unwrap();
        }

        let bounded = inventory_storage_root(&root, 3, Duration::from_secs(10));
        let timed_out = inventory_storage_root(&root, 100, Duration::ZERO);
        let missing = inventory_storage_root(&root.join("missing"), 3, Duration::from_secs(10));

        assert!(bounded.root_exists);
        assert_eq!(bounded.scanned_entries, 3);
        assert!(!bounded.scan_complete);
        assert!(!bounded.time_limit_reached);
        assert!(timed_out.root_exists);
        assert_eq!(timed_out.scanned_entries, 1);
        assert!(!timed_out.scan_complete);
        assert!(timed_out.time_limit_reached);
        assert_eq!(
            missing,
            StorageRootInventory {
                root_exists: false,
                file_count: 0,
                directory_count: 0,
                symlink_count: 0,
                other_entry_count: 0,
                total_file_bytes: 0,
                scanned_entries: 0,
                max_entries: 3,
                max_scan_milliseconds: 10_000,
                scan_complete: true,
                time_limit_reached: false,
                error_count: 0,
            }
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn storage_inventory_counts_but_never_follows_symlinks() {
        let root =
            std::env::temp_dir().join(format!("llm-studio-storage-test-{}", unique_suffix()));
        let outside =
            std::env::temp_dir().join(format!("llm-studio-storage-outside-{}", unique_suffix()));
        fs::create_dir_all(&root).unwrap();
        fs::create_dir_all(&outside).unwrap();
        fs::write(outside.join("secret.bin"), vec![0_u8; 4096]).unwrap();
        std::os::unix::fs::symlink(&outside, root.join("outside-link")).unwrap();

        let inventory = inventory_storage_root(&root, 100, Duration::from_secs(10));

        assert_eq!(inventory.directory_count, 1);
        assert_eq!(inventory.file_count, 0);
        assert_eq!(inventory.symlink_count, 1);
        assert_eq!(inventory.total_file_bytes, 0);
        assert!(inventory.scan_complete);
        fs::remove_dir_all(root).unwrap();
        fs::remove_dir_all(outside).unwrap();
    }

    #[test]
    fn runtime_paths_reject_absolute_and_parent_paths() {
        let root = Path::new("/runtime");
        for unsafe_path in [
            "",
            ".",
            "../secret",
            "source/../secret",
            "/etc/passwd",
            "//server/share",
            r"\server\share",
            r"C:\secret",
            "C:/secret",
            "C:secret",
            "source\\secret",
            "source//secret",
            "source/./secret",
        ] {
            assert!(runtime_path(root, unsafe_path).is_err(), "{unsafe_path}");
        }
        assert_eq!(
            runtime_path(root, "source/model/model.py").unwrap(),
            root.join("source/model/model.py")
        );
    }

    #[test]
    fn release_runtime_paths_reject_symlink_escape() {
        let root =
            std::env::temp_dir().join(format!("llm-studio-runtime-test-{}", unix_timestamp()));
        let outside =
            std::env::temp_dir().join(format!("llm-studio-outside-test-{}", unix_timestamp()));
        fs::create_dir_all(&root).unwrap();
        fs::write(&outside, b"outside").unwrap();
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&outside, root.join("linked")).unwrap();
            assert!(
                ensure_runtime_path_is_contained(&root, &root.join("linked"))
                    .unwrap_err()
                    .contains("outside packaged resources")
            );
        }
        let _ = fs::remove_dir_all(root);
        let _ = fs::remove_file(outside);
    }

    #[test]
    fn close_warning_mentions_runpod_billing() {
        let mut jobs = empty_active_jobs();
        jobs.active = true;
        jobs.has_active_runpod_training = true;
        let warning = close_warning(&jobs);
        assert!(warning.contains("billing"));
        assert!(warning.contains("explicitly exit"));
        assert!(!warning.contains("stop and exit"));
    }

    #[test]
    fn retries_are_bounded_and_rate_limited() {
        let mut supervisor = Supervisor {
            attempts: 1,
            last_attempt_at: Some(Instant::now()),
            ..Supervisor::default()
        };
        assert!(validate_start_attempt(&supervisor)
            .unwrap_err()
            .contains("rapid crash loop"));

        supervisor.attempts = MAX_START_ATTEMPTS;
        supervisor.last_attempt_at = None;
        assert!(validate_start_attempt(&supervisor)
            .unwrap_err()
            .contains("failed to start"));
    }

    #[test]
    fn startup_cancellation_is_explicit_and_resettable() {
        let cancellation = Arc::new(AtomicBool::new(false));
        assert!(ensure_startup_not_cancelled(&cancellation).is_ok());
        cancellation.store(true, Ordering::SeqCst);
        assert!(ensure_startup_not_cancelled(&cancellation)
            .unwrap_err()
            .contains("cancelled"));
        cancellation.store(false, Ordering::SeqCst);
        assert!(ensure_startup_not_cancelled(&cancellation).is_ok());
    }

    #[cfg(windows)]
    #[test]
    fn windows_supervisor_state_is_send_and_sync() {
        fn assert_send_and_sync<T: Send + Sync>() {}

        assert_send_and_sync::<WindowsJob>();
        assert_send_and_sync::<SharedSupervisor>();
    }

    #[test]
    fn shell_error_redaction_covers_common_secret_markers() {
        assert_eq!(
            redact_error("Authorization: Bearer value\nordinary failure\nrunpod api key=value"),
            "[redacted secret-related error]\nordinary failure\n[redacted secret-related error]"
        );
    }

    #[test]
    fn shell_error_redaction_removes_home_and_repository_paths() {
        let home = env::var("HOME").unwrap();
        let cwd = env::current_dir().unwrap();
        let message = format!("{home}/private/file\n{}/runtime", cwd.display());
        let redacted = redact_error(&message);
        assert!(!redacted.contains(&home));
        assert!(!redacted.contains(&cwd.display().to_string()));
        assert!(redacted.contains("[REDACTED_PATH]"));
    }

    #[test]
    fn manifest_compatibility_enforces_shell_and_data_versions() {
        let mut manifest = compatible_manifest();
        assert!(validate_manifest_compatibility(&manifest, false).is_ok());

        manifest.data_schema_version = "4".to_string();
        assert!(validate_manifest_compatibility(&manifest, false)
            .unwrap_err()
            .contains("data schema"));

        manifest = compatible_manifest();
        manifest.shell_compatibility.minimum = "0.2.0".to_string();
        assert!(validate_manifest_compatibility(&manifest, false)
            .unwrap_err()
            .contains("current shell"));

        manifest = compatible_manifest();
        manifest
            .provenance
            .insert("build_mode".to_string(), "linked-development".to_string());
        assert!(validate_manifest_compatibility(&manifest, false)
            .unwrap_err()
            .contains("not release-portable"));
        assert!(validate_manifest_compatibility(&manifest, true).is_ok());
    }

    #[test]
    fn fake_sidecar_ready_handshake_reaches_readiness() {
        let (mut child, handshake) = spawn_fake_sidecar("ready", "test-token");
        let cancellation = Arc::new(AtomicBool::new(false));
        let mut stages = Vec::new();
        let result = wait_for_handshake_and_readiness_with_progress(
            &mut child,
            &handshake,
            "test-token",
            &cancellation,
            Duration::from_secs(3),
            |stage, _message| stages.push(stage),
        );
        let child_pid = child.id();
        terminate_child_for_test(&mut child);
        let _ = fs::remove_file(handshake);
        let result = result.unwrap();

        assert_eq!(result.pid, child_pid);
        assert!(stages.contains(&"backend_bind"));
    }

    #[test]
    fn fake_sidecar_crash_is_reported() {
        let (mut child, handshake) = spawn_fake_sidecar("crash", "test-token");
        let result =
            wait_for_fake_sidecar(&mut child, &handshake, "test-token", Duration::from_secs(3));
        terminate_child_for_test(&mut child);
        let _ = fs::remove_file(handshake);
        let error = result.unwrap_err();

        assert!(error.contains("exited during startup"), "{error}");
    }

    #[test]
    fn fake_sidecar_timeout_and_bad_token_fail_closed() {
        let (mut child, handshake) = spawn_fake_sidecar("timeout", "test-token");
        let result = wait_for_fake_sidecar(
            &mut child,
            &handshake,
            "test-token",
            Duration::from_millis(350),
        );
        terminate_child_for_test(&mut child);
        let _ = fs::remove_file(handshake);
        let error = result.unwrap_err();
        assert!(error.contains("timed out"), "{error}");

        let (mut child, handshake) = spawn_fake_sidecar("bad_token", "test-token");
        let mut stages = Vec::new();
        let result = wait_for_handshake_and_readiness_with_progress(
            &mut child,
            &handshake,
            "wrong-token",
            &Arc::new(AtomicBool::new(false)),
            Duration::from_secs(3),
            |stage, _message| stages.push(stage),
        );
        terminate_child_for_test(&mut child);
        let _ = fs::remove_file(handshake);
        let error = result.unwrap_err();
        assert!(error.contains("timed out"), "{error}");
        assert!(
            stages.contains(&"backend_bind"),
            "bad-token test never reached authenticated readiness checks"
        );
    }

    #[test]
    fn fake_sidecar_invalid_handshake_and_port_are_rejected() {
        for (mode, expected) in [
            ("invalid_schema", "Unsupported startup handshake schema"),
            ("invalid_port", "invalid startup URL"),
        ] {
            let (mut child, handshake) = spawn_fake_sidecar(mode, "test-token");
            let result =
                wait_for_fake_sidecar(&mut child, &handshake, "test-token", Duration::from_secs(3));
            terminate_child_for_test(&mut child);
            let _ = fs::remove_file(handshake);
            let error = result.unwrap_err();
            assert!(error.contains(expected), "{error}");
        }
    }

    #[test]
    fn owned_fake_sidecar_is_terminated() {
        let (mut child, handshake) = spawn_fake_sidecar("timeout", "test-token");
        terminate_child_for_test(&mut child);

        assert!(child.try_wait().unwrap().is_some());
        let _ = fs::remove_file(handshake);
    }

    #[cfg(unix)]
    #[test]
    fn owned_process_group_termination_kills_descendants_only() {
        let descendant_pid_path = std::env::temp_dir().join(format!(
            "llm-studio-fake-descendant-{}.pid",
            unique_suffix()
        ));
        let (mut owned, owned_handshake) =
            spawn_fake_sidecar_with_descendant("test-token", &descendant_pid_path);
        let descendant_pid = wait_for_descendant_pid(&descendant_pid_path);
        let owned_group = owned.id() as i32;
        assert_eq!(
            unsafe { libc::getpgid(descendant_pid) },
            owned_group,
            "descendant must inherit the owned backend process group"
        );

        let (mut unrelated, unrelated_handshake) = spawn_fake_sidecar("timeout", "test-token");
        unsafe {
            libc::kill(-owned_group, libc::SIGTERM);
        }
        let leader_deadline = Instant::now() + Duration::from_secs(1);
        while owned.try_wait().unwrap().is_none() && Instant::now() < leader_deadline {
            thread::sleep(Duration::from_millis(10));
        }
        assert!(owned.try_wait().unwrap().is_some());
        assert!(
            unix_process_group_exists(owned_group),
            "SIGTERM-resistant descendant did not remain for forced-shutdown coverage"
        );

        terminate_child_with_grace(&mut owned, Duration::from_millis(100));

        assert!(owned.try_wait().unwrap().is_some());
        assert!(
            wait_for_process_group_exit(owned_group, Duration::from_secs(3)),
            "owned descendant process group remained alive"
        );
        assert!(
            unrelated.try_wait().unwrap().is_none(),
            "termination escaped the owned process group"
        );

        terminate_child_for_test(&mut unrelated);
        let _ = fs::remove_file(owned_handshake);
        let _ = fs::remove_file(unrelated_handshake);
        let _ = fs::remove_file(descendant_pid_path);
    }

    #[test]
    fn fake_sidecar_process() {
        let Ok(mode) = env::var("LLM_STUDIO_FAKE_SIDECAR_MODE") else {
            return;
        };
        #[cfg(unix)]
        if env::var_os("LLM_STUDIO_FAKE_IGNORE_TERM").is_some() {
            unsafe {
                libc::signal(libc::SIGTERM, libc::SIG_IGN);
            }
            fs::write(
                env::var("LLM_STUDIO_FAKE_DESCENDANT_PID").unwrap(),
                std::process::id().to_string(),
            )
            .unwrap();
        }
        if mode == "crash" {
            return;
        }
        if mode == "timeout" {
            thread::sleep(Duration::from_secs(10));
            return;
        }
        if mode == "process_tree" {
            let mut descendant = Command::new(std::env::current_exe().unwrap())
                .arg("--exact")
                .arg("tests::fake_sidecar_process")
                .arg("--nocapture")
                .env("LLM_STUDIO_FAKE_SIDECAR_MODE", "timeout")
                .env("LLM_STUDIO_FAKE_IGNORE_TERM", "1")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .unwrap();
            let _ = descendant.wait();
            return;
        }

        let handshake = PathBuf::from(env::var("LLM_STUDIO_FAKE_HANDSHAKE").unwrap());
        if mode == "invalid_port" {
            write_fake_handshake(&handshake, 1, 0, "http://127.0.0.1:0");
            thread::sleep(Duration::from_secs(10));
            return;
        }

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let port = listener.local_addr().unwrap().port();
        write_fake_handshake(
            &handshake,
            if mode == "invalid_schema" { 2 } else { 1 },
            port,
            &format!("http://127.0.0.1:{port}"),
        );
        if mode == "invalid_schema" {
            thread::sleep(Duration::from_secs(10));
            return;
        }

        let expected_token = env::var("LLM_STUDIO_FAKE_TOKEN").unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            match listener.accept() {
                Ok((mut stream, _address)) => {
                    let mut request = [0_u8; 4096];
                    let count = stream.read(&mut request).unwrap_or(0);
                    let request = String::from_utf8_lossy(&request[..count]).to_ascii_lowercase();
                    let authorized = request.contains(&format!(
                        "x-llm-studio-token: {}",
                        expected_token.to_ascii_lowercase()
                    ));
                    let (status, body) = if authorized {
                        ("200 OK", r#"{"ready":true,"startup_detail":"ready"}"#)
                    } else {
                        ("401 Unauthorized", r#"{"detail":"Unauthorized"}"#)
                    };
                    let response = format!(
                        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    stream.write_all(response.as_bytes()).unwrap();
                    if authorized && mode == "ready" {
                        return;
                    }
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(10));
                }
                Err(error) => panic!("fake sidecar accept failed: {error}"),
            }
        }
    }

    fn spawn_fake_sidecar(mode: &str, token: &str) -> (Child, PathBuf) {
        let handshake =
            std::env::temp_dir().join(format!("llm-studio-fake-sidecar-{}.json", unique_suffix()));
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .arg("--exact")
            .arg("tests::fake_sidecar_process")
            .arg("--nocapture")
            .env("LLM_STUDIO_FAKE_SIDECAR_MODE", mode)
            .env("LLM_STUDIO_FAKE_HANDSHAKE", &handshake)
            .env("LLM_STUDIO_FAKE_TOKEN", token)
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        configure_child_process_group(&mut command);
        (command.spawn().unwrap(), handshake)
    }

    #[cfg(unix)]
    fn spawn_fake_sidecar_with_descendant(token: &str, descendant_pid: &Path) -> (Child, PathBuf) {
        let handshake =
            std::env::temp_dir().join(format!("llm-studio-fake-sidecar-{}.json", unique_suffix()));
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .arg("--exact")
            .arg("tests::fake_sidecar_process")
            .arg("--nocapture")
            .env("LLM_STUDIO_FAKE_SIDECAR_MODE", "process_tree")
            .env("LLM_STUDIO_FAKE_HANDSHAKE", &handshake)
            .env("LLM_STUDIO_FAKE_TOKEN", token)
            .env("LLM_STUDIO_FAKE_DESCENDANT_PID", descendant_pid)
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        configure_child_process_group(&mut command);
        (command.spawn().unwrap(), handshake)
    }

    #[cfg(unix)]
    fn wait_for_descendant_pid(path: &Path) -> i32 {
        let deadline = Instant::now() + Duration::from_secs(3);
        while Instant::now() < deadline {
            if let Ok(value) = fs::read_to_string(path) {
                return value.parse().unwrap();
            }
            thread::sleep(Duration::from_millis(10));
        }
        panic!("fake sidecar descendant PID was not written");
    }

    #[cfg(unix)]
    fn wait_for_process_group_exit(group: i32, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if !unix_process_group_exists(group) {
                return true;
            }
            thread::sleep(Duration::from_millis(10));
        }
        !unix_process_group_exists(group)
    }

    fn wait_for_fake_sidecar(
        child: &mut Child,
        handshake: &Path,
        token: &str,
        timeout: Duration,
    ) -> Result<StartupHandshake, String> {
        wait_for_handshake_and_readiness_with_progress(
            child,
            handshake,
            token,
            &Arc::new(AtomicBool::new(false)),
            timeout,
            |_stage, _message| {},
        )
    }

    fn write_fake_handshake(path: &Path, schema_version: u32, port: u16, base_url: &str) {
        fs::write(
            path,
            serde_json::to_vec(&serde_json::json!({
                "schema_version": schema_version,
                "host": "127.0.0.1",
                "port": port,
                "base_url": base_url,
                "pid": std::process::id(),
            }))
            .unwrap(),
        )
        .unwrap();
    }

    #[cfg(unix)]
    fn terminate_child_for_test(child: &mut Child) {
        terminate_child(child);
    }

    #[cfg(windows)]
    fn terminate_child_for_test(child: &mut Child) {
        if child.try_wait().ok().flatten().is_some() {
            return;
        }
        if let Ok(job) = assign_windows_job_or_terminate(child) {
            terminate_child(child, job);
        }
    }

    fn compatible_manifest() -> RuntimeManifest {
        RuntimeManifest {
            schema_version: SUPPORTED_MANIFEST_SCHEMA,
            runtime_version: "test".to_string(),
            shell_compatibility: ShellCompatibility {
                minimum: "0.1.0".to_string(),
                maximum_exclusive: "0.2.0".to_string(),
            },
            api_contract_version: SUPPORTED_API_CONTRACT.to_string(),
            data_schema_version: SUPPORTED_DATA_SCHEMA.to_string(),
            platform: normalize_platform(env::consts::OS).to_string(),
            architecture: normalize_architecture(env::consts::ARCH).to_string(),
            python_version: "3.12.0".to_string(),
            source_root: "source".to_string(),
            python_executable: "python/bin/python".to_string(),
            required_files: Vec::new(),
            file_hashes: BTreeMap::new(),
            dependency_versions: BTreeMap::new(),
            provenance: BTreeMap::from([("build_mode".to_string(), "portable".to_string())]),
            size: None,
        }
    }

    #[test]
    fn runtime_manifest_preserves_size_policy_metadata() {
        let mut manifest = compatible_manifest();
        let size = RuntimeSize {
            threshold_kind: "release_threshold".to_string(),
            target: "macos-aarch64".to_string(),
            max_payload_bytes: 1_000,
            max_payload_files: 100,
            payload_total_bytes: 900,
            payload_file_count: 90,
            policy_file: "scripts/desktop/runtime-size-policy.json".to_string(),
        };
        manifest.size = Some(size.clone());

        let serialized = serde_json::to_vec(&manifest).unwrap();
        let decoded: RuntimeManifest = serde_json::from_slice(&serialized).unwrap();

        assert_eq!(decoded.size, Some(size));
    }
}
