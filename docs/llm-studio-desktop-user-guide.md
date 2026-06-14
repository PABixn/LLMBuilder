# LLM Studio Desktop User Guide

## Install and Launch

Install the signed package for your OS and launch **LLM Studio** normally. No
terminal, Python, Node.js, or separate web server is required by a release build.
The startup screen remains visible while managed paths, runtime checksums,
Python/compute dependencies, backend bind, migrations, and API readiness
complete. The current stage is shown and startup can be cancelled.

If startup fails, use **Retry backend**, **Open logs**, **Open data folder**, or
**Export diagnostics**. Diagnostics are local and do not contain project data or
secrets. They report only aggregate data/cache/log file counts and bytes, never
filenames or full paths; very large trees may be reported as a bounded,
incomplete scan.

## Data and Offline Behavior

Projects, tokenizer jobs, training history, artifacts, and caches persist in the
OS application-data directory. Local workflows work offline when all selected
datasets and dependencies are already local. Hugging Face, RunPod, and updates
still require network access.

On the first launch after upgrading from historical Tokenizer Studio data, LLM
Studio safely copies legacy default databases and browser preferences into the
current LLM Studio names. Existing current data wins, and retained legacy
database files provide rollback evidence.

Uninstalling the application preserves user data by default.

## Backup and Cleanup

Use **Open data folder** before backup or cleanup. Quit LLM Studio first, then
copy the complete data folder to preserve projects, job history, databases,
artifacts, and migration backups. The cache folder contains reproducible
downloads and may be removed while the app is closed; it will be recreated as
needed. Do not selectively edit SQLite databases or active job directories.

To remove LLM Studio completely, uninstall the application, then manually
remove its data, cache, and log folders only after confirming backups are no
longer needed. Reinstalling or uninstalling the application does not remove user
data by default.

## Saving Files

Model exports, tokenizer artifacts, training artifacts, and diagnostics use the
same native **Save As** dialog. Cancelling the dialog leaves data unchanged.
In the Workspace, the folder action reveals an existing managed artifact without
exposing an arbitrary filesystem path to the interface.

## Closing During Work

When local work is active, closing the window is blocked until you return to the
app or choose **Stop local work and exit**. When RunPod work is active, the exit
action explicitly warns that the remote pod may continue billing. Stop or clean
up RunPod resources explicitly before exiting. Automatic reattach after a
crash/restart is currently limited.

## Compute Support

- macOS arm64: CPU and MPS when the packaged PyTorch runtime reports MPS.
- Windows x64: CPU in the v1 support promise.
- Linux x64 beta: CPU.
- RunPod: remote GPU behavior follows the existing web workflow.

## Dataset Credentials

Hugging Face dataset tokens entered in the interface are held in process memory
only. LLM Studio excludes them from browser storage, normalized configuration,
job databases, job input files, diagnostics, logs, and remote bundle archives.
Upgrading to data schema v3 also removes credentials retained by older managed
database rows, backups, and job inputs.

While a job is active, LLM Studio also tracks the exact token value only in
process memory so arbitrary legacy token formats are redacted from errors and
logs. Training completion, failure, or cancellation scrubs managed text outputs
before that in-memory redaction scope is released.

A RunPod job that reads a private Hugging Face dataset must receive that token
in the created pod environment. Treat RunPod as a credential trust boundary:
use a least-privilege, short-lived Hugging Face token and stop or delete remote
resources according to your organization policy.
