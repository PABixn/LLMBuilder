# LLM Studio Desktop Troubleshooting

## Runtime Does Not Start

1. Select **Retry backend** once.
2. Open the logs folder and inspect the newest `backend-*.log`.
3. Export diagnostics and retain the JSON for support.
4. Reinstall the matching target package if diagnostics report a missing,
   incompatible, or checksum-invalid runtime.

Development builds can validate an override with:

```bash
make -C apps/llm-studio desktop-build-runtime
make -C apps/llm-studio desktop-smoke-runtime
```

## Loopback or Endpoint-Security Failure

The backend binds an ephemeral `127.0.0.1` port and requires a launch token.
Endpoint-security software must allow local process-to-process loopback traffic.
The app never requires a public firewall exception. If startup reports that a
configured port is already in use, close the conflicting process or restore the
desktop default of port `0` so LLM Studio can reserve an available ephemeral
port.

## Storage or Database Startup Failure

LLM Studio validates that its data, cache, and log roots are writable and have
adequate free space before desktop startup. A storage failure names the affected
managed path. Check folder permissions, free disk space, and security software,
then retry.

Exported diagnostics include aggregate file counts and bytes for the managed
data, cache, and log roots without exposing filenames or paths. A
`scan_complete` value of `false` means the bounded inventory reached its entry
or time limit, or encountered an unreadable entry; inspect the corresponding
managed folder while LLM Studio is closed.

A database-unavailable error means a managed SQLite database may be locked by
another process, read-only, or corrupt. Quit every LLM Studio instance before
retrying. If the failure persists, preserve the entire data folder and its
`backups/` directory before restoring a coherent backup. Do not delete or edit a
database, `-wal`, or `-shm` file independently.

An upgrade from historical Tokenizer Studio defaults may also leave
`tokenizer_studio.db`, `training_studio.db`, and
`legacy-database-name-migration.json` beside the active LLM Studio databases.
This is intentional rollback evidence. Do not delete it until the upgraded data
has been reviewed and backed up.

## Corporate Proxy or Custom CA

The runtime preserves standard proxy and trust variables such as
`HTTPS_PROXY`, `NO_PROXY`, `SSL_CERT_FILE`, and `REQUESTS_CA_BUNDLE`. Configure
them through your managed environment before launch. Do not put API keys in proxy
URLs or support logs.

## Missing History or Artifacts

Use **Open data folder**. Desktop data is not stored in the repository or
application bundle. Reinstall/uninstall should not remove it. Avoid moving or
editing SQLite files while the app is running.

## Backup, Cache Cleanup, and Complete Uninstall

Quit LLM Studio before filesystem maintenance. Use **Open data folder** and copy
the entire folder for a coherent backup. Preserve `backups/` and database
sidecar files (`-wal`/`-shm`) together with their databases. Cache contents may
be deleted while the app is closed; they are recreated on demand, though
network-backed datasets or models may need to be downloaded again.

Normal uninstall removes application binaries and preserves user data. For a
complete removal, uninstall first, then delete the LLM Studio data, cache, and
log folders manually after retaining any required artifacts or backups.

## RunPod Job After Crash

A remote pod may still be running and billing. Open RunPod directly, inspect the
job/pod, and stop or delete it as appropriate. Automatic remote reattach is
recovery-limited because LLM Studio does not persist the raw pod-agent token.

For private Hugging Face datasets, LLM Studio keeps the dataset token
memory-only on the local machine and excludes it from local job state and remote
bundle files. The token is necessarily sent to the created RunPod pod as an
environment variable so the pod can access the dataset. Use a least-privilege,
short-lived token, and rotate it if a remote provider account or pod may have
been exposed.

Locally, active jobs use an exact-value, process-memory-only redaction scope so
legacy tokens without a recognizable provider prefix are still removed from
errors and logs. Terminal training transitions scrub managed text outputs before
that scope is released.

## Unsupported Compute

The readiness diagnostics report CPU, MPS, and CUDA availability from the
packaged runtime. Availability outside the documented support matrix is not a
release guarantee.
