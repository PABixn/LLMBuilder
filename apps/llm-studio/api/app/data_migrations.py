from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sqlite3
from typing import Any

from .config import RuntimeSettings
from .dataset_credentials import strip_hf_tokens
from .logging_config import redact_secrets
from .managed_locations import encode_managed_location
from .storage_safety import UnsafeManagedPathError, ensure_free_space, require_managed_path

DATA_SCHEMA_VERSION = 3
_SCHEMA_FILE = "data-schema.json"
_LEGACY_IMPORT_FILE = "legacy-source-tree-import.json"
_LEGACY_DATABASE_NAME_MIGRATION_FILE = "legacy-database-name-migration.json"
_LEGACY_DATABASE_NAMES = {
    "tokenizer": ("llm_studio_tokenizer.db", "tokenizer_studio.db"),
    "training": ("llm_studio_training.db", "training_studio.db"),
}
_TOKENIZER_MANAGED_PATHS = {
    "training_jobs": ("evaluation_text_path", "artifact_path"),
    "uploaded_files": ("file_path",),
}
_TRAINING_MANAGED_PATHS = {
    "llm_training_jobs": (
        "artifact_dir",
        "stats_path",
        "samples_path",
        "stdout_path",
        "stderr_path",
    ),
}
_TRAINING_JOB_JSON_FILES = ("dataloader_config.json", "resolved_preflight.json")
_TRAINING_JOB_LOG_FILES = (
    "stdout.log",
    "stderr.log",
    "runpod_lifecycle.jsonl",
    "runpod_lifecycle.log",
)
_TRANSIENT_REMOTE_BUNDLE_FILES = ("bundle.tar.gz", "bundle.tar.zst")


def prepare_data_schema(settings: RuntimeSettings) -> dict[str, Any]:
    """Back up existing SQLite state, import legacy managed files, then commit metadata."""
    database_name_migration = _migrate_legacy_database_names(settings)
    schema_path = settings.data_dir / _SCHEMA_FILE
    current = _read_schema_version(schema_path)
    if current > DATA_SCHEMA_VERSION:
        raise RuntimeError(
            f"Data schema {current} is newer than supported schema {DATA_SCHEMA_VERSION}; "
            "refusing an unsafe downgrade."
        )

    backup_dir: Path | None = None
    migrated_paths = 0
    credential_cleanup = {
        "database_fields": 0,
        "legacy_database_fields": 0,
        "backup_database_fields": 0,
        "job_files": 0,
        "transient_bundles_removed": 0,
    }
    if current < DATA_SCHEMA_VERSION:
        backup_dir = _backup_databases(settings, from_version=current)
        migrated_paths = _migrate_managed_path_records(settings)
        credential_cleanup = _migrate_persisted_credentials(settings)
        _write_json_atomic(
            schema_path,
            {
                "schema_version": DATA_SCHEMA_VERSION,
                "previous_schema_version": current,
                "migrated_at": _timestamp(),
                "backup_dir": (
                    backup_dir.relative_to(settings.data_dir).as_posix()
                    if backup_dir is not None
                    else None
                ),
                "credential_cleanup": credential_cleanup,
            },
            managed_root=settings.data_dir,
            description="data schema metadata",
        )

    import_summary = _import_legacy_source_tree_data(settings)
    return {
        "schema_version": DATA_SCHEMA_VERSION,
        "previous_schema_version": current,
        "backup_created": backup_dir is not None,
        "managed_paths_migrated": migrated_paths,
        "credential_cleanup": credential_cleanup,
        "legacy_database_names": database_name_migration,
        "legacy_import": import_summary,
    }


def _read_schema_version(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return int(payload["schema_version"])
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Data schema metadata is corrupt: {path}") from exc


def _backup_databases(settings: RuntimeSettings, *, from_version: int) -> Path | None:
    sources = [
        ("tokenizer", settings.tokenizer_database_path),
        ("training", settings.training_database_path),
    ]
    existing = [(name, path) for name, path in sources if path.is_file()]
    if not existing:
        return None
    required_bytes = sum(
        candidate.stat().st_size
        for _database_name, source in existing
        for candidate in (source, Path(f"{source}-wal"), Path(f"{source}-shm"))
        if candidate.is_file()
    )
    ensure_free_space(
        settings.data_dir,
        minimum_free_bytes=max(64 * 1024 * 1024, required_bytes * 2),
        operation="database migration backup",
    )
    backup_dir = settings.data_dir / "backups" / (
        f"schema-{from_version}-to-{DATA_SCHEMA_VERSION}-{_timestamp(compact=True)}"
    )
    backup_dir = require_managed_path(
        backup_dir,
        settings.data_dir,
        description="database backup directory",
    )
    backup_dir.mkdir(parents=True, exist_ok=False)
    try:
        for database_name, source in existing:
            database_backup_dir = require_managed_path(
                backup_dir / database_name,
                settings.data_dir,
                description=f"{database_name} database backup directory",
            )
            database_backup_dir.mkdir()
            for candidate in (source, Path(f"{source}-wal"), Path(f"{source}-shm")):
                if candidate.is_file():
                    shutil.copy2(candidate, database_backup_dir / candidate.name)
    except Exception:
        shutil.rmtree(backup_dir, ignore_errors=True)
        raise
    return backup_dir


def _migrate_legacy_database_names(settings: RuntimeSettings) -> dict[str, Any]:
    marker = settings.data_dir / _LEGACY_DATABASE_NAME_MIGRATION_FILE
    completed = _read_completed_database_name_migrations(marker)
    previously_completed = set(completed)
    copied: list[str] = []
    targets = {
        "tokenizer": settings.tokenizer_database_path,
        "training": settings.training_database_path,
    }
    for database_name, target in targets.items():
        current_name, legacy_name = _LEGACY_DATABASE_NAMES[database_name]
        expected_target = (settings.data_dir / "db" / current_name).absolute()
        if target.absolute() != expected_target or database_name in completed:
            continue
        if target.is_symlink():
            raise UnsafeManagedPathError(
                f"Refusing symlinked {database_name} database migration target."
            )
        target = require_managed_path(
            target,
            settings.data_dir,
            description=f"{database_name} database name migration target",
        )
        legacy = settings.data_dir / "db" / legacy_name
        if target.exists():
            completed.add(database_name)
            continue
        if legacy.is_file():
            if legacy.is_symlink():
                raise UnsafeManagedPathError(
                    f"Refusing symlinked legacy {database_name} database."
                )
            legacy = require_managed_path(
                legacy,
                settings.data_dir,
                description=f"legacy {database_name} database",
                must_exist=True,
            )
            _copy_sqlite_database(legacy, target, data_root=settings.data_dir)
            copied.append(database_name)
        completed.add(database_name)

    if completed and (completed != previously_completed or not marker.is_file()):
        _write_json_atomic(
            marker,
            {
                "schema_version": 1,
                "completed_at": _timestamp(),
                "completed_databases": sorted(completed),
                "copied_databases": copied,
                "legacy_sources_retained": True,
            },
            managed_root=settings.data_dir,
            description="legacy database name migration marker",
        )
    return {
        "copied": copied,
        "completed": sorted(completed),
        "legacy_sources_retained": True,
    }


def _read_completed_database_name_migrations(marker: Path) -> set[str]:
    if not marker.is_file():
        return set()
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        completed = payload["completed_databases"]
        if not isinstance(completed, list):
            raise TypeError("completed_databases must be a list")
        values = {str(item) for item in completed}
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Legacy database name migration metadata is corrupt: {marker}") from exc
    unknown = values - _LEGACY_DATABASE_NAMES.keys()
    if unknown:
        raise RuntimeError(
            "Legacy database name migration metadata contains unknown databases: "
            + ", ".join(sorted(unknown))
        )
    return values


def _copy_sqlite_database(source: Path, target: Path, *, data_root: Path) -> None:
    ensure_free_space(
        data_root,
        minimum_free_bytes=max(64 * 1024 * 1024, source.stat().st_size * 2),
        operation="legacy database name migration",
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = require_managed_path(
        target.with_suffix(f"{target.suffix}.migration.tmp"),
        data_root,
        description="temporary legacy database name migration target",
    )
    temporary.unlink(missing_ok=True)
    try:
        with closing(sqlite3.connect(source)) as source_connection, closing(
            sqlite3.connect(temporary)
        ) as target_connection:
            source_connection.backup(target_connection)
            result = target_connection.execute("PRAGMA quick_check").fetchone()
            if result != ("ok",):
                raise RuntimeError(
                    f"Legacy database name migration integrity check failed: {source}"
                )
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _import_legacy_source_tree_data(settings: RuntimeSettings) -> dict[str, int]:
    marker = settings.data_dir / _LEGACY_IMPORT_FILE
    if marker.is_file():
        return {"artifacts": 0, "uploads": 0}

    api_root = settings.source_root / "apps" / "llm-studio" / "api"
    mappings = (
        (api_root / "artifacts" / "tokenizers", settings.tokenizer_output_dir, "artifacts"),
        (api_root / "datasets" / "uploads", settings.tokenizer_upload_dir, "uploads"),
    )
    summary = {"artifacts": 0, "uploads": 0}
    for source, destination, key in mappings:
        if not source.is_dir() or source.resolve() == destination.resolve():
            continue
        destination.mkdir(parents=True, exist_ok=True)
        for item in source.iterdir():
            if not item.is_file():
                continue
            target = destination / item.name
            if target.exists():
                continue
            target = require_managed_path(
                target,
                destination,
                description=f"legacy {key} import target",
            )
            shutil.copy2(item, target)
            summary[key] += 1

    _write_json_atomic(
        marker,
        {
            "schema_version": 1,
            "completed_at": _timestamp(),
            "copy_only": True,
            "summary": summary,
        },
        managed_root=settings.data_dir,
        description="legacy import marker",
    )
    return summary


def _migrate_managed_path_records(settings: RuntimeSettings) -> int:
    migrated = 0
    migrated += _migrate_sqlite_managed_paths(
        settings.tokenizer_database_path,
        settings.data_dir,
        _TOKENIZER_MANAGED_PATHS,
    )
    migrated += _migrate_sqlite_managed_paths(
        settings.training_database_path,
        settings.data_dir,
        _TRAINING_MANAGED_PATHS,
    )
    return migrated


def _migrate_persisted_credentials(settings: RuntimeSettings) -> dict[str, int]:
    database_fields = _migrate_sqlite_secret_fields(
        settings.tokenizer_database_path,
        table="training_jobs",
        json_columns=("dataloader_config",),
        text_columns=("error",),
    )
    database_fields += _migrate_sqlite_secret_fields(
        settings.training_database_path,
        table="llm_training_jobs",
        json_columns=("dataloader_config",),
        text_columns=("error", "remote_error"),
    )
    legacy_database_fields = _migrate_legacy_database_credentials(settings)
    backup_database_fields = _migrate_backup_database_credentials(settings)
    job_files, transient_bundles_removed = _sanitize_training_job_files(
        settings.training_jobs_dir
    )
    return {
        "database_fields": database_fields,
        "legacy_database_fields": legacy_database_fields,
        "backup_database_fields": backup_database_fields,
        "job_files": job_files,
        "transient_bundles_removed": transient_bundles_removed,
    }


def _migrate_legacy_database_credentials(settings: RuntimeSettings) -> int:
    database_dir = settings.data_dir / "db"
    if not database_dir.is_dir() or database_dir.is_symlink():
        return 0
    migrated = 0
    migrated += _migrate_sqlite_secret_fields(
        database_dir / _LEGACY_DATABASE_NAMES["tokenizer"][1],
        table="training_jobs",
        json_columns=("dataloader_config",),
        text_columns=("error",),
    )
    migrated += _migrate_sqlite_secret_fields(
        database_dir / _LEGACY_DATABASE_NAMES["training"][1],
        table="llm_training_jobs",
        json_columns=("dataloader_config",),
        text_columns=("error", "remote_error"),
    )
    return migrated


def _migrate_backup_database_credentials(settings: RuntimeSettings) -> int:
    backups_root = settings.data_dir / "backups"
    if not backups_root.is_dir() or backups_root.is_symlink():
        return 0

    migrated = 0
    for backup_dir in backups_root.iterdir():
        if backup_dir.is_symlink() or not backup_dir.is_dir():
            continue
        migrated += _migrate_sqlite_secret_fields(
            backup_dir / "tokenizer" / settings.tokenizer_database_path.name,
            table="training_jobs",
            json_columns=("dataloader_config",),
            text_columns=("error",),
        )
        migrated += _migrate_sqlite_secret_fields(
            backup_dir / "training" / settings.training_database_path.name,
            table="llm_training_jobs",
            json_columns=("dataloader_config",),
            text_columns=("error", "remote_error"),
        )
    return migrated


def _migrate_sqlite_secret_fields(
    database_path: Path,
    *,
    table: str,
    json_columns: tuple[str, ...],
    text_columns: tuple[str, ...],
) -> int:
    if not database_path.is_file():
        return 0

    migrated = 0
    compacted = False
    with sqlite3.connect(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        existing_tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        if table not in existing_tables:
            return 0
        existing_columns = {
            str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")')
        }
        columns = tuple(
            column
            for column in (*json_columns, *text_columns)
            if column in existing_columns
        )
        if not columns:
            return 0
        compacted = True

        selected = ", ".join(f'"{column}"' for column in columns)
        for row in connection.execute(f'SELECT rowid, {selected} FROM "{table}"').fetchall():
            updates: dict[str, str] = {}
            for column, value in zip(columns, row[1:]):
                if value is None:
                    continue
                original = str(value)
                sanitized = (
                    _strip_hf_tokens_from_json_text(original)
                    if column in json_columns
                    else redact_secrets(original)
                )
                if sanitized != original:
                    updates[column] = sanitized
            if not updates:
                continue
            assignments = ", ".join(f'"{column}" = ?' for column in updates)
            connection.execute(
                f'UPDATE "{table}" SET {assignments} WHERE rowid = ?',
                (*updates.values(), int(row[0])),
            )
            migrated += len(updates)
        connection.commit()
        _checkpoint_sqlite_wal(connection, database_path)
        connection.execute("VACUUM")
        _checkpoint_sqlite_wal(connection, database_path)
    if compacted:
        _remove_sqlite_sidecars(database_path)
    return migrated


def _checkpoint_sqlite_wal(connection: sqlite3.Connection, database_path: Path) -> None:
    result = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    if result is not None and int(result[0]) != 0:
        raise RuntimeError(
            f"Could not checkpoint SQLite WAL during credential cleanup: {database_path}"
        )


def _remove_sqlite_sidecars(database_path: Path) -> None:
    for suffix in ("-wal", "-shm"):
        sidecar = Path(f"{database_path}{suffix}")
        if sidecar.is_symlink():
            raise UnsafeManagedPathError(
                f"Refusing symlinked SQLite sidecar during credential cleanup: {sidecar}"
            )
        sidecar.unlink(missing_ok=True)


def _strip_hf_tokens_from_json_text(value: str) -> str:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return value
    sanitized = strip_hf_tokens(payload)
    if sanitized == payload:
        return value
    return json.dumps(sanitized, ensure_ascii=True, separators=(",", ":"))


def _sanitize_training_job_files(training_jobs_dir: Path) -> tuple[int, int]:
    if not training_jobs_dir.is_dir() or training_jobs_dir.is_symlink():
        return 0, 0

    sanitized_files = 0
    removed_bundles = 0
    for job_dir in training_jobs_dir.iterdir():
        if job_dir.is_symlink() or not job_dir.is_dir():
            continue
        for file_name in _TRAINING_JOB_JSON_FILES:
            path = job_dir / file_name
            if path.is_symlink() or not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            sanitized = strip_hf_tokens(payload)
            if sanitized == payload:
                continue
            _write_json_atomic(
                path,
                sanitized,
                managed_root=training_jobs_dir,
                description="managed training job credential cleanup",
            )
            sanitized_files += 1

        for file_name in _TRAINING_JOB_LOG_FILES:
            path = job_dir / file_name
            if path.is_symlink() or not path.is_file():
                continue
            try:
                original = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            sanitized = redact_secrets(original)
            if sanitized == original:
                continue
            _write_text_atomic(
                path,
                sanitized,
                managed_root=training_jobs_dir,
                description="managed training job log credential cleanup",
            )
            sanitized_files += 1

        staging_dir = job_dir / ".remote_bundle"
        if staging_dir.is_dir() and not staging_dir.is_symlink():
            shutil.rmtree(staging_dir)
            removed_bundles += 1
        for file_name in _TRANSIENT_REMOTE_BUNDLE_FILES:
            path = job_dir / file_name
            if path.is_file() and not path.is_symlink():
                path.unlink()
                removed_bundles += 1
    return sanitized_files, removed_bundles


def _migrate_sqlite_managed_paths(
    database_path: Path,
    data_root: Path,
    tables: dict[str, tuple[str, ...]],
) -> int:
    if not database_path.is_file():
        return 0

    migrated = 0
    with sqlite3.connect(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        existing_tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        for table, configured_columns in tables.items():
            if table not in existing_tables:
                continue
            existing_columns = {
                str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")')
            }
            columns = tuple(column for column in configured_columns if column in existing_columns)
            if not columns:
                continue
            selected = ", ".join(f'"{column}"' for column in columns)
            for row in connection.execute(f'SELECT rowid, {selected} FROM "{table}"').fetchall():
                rowid = int(row[0])
                updates: dict[str, str] = {}
                for column, value in zip(columns, row[1:]):
                    if value is None:
                        continue
                    original = str(value)
                    encoded = encode_managed_location(original, data_root)
                    if encoded != original:
                        updates[column] = encoded
                if not updates:
                    continue
                assignments = ", ".join(f'"{column}" = ?' for column in updates)
                connection.execute(
                    f'UPDATE "{table}" SET {assignments} WHERE rowid = ?',
                    (*updates.values(), rowid),
                )
                migrated += len(updates)
    return migrated


def _write_json_atomic(
    path: Path,
    payload: Any,
    *,
    managed_root: Path,
    description: str,
) -> None:
    path = require_managed_path(path, managed_root, description=description)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = require_managed_path(
        path.with_suffix(f"{path.suffix}.tmp"),
        managed_root,
        description=f"temporary {description}",
    )
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_text_atomic(
    path: Path,
    value: str,
    *,
    managed_root: Path,
    description: str,
) -> None:
    path = require_managed_path(path, managed_root, description=description)
    temporary = require_managed_path(
        path.with_suffix(f"{path.suffix}.tmp"),
        managed_root,
        description=f"temporary {description}",
    )
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _timestamp(*, compact: bool = False) -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y%m%dT%H%M%S%fZ") if compact else now.isoformat()
