from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import sqlite3
from typing import Any

import pytest

from app import config
from app import data_migrations
from app.storage_safety import InsufficientStorageError, UnsafeManagedPathError

MIGRATION_FIXTURES = Path(__file__).with_name("fixtures") / "desktop-migrations"


def _settings(monkeypatch, tmp_path: Path) -> config.RuntimeSettings:
    monkeypatch.setenv("LLM_STUDIO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LLM_STUDIO_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("LLM_STUDIO_LOG_DIR", str(tmp_path / "logs"))
    config.reset_settings_cache()
    settings = config.get_settings()
    config.ensure_runtime_directories(settings)
    return settings


def _sqlite_database(path: Path, value: str = "fixture") -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE fixture (value TEXT NOT NULL)")
        connection.execute("INSERT INTO fixture (value) VALUES (?)", (value,))


def _expand_fixture(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        for placeholder, replacement in replacements.items():
            value = value.replace(placeholder, replacement)
        return value
    if isinstance(value, list):
        return [_expand_fixture(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _expand_fixture(item, replacements) for key, item in value.items()}
    return value


def test_prepare_data_schema_backs_up_existing_databases(monkeypatch, tmp_path: Path) -> None:
    settings = _settings(monkeypatch, tmp_path)
    _sqlite_database(settings.tokenizer_database_path, "tokenizer-db")
    _sqlite_database(settings.training_database_path, "training-db")

    status = data_migrations.prepare_data_schema(settings)

    metadata = json.loads((settings.data_dir / "data-schema.json").read_text(encoding="utf-8"))
    backup = settings.data_dir / metadata["backup_dir"]
    assert status["backup_created"] is True
    with sqlite3.connect(backup / "tokenizer" / settings.tokenizer_database_path.name) as connection:
        assert connection.execute("SELECT value FROM fixture").fetchone()[0] == "tokenizer-db"
    with sqlite3.connect(backup / "training" / settings.training_database_path.name) as connection:
        assert connection.execute("SELECT value FROM fixture").fetchone()[0] == "training-db"


def test_prepare_data_schema_copies_legacy_database_names_once(
    monkeypatch,
    tmp_path: Path,
) -> None:
    settings = _settings(monkeypatch, tmp_path)
    legacy_tokenizer = settings.data_dir / "db" / "tokenizer_studio.db"
    legacy_training = settings.data_dir / "db" / "training_studio.db"
    _sqlite_database(legacy_tokenizer, "legacy-tokenizer")
    _sqlite_database(legacy_training, "legacy-training")

    first = data_migrations.prepare_data_schema(settings)
    marker_path = settings.data_dir / "legacy-database-name-migration.json"
    marker = marker_path.read_text(encoding="utf-8")
    second = data_migrations.prepare_data_schema(settings)

    assert first["legacy_database_names"]["copied"] == ["tokenizer", "training"]
    assert second["legacy_database_names"]["copied"] == []
    assert marker_path.read_text(encoding="utf-8") == marker
    assert legacy_tokenizer.is_file()
    assert legacy_training.is_file()
    with sqlite3.connect(settings.tokenizer_database_path) as connection:
        assert connection.execute("SELECT value FROM fixture").fetchone()[0] == "legacy-tokenizer"
    with sqlite3.connect(settings.training_database_path) as connection:
        assert connection.execute("SELECT value FROM fixture").fetchone()[0] == "legacy-training"


def test_legacy_database_name_migration_never_overwrites_current_database(
    monkeypatch,
    tmp_path: Path,
) -> None:
    settings = _settings(monkeypatch, tmp_path)
    _sqlite_database(settings.data_dir / "db" / "tokenizer_studio.db", "legacy")
    _sqlite_database(settings.tokenizer_database_path, "current")

    status = data_migrations._migrate_legacy_database_names(settings)

    assert status["copied"] == []
    with sqlite3.connect(settings.tokenizer_database_path) as connection:
        assert connection.execute("SELECT value FROM fixture").fetchone()[0] == "current"


def test_legacy_database_name_migration_does_not_import_into_custom_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    settings = _settings(monkeypatch, tmp_path)
    _sqlite_database(settings.data_dir / "db" / "tokenizer_studio.db", "legacy")
    custom = settings.data_dir / "db" / "custom-tokenizer.db"
    settings = replace(settings, tokenizer_database_path=custom)

    status = data_migrations._migrate_legacy_database_names(settings)

    assert status["copied"] == []
    assert not custom.exists()
    assert "tokenizer" not in status["completed"]


def test_legacy_database_name_migration_rejects_symlinked_source(
    monkeypatch,
    tmp_path: Path,
) -> None:
    settings = _settings(monkeypatch, tmp_path)
    outside = tmp_path / "outside.db"
    _sqlite_database(outside, "outside")
    legacy = settings.data_dir / "db" / "tokenizer_studio.db"
    try:
        legacy.symlink_to(outside)
    except OSError:
        pytest.skip("Symlink creation is unavailable on this platform.")

    with pytest.raises(UnsafeManagedPathError, match="symlinked legacy tokenizer"):
        data_migrations._migrate_legacy_database_names(settings)

    assert not settings.tokenizer_database_path.exists()
    assert not (settings.data_dir / "legacy-database-name-migration.json").exists()


def test_legacy_database_name_migration_rejects_symlinked_target(
    monkeypatch,
    tmp_path: Path,
) -> None:
    settings = _settings(monkeypatch, tmp_path)
    _sqlite_database(settings.data_dir / "db" / "tokenizer_studio.db", "legacy")
    outside = tmp_path / "outside.db"
    try:
        settings.tokenizer_database_path.symlink_to(outside)
    except OSError:
        pytest.skip("Symlink creation is unavailable on this platform.")

    with pytest.raises(UnsafeManagedPathError, match="symlinked tokenizer database migration target"):
        data_migrations._migrate_legacy_database_names(settings)

    assert not outside.exists()
    assert not (settings.data_dir / "legacy-database-name-migration.json").exists()


def test_legacy_database_name_migration_removes_partial_target_after_invalid_source(
    monkeypatch,
    tmp_path: Path,
) -> None:
    settings = _settings(monkeypatch, tmp_path)
    legacy = settings.data_dir / "db" / "tokenizer_studio.db"
    legacy.write_text("not a SQLite database", encoding="utf-8")

    with pytest.raises(sqlite3.DatabaseError):
        data_migrations._migrate_legacy_database_names(settings)

    assert not settings.tokenizer_database_path.exists()
    assert not settings.tokenizer_database_path.with_suffix(".db.migration.tmp").exists()
    assert not (settings.data_dir / "legacy-database-name-migration.json").exists()


def test_legacy_database_name_migration_closes_database_handles(
    monkeypatch,
    tmp_path: Path,
) -> None:
    settings = _settings(monkeypatch, tmp_path)
    legacy = settings.data_dir / "db" / "tokenizer_studio.db"
    _sqlite_database(legacy, "legacy")
    real_connect = sqlite3.connect
    tracked_connections: list[Any] = []

    class TrackedConnection:
        def __init__(self, path: Path) -> None:
            self.connection = real_connect(path)
            self.closed = False
            tracked_connections.append(self)

        def backup(self, target: "TrackedConnection") -> None:
            self.connection.backup(target.connection)

        def execute(self, statement: str):
            return self.connection.execute(statement)

        def close(self) -> None:
            self.connection.close()
            self.closed = True

    monkeypatch.setattr(
        data_migrations.sqlite3,
        "connect",
        lambda path: TrackedConnection(path),
    )

    data_migrations._copy_sqlite_database(
        legacy,
        settings.tokenizer_database_path,
        data_root=settings.data_dir,
    )

    assert len(tracked_connections) == 2
    assert all(connection.closed for connection in tracked_connections)


def test_legacy_database_name_migration_rejects_corrupt_marker(
    monkeypatch,
    tmp_path: Path,
) -> None:
    settings = _settings(monkeypatch, tmp_path)
    marker = settings.data_dir / "legacy-database-name-migration.json"
    marker.write_text("{not-json", encoding="utf-8")
    _sqlite_database(settings.data_dir / "db" / "tokenizer_studio.db", "legacy")

    with pytest.raises(RuntimeError, match="migration metadata is corrupt"):
        data_migrations._migrate_legacy_database_names(settings)

    assert not settings.tokenizer_database_path.exists()


def test_prepare_data_schema_rejects_unsafe_downgrade(monkeypatch, tmp_path: Path) -> None:
    settings = _settings(monkeypatch, tmp_path)
    (settings.data_dir / "data-schema.json").write_text(
        json.dumps({"schema_version": data_migrations.DATA_SCHEMA_VERSION + 1}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="unsafe downgrade"):
        data_migrations.prepare_data_schema(settings)


def test_database_backup_keeps_same_named_databases_separate(monkeypatch, tmp_path: Path) -> None:
    settings = _settings(monkeypatch, tmp_path)
    tokenizer_database = tmp_path / "tokenizer-custom" / "studio.db"
    training_database = tmp_path / "training-custom" / "studio.db"
    tokenizer_database.parent.mkdir()
    training_database.parent.mkdir()
    _sqlite_database(tokenizer_database, "tokenizer")
    _sqlite_database(training_database, "training")
    settings = replace(
        settings,
        tokenizer_database_path=tokenizer_database,
        training_database_path=training_database,
    )

    status = data_migrations.prepare_data_schema(settings)

    metadata = json.loads((settings.data_dir / "data-schema.json").read_text(encoding="utf-8"))
    backup = settings.data_dir / metadata["backup_dir"]
    with sqlite3.connect(backup / "tokenizer" / "studio.db") as connection:
        assert connection.execute("SELECT value FROM fixture").fetchone()[0] == "tokenizer"
    with sqlite3.connect(backup / "training" / "studio.db") as connection:
        assert connection.execute("SELECT value FROM fixture").fetchone()[0] == "training"
    assert status["backup_created"] is True


def test_database_backup_checks_space_against_database_sidecars(monkeypatch, tmp_path: Path) -> None:
    settings = _settings(monkeypatch, tmp_path)
    settings.tokenizer_database_path.write_bytes(b"x")
    with settings.tokenizer_database_path.open("r+b") as database:
        database.truncate(40 * 1024 * 1024)
    wal = Path(f"{settings.tokenizer_database_path}-wal")
    wal.write_bytes(b"w")
    with wal.open("r+b") as sidecar:
        sidecar.truncate(10 * 1024 * 1024)
    captured: dict[str, object] = {}

    def reject_space(path: Path, *, minimum_free_bytes: int, operation: str) -> int:
        captured.update(
            path=path,
            minimum_free_bytes=minimum_free_bytes,
            operation=operation,
        )
        raise InsufficientStorageError("simulated migration capacity failure")

    monkeypatch.setattr(data_migrations, "ensure_free_space", reject_space)

    with pytest.raises(InsufficientStorageError, match="simulated migration capacity failure"):
        data_migrations.prepare_data_schema(settings)

    assert captured == {
        "path": settings.data_dir,
        "minimum_free_bytes": 100 * 1024 * 1024,
        "operation": "database migration backup",
    }
    assert not (settings.data_dir / "backups").exists()
    assert not (settings.data_dir / "data-schema.json").exists()


def test_prepare_data_schema_is_idempotent_and_rejects_corrupt_metadata(
    monkeypatch, tmp_path: Path
) -> None:
    settings = _settings(monkeypatch, tmp_path)
    _sqlite_database(settings.tokenizer_database_path, "tokenizer-db")

    first = data_migrations.prepare_data_schema(settings)
    metadata_path = settings.data_dir / "data-schema.json"
    metadata = metadata_path.read_text(encoding="utf-8")
    second = data_migrations.prepare_data_schema(settings)

    assert first["backup_created"] is True
    assert second["backup_created"] is False
    assert metadata_path.read_text(encoding="utf-8") == metadata

    metadata_path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="metadata is corrupt"):
        data_migrations.prepare_data_schema(settings)


def test_failed_backup_does_not_commit_schema_metadata(monkeypatch, tmp_path: Path) -> None:
    settings = _settings(monkeypatch, tmp_path)
    settings.tokenizer_database_path.write_text("tokenizer-db", encoding="utf-8")
    monkeypatch.setattr(
        data_migrations.shutil,
        "copy2",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("copy failed")),
    )

    with pytest.raises(OSError, match="copy failed"):
        data_migrations.prepare_data_schema(settings)

    assert not (settings.data_dir / "data-schema.json").exists()


def test_legacy_source_tree_import_is_copy_only_and_idempotent(
    monkeypatch, tmp_path: Path
) -> None:
    settings = replace(_settings(monkeypatch, tmp_path), source_root=tmp_path / "source")
    legacy = settings.source_root / "apps" / "llm-studio" / "api" / "artifacts" / "tokenizers"
    legacy.mkdir(parents=True, exist_ok=True)
    source = legacy / "desktop-migration-fixture.json"
    source.write_text('{"fixture": true}', encoding="utf-8")
    first = data_migrations.prepare_data_schema(settings)
    second = data_migrations.prepare_data_schema(settings)
    target = settings.tokenizer_output_dir / source.name

    assert first["legacy_import"]["artifacts"] == 1
    assert second["legacy_import"]["artifacts"] == 0
    assert source.exists()
    assert target.read_text(encoding="utf-8") == '{"fixture": true}'


def test_migration_rejects_backup_symlink_escape(monkeypatch, tmp_path: Path) -> None:
    settings = _settings(monkeypatch, tmp_path)
    settings.tokenizer_database_path.write_text("tokenizer-db", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (settings.data_dir / "backups").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation is unavailable on this platform.")

    with pytest.raises(UnsafeManagedPathError, match="outside"):
        data_migrations.prepare_data_schema(settings)

    assert list(outside.iterdir()) == []
    assert not (settings.data_dir / "data-schema.json").exists()


def test_migration_rejects_legacy_import_symlink_escape(monkeypatch, tmp_path: Path) -> None:
    settings = replace(_settings(monkeypatch, tmp_path), source_root=tmp_path / "source")
    legacy = settings.source_root / "apps" / "llm-studio" / "api" / "artifacts" / "tokenizers"
    legacy.mkdir(parents=True, exist_ok=True)
    (legacy / "fixture.json").write_text("{}", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (settings.tokenizer_output_dir / "fixture.json").symlink_to(outside / "escaped.json")
    except OSError:
        pytest.skip("Symlink creation is unavailable on this platform.")

    with pytest.raises(UnsafeManagedPathError, match="outside"):
        data_migrations.prepare_data_schema(settings)

    assert list(outside.iterdir()) == []
    assert not (settings.data_dir / "legacy-source-tree-import.json").exists()


def test_schema_v2_migrates_only_managed_absolute_paths_and_is_idempotent(
    monkeypatch,
    tmp_path: Path,
) -> None:
    settings = _settings(monkeypatch, tmp_path)
    managed_artifact = settings.tokenizer_output_dir / "tokenizer.json"
    managed_upload = settings.tokenizer_upload_dir / "train.txt"
    managed_job = settings.training_jobs_dir / "job-1"
    external = tmp_path / "external" / "artifact.json"

    with sqlite3.connect(settings.tokenizer_database_path) as connection:
        connection.execute(
            """
            CREATE TABLE training_jobs (
                evaluation_text_path TEXT NOT NULL,
                artifact_path TEXT
            )
            """
        )
        connection.execute(
            "INSERT INTO training_jobs VALUES (?, ?)",
            ("__training_dataset__", str(managed_artifact)),
        )
        connection.execute("CREATE TABLE uploaded_files (file_path TEXT NOT NULL)")
        connection.execute("INSERT INTO uploaded_files VALUES (?)", (str(managed_upload),))
        connection.execute("INSERT INTO uploaded_files VALUES (?)", (str(external),))
    with sqlite3.connect(settings.training_database_path) as connection:
        connection.execute(
            """
            CREATE TABLE llm_training_jobs (
                artifact_dir TEXT NOT NULL,
                stats_path TEXT NOT NULL,
                samples_path TEXT NOT NULL,
                stdout_path TEXT NOT NULL,
                stderr_path TEXT NOT NULL,
                remote_workspace_path TEXT
            )
            """
        )
        connection.execute(
            "INSERT INTO llm_training_jobs VALUES (?, ?, ?, ?, ?, ?)",
            (
                str(managed_job),
                str(managed_job / "stats.jsonl"),
                str(managed_job / "samples.jsonl"),
                str(managed_job / "stdout.log"),
                str(managed_job / "stderr.log"),
                "/workspace/llm-studio/jobs/job-1",
            ),
        )
    (settings.data_dir / "data-schema.json").write_text(
        json.dumps({"schema_version": 1}),
        encoding="utf-8",
    )

    first = data_migrations.prepare_data_schema(settings)
    second = data_migrations.prepare_data_schema(settings)

    with sqlite3.connect(settings.tokenizer_database_path) as connection:
        tokenizer_row = connection.execute(
            "SELECT evaluation_text_path, artifact_path FROM training_jobs"
        ).fetchone()
        upload_rows = [
            row[0] for row in connection.execute("SELECT file_path FROM uploaded_files ORDER BY rowid")
        ]
    with sqlite3.connect(settings.training_database_path) as connection:
        training_row = connection.execute(
            """
            SELECT artifact_dir, stats_path, samples_path, stdout_path, stderr_path,
                   remote_workspace_path
            FROM llm_training_jobs
            """
        ).fetchone()
    metadata = json.loads((settings.data_dir / "data-schema.json").read_text(encoding="utf-8"))
    backup = settings.data_dir / metadata["backup_dir"]
    with sqlite3.connect(backup / "tokenizer" / settings.tokenizer_database_path.name) as connection:
        backed_up_artifact = connection.execute(
            "SELECT artifact_path FROM training_jobs"
        ).fetchone()[0]

    assert first["managed_paths_migrated"] == 7
    assert second["managed_paths_migrated"] == 0
    assert tokenizer_row == (
        "__training_dataset__",
        "llm-studio-data:v1/artifacts/tokenizers/tokenizer.json",
    )
    assert upload_rows == [
        "llm-studio-data:v1/uploads/train.txt",
        str(external),
    ]
    assert training_row == (
        "llm-studio-data:v1/training/jobs/job-1",
        "llm-studio-data:v1/training/jobs/job-1/stats.jsonl",
        "llm-studio-data:v1/training/jobs/job-1/samples.jsonl",
        "llm-studio-data:v1/training/jobs/job-1/stdout.log",
        "llm-studio-data:v1/training/jobs/job-1/stderr.log",
        "/workspace/llm-studio/jobs/job-1",
    )
    assert backed_up_artifact == str(managed_artifact)


def test_schema_v3_removes_legacy_credentials_from_managed_state(
    monkeypatch,
    tmp_path: Path,
) -> None:
    settings = _settings(monkeypatch, tmp_path)
    hf_token = "hf_0123456789abcdef0123456789abcdef"
    runpod_token = "rpa_0123456789abcdef0123456789abcdef"
    dataloader = json.dumps(
        {"datasets": [{"name": "private", "hf_token": hf_token}]}
    )
    with sqlite3.connect(settings.tokenizer_database_path) as connection:
        connection.execute(
            "CREATE TABLE training_jobs (dataloader_config JSON NOT NULL, error TEXT)"
        )
        connection.execute(
            "INSERT INTO training_jobs VALUES (?, ?)",
            (dataloader, f"provider echoed {runpod_token}"),
        )
    with sqlite3.connect(settings.training_database_path) as connection:
        connection.execute(
            """
            CREATE TABLE llm_training_jobs (
                dataloader_config JSON NOT NULL,
                error TEXT,
                remote_error TEXT
            )
            """
        )
        connection.execute(
            "INSERT INTO llm_training_jobs VALUES (?, ?, ?)",
            (dataloader, f"failed with {hf_token}", f"provider echoed {runpod_token}"),
        )
    legacy_tokenizer = settings.data_dir / "db" / "tokenizer_studio.db"
    legacy_training = settings.data_dir / "db" / "training_studio.db"
    with sqlite3.connect(legacy_tokenizer) as connection:
        connection.execute(
            "CREATE TABLE training_jobs (dataloader_config JSON NOT NULL, error TEXT)"
        )
        connection.execute(
            "INSERT INTO training_jobs VALUES (?, ?)",
            (dataloader, f"provider echoed {runpod_token}"),
        )
    with sqlite3.connect(legacy_training) as connection:
        connection.execute(
            """
            CREATE TABLE llm_training_jobs (
                dataloader_config JSON NOT NULL,
                error TEXT,
                remote_error TEXT
            )
            """
        )
        connection.execute(
            "INSERT INTO llm_training_jobs VALUES (?, ?, ?)",
            (dataloader, f"failed with {hf_token}", f"provider echoed {runpod_token}"),
        )

    job_dir = settings.training_jobs_dir / "job-1"
    job_dir.mkdir(parents=True)
    (job_dir / "dataloader_config.json").write_text(dataloader, encoding="utf-8")
    (job_dir / "resolved_preflight.json").write_text(
        json.dumps({"normalized": json.loads(dataloader)}),
        encoding="utf-8",
    )
    (job_dir / "stdout.log").write_text(f"provider echoed {runpod_token}", encoding="utf-8")
    (job_dir / "bundle.tar.gz").write_bytes(hf_token.encode())
    staging = job_dir / ".remote_bundle"
    staging.mkdir()
    (staging / "secret.txt").write_text(hf_token, encoding="utf-8")
    (settings.data_dir / "data-schema.json").write_text(
        json.dumps({"schema_version": 2}),
        encoding="utf-8",
    )

    first = data_migrations.prepare_data_schema(settings)
    second = data_migrations.prepare_data_schema(settings)

    assert first["credential_cleanup"] == {
        "database_fields": 5,
        "legacy_database_fields": 5,
        "backup_database_fields": 5,
        "job_files": 3,
        "transient_bundles_removed": 2,
    }
    assert second["credential_cleanup"] == {
        "database_fields": 0,
        "legacy_database_fields": 0,
        "backup_database_fields": 0,
        "job_files": 0,
        "transient_bundles_removed": 0,
    }
    assert hf_token not in (job_dir / "dataloader_config.json").read_text(encoding="utf-8")
    assert hf_token not in (job_dir / "resolved_preflight.json").read_text(encoding="utf-8")
    assert runpod_token not in (job_dir / "stdout.log").read_text(encoding="utf-8")
    assert not (job_dir / "bundle.tar.gz").exists()
    assert not staging.exists()
    for database_path in (
        settings.tokenizer_database_path,
        settings.training_database_path,
        legacy_tokenizer,
        legacy_training,
    ):
        assert hf_token.encode() not in database_path.read_bytes()
        assert runpod_token.encode() not in database_path.read_bytes()
    metadata = json.loads((settings.data_dir / "data-schema.json").read_text(encoding="utf-8"))
    backup = settings.data_dir / metadata["backup_dir"]
    for database_path in (
        backup / "tokenizer" / settings.tokenizer_database_path.name,
        backup / "training" / settings.training_database_path.name,
    ):
        assert hf_token.encode() not in database_path.read_bytes()
        assert runpod_token.encode() not in database_path.read_bytes()


def test_credential_cleanup_removes_sqlite_sidecars_and_rejects_symlinks(
    tmp_path: Path,
) -> None:
    database = tmp_path / "training.db"
    _sqlite_database(database)
    wal = Path(f"{database}-wal")
    shm = Path(f"{database}-shm")
    wal.write_text("hf_0123456789abcdef0123456789abcdef", encoding="utf-8")
    shm.write_text("rpa_0123456789abcdef0123456789abcdef", encoding="utf-8")

    data_migrations._remove_sqlite_sidecars(database)

    assert not wal.exists()
    assert not shm.exists()

    outside = tmp_path / "outside"
    outside.write_text("keep", encoding="utf-8")
    try:
        wal.symlink_to(outside)
    except OSError:
        pytest.skip("Symlink creation is unavailable on this platform.")
    with pytest.raises(UnsafeManagedPathError, match="symlinked SQLite sidecar"):
        data_migrations._remove_sqlite_sidecars(database)
    assert outside.read_text(encoding="utf-8") == "keep"


def test_golden_pre_desktop_schema_one_fixture_migrates_and_imports(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source-tree"
    settings = replace(_settings(monkeypatch, tmp_path), source_root=source_root)
    fixture = _expand_fixture(
        json.loads(
            (MIGRATION_FIXTURES / "pre-desktop-schema-1.json").read_text(encoding="utf-8")
        ),
        {
            "{DATA_ROOT}": settings.data_dir.as_posix(),
            "{EXTERNAL_ROOT}": (tmp_path / "external").as_posix(),
        },
    )
    assert fixture["fixture_schema_version"] == 1
    assert fixture["data_schema_version"] == 1

    source_api = source_root / "apps" / "llm-studio" / "api"
    for kind, relative in (
        ("artifacts", Path("artifacts/tokenizers")),
        ("uploads", Path("datasets/uploads")),
    ):
        directory = source_api / relative
        directory.mkdir(parents=True, exist_ok=True)
        for item in fixture["legacy_source_tree"][kind]:
            (directory / item["name"]).write_text(item["content"], encoding="utf-8")

    tokenizer_fixture = fixture["tokenizer_database"]
    with sqlite3.connect(settings.tokenizer_database_path) as connection:
        connection.execute(
            """
            CREATE TABLE training_jobs (
                evaluation_text_path TEXT NOT NULL,
                artifact_path TEXT
            )
            """
        )
        connection.executemany(
            "INSERT INTO training_jobs VALUES (:evaluation_text_path, :artifact_path)",
            tokenizer_fixture["training_jobs"],
        )
        connection.execute("CREATE TABLE uploaded_files (file_path TEXT NOT NULL)")
        connection.executemany(
            "INSERT INTO uploaded_files VALUES (:file_path)",
            tokenizer_fixture["uploaded_files"],
        )

    training_fixture = fixture["training_database"]
    with sqlite3.connect(settings.training_database_path) as connection:
        connection.execute(
            """
            CREATE TABLE llm_training_jobs (
                artifact_dir TEXT NOT NULL,
                stats_path TEXT NOT NULL,
                samples_path TEXT NOT NULL,
                stdout_path TEXT NOT NULL,
                stderr_path TEXT NOT NULL,
                remote_workspace_path TEXT
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO llm_training_jobs VALUES (
                :artifact_dir, :stats_path, :samples_path, :stdout_path,
                :stderr_path, :remote_workspace_path
            )
            """,
            training_fixture["llm_training_jobs"],
        )
    (settings.data_dir / "data-schema.json").write_text(
        json.dumps({"schema_version": fixture["data_schema_version"]}),
        encoding="utf-8",
    )

    first = data_migrations.prepare_data_schema(settings)
    second = data_migrations.prepare_data_schema(settings)

    with sqlite3.connect(settings.tokenizer_database_path) as connection:
        tokenizer_training_job = connection.execute(
            "SELECT evaluation_text_path, artifact_path FROM training_jobs"
        ).fetchone()
        uploaded_files = [
            row[0] for row in connection.execute("SELECT file_path FROM uploaded_files ORDER BY rowid")
        ]
    with sqlite3.connect(settings.training_database_path) as connection:
        training_job = connection.execute(
            """
            SELECT artifact_dir, stats_path, samples_path, stdout_path, stderr_path,
                   remote_workspace_path
            FROM llm_training_jobs
            """
        ).fetchone()

    expected = fixture["expected"]
    assert first["managed_paths_migrated"] == expected["managed_paths_migrated"]
    assert first["legacy_import"] == expected["legacy_import"]
    assert second["managed_paths_migrated"] == 0
    assert second["legacy_import"] == {"artifacts": 0, "uploads": 0}
    assert list(tokenizer_training_job) == expected["tokenizer_training_job"]
    assert uploaded_files == expected["uploaded_files"]
    assert list(training_job) == expected["training_job"]
    for kind, directory in (
        ("artifacts", settings.tokenizer_output_dir),
        ("uploads", settings.tokenizer_upload_dir),
    ):
        for item in fixture["legacy_source_tree"][kind]:
            assert (directory / item["name"]).read_text(encoding="utf-8") == item["content"]


def test_managed_path_database_migration_rolls_back_on_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    database = data_root / "db" / "tokenizer.db"
    database.parent.mkdir(parents=True)
    paths = [
        data_root / "artifacts" / "one.json",
        data_root / "artifacts" / "two.json",
    ]
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE uploaded_files (file_path TEXT NOT NULL)")
        connection.executemany(
            "INSERT INTO uploaded_files VALUES (?)",
            [(str(path),) for path in paths],
        )

    original_encode = data_migrations.encode_managed_location

    def fail_second(value: str, root: Path) -> str:
        if value.endswith("two.json"):
            raise RuntimeError("simulated migration failure")
        return original_encode(value, root)

    monkeypatch.setattr(data_migrations, "encode_managed_location", fail_second)

    with pytest.raises(RuntimeError, match="simulated migration failure"):
        data_migrations._migrate_sqlite_managed_paths(
            database,
            data_root,
            {"uploaded_files": ("file_path",)},
        )

    with sqlite3.connect(database) as connection:
        stored = [
            row[0] for row in connection.execute("SELECT file_path FROM uploaded_files ORDER BY rowid")
        ]
    assert stored == [str(path) for path in paths]
