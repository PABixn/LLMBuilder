from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

# Documented SQLite migration registry.
#
# Keys are column names that must exist on `llm_training_jobs`; values are the
# idempotent ALTER statements used when opening older local databases. Keep this
# append-only so future migrations can be audited without reverse-engineering
# historical table shapes.
SQLITE_SCHEMA_MIGRATIONS: dict[str, str] = {
    "executor_kind": "ALTER TABLE llm_training_jobs ADD COLUMN executor_kind VARCHAR(32) NOT NULL DEFAULT 'local'",
    "executor_status": "ALTER TABLE llm_training_jobs ADD COLUMN executor_status VARCHAR(64)",
    "runpod_pod_id": "ALTER TABLE llm_training_jobs ADD COLUMN runpod_pod_id VARCHAR(128)",
    "runpod_pod_name": "ALTER TABLE llm_training_jobs ADD COLUMN runpod_pod_name VARCHAR(255)",
    "runpod_network_volume_id": "ALTER TABLE llm_training_jobs ADD COLUMN runpod_network_volume_id VARCHAR(128)",
    "runpod_data_center_id": "ALTER TABLE llm_training_jobs ADD COLUMN runpod_data_center_id VARCHAR(128)",
    "runpod_gpu_type_id": "ALTER TABLE llm_training_jobs ADD COLUMN runpod_gpu_type_id VARCHAR(255)",
    "runpod_gpu_count": "ALTER TABLE llm_training_jobs ADD COLUMN runpod_gpu_count INTEGER NOT NULL DEFAULT 1",
    "runpod_cloud_type": "ALTER TABLE llm_training_jobs ADD COLUMN runpod_cloud_type VARCHAR(32)",
    "runpod_interruptible": "ALTER TABLE llm_training_jobs ADD COLUMN runpod_interruptible BOOLEAN NOT NULL DEFAULT 0",
    "runpod_cost_per_hr": "ALTER TABLE llm_training_jobs ADD COLUMN runpod_cost_per_hr FLOAT",
    "runpod_public_ip": "ALTER TABLE llm_training_jobs ADD COLUMN runpod_public_ip VARCHAR(255)",
    "runpod_port_mappings": "ALTER TABLE llm_training_jobs ADD COLUMN runpod_port_mappings JSON",
    "runpod_agent_base_url": "ALTER TABLE llm_training_jobs ADD COLUMN runpod_agent_base_url TEXT",
    "runpod_agent_token_hash": "ALTER TABLE llm_training_jobs ADD COLUMN runpod_agent_token_hash VARCHAR(128)",
    "runpod_last_heartbeat_at": "ALTER TABLE llm_training_jobs ADD COLUMN runpod_last_heartbeat_at DATETIME",
    "runpod_last_sync_at": "ALTER TABLE llm_training_jobs ADD COLUMN runpod_last_sync_at DATETIME",
    "runpod_cleanup_policy": "ALTER TABLE llm_training_jobs ADD COLUMN runpod_cleanup_policy JSON",
    "remote_workspace_path": "ALTER TABLE llm_training_jobs ADD COLUMN remote_workspace_path TEXT",
    "remote_error": "ALTER TABLE llm_training_jobs ADD COLUMN remote_error TEXT",
}


def apply_sqlite_migrations(engine: Engine) -> None:
    existing = {column["name"] for column in inspect(engine).get_columns("llm_training_jobs")}
    with engine.begin() as connection:
        for column_name, statement in SQLITE_SCHEMA_MIGRATIONS.items():
            if column_name not in existing:
                connection.execute(text(statement))
