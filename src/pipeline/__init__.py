"""Formula Student EV Season 2025/2026 - Data Ingestion & Pipeline Package.

Module: src.pipeline
Exports cataloging engine, vehicle telemetry normalizer, BMS parser, and IMU processor.
"""

from src.pipeline.bms import (
    FsmTransition,
    extract_fsm_transitions,
    filter_c1_artifact,
    load_ams_log,
    load_ams_session,
    resolve_bms_path,
)
from src.pipeline.catalog import (
    CatalogEntry,
    build_season_catalog,
    compute_file_hash,
    detect_schema_id,
    export_catalog_artifacts,
    extract_provenance_tags,
    extract_session_metadata,
    get_active_sessions,
    get_stubs,
)
from src.pipeline.imu import (
    ShockEvent,
    correlate_imu_bms,
    detect_mechanical_shocks,
    load_imu_log,
    resample_bms_to_imu,
    resolve_imu_path,
)
from src.pipeline.telemetry import (
    CANONICAL_COLUMNS,
    compute_calibrated_speed,
    detect_telemetry_schema,
    extract_standardized_time,
    load_session,
    load_telemetry_file,
    normalize_dcdc_current,
    normalize_pedals,
    normalize_telemetry_dataframe,
    resolve_telemetry_path,
)

__all__ = [
    # Catalog
    "CatalogEntry",
    "build_season_catalog",
    "export_catalog_artifacts",
    "get_active_sessions",
    "get_stubs",
    "compute_file_hash",
    "detect_schema_id",
    "extract_session_metadata",
    "extract_provenance_tags",
    # Telemetry
    "load_session",
    "load_telemetry_file",
    "detect_telemetry_schema",
    "compute_calibrated_speed",
    "normalize_dcdc_current",
    "extract_standardized_time",
    "normalize_pedals",
    "normalize_telemetry_dataframe",
    "resolve_telemetry_path",
    "CANONICAL_COLUMNS",
    # BMS
    "load_ams_log",
    "load_ams_session",
    "extract_fsm_transitions",
    "filter_c1_artifact",
    "resolve_bms_path",
    "FsmTransition",
    # IMU
    "load_imu_log",
    "detect_mechanical_shocks",
    "correlate_imu_bms",
    "resample_bms_to_imu",
    "resolve_imu_path",
    "ShockEvent",
]
