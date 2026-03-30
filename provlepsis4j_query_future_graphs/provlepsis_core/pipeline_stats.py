from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


CSV_FILENAME = "provlepsis4j_pipeline_stats.csv"
SPLIT_STATE_FILENAME = ".provlepsis4j_last_split.json"


def _run_dir() -> Path:
    return Path.cwd()


def csv_output_path() -> Path:
    return _run_dir() / CSV_FILENAME


def _split_state_path() -> Path:
    return _run_dir() / SPLIT_STATE_FILENAME


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _num(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return round(float(v), 6)
    except Exception:
        return None


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def remember_split_run(
    *,
    base_db: str,
    split_seconds: float,
    temporal_mode: bool,
    test_holdout: float,
    val_holdout: float,
) -> None:
    data = _read_json(_split_state_path())
    data[base_db] = {
        "recorded_at_utc": _utc_now(),
        "split_seconds": _num(split_seconds),
        "temporal_mode": bool(temporal_mode),
        "test_holdout": _num(test_holdout),
        "val_holdout": _num(val_holdout),
    }
    _write_json(_split_state_path(), data)


def get_last_split_run(base_db: str) -> Dict[str, Any]:
    data = _read_json(_split_state_path())
    out = data.get(base_db, {})
    return out if isinstance(out, dict) else {}


CSV_COLUMNS = [
    "timestamp_utc",
    "base_db",
    "embedding_family",
    "embedding_property",
    "predict_k",
    "prob_threshold",
    "predicted_db_name",
    "split_recorded_at_utc",
    "split_seconds",
    "split_temporal_mode",
    "split_test_holdout",
    "split_val_holdout",
    "embedding_seconds",
    "link_prediction_seconds",
    "predicted_db_creation_seconds",
    "predicted_graph_projection_seconds",
    "materialize_prediction_db_seconds",
    "test_auc",
    "test_ap",
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_f1",
]


def append_pipeline_row(
    *,
    base_db: str,
    embedding_family: str,
    embedding_property: str,
    predict_k: int,
    prob_threshold: float,
    split_info: Optional[Dict[str, Any]],
    embedding_seconds: float,
    lp_timings: Optional[Dict[str, Any]],
    test_metrics: Optional[Dict[str, Any]],
    predicted_db_name: Optional[str],
) -> Path:
    split_info = split_info or {}
    lp_timings = lp_timings or {}
    test_metrics = test_metrics or {}

    row = {
        "timestamp_utc": _utc_now(),
        "base_db": base_db,
        "embedding_family": embedding_family,
        "embedding_property": embedding_property,
        "predict_k": int(predict_k),
        "prob_threshold": _num(prob_threshold),
        "predicted_db_name": predicted_db_name or "",
        "split_recorded_at_utc": split_info.get("recorded_at_utc", ""),
        "split_seconds": _num(split_info.get("split_seconds")),
        "split_temporal_mode": split_info.get("temporal_mode", ""),
        "split_test_holdout": _num(split_info.get("test_holdout")),
        "split_val_holdout": _num(split_info.get("val_holdout")),
        "embedding_seconds": _num(embedding_seconds),
        "link_prediction_seconds": _num(lp_timings.get("link_prediction_seconds")),
        "predicted_db_creation_seconds": _num(lp_timings.get("predicted_db_creation_seconds")),
        "predicted_graph_projection_seconds": _num(lp_timings.get("predicted_graph_projection_seconds")),
        "materialize_prediction_db_seconds": _num(lp_timings.get("materialize_prediction_db_seconds")),
        "test_auc": _num(test_metrics.get("auc")),
        "test_ap": _num(test_metrics.get("ap")),
        "test_accuracy": _num(test_metrics.get("accuracy")),
        "test_precision": _num(test_metrics.get("precision")),
        "test_recall": _num(test_metrics.get("recall")),
        "test_f1": _num(test_metrics.get("f1")),
    }

    path = csv_output_path()
    write_header = not path.exists()

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})

    return path