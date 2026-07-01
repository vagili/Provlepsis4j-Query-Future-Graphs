from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .lp_model_state import normalize_prediction_model
from .prediction_timeline import normalize_embedding_family
from .storage_repr import SEPARATE_DBS, normalize_storage_representation


def _safe_name(value: Any) -> str:
    raw = str(value or "default")
    return "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_"
        for ch in raw
    )


def _config_path(base_db: str) -> Path:
    return (
        Path("provlepsis_runtime")
        / _safe_name(base_db)
        / "admin_extension_config.json"
    )


DEFAULT_ADMIN_EXTENSION_CONFIG = {
    "storageRepresentation": SEPARATE_DBS,
    "predictionModel": "logistic_regression",
    "embeddingFamily": "Node2Vec",
    "embeddingProperty": "Node2Vec",
    "predictK": 100,
    "candidateMultiplier": 20,
    "probThreshold": 0.50,
    "negativeRatio": 1.0,
    "predictedEdgeInclusion": True,
    "retrainingInterval": 1,
    "additionalTimesteps": 1,
    "tgnDirectParams": {},
}


def load_admin_extension_config(base_db: str) -> Dict[str, Any]:
    config = dict(DEFAULT_ADMIN_EXTENSION_CONFIG)
    path = _config_path(base_db)

    if path.exists():
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(stored, dict):
                config.update(stored)
        except Exception:
            pass

    config["storageRepresentation"] = normalize_storage_representation(
        config.get("storageRepresentation")
    )
    config["predictionModel"] = normalize_prediction_model(
        config.get("predictionModel")
    )

    if config["predictionModel"] == "tgn_direct":
        config["embeddingFamily"] = "TGN"
        config["embeddingProperty"] = "TGN"
    else:
        config["embeddingFamily"] = normalize_embedding_family(
            config.get("embeddingFamily")
        )

    config["predictK"] = max(1, int(config.get("predictK", 100)))
    config["additionalTimesteps"] = max(
        1,
        int(config.get("additionalTimesteps", 1)),
    )
    config["candidateMultiplier"] = max(
        2,
        int(config.get("candidateMultiplier", 20)),
    )
    config["probThreshold"] = float(config.get("probThreshold", 0.50))
    config["negativeRatio"] = float(config.get("negativeRatio", 1.0))
    include_predictions = bool(config.get("predictedEdgeInclusion", True))
    config["predictedEdgeInclusion"] = include_predictions
    config["retrainingInterval"] = 1 if include_predictions else 0
    params = config.get("tgnDirectParams")
    config["tgnDirectParams"] = dict(params) if isinstance(params, dict) else {}
    return config


def save_admin_extension_config(
    base_db: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    merged = dict(DEFAULT_ADMIN_EXTENSION_CONFIG)
    merged.update(config)
    merged["storageRepresentation"] = normalize_storage_representation(
        merged.get("storageRepresentation")
    )
    merged["predictionModel"] = normalize_prediction_model(
        merged.get("predictionModel")
    )

    if merged["predictionModel"] == "tgn_direct":
        merged["embeddingFamily"] = "TGN"
        merged["embeddingProperty"] = "TGN"
    else:
        merged["embeddingFamily"] = normalize_embedding_family(
            merged.get("embeddingFamily")
        )

    merged["predictK"] = max(1, int(merged.get("predictK", 100)))
    merged["candidateMultiplier"] = max(
        2,
        int(merged.get("candidateMultiplier", 20)),
    )
    merged["additionalTimesteps"] = max(
        1,
        int(merged.get("additionalTimesteps", 1)),
    )
    merged["probThreshold"] = float(merged.get("probThreshold", 0.50))
    merged["negativeRatio"] = float(merged.get("negativeRatio", 1.0))
    include_predictions = bool(merged.get("predictedEdgeInclusion", True))
    merged["predictedEdgeInclusion"] = include_predictions
    merged["retrainingInterval"] = 1 if include_predictions else 0
    params = merged.get("tgnDirectParams")
    merged["tgnDirectParams"] = dict(params) if isinstance(params, dict) else {}

    path = _config_path(base_db)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(merged, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(path)
    return merged
