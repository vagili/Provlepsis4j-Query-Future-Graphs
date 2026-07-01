# provlepsis_core/prediction_artifacts.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .lp_model_state import normalize_prediction_model


def _safe_name(value: Any) -> str:
    s = str(value or "default")
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)


def _base_dir(base_db: str, storage_representation: str) -> Path:
    return (
        Path("provlepsis_runtime")
        / _safe_name(base_db)
        / _safe_name(storage_representation)
    )


def _candidate_dir(base_db: str, storage_representation: str) -> Path:
    return _base_dir(base_db, storage_representation) / "candidate_pools"


def _evaluation_dir(base_db: str, storage_representation: str) -> Path:
    return _base_dir(base_db, storage_representation) / "evaluation"


def _pool_path(base_db: str, storage_representation: str, timestep: int) -> Path:
    return _candidate_dir(base_db, storage_representation) / f"candidate_pool_t{int(timestep)}.json"


def _evaluation_path(base_db: str, storage_representation: str, prediction_model: str) -> Path:
    model = normalize_prediction_model(prediction_model)
    return _evaluation_dir(base_db, storage_representation) / f"{model}.json"


def pair_key(s: Any, t: Any) -> str:
    a = str(s)
    b = str(t)
    return f"{a}|{b}" if a < b else f"{b}|{a}"


def normalize_candidate(row: Any) -> Dict[str, Any]:
    if isinstance(row, dict):
        s = row.get("s")
        t = row.get("t")
    else:
        s, t = row[0], row[1]

    key = pair_key(s, t)

    return {
        "s": str(s),
        "t": str(t),
        "pairKey": key,
    }


def load_candidate_pool(
    *,
    base_db: str,
    storage_representation: str,
    timestep: int,
) -> Dict[str, Any]:
    path = _pool_path(base_db, storage_representation, timestep)

    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_candidate_pool(
    *,
    base_db: str,
    storage_representation: str,
    timestep: int,
    payload: Dict[str, Any],
) -> None:
    path = _pool_path(base_db, storage_representation, timestep)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def get_or_create_candidate_pool(
    *,
    base_db: str,
    storage_representation: str,
    timestep: int,
    generated_candidates: Iterable[Any],
    pool_size: int,
    force_new: bool = False,
    **_ignored: Any,
) -> List[Dict[str, Any]]:
    existing_payload: Dict[str, Any] = {}

    if not force_new:
        existing_payload = load_candidate_pool(
            base_db=base_db,
            storage_representation=storage_representation,
            timestep=timestep,
        )

    existing_candidates = list(
        existing_payload.get("candidates") or []
    )

    seen = {
        str(row.get("pairKey"))
        for row in existing_candidates
        if row.get("pairKey")
    }

    candidates = list(existing_candidates)

    for raw in generated_candidates:
        if len(candidates) >= int(pool_size):
            break

        candidate = normalize_candidate(raw)
        key = candidate["pairKey"]

        if key in seen:
            continue

        seen.add(key)
        candidates.append(candidate)

    payload = {
        "baseDb": base_db,
        "storageRepresentation": storage_representation,
        "timestep": int(timestep),
        "poolSize": len(candidates),
        "candidates": candidates,
        "scores": existing_payload.get("scores") or {},
    }

    save_candidate_pool(
        base_db=base_db,
        storage_representation=storage_representation,
        timestep=timestep,
        payload=payload,
    )

    return candidates


def record_candidate_scores(
    *,
    base_db: str,
    storage_representation: str,
    timestep: int,
    prediction_model: str,
    embedding_family: str,
    scored_candidates: Iterable[Dict[str, Any]],
    **_ignored: Any,
) -> None:

    model_key = normalize_prediction_model(prediction_model)

    payload = load_candidate_pool(
        base_db=base_db,
        storage_representation=storage_representation,
        timestep=timestep,
    )

    if not payload:
        payload = {
            "baseDb": base_db,
            "storageRepresentation": storage_representation,
            "timestep": int(timestep),
            "candidates": [],
            "scores": {},
        }

    payload.setdefault("scores", {})
    payload["scores"].setdefault(model_key, {})
    payload["scores"][model_key].setdefault(embedding_family, {})

    fam_scores = {}

    for row in scored_candidates:
        key = row.get("pairKey") or pair_key(row.get("s"), row.get("t"))
        fam_scores[key] = {
            "probability": float(row.get("probability", 0.0) or 0.0),
            "rank": int(row.get("rank", 0) or 0),
            "selected": bool(row.get("selected", False)),
        }

    payload["scores"][model_key][embedding_family] = fam_scores

    save_candidate_pool(
        base_db=base_db,
        storage_representation=storage_representation,
        timestep=timestep,
        payload=payload,
    )


def load_evaluation_artifacts(
    *,
    base_db: str,
    storage_representation: str,
    prediction_model: str,
) -> Dict[str, Any]:
    path = _evaluation_path(base_db, storage_representation, prediction_model)

    if not path.exists():
        return {
            "baseDb": base_db,
            "storageRepresentation": storage_representation,
            "predictionModel": normalize_prediction_model(prediction_model),
            "families": {},
            "groundTruth": [],
        }

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "baseDb": base_db,
            "storageRepresentation": storage_representation,
            "predictionModel": normalize_prediction_model(prediction_model),
            "families": {},
            "groundTruth": [],
        }


def save_evaluation_artifacts(
    *,
    base_db: str,
    storage_representation: str,
    prediction_model: str,
    payload: Dict[str, Any],
) -> None:
    path = _evaluation_path(base_db, storage_representation, prediction_model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def update_family_evaluation_artifact(
    *,
    base_db: str,
    storage_representation: str,
    prediction_model: str,
    embedding_family: str,
    metrics: Dict[str, Any],
    eval_pairs: Iterable[Tuple[Any, Any]],
    materialized_pairs: Iterable[Tuple[Any, Any]],
    ground_truth_pairs: Iterable[Tuple[Any, Any]],
) -> Dict[str, Any]:

    payload = load_evaluation_artifacts(
        base_db=base_db,
        storage_representation=storage_representation,
        prediction_model=prediction_model,
    )

    payload["baseDb"] = base_db
    payload["storageRepresentation"] = storage_representation
    payload["predictionModel"] = normalize_prediction_model(prediction_model)
    payload.setdefault("families", {})

    payload["families"][embedding_family] = {
        "metrics": metrics,
        "evalPairs": [
            {"s": str(s), "t": str(t), "pairKey": pair_key(s, t)}
            for s, t in eval_pairs
        ],
        "materializedPairs": [
            {"s": str(s), "t": str(t), "pairKey": pair_key(s, t)}
            for s, t in materialized_pairs
        ],
    }

    payload["groundTruth"] = [
        {"s": str(s), "t": str(t), "pairKey": pair_key(s, t)}
        for s, t in ground_truth_pairs
    ]

    save_evaluation_artifacts(
        base_db=base_db,
        storage_representation=storage_representation,
        prediction_model=prediction_model,
        payload=payload,
    )

    return payload