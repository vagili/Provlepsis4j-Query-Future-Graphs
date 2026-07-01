from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional

def normalize_prediction_model(value: Any) -> str:
    raw = str(value or "logistic_regression").strip().lower()
    raw = raw.replace("-", "_").replace(" ", "_")

    if raw in {"logistic", "logistic_regression", "logisticregression", "lr"}:
        return "logistic_regression"

    if raw in {"lasso", "lasso_regression", "lasso_logistic", "l1", "l1_logistic"}:
        return "lasso_regression"

    if raw in {"tgn", "tgn_direct", "tgn_model", "tgn_link_prediction", "temporal_graph_network"}:
        return "tgn_direct"

    return "logistic_regression"


def prediction_model_display_name(value: Any) -> str:
    model = normalize_prediction_model(value)
    if model == "lasso_regression":
        return "LASSO regression"
    if model == "tgn_direct":
        return "TGN"
    return "Logistic regression"


def prediction_model_algorithm_name(value: Any) -> str:
    model = normalize_prediction_model(value)
    if model == "lasso_regression":
        return "L1LogisticRegression"
    if model == "tgn_direct":
        return "TGN"
    return "LogisticRegression"

def _safe_name(value: Any) -> str:
    s = str(value or "default")
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)


def _state_dir(
    base_db: str,
    storage_representation: str,
    embedding_family: str,
    prediction_model: str = "logistic_regression",
) -> Path:
    return (
        Path("provlepsis_model_states")
        / _safe_name(base_db)
        / _safe_name(storage_representation)
        / _safe_name(embedding_family)
        / _safe_name(normalize_prediction_model(prediction_model))
    )


def _state_path(
    *,
    base_db: str,
    storage_representation: str,
    embedding_family: str,
    state_name: str,
    prediction_model: str = "logistic_regression",
) -> Path:
    return (
        _state_dir(base_db, storage_representation, embedding_family, prediction_model)
        / f"{_safe_name(state_name)}.pkl"
    )


def save_lp_learning_state(
    *,
    base_db: str,
    storage_representation: str,
    embedding_family: str,
    embedding_property: str,
    model: Any,
    trained_at_timestep: int,
    prediction_model: str = "logistic_regression",
    state_name: str = "latest",
    embeddings: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    prediction_model = normalize_prediction_model(prediction_model)
    path = _state_path(
        base_db=base_db,
        storage_representation=storage_representation,
        embedding_family=embedding_family,
        prediction_model=prediction_model,
        state_name=state_name,
    )

    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "baseDb": base_db,
        "storageRepresentation": storage_representation,
        "embeddingFamily": embedding_family,
        "embeddingProperty": embedding_property,
        "trainedAtTimestep": int(trained_at_timestep),
        "stateName": state_name,
        "metadata": metadata or {},
        "model": model,
        "embeddings": embeddings or {},
        "predictionModel": prediction_model,
        "predictionModelDisplayName": prediction_model_display_name(prediction_model),
        "predictionModelAlgorithm": prediction_model_algorithm_name(prediction_model),
    }

    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump(payload, f)

    tmp.replace(path)

    return {
        "ok": True,
        "path": str(path),
        "stateName": state_name,
        "embeddingFamily": embedding_family,
        "embeddingProperty": embedding_property,
        "trainedAtTimestep": int(trained_at_timestep),
        "predictionModel": prediction_model,
        "predictionModelDisplayName": prediction_model_display_name(prediction_model),
    }


def load_lp_learning_state(
    *,
    base_db: str,
    storage_representation: str,
    embedding_family: str,
    prediction_model: str = "logistic_regression",
    state_name: str = "latest",
) -> Optional[Dict[str, Any]]:
    path = _state_path(
        base_db=base_db,
        storage_representation=storage_representation,
        embedding_family=embedding_family,
        prediction_model=prediction_model,
        state_name=state_name,
    )

    if not path.exists():
        return None

    with path.open("rb") as f:
        return pickle.load(f)


def delete_lp_learning_state(
    *,
    base_db: str,
    storage_representation: str,
    embedding_family: str,
    prediction_model: str = "logistic_regression",
    state_name: str = "latest",
) -> bool:
    path = _state_path(
        base_db=base_db,
        storage_representation=storage_representation,
        embedding_family=embedding_family,
        prediction_model=prediction_model,
        state_name=state_name,
    )

    if path.exists():
        path.unlink()
        return True

    return False