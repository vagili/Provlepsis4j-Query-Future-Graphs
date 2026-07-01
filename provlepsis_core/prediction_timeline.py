from __future__ import annotations

from typing import Any, List, Tuple

from .lp_model_state import (
    normalize_prediction_model,
    prediction_model_display_name,
)


EMBEDDING_FAMILIES = (
    "Node2Vec",
    "FastRP",
    "GraphSAGE",
    "HashGNN",
    "TGN",
)

LINEAR_PREDICTION_MODELS = (
    "logistic_regression",
    "lasso_regression",
)

FAMILY_SLUGS = {
    "Node2Vec": "node2vec",
    "FastRP": "fastrp",
    "GraphSAGE": "graphsage",
    "HashGNN": "hashgnn",
    "TGN": "tgn",
}

MODEL_SLUGS = {
    "logistic_regression": "logistic",
    "lasso_regression": "lasso",
    "tgn_direct": "tgn",
}


def normalize_embedding_family(value: Any) -> str:
    raw = "".join(
        ch for ch in str(value or "").lower()
        if ch.isalnum()
    )

    aliases = {
        "node2vec": "Node2Vec",
        "n2v": "Node2Vec",
        "fastrp": "FastRP",
        "graphsage": "GraphSAGE",
        "sage": "GraphSAGE",
        "hashgnn": "HashGNN",
        "tgn": "TGN",
    }

    if raw not in aliases:
        raise ValueError(f"Unsupported embedding family: {value}")

    return aliases[raw]


def validate_timeline(
    embedding_family: Any,
    prediction_model: Any,
) -> Tuple[str, str]:
    family = normalize_embedding_family(embedding_family)
    model = normalize_prediction_model(prediction_model)

    if model == "tgn_direct" and family != "TGN":
        raise ValueError(
            "The direct TGN prediction model can only be used "
            "with the TGN embedding family."
        )

    return family, model


def timeline_key(
    embedding_family: Any,
    prediction_model: Any,
) -> str:
    family, model = validate_timeline(
        embedding_family,
        prediction_model,
    )

    return f"{family}|{model}"


def predicted_db_name(
    base_db: str,
    embedding_family: Any,
    prediction_model: Any,
) -> str:
    family, model = validate_timeline(
        embedding_family,
        prediction_model,
    )

    return (
        f"{base_db}-"
        f"{FAMILY_SLUGS[family]}-"
        f"{MODEL_SLUGS[model]}"
    )


def valid_timeline_combinations() -> List[Tuple[str, str]]:
    combinations: List[Tuple[str, str]] = []

    for family in EMBEDDING_FAMILIES:
        for model in LINEAR_PREDICTION_MODELS:
            combinations.append((family, model))

    combinations.append(("TGN", "tgn_direct"))

    return combinations


def timeline_metadata(
    base_db: str,
    embedding_family: Any,
    prediction_model: Any,
) -> dict:
    family, model = validate_timeline(
        embedding_family,
        prediction_model,
    )

    return {
        "name": predicted_db_name(base_db, family, model),
        "family": family,
        "type": family,
        "predictionModel": model,
        "predictionModelDisplayName": prediction_model_display_name(model),
    }