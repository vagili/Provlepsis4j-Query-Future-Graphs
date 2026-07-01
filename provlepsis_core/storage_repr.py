# provlepsis_core/storage_repr.py
from __future__ import annotations

from typing import Any, Optional

SEPARATE_DBS = "separate_dbs"
SINGLE_DB_MULTI_EDGE = "single_db_multi_edge"


def normalize_storage_representation(value: Optional[str]) -> str:
    raw = str(value or SEPARATE_DBS).strip().lower()
    raw = raw.replace("-", "_").replace(" ", "_")

    aliases = {
        "separate_dbs": SEPARATE_DBS,
        "separate_db": SEPARATE_DBS,
        "two_dbs": SEPARATE_DBS,
        "single_edge": SEPARATE_DBS,

        "single_db": SINGLE_DB_MULTI_EDGE,
        "single_db_multi_edge": SINGLE_DB_MULTI_EDGE,
        "single_db_multiedge": SINGLE_DB_MULTI_EDGE,
        "multi_edge": SINGLE_DB_MULTI_EDGE,
        "multiedge": SINGLE_DB_MULTI_EDGE,
    }

    return aliases.get(raw, SEPARATE_DBS)


def is_single_db_multi_edge(value: Optional[str]) -> bool:
    return normalize_storage_representation(value) == SINGLE_DB_MULTI_EDGE


def sanitize_rel_type_base(value: Any) -> str:
    s = "".join(c for c in str(value or "REL") if c.isalnum() or c == "_")
    return s.upper() or "REL"


def normalize_prediction_model_for_type(value: Any) -> str:
    raw = str(value or "logistic_regression").strip().lower()
    raw = raw.replace("-", "_").replace(" ", "_")

    if raw in {"logistic", "logistic_regression", "logisticregression", "lr"}:
        return "LOGISTIC"

    if raw in {"lasso", "lasso_regression", "lasso_logistic", "l1", "l1_logistic"}:
        return "LASSO"

    if raw in {"tgn", "tgn_direct", "tgn_model", "tgn_link_prediction", "temporal_graph_network"}:
        return "TGN"

    return "LOGISTIC"


def observed_rel_type(base_type: Any, time_point: Any = None) -> str:

    base = sanitize_rel_type_base(base_type)

    if time_point is None:
        return base

    try:
        tp = _canon_like_rel_suffix(time_point)
    except Exception:
        tp = str(time_point)

    tp = "".join(c for c in str(tp) if c.isalnum() or c == "_")

    if not tp:
        return base

    return f"{base}_T{tp}"


def _canon_like_rel_suffix(value: Any) -> Any:
    try:
        f = float(value)
        return int(f) if f.is_integer() else str(value).replace(".", "_")
    except Exception:
        return value


def predicted_rel_type(
    embedding_family: Any,
    level: Any = 1,
    prediction_model: Any = "logistic_regression",
    **_ignored: Any,
) -> str:

    model = normalize_prediction_model_for_type(prediction_model)
    fam = sanitize_rel_type_base(embedding_family or "PREDICTED")

    try:
        lvl = int(level)
    except Exception:
        lvl = 1

    if lvl < 1:
        lvl = 1

    return f"PREDICTED_{model}_{fam}_T{lvl}"