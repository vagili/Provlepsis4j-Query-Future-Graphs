from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import sys
import math
import datetime
import re
from .emb_graphsage import _resolve_graphsage_prefix
from ..db import run, run_data, current_database
from . import split as split_mod
from neo4j.exceptions import Neo4jError
from . import emb_tgn
from ..storage_repr import (
    SEPARATE_DBS,
    SINGLE_DB_MULTI_EDGE,
    normalize_storage_representation,
    predicted_rel_type,
)
from ..prediction_artifacts import (
    get_or_create_candidate_pool,
    record_candidate_scores,
)

from ..lp_model_state import (
    save_lp_learning_state,
    load_lp_learning_state,
    normalize_prediction_model,
    prediction_model_display_name,
    prediction_model_algorithm_name,
)

from ..prediction_timeline import (
    EMBEDDING_FAMILIES,
    predicted_db_name,
    timeline_key,
    valid_timeline_combinations,
    validate_timeline,
)

from . import lp as lp_mod
# ----------------- sklearn import just for iterative LP -----------------

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except Exception:  
    LogisticRegression = None
    make_pipeline = None
    StandardScaler = None

router = APIRouter()

EMB_FAMILIES = ("FastRP", "Node2Vec", "GraphSAGE", "HashGNN", "TGN")

_PROP_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# ----------------- Debug helper -----------------

def _dbg_iter(tag: str, **fields: Any) -> None:

    try:
        db = current_database()
    except Exception:
        db = "<unknown>"

    ts = datetime.datetime.utcnow().isoformat()
    msg = " ".join(f"{k}={repr(v)}" for k, v in fields.items())
    line = f"{ts} [LP_ITER DEBUG] {tag} db={db} {msg}"
    print(line, file=sys.stderr, flush=True)


# ----------------- DB binding helper -----------------

def _with_db_bound(module, target_db: str):

    class _Ctx:
        def __enter__(self_inner):
            # Save originals
            self_inner._orig_run = getattr(module, "run", None)
            self_inner._orig_run_data = getattr(module, "run_data", None)

            # Wrappers
            def _r(q, p=None, **kw):
                return run(q, p or {}, db=target_db)

            def _rd(q, p=None, **kw):
                return run_data(q, p or {}, db=target_db)

            if self_inner._orig_run is not None:
                module.run = _r
            if self_inner._orig_run_data is not None:
                module.run_data = _rd

            return self_inner

        def __exit__(self_inner, exc_type, exc, tb):
            # Restore originals
            if getattr(self_inner, "_orig_run", None) is not None:
                module.run = self_inner._orig_run
            if getattr(self_inner, "_orig_run_data", None) is not None:
                module.run_data = self_inner._orig_run_data

    return _Ctx()


# ----------------- Request model -----------------

class IterateBody(BaseModel):
    embeddingFamily: str = Field(..., description="FastRP | Node2Vec | GraphSAGE | HashGNN | TGN")
    embeddingProperty: str

    predictionModel: str = Field(default="logistic_regression")
    storageRepresentation: str = Field(default=SEPARATE_DBS)

    predictK: int = Field(default=100, ge=1)
    candidateMultiplier: int = Field(default=20, ge=2, le=200)
    probThreshold: float = Field(default=0.80, ge=0.0, le=1.0)
    negativeRatio: float = Field(default=1.0, ge=0.1, le=10.0)

    retrainingInterval: int = Field(
        default=1,
        ge=0,
        description="0 = use initial model state, 1 = retrain before every iterative step",
    )

# ----------------- Small utility helpers -----------------
def _drop_model_if_exists_for_db(dbname: str, model_name: str) -> None:
    try:
        run_data(
            "CALL gds.model.drop($name) YIELD modelName RETURN modelName",
            {"name": model_name},
            db=dbname,
        )
        _dbg_iter("_drop_model_if_exists_for_db.dropped", dbname=dbname, model=model_name)
    except Exception as e:
        _dbg_iter("_drop_model_if_exists_for_db.skip", dbname=dbname, model=model_name, error=str(e))


def _graphsage_property_exists_in_projection_for_db(dbname: str, graph: str, prop: str) -> bool:
    try:
        rows = run_data(
            """
            CALL gds.graph.nodeProperty.stream($g, $p)
            YIELD nodeId, propertyValue
            RETURN 1 AS ok
            LIMIT 1
            """,
            {"g": graph, "p": prop},
            db=dbname,
        )
        return bool(rows)
    except Neo4jError as e:
        msg = f"{getattr(e, 'code', '')}: {getattr(e, 'message', str(e))}"
        if "Expecting at least one node projection to contain property key" in msg:
            return False
        raise


def _graphsage_validate_props_exist_in_projection_for_db(
    dbname: str,
    graph: str,
    props: List[str],
) -> List[str]:
    if not props:
        return []
    ok: List[str] = []
    for p in props:
        if _graphsage_property_exists_in_projection_for_db(dbname, graph, p):
            ok.append(p)
    return ok


def _graphsage_ensure_topology_fallback_feature_for_db(
    dbname: str,
    graph: str,
    prop: str = "__graphsage_degree__",
) -> List[str]:
    if _graphsage_property_exists_in_projection_for_db(dbname, graph, prop):
        return [prop]

    run_data(
        """
        CALL gds.degree.mutate($g, {
          mutateProperty: $prop
        })
        YIELD nodePropertiesWritten
        RETURN nodePropertiesWritten
        """,
        {"g": graph, "prop": prop},
        db=dbname,
    )

    if not _graphsage_property_exists_in_projection_for_db(dbname, graph, prop):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Failed to create GraphSAGE fallback feature '{prop}' "
                f"in graph '{graph}' of database '{dbname}'."
            ),
        )

    _dbg_iter(
        "_graphsage_ensure_topology_fallback_feature_for_db.done",
        dbname=dbname,
        graph=graph,
        prop=prop,
    )
    return [prop]


def _predicted_db_name(
    base: str,
    family: str,
    prediction_model: str,
) -> str:
    return predicted_db_name(
        base,
        family,
        prediction_model,
    )


def _db_exists(dbname: str) -> bool:

    try:
        rows = run_data(
            "SHOW DATABASE $db YIELD name, currentStatus RETURN currentStatus",
            {"db": dbname},
            db="system",
        )
        exists = bool(rows)
        _dbg_iter("_db_exists", dbname=dbname, exists=exists)
        return exists
    except Exception as e:
        _dbg_iter("_db_exists.error", dbname=dbname, error=str(e))
        return False


def _compute_level_and_ts(dbname: str) -> Tuple[int, int]:

    _dbg_iter("_compute_level_and_ts.start", dbname=dbname)
    rows = run_data(
        """
        MATCH ()-[r]-()
        WITH
          CASE
            WHEN r.timestamps IS NOT NULL
                 AND valueType(r.timestamps) STARTS WITH 'LIST'
                 AND size(r.timestamps) > 0
              THEN reduce(
                     m = 0.0,
                     x IN [y IN r.timestamps | toFloat(y)] |
                     CASE WHEN x > m THEN x ELSE m END
                   )
            ELSE toFloat(coalesce(r.lastTimestamp, r.firstTimestamp, r.timestamp, 0))
          END AS rel_max_ts,
          toInteger(coalesce(r.predicted, 0)) AS lvl
        RETURN
          coalesce(max(rel_max_ts), 0) AS max_ts,
          coalesce(max(lvl), 0) AS max_pred
        """,
        db=dbname,
    )
    if not rows:
        _dbg_iter("_compute_level_and_ts.empty", dbname=dbname)
        return (0, 0)

    max_ts = int(float(rows[0]["max_ts"] or 0))
    max_pred = int(rows[0]["max_pred"] or 0)

    _dbg_iter(
        "_compute_level_and_ts.done",
        dbname=dbname,
        max_ts=max_ts,
        max_pred=max_pred,
    )
    return max_pred, max_ts


def _assert_sklearn_iter() -> None:
    if LogisticRegression is None:
        _dbg_iter("_assert_sklearn_iter.missing")
        raise HTTPException(
            status_code=500,
            detail=(
                "scikit-learn is not installed in the backend image. "
                "Add it to requirements.txt (e.g., scikit-learn==1.4.*) and rebuild."
            ),
        )

def _make_iter_classifier(prediction_model: str):

    _assert_sklearn_iter()
    model_key = normalize_prediction_model(prediction_model)

    if model_key == "lasso_regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty="l1",
                solver="liblinear",
                C=1.0,
                max_iter=1000,
            ),
        )

    return LogisticRegression(max_iter=200, n_jobs=None)

def _sanitize_prop(name: str) -> str:
    if not _PROP_RE.match(name):
        _dbg_iter("_sanitize_prop.invalid", name=name)
        raise HTTPException(status_code=400, detail=f"Invalid embedding property name: {name}")
    return name


def _hadamard(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a * b

def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)

# ----------------- Embedding step on trainGraph -----------------

def _infer_embedding_dimension(dbname: str, prop: str, default: int = 128) -> int:

    try:
        rows = run_data(
            f"""
            MATCH (n)
            WHERE n.{prop} IS NOT NULL AND size(n.{prop}) > 0
            RETURN size(n.{prop}) AS dim
            LIMIT 1
            """,
            db=dbname,
        )
        if rows and rows[0].get("dim") is not None:
            dim = int(rows[0]["dim"])
            _dbg_iter("_infer_embedding_dimension.found", dbname=dbname, prop=prop, dim=dim)
            return dim
    except Exception as e:
        _dbg_iter("_infer_embedding_dimension.error", dbname=dbname, prop=prop, error=str(e))
    _dbg_iter("_infer_embedding_dimension.default", dbname=dbname, prop=prop, dim=default)
    return default

# ---- HashGNN helpers for iterative LP ----

def _sanitize_hashgnn_prop(name: str) -> str:

    cleaned = "".join(ch for ch in str(name) if ch.isalnum() or ch == "_")
    return cleaned or "HashGNN"


def _resolve_hashgnn_stream_for_db(dbname: str) -> str:

    rows = run_data(
        "CALL gds.list() YIELD name "
        "WHERE name IN ['gds.hashgnn.stream'] "
        "RETURN collect(name) AS present",
        {},
        db=dbname,
    )
    present = set(rows[0]["present"] if rows else [])
    if "gds.hashgnn.stream" in present:
        return "gds.hashgnn.stream"

    raise HTTPException(
        status_code=400,
        detail=(
            "HashGNN procedure 'gds.hashgnn.stream' is not available in this database. "
            "Check your GDS version/edition and `CALL gds.list()`."
        ),
    )


# ---- GraphSAGE helpers for iterative LP ----
def _resolve_graphsage_prefix_for_db(dbname: str) -> str:

    names = [
        "gds.graphSage.train",
        "gds.beta.graphSage.train",
        "gds.graphSage.write",
        "gds.beta.graphSage.write",
    ]
    rows = run_data(
        "CALL gds.list() YIELD name "
        "WHERE name IN $names "
        "RETURN collect(name) AS present",
        {"names": names},
        db=dbname,
    )
    present = set(rows[0]["present"] if rows else [])
    train = (
        "gds.graphSage.train"
        if "gds.graphSage.train" in present
        else ("gds.beta.graphSage.train" if "gds.beta.graphSage.train" in present else None)
    )
    write = (
        "gds.graphSage.write"
        if "gds.graphSage.write" in present
        else ("gds.beta.graphSage.write" if "gds.beta.graphSage.write" in present else None)
    )
    if not train or not write:
        raise HTTPException(
            status_code=400,
            detail=(
                "GraphSAGE procedures not found in this database. "
                "Your GDS version/edition may not include GraphSAGE "
                "(look for gds.graphSage.* or gds.beta.graphSage.* in CALL gds.list())."
            ),
        )
    return train.rsplit(".", 1)[0]


def _graphs_age_model_list_proc_for_db(dbname: str) -> str:

    rows = run_data(
        "CALL gds.list() YIELD name "
        "WHERE name IN ['gds.model.list','gds.beta.model.list'] "
        "RETURN collect(name) AS present",
        {},
        db=dbname,
    )
    present = set(rows[0]["present"] if rows else [])
    if "gds.model.list" in present:
        return "gds.model.list"
    if "gds.beta.model.list" in present:
        return "gds.beta.model.list"
    raise HTTPException(
        status_code=400,
        detail=(
            "GDS model catalog procedures (gds.model.list / gds.beta.model.list) "
            "are not available in this database; cannot manage GraphSAGE models."
        ),
    )


def _graphs_age_model_exists(dbname: str, model_name: str) -> bool:

    proc = _graphs_age_model_list_proc_for_db(dbname)
    rows = run_data(
        f"CALL {proc}() YIELD modelName "
        "WHERE modelName = $name "
        "RETURN modelName",
        {"name": model_name},
        db=dbname,
    )
    return bool(rows)


def _graphs_age_discover_feature_props(dbname: str, graph: str) -> List[str]:

    # Step 1: sample keys from a projected node
    rows = run_data(
        """
        WITH $g AS g
        CALL gds.degree.stream(g) YIELD nodeId
        WITH g, nodeId LIMIT 1
        WITH g, gds.util.asNode(nodeId) AS n
        RETURN [p IN keys(n)
                WHERE NOT p IN ['FastRP','Node2Vec','GraphSAGE','HashGNN', 'TGN']
                  AND NOT p STARTS WITH 'embedding_'
                  AND (
                    toFloatOrNull(n[p]) IS NOT NULL
                    OR (
                      valueType(n[p]) STARTS WITH 'LIST'
                      AND all(x IN n[p] WHERE x IS NULL OR toFloatOrNull(x) IS NOT NULL)
                    )
                  )
               ] AS numericProjectedProps
        """,
        {"g": graph},
        db=dbname,
    )
    props = rows[0]["numericProjectedProps"] if rows else []

    # Step 2: exclude obvious embedding props and internal keys
    EMBEDDING_PREFIXES = ["fastrp_", "node2vec_", "graphsage_", "hashgnn_", "tgn_"]
    EMBEDDING_EXACTS = {"FastRP", "Node2Vec", "GraphSAGE", "HashGNN", "TGN"}
    INTERNAL_KEYS = {"_id", "_tmp", "_ts", "id", "ID"}

    def _exclude_embedding_props(props_in: List[str]) -> List[str]:
        out: List[str] = []
        for p in props_in:
            if p in INTERNAL_KEYS or p in EMBEDDING_EXACTS:
                continue
            if any(p.startswith(pref) for pref in EMBEDDING_PREFIXES):
                continue
            out.append(p)
        return out

    props = _exclude_embedding_props(props)

    # Step 3: filter values
    _ALLOWED_SCALARS = (int, float, bool, str)

    def _looks_supported_value(v) -> bool:
        if v is None:
            return True
        if isinstance(v, _ALLOWED_SCALARS):
            return True
        if isinstance(v, list):
            for x in v:
                if x is None:
                    continue
                if not isinstance(x, (int, float)):
                    return False
            return True
        return False

    def _filter_supported_props(props_in: List[str]) -> List[str]:
        safe: List[str] = []
        for p in props_in:
            rows2 = run_data(
                "WITH $k AS k MATCH (n) WHERE n[k] IS NOT NULL RETURN n[k] AS v LIMIT 500",
                {"k": p},
                db=dbname,
            )
            if not rows2:
                safe.append(p)
                continue
            ok = True
            for r2 in rows2:
                if not _looks_supported_value(r2.get("v")):
                    ok = False
                    break
            if ok:
                safe.append(p)
        return safe

    props = _filter_supported_props(props)

    props = _graphsage_validate_props_exist_in_projection_for_db(dbname, graph, props)
    return props


def _ensure_graphsage_model_for_db(
    dbname: str,
    graph: str,
    model_name: str,
    dim: int,
) -> Dict[str, Any]:

    prefix = _resolve_graphsage_prefix_for_db(dbname)

    _drop_model_if_exists_for_db(dbname, model_name)

    used_features = _graphs_age_discover_feature_props(dbname, graph)
    note = None

    if not used_features:
        used_features = _graphsage_ensure_topology_fallback_feature_for_db(
            dbname=dbname,
            graph=graph,
            prop="__graphsage_degree__",
        )
        note = (
            "No projected numeric node features were found. "
            "GraphSAGE used topology-derived fallback feature '__graphsage_degree__'."
        )

    if not used_features:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No suitable featureProperties were found for GraphSAGE in graph '{graph}' "
                f"of database '{dbname}', and fallback feature creation failed."
            ),
        )

    train_cfg: Dict[str, Any] = {
        "modelName": model_name,
        "featureProperties": used_features,
        "aggregator": "mean",
        "sampleSizes": [25, 10],
        "embeddingDimension": int(dim),
        "epochs": 5,
        "learningRate": 0.01,
    }

    _dbg_iter(
        "_ensure_graphsage_model_for_db.train",
        dbname=dbname,
        graph=graph,
        model=model_name,
        featureProperties=used_features,
        dim=dim,
    )

    run_data(
        f"CALL {prefix}.train($g, $cfg)",
        {"g": graph, "cfg": train_cfg},
        db=dbname,
    )

    _dbg_iter(
        "_ensure_graphsage_model_for_db.trained",
        dbname=dbname,
        graph=graph,
        model=model_name,
        featureProperties=used_features,
    )

    return {
        "prefix": prefix,
        "modelName": model_name,
        "featureProperties": used_features,
        "note": note,
    }

def _compute_embeddings_for_family(
    dbname: str,
    family: str,
    prop: str,
    graph_name: str = "trainGraph",
) -> Dict[str, Any]:

    dim = _infer_embedding_dimension(dbname, prop, default=128)
    _dbg_iter(
        "_compute_embeddings_for_family.start",
        dbname=dbname,
        family=family,
        prop=prop,
        dim=dim,
        graph_name=graph_name,
    )

    # ---------- FastRP ----------
    if family == "FastRP":
        cfg = {
            "embeddingDimension": dim,
            "writeProperty": prop,
        }
        rows = run_data(
            """
            CALL gds.fastRP.write($g, $cfg)
            YIELD nodePropertiesWritten, nodeCount, computeMillis, writeMillis
            RETURN nodePropertiesWritten, nodeCount, computeMillis, writeMillis
            """,
            {"g": graph_name, "cfg": cfg},
            db=dbname,
        )
        info = rows[0] if rows else {}
        _dbg_iter("_compute_embeddings_for_family.fastrp_done", dbname=dbname, info=info)
        return {"family": family, "config": cfg, "info": info}

    # ---------- Node2Vec ----------
    if family == "Node2Vec":
        cfg = {
            "embeddingDimension": dim,
            "walkLength": 80,
            "walksPerNode": 10,
            "returnFactor": 1.0,
            "inOutFactor": 1.0,
            "writeProperty": prop,
        }
        rows = run_data(
            """
            CALL gds.node2vec.write($g, $cfg)
            YIELD nodePropertiesWritten, nodeCount, computeMillis, writeMillis
            RETURN nodePropertiesWritten, nodeCount, computeMillis, writeMillis
            """,
            {"g": graph_name, "cfg": cfg},
            db=dbname,
        )
        info = rows[0] if rows else {}
        _dbg_iter("_compute_embeddings_for_family.n2v_done", dbname=dbname, info=info)
        return {"family": family, "config": cfg, "info": info}

    # ---------- GraphSAGE ----------
    if family == "GraphSAGE":
        model_name = "sage_model"

        model_info = _ensure_graphsage_model_for_db(
            dbname=dbname,
            graph=graph_name,
            model_name=model_name,
            dim=dim,
        )
        prefix = model_info["prefix"]

        write_cfg: Dict[str, Any] = {
            "modelName": model_name,
            "writeProperty": prop,
        }

        try:
            rows = run_data(
                f"""
                CALL {prefix}.write($g, $cfg)
                YIELD nodePropertiesWritten, computeMillis, writeMillis
                RETURN nodePropertiesWritten, computeMillis, writeMillis
                """,
                {"g": graph_name, "cfg": write_cfg},
                db=dbname,
            )
        except Neo4jError as e:
            msg = getattr(e, "message", str(e))

            if "Model with name" in msg and "does not exist" in msg:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"GraphSAGE model '{model_name}' does not exist in database '{dbname}'. "
                        "Tried to auto-create it but the operation failed. "
                        "Please run the initial GraphSAGE /trainWrite step for this DB."
                    ),
                )
            raise

        info = rows[0] if rows else {}
        _dbg_iter(
            "_compute_embeddings_for_family.graphsage_done",
            dbname=dbname,
            model=model_name,
            writeProperty=prop,
            info=info,
        )
        return {"family": family, "config": write_cfg, "info": info, "modelInfo": model_info}


    # ---------- HashGNN ----------
    if family == "HashGNN":

        proc = _resolve_hashgnn_stream_for_db(dbname)
        write_prop = _sanitize_hashgnn_prop(prop)

        
        it = 10
        dens = 10  
        params = {
            "g": graph_name,
            "it": it,
            "dim": int(dim),
            "dens": dens,
        }

        cypher = (
            f"CALL {proc}($g, {{"
            "  iterations: $it,"
            "  outputDimension: $dim,"
            "  embeddingDensity: $dens,"
            "  generateFeatures: { dimension: $dim, densityLevel: $dens }"
            "}) "
            "YIELD nodeId, embedding "
            "WITH gds.util.asNode(nodeId) AS n, [x IN embedding | toFloat(x)] AS v "
            f"SET n.{write_prop} = v "
            "RETURN count(*) AS wrote"
        )

        _dbg_iter(
            "_compute_embeddings_for_family.hashgnn_start",
            dbname=dbname,
            graph=graph_name,
            prop=write_prop,
            dim=dim,
            it=it,
            dens=dens,
            proc=proc,
        )

        rows = run_data(cypher, params, db=dbname)
        wrote = int(rows[0]["wrote"]) if rows else 0

        info = {
            "wrote": wrote,
            "iterations": it,
            "outputDimension": int(dim),
            "embeddingDensity": dens,
            "generateFeatures": {"dimension": int(dim), "densityLevel": dens},
            "writeProperty": write_prop,
        }

        _dbg_iter(
            "_compute_embeddings_for_family.hashgnn_done",
            dbname=dbname,
            graph=graph_name,
            prop=write_prop,
            info=info,
        )

        return {"family": family, "config": info, "info": info}

    # ---------- TGN ----------
    if family == "TGN":
        body = emb_tgn.TGNWriteBody(
            graphName=graph_name,
            writeProperty=prop,
            embeddingDimension=dim,
            memoryDimension=dim,
            timeDimension=min(64, dim),
            epochs=3,
            batchSize=200,
            learningRate=1e-4,
            numNeighbors=10,
        )
        info = emb_tgn.tgn_write_for_db(body, dbname=dbname)
        _dbg_iter("_compute_embeddings_for_family.tgn_done", dbname=dbname, info=info)
        return {"family": family, "config": info, "info": info}


    # ---------- Fallback ----------
    _dbg_iter("_compute_embeddings_for_family.unsupported", dbname=dbname, family=family)
    raise HTTPException(
        status_code=400,
        detail=(
            "Iterative LP currently supports 'FastRP', 'Node2Vec', 'GraphSAGE' "
            f"and 'HashGNN' and 'TGN' only. Got embeddingFamily='{family}'."
        ),
    )



# ----------------- Embeddings + pairs helpers inside chosen_db -----------------

def _load_embeddings_from_db(dbname: str, prop: str) -> Dict[str, np.ndarray]:
    prop = _sanitize_prop(prop)
    rows = run_data(
        f"""
        MATCH (n)
        WHERE n.`{prop}` IS NOT NULL
        RETURN elementId(n) AS eid, n.`{prop}` AS emb
        """,
        db=dbname,
    )
    out: Dict[str, np.ndarray] = {}
    for row in rows:
        value = row.get("emb")
        if isinstance(value, list):
            out[str(row["eid"])] = np.asarray(value, dtype=float)
    return out


def _load_embeddings_from_db_stable(
    dbname: str,
    prop: str,
) -> Dict[str, np.ndarray]:
    prop = _sanitize_prop(prop)
    rows = run_data(
        f"""
        MATCH (n)
        WHERE n.`{prop}` IS NOT NULL
        RETURN
          toString(coalesce(n.eid, elementId(n))) AS stableId,
          n.`{prop}` AS embedding
        """,
        db=dbname,
    )
    out: Dict[str, np.ndarray] = {}
    for row in rows:
        value = row.get("embedding")
        if isinstance(value, list):
            out[str(row["stableId"])] = np.asarray(value, dtype=float)
    return out


def _load_timeline_pairs_stable(
    *,
    dbname: str,
    family: str,
    prediction_model: str,
) -> set[Tuple[str, str]]:
    model_key = normalize_prediction_model(prediction_model)
    rows = run_data(
        """
        MATCH (a)-[r]-(b)
        WHERE coalesce(r.edgeOrigin, '') = 'predicted'
          AND coalesce(r.embeddingFamily, '') = $family
          AND coalesce(
                r.predictionModel,
                'logistic_regression'
              ) = $predictionModel
        WITH
          toString(coalesce(a.eid, elementId(a))) AS aId,
          toString(coalesce(b.eid, elementId(b))) AS bId
        RETURN DISTINCT
          CASE WHEN aId < bId THEN aId ELSE bId END AS s,
          CASE WHEN aId < bId THEN bId ELSE aId END AS t
        """,
        {
            "family": family,
            "predictionModel": model_key,
        },
        db=dbname,
    )
    return {_pair_key(str(row["s"]), str(row["t"])) for row in rows}


def _load_observed_pairs_stable(dbname: str) -> set[Tuple[str, str]]:
    rows = run_data(
        """
        MATCH (a)-[r]-(b)
        WHERE coalesce(r.edgeOrigin, 'observed') = 'observed'
          AND NOT type(r) STARTS WITH 'SPLIT_'
        WITH
          toString(coalesce(a.eid, elementId(a))) AS aId,
          toString(coalesce(b.eid, elementId(b))) AS bId
        RETURN DISTINCT
          CASE WHEN aId < bId THEN aId ELSE bId END AS s,
          CASE WHEN aId < bId THEN bId ELSE aId END AS t
        """,
        db=dbname,
    )
    return {_pair_key(str(row["s"]), str(row["t"])) for row in rows}


def _observed_database_max_timestamp(dbname: str) -> int:
    rows = run_data(
        """
        MATCH ()-[r]-()
        WHERE coalesce(r.edgeOrigin, 'observed') = 'observed'
        WITH CASE
            WHEN r.timestamps IS NOT NULL
                 AND valueType(r.timestamps) STARTS WITH 'LIST'
                 AND size(r.timestamps) > 0
              THEN reduce(
                     m = 0.0,
                     x IN [y IN r.timestamps | toFloat(y)] |
                     CASE WHEN x > m THEN x ELSE m END
                   )
            WHEN r.lastTimestamp IS NOT NULL THEN toFloat(r.lastTimestamp)
            WHEN r.firstTimestamp IS NOT NULL THEN toFloat(r.firstTimestamp)
            WHEN r.timestamp IS NOT NULL THEN toFloat(r.timestamp)
            ELSE 0.0
        END AS ts
        RETURN coalesce(max(ts), 0) AS max_ts
        """,
        db=dbname,
    )

    return int(float(rows[0]["max_ts"] or 0)) if rows else 0

def _get_train_positive_pairs(dbname: str, graph_name: str = "trainGraph") -> List[Tuple[str, str]]:

    _dbg_iter("_get_train_positive_pairs.start", dbname=dbname, graph_name=graph_name)
    rows = run_data(
        """
        CALL gds.graph.relationships.stream($g)
        YIELD sourceNodeId, targetNodeId
        WITH gds.util.asNode(sourceNodeId) AS sN,
             gds.util.asNode(targetNodeId) AS tN
        WITH elementId(sN) AS a, elementId(tN) AS b
        WITH CASE WHEN a < b THEN a ELSE b END AS s,
             CASE WHEN a < b THEN b ELSE a END AS t
        RETURN DISTINCT s, t
        """,
        {"g": graph_name},
        db=dbname,
    )
    pairs = [(r["s"], r["t"]) for r in rows]
    _dbg_iter("_get_train_positive_pairs.done", dbname=dbname, graph_name=graph_name, count=len(pairs))
    return pairs


def _sample_negative_pairs(dbname: str, prop: str, k: int) -> List[Tuple[str, str]]:

    k = max(0, int(k))
    if k == 0:
        _dbg_iter("_sample_negative_pairs.skip", dbname=dbname, k=k)
        return []
    oversample = max(k * 3, 50)
    _dbg_iter("_sample_negative_pairs.start", dbname=dbname, prop=prop, k=k, oversample=oversample)

    cypher = f"""
    // (1) restrict to nodes with the embedding property
    MATCH (n)
    WHERE n.{prop} IS NOT NULL
    WITH collect(elementId(n)) AS ids, size(collect(n)) AS N

    // (2) sample candidate pairs (by elementId) via rand()
    CALL {{
      WITH ids, N
      UNWIND range(1, $oversample) AS _
      WITH ids, N,
           ids[toInteger(rand() * N)] AS a,
           ids[toInteger(rand() * N)] AS b
      WHERE a <> b
      WITH CASE WHEN a < b THEN a ELSE b END AS s,
           CASE WHEN a < b THEN b ELSE a END AS t
      RETURN DISTINCT s, t
      LIMIT $oversample
    }}

    // (3) keep only those with no edge of any type between them
    WITH s, t
    WHERE NOT EXISTS {{
      MATCH (x)--(y)
      WHERE (elementId(x) = s AND elementId(y) = t)
         OR (elementId(x) = t AND elementId(y) = s)
    }}

    RETURN DISTINCT s, t
    LIMIT $k
    """

    rows = run_data(
        cypher,
        {"k": k, "oversample": oversample},
        db=dbname,
    )
    pairs = [(r["s"], r["t"]) for r in rows]
    _dbg_iter("_sample_negative_pairs.done", dbname=dbname, prop=prop, requested=k, returned=len(pairs))
    return pairs


def _build_train_matrices(
    dbname: str,
    prop: str,
    pos_pairs: List[Tuple[str, str]],
    emb: Dict[str, np.ndarray],
    negative_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, int, int]:

    _dbg_iter(
        "_build_train_matrices.start",
        dbname=dbname,
        prop=prop,
        pos_pairs=len(pos_pairs),
        negative_ratio=negative_ratio,
    )
    # Positive features
    X_pos: List[np.ndarray] = []
    for s, t in pos_pairs:
        es = emb.get(s)
        et = emb.get(t)
        if es is None or et is None:
            continue
        X_pos.append(_hadamard(es, et))
    cnt_pos = len(X_pos)

    if cnt_pos == 0:
        _dbg_iter("_build_train_matrices.no_pos_after_emb", dbname=dbname, prop=prop)
        return np.zeros((0, 1)), np.zeros((0,), dtype=int), 0, 0

    want_neg = int(math.ceil(cnt_pos * max(0.0, min(10.0, float(negative_ratio)))))
    neg_pairs = _sample_negative_pairs(dbname, prop, want_neg)

    # Negative features
    X_neg: List[np.ndarray] = []
    for s, t in neg_pairs:
        es = emb.get(s)
        et = emb.get(t)
        if es is None or et is None:
            continue
        X_neg.append(_hadamard(es, et))
    cnt_neg = len(X_neg)

    if not X_neg:
        _dbg_iter("_build_train_matrices.no_neg", dbname=dbname, prop=prop, cnt_pos=cnt_pos)
        return np.zeros((0, 1)), np.zeros((0,), dtype=int), cnt_pos, 0

    X = np.vstack([X_pos, X_neg])
    y = np.hstack([
        np.ones(cnt_pos, dtype=int),
        np.zeros(cnt_neg, dtype=int),
    ])

    _dbg_iter(
        "_build_train_matrices.done",
        dbname=dbname,
        prop=prop,
        cnt_pos=cnt_pos,
        cnt_neg=cnt_neg,
        X_shape=X.shape,
    )
    return X, y, cnt_pos, cnt_neg


def _write_predicted_edges_inplace(
    dbname: str,
    triples: List[Tuple[str, str, float]],
    level: int,
    ts: int,
    embedding_family: str,
    prediction_model: str,
) -> int:
    if not triples:
        return 0

    rel_type = predicted_rel_type(
        embedding_family,
        level=level,
        prediction_model=prediction_model,
    )

    rows = run_data(
        f"""
        UNWIND $triples AS row
        MATCH (s)
        WHERE toString(coalesce(s.eid, elementId(s))) = toString(row.s)
        MATCH (t)
        WHERE toString(coalesce(t.eid, elementId(t))) = toString(row.t)
        CREATE (s)-[r:`{rel_type}`]->(t)
        SET r.predicted = $level,
            r.predicted_timestep = $level,
            r.probability = row.p,
            r.timestamp = $ts,
            r.firstTimestamp = $ts,
            r.lastTimestamp = $ts,
            r.timestamps = [$ts],
            r.interactionCount = 1,
            r.edgeOrigin = 'predicted',
            r.embeddingFamily = $embeddingFamily,
            r.predictionModel = $predictionModel,
            r.predictionModelDisplayName = $predictionModelDisplayName,
            r.storageRepresentation = $storageRepresentation
        RETURN count(r) AS relCount
        """,
        {
            "triples": [
                {"s": s, "t": t, "p": float(p)}
                for s, t, p in triples
            ],
            "level": int(level),
            "ts": int(ts),
            "embeddingFamily": embedding_family,
            "predictionModel": normalize_prediction_model(prediction_model),
            "predictionModelDisplayName": prediction_model_display_name(
                prediction_model
            ),
            "storageRepresentation": SEPARATE_DBS,
        },
        db=dbname,
    )
    return int(rows[0]["relCount"]) if rows else 0



# ----------------- Main iterative endpoint -----------------

@router.post("/iterate")
def iterate_prediction(body: IterateBody) -> Dict[str, Any]:

    _dbg_iter("iterate_prediction.enter", body=body.dict())

    prediction_model_key = normalize_prediction_model(getattr(body, "predictionModel", "logistic_regression"))
    if prediction_model_key == "tgn_direct":
        from . import tgn_direct
        return tgn_direct.iterate_tgn_direct(
            tgn_direct.TGNDirectIterateBody(
                storageRepresentation=body.storageRepresentation,
                predictK=body.predictK,
                candidateMultiplier=body.candidateMultiplier,
                probThreshold=body.probThreshold,
                negativeRatio=body.negativeRatio,
                retrainingInterval=body.retrainingInterval,
            )
        )

    fam = body.embeddingFamily.strip()
    if fam not in EMB_FAMILIES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown embeddingFamily '{fam}'. Choose one of {EMB_FAMILIES}.",
        )
    
    storage_mode = normalize_storage_representation(body.storageRepresentation)

    if storage_mode == SINGLE_DB_MULTI_EDGE:
        return _iterate_prediction_single_db_multi_edge(body)

    return _iterate_prediction_separate_family_db(body)

def _iterate_prediction_separate_family_db(body: IterateBody) -> Dict[str, Any]:
    try:
        fam, prediction_model = validate_timeline(
            body.embeddingFamily,
            body.predictionModel,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if prediction_model == "tgn_direct":
        raise HTTPException(
            status_code=400,
            detail="Direct TGN iteration must use the TGN direct endpoint.",
        )

    base = current_database()
    target_db = _predicted_db_name(base, fam, prediction_model)

    if not _db_exists(target_db):
        raise HTTPException(
            status_code=404,
            detail=(
                f"Predicted database '{target_db}' was not found. "
                f"Run the initial {fam} prediction for "
                f"{prediction_model_display_name(prediction_model)} first."
            ),
        )

    prev_level, prev_ts = _compute_level_and_ts(target_db)
    observed_max_ts = _observed_database_max_timestamp(target_db)
    new_level = prev_level + 1
    new_ts = max(prev_ts, observed_max_ts) + 1
    prop = _sanitize_prop(body.embeddingProperty)
    retraining_interval = max(0, int(body.retrainingInterval))
    state_name_to_load = "initial" if retraining_interval == 0 else "latest"

    learning_state = load_lp_learning_state(
        base_db=base,
        storage_representation=SEPARATE_DBS,
        embedding_family=fam,
        prediction_model=prediction_model,
        state_name=state_name_to_load,
    )

    has_learning_state = bool(
        learning_state
        and learning_state.get("model") is not None
        and learning_state.get("embeddingProperty")
    )
    did_retrain_model = (not has_learning_state) or retraining_interval > 0

    training_policy = {
        "retrainingInterval": retraining_interval,
        "didRetrainModel": bool(did_retrain_model),
        "loadedStateName": state_name_to_load,
        "hadSavedLearningState": bool(has_learning_state),
        "mode": "frozen" if retraining_interval == 0 else "retrain_every_timestep",
    }

    if did_retrain_model:
        with _with_db_bound(split_mod, target_db):
            split_mod.clear_feature_cache(db=target_db)
            split_result = split_mod.execute_master_split(
                split_mod.ExecuteSplitBody(
                    trainGraphName="trainGraph",
                    testGraphName=None,
                    valGraphName=None,
                    testHoldout=0.0,
                    valHoldout=0.0,
                    ensureConnected=True,
                    includeEmbeddingProps=False,
                    reFreshFeatureCache=True,
                    temporalMode=True,
                    explicitTestTimestamps=[],
                    explicitValTimestamps=[],
                    storageRepresentation=SEPARATE_DBS,
                )
            )

        split_payload = (
            split_result.model_dump()
            if hasattr(split_result, "model_dump")
            else split_result.dict()
            if hasattr(split_result, "dict")
            else split_result
        )

        emb_info = _compute_embeddings_for_family(
            dbname=target_db,
            family=fam,
            prop=prop,
            graph_name="trainGraph",
        )
        pos_pairs = _get_train_positive_pairs(target_db, "trainGraph")
        train_emb = _load_embeddings_from_db(target_db, prop)

        if not train_emb:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"No nodes with embedding property '{prop}' were found "
                    f"in '{target_db}'."
                ),
            )

        Xtr, ytr, npos_tr, nneg_tr = _build_train_matrices(
            dbname=target_db,
            prop=prop,
            pos_pairs=pos_pairs,
            emb=train_emb,
            negative_ratio=body.negativeRatio,
        )

        if Xtr.shape[0] == 0 or npos_tr == 0 or nneg_tr == 0:
            raise HTTPException(
                status_code=400,
                detail="No usable training pairs for separate-database iterative prediction.",
            )

        model = _make_iter_classifier(prediction_model)
        model.fit(Xtr, ytr)
        stable_emb = _load_embeddings_from_db_stable(target_db, prop)
        saved_state = save_lp_learning_state(
            base_db=base,
            storage_representation=SEPARATE_DBS,
            embedding_family=fam,
            embedding_property=prop,
            model=model,
            trained_at_timestep=new_level,
            prediction_model=prediction_model,
            state_name="latest",
            embeddings=stable_emb,
            metadata={
                "source": "iterative_lp_separate_db",
                "targetPredictedDb": target_db,
                "trainPairs": npos_tr,
                "negTrainPairs": nneg_tr,
                "probThreshold": float(body.probThreshold),
                "predictK": int(body.predictK),
                "retrainingInterval": retraining_interval,
                "predictionModel": prediction_model,
                "predictionModelDisplayName": prediction_model_display_name(
                    prediction_model
                ),
                "predictionModelAlgorithm": prediction_model_algorithm_name(
                    prediction_model
                ),
            },
        )
        training_policy["savedLearningState"] = saved_state
    else:
        prop = str(learning_state["embeddingProperty"])
        model = learning_state["model"]
        stable_emb = learning_state.get("embeddings") or {}
        if not stable_emb:
            stable_emb = _load_embeddings_from_db_stable(target_db, prop)
        if not stable_emb:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"No saved or stored embeddings for property '{prop}' "
                    f"were found for '{target_db}'."
                ),
            )
        split_payload = {
            "reused": True,
            "reason": "learning_state_reused",
            "trainedAtTimestep": learning_state.get("trainedAtTimestep"),
        }
        emb_info = {
            "reused": True,
            "embeddingProperty": prop,
            "trainedAtTimestep": learning_state.get("trainedAtTimestep"),
        }
        metadata = learning_state.get("metadata") or {}
        npos_tr = int(metadata.get("trainPairs", 0) or 0)
        nneg_tr = int(metadata.get("negTrainPairs", 0) or 0)

    candidate_pool_size = max(
        int(body.predictK) * 20,
        int(body.predictK) + 50,
        1000,
    )
    generated_candidates = lp_mod._build_canonical_candidate_pairs(
        pool_size=candidate_pool_size * 2,
        version=new_level,
    )
    shared_pool = get_or_create_candidate_pool(
        base_db=base,
        storage_representation=SEPARATE_DBS,
        timestep=new_level,
        generated_candidates=generated_candidates,
        pool_size=candidate_pool_size,
        force_new=False,
    )
    already_predicted = _load_timeline_pairs_stable(
        dbname=target_db,
        family=fam,
        prediction_model=prediction_model,
    )
    observed_pairs = _load_observed_pairs_stable(base)
    candidate_pairs: List[Tuple[str, str]] = []
    seen: set[Tuple[str, str]] = set()
    for row in shared_pool:
        pair = _pair_key(str(row["s"]), str(row["t"]))
        if pair in seen or pair in already_predicted or pair in observed_pairs:
            continue
        seen.add(pair)
        candidate_pairs.append(pair)

    scored_candidates = _score_candidate_pairs_for_iterative_lp(
        candidate_pairs=candidate_pairs,
        emb=stable_emb,
        model=model,
    )
    selected_rows = _select_top_prediction_rows(
        scored_candidates=scored_candidates,
        predict_k=int(body.predictK),
        prob_threshold=float(body.probThreshold),
    )
    selected_triples = _rows_to_triples(selected_rows)
    selected_keys = {
        _pair_key(str(row["s"]), str(row["t"]))
        for row in selected_rows
    }
    artifact_rows = [
        {
            "s": key[0],
            "t": key[1],
            "pairKey": f"{key[0]}|{key[1]}",
            "probability": float(row.get("probability", 0.0) or 0.0),
            "rank": rank,
            "selected": key in selected_keys,
        }
        for rank, row in enumerate(scored_candidates, start=1)
        for key in [_pair_key(str(row["s"]), str(row["t"]))]
    ]
    record_candidate_scores(
        base_db=base,
        storage_representation=SEPARATE_DBS,
        timestep=new_level,
        prediction_model=prediction_model,
        embedding_family=fam,
        scored_candidates=artifact_rows,
    )
    rel_written = _write_predicted_edges_inplace(
        dbname=target_db,
        triples=selected_triples,
        level=new_level,
        ts=new_ts,
        embedding_family=fam,
        prediction_model=prediction_model,
    )
    count_above = sum(
        1
        for row in scored_candidates
        if float(row.get("probability", 0.0) or 0.0)
        >= float(body.probThreshold)
    )

    return {
        "ok": True,
        "baseDatabase": base,
        "storageRepresentation": SEPARATE_DBS,
        "embeddingFamily": fam,
        "predictionModel": prediction_model,
        "predictionModelDisplayName": prediction_model_display_name(
            prediction_model
        ),
        "targetPredictedDb": target_db,
        "targetExists": True,
        "embeddingProperty": prop,
        "splitResult": split_payload,
        "prevPredictedLevel": prev_level,
        "prevTimestamp": prev_ts,
        "newPredictedLevel": new_level,
        "newTimestamp": new_ts,
        "trainPairs": npos_tr,
        "negTrainPairs": nneg_tr,
        "embeddingInfo": emb_info,
        "trainingPolicy": training_policy,
        "prediction": {
            "requestedK": int(body.predictK),
            "candidateMultiplier": int(body.candidateMultiplier),
            "probThreshold": float(body.probThreshold),
            "totalCandidatesScored": len(scored_candidates),
            "candidatesAboveThreshold": count_above,
            "selectedNewPairs": len(selected_triples),
            "relationshipsWritten": rel_written,
        },
    }


def _score_candidate_pairs_for_iterative_lp(
    *,
    candidate_pairs: List[Tuple[str, str]],
    emb: Dict[str, np.ndarray],
    model: LogisticRegression,
) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []

    for s, t in candidate_pairs:
        es = emb.get(s)
        et = emb.get(t)

        if es is None or et is None:
            continue

        p = float(model.predict_proba((es * et).reshape(1, -1))[0, 1])

        scored.append({
            "s": s,
            "t": t,
            "probability": p,
        })

    scored.sort(
        key=lambda row: float(row.get("probability", 0.0) or 0.0),
        reverse=True,
    )

    return scored


def _select_top_prediction_rows(
    *,
    scored_candidates: List[Dict[str, Any]],
    predict_k: int,
    prob_threshold: float,
) -> List[Dict[str, Any]]:
    above = [
        row
        for row in scored_candidates
        if float(row.get("probability", 0.0) or 0.0) >= float(prob_threshold)
    ]

    if len(above) >= int(predict_k):
        return above[: int(predict_k)]

    return scored_candidates[: int(predict_k)]


def _rows_to_triples(rows: List[Dict[str, Any]]) -> List[Tuple[str, str, float]]:
    return [
        (
            str(row["s"]),
            str(row["t"]),
            float(row.get("probability", 0.0) or 0.0),
        )
        for row in rows
    ]


def _iterate_prediction_single_db_multi_edge(body: IterateBody) -> Dict[str, Any]:
    try:
        fam, prediction_model = validate_timeline(
            body.embeddingFamily,
            body.predictionModel,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if prediction_model == "tgn_direct":
        raise HTTPException(
            status_code=400,
            detail="Direct TGN iteration must use the TGN direct endpoint.",
        )

    base = current_database()
    prev_level, prev_family_ts = _compute_level_and_ts_single_db(
        base,
        fam,
        prediction_model,
    )
    observed_max_ts = _observed_database_max_timestamp(base)
    new_level = prev_level + 1
    new_ts = max(prev_family_ts, observed_max_ts) + 1
    prop = _sanitize_prop(body.embeddingProperty)
    retraining_interval = max(0, int(body.retrainingInterval))
    state_name_to_load = "initial" if retraining_interval == 0 else "latest"

    learning_state = load_lp_learning_state(
        base_db=base,
        storage_representation=SINGLE_DB_MULTI_EDGE,
        embedding_family=fam,
        prediction_model=prediction_model,
        state_name=state_name_to_load,
    )
    has_learning_state = bool(
        learning_state
        and learning_state.get("model") is not None
        and learning_state.get("embeddingProperty")
    )

    if not has_learning_state:
        did_retrain_model = True
    elif retraining_interval == 0:
        did_retrain_model = False
    elif retraining_interval == 1:
        did_retrain_model = True
    else:
        did_retrain_model = ((new_level - 2) % retraining_interval) == 0

    training_policy = {
        "retrainingInterval": retraining_interval,
        "didRetrainModel": bool(did_retrain_model),
        "loadedStateName": state_name_to_load,
        "hadSavedLearningState": bool(has_learning_state),
        "mode": (
            "frozen"
            if retraining_interval == 0
            else "retrain_every_timestep"
            if retraining_interval == 1
            else f"retrain_every_{retraining_interval}_timesteps"
        ),
    }

    if did_retrain_model:
        with _with_db_bound(split_mod, base):
            split_mod.clear_feature_cache(db=base)
            split_result = split_mod.execute_master_split(
                split_mod.ExecuteSplitBody(
                    trainGraphName="trainGraph",
                    testGraphName=None,
                    valGraphName=None,
                    testHoldout=0.0,
                    valHoldout=0.0,
                    ensureConnected=True,
                    includeEmbeddingProps=False,
                    reFreshFeatureCache=True,
                    temporalMode=True,
                    explicitTestTimestamps=[],
                    explicitValTimestamps=[],
                    storageRepresentation=SINGLE_DB_MULTI_EDGE,
                    includePredictedFamily=fam,
                    includePredictedModel=prediction_model,
                    includePredictedBeforeLevel=new_level,
                )
            )

        emb_info = _compute_embeddings_for_family(
            dbname=base,
            family=fam,
            prop=prop,
            graph_name="trainGraph",
        )
        pos_pairs = _get_train_positive_pairs(base, "trainGraph")
        emb = _load_embeddings_from_db(base, prop)
        Xtr, ytr, npos_tr, nneg_tr = _build_train_matrices(
            dbname=base,
            prop=prop,
            pos_pairs=pos_pairs,
            emb=emb,
            negative_ratio=body.negativeRatio,
        )

        if Xtr.shape[0] == 0 or npos_tr == 0 or nneg_tr == 0:
            raise HTTPException(
                status_code=400,
                detail="No usable training pairs for single-database iterative prediction.",
            )

        model = _make_iter_classifier(prediction_model)
        model.fit(Xtr, ytr)
        saved_state = save_lp_learning_state(
            base_db=base,
            storage_representation=SINGLE_DB_MULTI_EDGE,
            embedding_family=fam,
            embedding_property=prop,
            model=model,
            trained_at_timestep=new_level,
            prediction_model=prediction_model,
            state_name="latest",
            embeddings=emb,
            metadata={
                "source": "iterative_lp",
                "trainPairs": npos_tr,
                "negTrainPairs": nneg_tr,
                "probThreshold": float(body.probThreshold),
                "predictK": int(body.predictK),
                "retrainingInterval": retraining_interval,
                "predictionModel": prediction_model,
                "predictionModelDisplayName": prediction_model_display_name(
                    prediction_model
                ),
                "predictionModelAlgorithm": prediction_model_algorithm_name(
                    prediction_model
                ),
            },
        )
        training_policy["savedLearningState"] = saved_state
    else:
        prop = str(learning_state["embeddingProperty"])
        model = learning_state["model"]
        emb = learning_state.get("embeddings") or {}
        if not emb:
            emb = _load_embeddings_from_db(base, prop)
        if not emb:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"No saved or stored embeddings for property '{prop}' "
                    f"were found in '{base}'."
                ),
            )
        split_result = {
            "reused": True,
            "reason": "learning_state_reused",
            "trainedAtTimestep": learning_state.get("trainedAtTimestep"),
        }
        emb_info = {
            "reused": True,
            "embeddingProperty": prop,
            "trainedAtTimestep": learning_state.get("trainedAtTimestep"),
        }
        metadata = learning_state.get("metadata") or {}
        npos_tr = int(metadata.get("trainPairs", 0) or 0)
        nneg_tr = int(metadata.get("negTrainPairs", 0) or 0)

    candidate_pool_size = max(
        int(body.predictK) * 20,
        int(body.predictK) + 50,
        1000,
    )
    generated_candidates = lp_mod._build_canonical_candidate_pairs(
        pool_size=candidate_pool_size * 2,
        version=new_level,
    )
    shared_pool = get_or_create_candidate_pool(
        base_db=base,
        storage_representation=SINGLE_DB_MULTI_EDGE,
        timestep=new_level,
        generated_candidates=generated_candidates,
        pool_size=candidate_pool_size,
        force_new=False,
    )
    already_predicted = lp_mod._load_single_db_timeline_pairs(
        dbname=base,
        embedding_family=fam,
        prediction_model=prediction_model,
    )
    observed_pairs = _load_observed_pairs_stable(base)
    candidate_pairs: List[Tuple[str, str]] = []
    seen: set[Tuple[str, str]] = set()
    for row in shared_pool:
        pair = _pair_key(str(row["s"]), str(row["t"]))
        if pair in seen or pair in already_predicted or pair in observed_pairs:
            continue
        seen.add(pair)
        candidate_pairs.append(pair)

    scored_candidates = _score_candidate_pairs_for_iterative_lp(
        candidate_pairs=candidate_pairs,
        emb=emb,
        model=model,
    )
    selected_rows = _select_top_prediction_rows(
        scored_candidates=scored_candidates,
        predict_k=int(body.predictK),
        prob_threshold=float(body.probThreshold),
    )
    selected_triples = _rows_to_triples(selected_rows)
    selected_keys = {
        _pair_key(str(row["s"]), str(row["t"]))
        for row in selected_rows
    }
    artifact_rows = [
        {
            "s": key[0],
            "t": key[1],
            "pairKey": f"{key[0]}|{key[1]}",
            "probability": float(row.get("probability", 0.0) or 0.0),
            "rank": rank,
            "selected": key in selected_keys,
        }
        for rank, row in enumerate(scored_candidates, start=1)
        for key in [_pair_key(str(row["s"]), str(row["t"]))]
    ]
    record_candidate_scores(
        base_db=base,
        storage_representation=SINGLE_DB_MULTI_EDGE,
        timestep=new_level,
        embedding_family=fam,
        prediction_model=prediction_model,
        scored_candidates=artifact_rows,
    )
    rel_written = _write_predicted_edges_single_db(
        dbname=base,
        family=fam,
        prediction_model=prediction_model,
        triples=selected_triples,
        level=new_level,
        ts=new_ts,
    )
    count_above = sum(
        1
        for row in scored_candidates
        if float(row.get("probability", 0.0) or 0.0)
        >= float(body.probThreshold)
    )

    return {
        "ok": True,
        "baseDatabase": base,
        "storageRepresentation": SINGLE_DB_MULTI_EDGE,
        "embeddingFamily": fam,
        "predictionModel": prediction_model,
        "predictionModelDisplayName": prediction_model_display_name(
            prediction_model
        ),
        "targetPredictedDb": None,
        "materializedInPlace": True,
        "embeddingProperty": prop,
        "splitResult": split_result,
        "prevPredictedLevel": prev_level,
        "prevTimestamp": prev_family_ts,
        "newPredictedLevel": new_level,
        "newTimestamp": new_ts,
        "trainPairs": npos_tr,
        "negTrainPairs": nneg_tr,
        "embeddingInfo": emb_info,
        "trainingPolicy": training_policy,
        "prediction": {
            "requestedK": int(body.predictK),
            "candidateMultiplier": int(body.candidateMultiplier),
            "probThreshold": float(body.probThreshold),
            "totalCandidatesScored": len(scored_candidates),
            "candidatesAboveThreshold": count_above,
            "selectedNewPairs": len(selected_triples),
            "relationshipsWritten": rel_written,
        },
    }


def _compute_level_and_ts_single_db(
    dbname: str,
    family: str,
    prediction_model: str,
) -> Tuple[int, int]:
    model_key = normalize_prediction_model(prediction_model)
    rows = run_data(
        """
        MATCH ()-[r]-()
        WHERE coalesce(r.edgeOrigin, '') = 'predicted'
          AND coalesce(r.embeddingFamily, '') = $family
          AND coalesce(
                r.predictionModel,
                'logistic_regression'
              ) = $predictionModel
        WITH
          toInteger(
            coalesce(
              r.predicted_timestep,
              r.predicted,
              0
            )
          ) AS lvl,
          toFloat(
            coalesce(
              r.lastTimestamp,
              r.timestamp,
              r.predictedTimestamp,
              0
            )
          ) AS ts
        RETURN
          coalesce(max(lvl), 0) AS maxPredicted,
          coalesce(max(ts), 0) AS maxTimestamp
        """,
        {
            "family": family,
            "predictionModel": model_key,
        },
        db=dbname,
    )
    if not rows:
        return 0, 0
    return (
        int(rows[0].get("maxPredicted", 0) or 0),
        int(float(rows[0].get("maxTimestamp", 0) or 0)),
    )



def _write_predicted_edges_single_db(
    dbname: str,
    family: str,
    prediction_model: str,
    triples: List[Tuple[str, str, float]],
    level: int,
    ts: int,
) -> int:
    if not triples:
        return 0

    rel_type = predicted_rel_type(
        family,
        level=level,
        prediction_model=prediction_model,
    )

    rows = run_data(
        f"""
        UNWIND $triples AS row
        MATCH (s) WHERE elementId(s) = row.s
        MATCH (t) WHERE elementId(t) = row.t

        CREATE (s)-[r:`{rel_type}`]->(t)
        SET r.edgeOrigin = 'predicted',
            r.embeddingFamily = $family,
            r.predictionModel = $predictionModel,
            r.predictionModelDisplayName = $predictionModelDisplayName,
            r.predicted = $level,
            r.predicted_timestep = $level,
            r.probability = row.p,
            r.timestamp = $ts,
            r.predictedTimestamp = $ts,
            r.firstTimestamp = $ts,
            r.lastTimestamp = $ts,
            r.timestamps = [$ts],
            r.interactionCount = 1,
            r.storageRepresentation = 'single_db_multi_edge'

        RETURN count(r) AS relCount
        """,
        {
            "triples": [{"s": s, "t": t, "p": float(p)} for s, t, p in triples],
            "family": family,
            "level": int(level),
            "ts": int(ts),
            "predictionModel": normalize_prediction_model(prediction_model),
            "predictionModelDisplayName": prediction_model_display_name(prediction_model),
        },
        db=dbname,
    )

    return int(rows[0]["relCount"]) if rows else 0