from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import sys
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..db import run_data, current_database
from ..storage_repr import (
    SEPARATE_DBS,
    SINGLE_DB_MULTI_EDGE,
    normalize_storage_representation,
    predicted_rel_type,
)

from ..prediction_timeline import (
    EMBEDDING_FAMILIES,
    predicted_db_name,
    timeline_key,
    valid_timeline_combinations,
    validate_timeline,
)

from ..prediction_artifacts import get_or_create_candidate_pool, record_candidate_scores, update_family_evaluation_artifact
from ..lp_model_state import save_lp_learning_state, load_lp_learning_state, prediction_model_display_name, prediction_model_algorithm_name
from .. import tgn_adapter
from . import lp as lp_mod
from . import split as split_mod
from .lp_iter import _with_db_bound

router = APIRouter()

TGN_FAMILY = "TGN"
TGN_MODEL = "tgn_direct"


def _tgn_predicted_db_name(base_db: str) -> str:
    return predicted_db_name(
        base_db,
        TGN_FAMILY,
        TGN_MODEL,
    )


def _dbg(tag: str, **fields: Any) -> None:
    try:
        db = current_database()
    except Exception:
        db = "<unknown>"
    msg = " ".join(f"{k}={v!r}" for k, v in fields.items())
    print(f"[TGN_DIRECT DEBUG] {tag} db={db} {msg}", file=sys.stderr, flush=True)

def _cleanup_tgn_split_artifacts() -> None:
    try:
        split_mod._delete_all_split_rels()
    except Exception:
        pass

    try:
        split_mod._delete_all_split_endpoint_labels()
    except Exception:
        pass

class TGNDirectBody(BaseModel):
    trainGraphName: str = "trainGraph"
    valGraphName: Optional[str] = "valGraph"
    testGraphName: Optional[str] = "testGraph"
    storageRepresentation: str = SINGLE_DB_MULTI_EDGE
    negativeRatio: float = Field(default=1.0, ge=0.1, le=10.0)
    predictK: int = Field(default=100, ge=1)
    candidateMultiplier: int = Field(default=20, ge=2, le=200)
    probThreshold: float = Field(default=0.50, ge=0.0, le=1.0)
    evaluationThreshold: float = Field(default=0.50, ge=0.0, le=1.0)
    outputGraphName: str = "predictedGraph"
    force_new_canonical_pool: bool = False

    nEpoch: int = Field(default=5, ge=1, le=200)
    batchSize: int = Field(default=200, ge=1, le=100000)
    numNeighbors: int = Field(default=10, ge=1, le=1000)
    nodeDimension: int = Field(default=100, ge=2, le=2048)
    memoryDimension: int = Field(default=100, ge=2, le=2048)
    messageDimension: int = Field(default=100, ge=2, le=2048)
    learningRate: float = Field(default=0.0001, ge=0.000001, le=1.0)
    nHeads: int = Field(default=2, ge=1, le=16)
    nLayers: int = Field(default=1, ge=1, le=8)
    dropout: float = Field(default=0.10, ge=0.0, le=0.90)
    useMemory: bool = True

class TGNDirectIterateBody(BaseModel):
    storageRepresentation: str = SINGLE_DB_MULTI_EDGE
    predictK: int = Field(default=100, ge=1)
    candidateMultiplier: int = Field(default=20, ge=2, le=200)
    probThreshold: float = Field(default=0.50, ge=0.0, le=1.0)
    negativeRatio: float = Field(default=1.0, ge=0.1, le=10.0)
    retrainingInterval: int = Field(default=1, ge=0)

    nEpoch: int = Field(default=5, ge=1, le=200)
    batchSize: int = Field(default=200, ge=1, le=100000)
    numNeighbors: int = Field(default=10, ge=1, le=1000)
    nodeDimension: int = Field(default=100, ge=2, le=2048)
    memoryDimension: int = Field(default=100, ge=2, le=2048)
    messageDimension: int = Field(default=100, ge=2, le=2048)
    learningRate: float = Field(default=0.0001, ge=0.000001, le=1.0)
    nHeads: int = Field(default=2, ge=1, le=16)
    nLayers: int = Field(default=1, ge=1, le=8)
    dropout: float = Field(default=0.10, ge=0.0, le=0.90)
    useMemory: bool = True

def _canonical(a: Any, b: Any) -> Tuple[str, str]:
    sa = str(a)
    sb = str(b)
    return (sa, sb) if sa < sb else (sb, sa)


def _select(scored: List[Dict[str, Any]], predict_k: int, threshold: float) -> List[Dict[str, Any]]:
    above = [r for r in scored if float(r.get("probability", 0.0) or 0.0) >= float(threshold)]
    return above[: int(predict_k)] if len(above) >= int(predict_k) else scored[: int(predict_k)]


def _rows_to_triples(rows: List[Dict[str, Any]]) -> List[Tuple[str, str, float]]:
    return [(str(r["s"]), str(r["t"]), float(r.get("probability", 0.0) or 0.0)) for r in rows]


def _artifact_rows(scored: List[Dict[str, Any]], selected: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected_keys = {_canonical(r["s"], r["t"]) for r in selected}
    rows = []
    for rank, r in enumerate(scored, start=1):
        s, t = _canonical(r["s"], r["t"])
        rows.append({
            "s": s,
            "t": t,
            "pairKey": f"{s}|{t}",
            "probability": float(r.get("probability", 0.0) or 0.0),
            "rank": int(rank),
            "selected": bool((s, t) in selected_keys),
        })
    return rows


def _pairs_from_tgn_data(state: tgn_adapter.TGNState, data: tgn_adapter.TGNEventData) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    seen: set[Tuple[str, str]] = set()
    for s_idx, t_idx in zip(data.sources.tolist(), data.destinations.tolist()):
        s = state.idx_to_node_id.get(int(s_idx))
        t = state.idx_to_node_id.get(int(t_idx))
        if s is None or t is None:
            continue
        key = _canonical(s, t)
        if key not in seen:
            seen.add(key)
            pairs.append(key)
    return pairs


def _write_predicted_edges_single_db(
    *,
    dbname: str,
    family: str,
    triples: List[Tuple[str, str, float]],
    level: int,
    ts: int,
    storage_representation: str = SINGLE_DB_MULTI_EDGE,
) -> int:
    if not triples:
        return 0

    rel_type = predicted_rel_type(
        family,
        level=level,
        prediction_model=TGN_MODEL,
    )
    rows = run_data(
        f"""
        UNWIND $triples AS row
        MATCH (s)
        WHERE toString(coalesce(s.eid, elementId(s))) = toString(row.s)
        MATCH (t)
        WHERE toString(coalesce(t.eid, elementId(t))) = toString(row.t)
        CREATE (s)-[r:`{rel_type}`]->(t)
        SET r.edgeOrigin = 'predicted',
            r.embeddingFamily = $family,
            r.predictionModel = $predictionModel,
            r.predictionModelDisplayName = $predictionModelDisplayName,
            r.predicted = $level,
            r.predicted_timestep = $level,
            r.probability = row.p,
            r.predictedTimestamp = $ts,
            r.timestamp = $ts,
            r.firstTimestamp = $ts,
            r.lastTimestamp = $ts,
            r.timestamps = [$ts],
            r.interactionCount = 1,
            r.storageRepresentation = $storageRepresentation
        RETURN count(r) AS relCount
        """,
        {
            "triples": [
                {"s": s, "t": t, "p": float(p)}
                for s, t, p in triples
            ],
            "family": family,
            "level": int(level),
            "ts": int(ts),
            "predictionModel": TGN_MODEL,
            "predictionModelDisplayName": prediction_model_display_name(
                TGN_MODEL
            ),
            "storageRepresentation": normalize_storage_representation(
                storage_representation
            ),
        },
        db=dbname,
    )
    return int(rows[0]["relCount"]) if rows else 0



def _current_family_level_and_ts(
    dbname: str,
    family: str = TGN_FAMILY,
) -> Tuple[int, int]:
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
            "predictionModel": TGN_MODEL,
        },
        db=dbname,
    )
    if not rows:
        return 0, 0
    return (
        int(rows[0].get("maxPredicted", 0) or 0),
        int(float(rows[0].get("maxTimestamp", 0) or 0)),
    )


def _load_tgn_timeline_pairs(dbname: str) -> set[Tuple[str, str]]:
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
            "family": TGN_FAMILY,
            "predictionModel": TGN_MODEL,
        },
        db=dbname,
    )
    return {_canonical(row["s"], row["t"]) for row in rows}


def _load_observed_pairs(dbname: str) -> set[Tuple[str, str]]:
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
    return {_canonical(row["s"], row["t"]) for row in rows}


def _prediction_timestamp_for_level(base_db: str, level: int) -> int:
    observed_max_ts = lp_mod._observed_database_max_timestamp(base_db)
    return int(observed_max_ts) + int(level)

def _train_state_for_current_split(
    dbname: str,
    body: Any,
) -> Tuple[tgn_adapter.TGNState, Dict[str, Any], Dict[str, tgn_adapter.TGNEventData]]:
    try:
        return tgn_adapter.train_tgn(
            dbname=dbname,
            family=TGN_FAMILY,
            n_epoch=int(getattr(body, "nEpoch", 5)),
            batch_size=int(getattr(body, "batchSize", 200)),
            n_neighbors=int(getattr(body, "numNeighbors", 10)),
            node_dim=int(getattr(body, "nodeDimension", 100)),
            memory_dim=int(getattr(body, "memoryDimension", 100)),
            message_dim=int(getattr(body, "messageDimension", 100)),
            learning_rate=float(getattr(body, "learningRate", 0.0001)),
            n_heads=int(getattr(body, "nHeads", 2)),
            n_layers=int(getattr(body, "nLayers", 1)),
            dropout=float(getattr(body, "dropout", 0.10)),
            use_memory=bool(getattr(body, "useMemory", True)),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"TGN training failed: {e}")


def _score_for_db(
    state: tgn_adapter.TGNState,
    pairs: List[Tuple[str, str]],
    ts: int,
    dbname: str,
    use_runtime_map: bool,
) -> List[Dict[str, Any]]:
    id_map = tgn_adapter.load_runtime_to_stable_id_map(dbname) if use_runtime_map else None
    return tgn_adapter.score_pairs(state, pairs, timestamp=float(ts), runtime_to_stable=id_map)


def _eval_block(
    state: tgn_adapter.TGNState,
    pos_pairs: List[Tuple[str, str]],
    ts: int,
    negative_ratio: float,
) -> Optional[Dict[str, Any]]:
    if not pos_pairs:
        return None
    neg_pairs = lp_mod._sample_negatives(max(1, int(round(len(pos_pairs) * float(negative_ratio)))), as_of_ts=float(ts))
    return tgn_adapter.evaluate_pairs(state, pos_pairs, timestamp=float(ts), negative_pairs=neg_pairs)


@router.post("/run")
def tgn_direct_run(body: TGNDirectBody) -> Dict[str, Any]:
    storage_mode = normalize_storage_representation(body.storageRepresentation)
    if storage_mode not in {SINGLE_DB_MULTI_EDGE, SEPARATE_DBS}:
        raise HTTPException(
            status_code=400,
            detail="Unsupported storage representation for TGN.",
        )

    base = current_database()
    started = time.perf_counter()

    try:
        state, training_meta, data_parts = _train_state_for_current_split(
            base,
            body,
        )
    except Exception:
        _cleanup_tgn_split_artifacts()
        raise

    pos_train = _pairs_from_tgn_data(state, data_parts["train"])
    pos_val = _pairs_from_tgn_data(state, data_parts["validation"])
    pos_test = _pairs_from_tgn_data(state, data_parts["test"])
    train_cutoff = lp_mod._graph_snapshot_ts(
        body.trainGraphName,
        time_prop="firstTimestamp",
    )
    val_cutoff = (
        lp_mod._graph_snapshot_ts(
            body.valGraphName,
            time_prop="firstTimestamp",
        )
        if body.valGraphName
        else train_cutoff
    )
    test_cutoff = (
        lp_mod._graph_snapshot_ts(
            body.testGraphName,
            time_prop="firstTimestamp",
        )
        if body.testGraphName
        else train_cutoff
    )
    mtr = _eval_block(
        state,
        pos_train,
        int(train_cutoff or 0),
        float(body.negativeRatio),
    )
    mval = (
        _eval_block(
            state,
            pos_val,
            int(val_cutoff or train_cutoff or 0),
            float(body.negativeRatio),
        )
        if pos_val
        else None
    )
    mts = (
        _eval_block(
            state,
            pos_test,
            int(test_cutoff or train_cutoff or 0),
            float(body.negativeRatio),
        )
        if pos_test
        else None
    )
    diag_scored = _score_for_db(
        state,
        pos_test,
        int(test_cutoff or train_cutoff or 0),
        base,
        use_runtime_map=False,
    )
    diag = [
        row
        for row in diag_scored
        if float(row.get("probability", 0.0) or 0.0)
        >= float(body.evaluationThreshold)
    ]
    diag_pairs = [(str(row["s"]), str(row["t"])) for row in diag]
    prediction_timestep = 1
    predicted_ts = _prediction_timestamp_for_level(base, prediction_timestep)

    if storage_mode == SINGLE_DB_MULTI_EDGE:
        timeline_id = timeline_key(TGN_FAMILY, TGN_MODEL)
        lp_mod._delete_single_db_timeline_predictions(
            dbname=base,
            embedding_family=TGN_FAMILY,
            prediction_model=TGN_MODEL,
        )
        lp_mod._clear_prediction_memory_for_family(timeline_id)

    for state_name in ("initial", "latest"):
        save_lp_learning_state(
            base_db=base,
            storage_representation=storage_mode,
            embedding_family=TGN_FAMILY,
            embedding_property="TGN",
            model=state,
            trained_at_timestep=prediction_timestep,
            prediction_model=TGN_MODEL,
            state_name=state_name,
            embeddings={},
            metadata={
                "source": "twitter_research_tgn_initial",
                **training_meta,
                "predictK": int(body.predictK),
                "probThreshold": float(body.probThreshold),
                "evaluationThreshold": float(body.evaluationThreshold),
                "predictionModel": TGN_MODEL,
                "predictionModelDisplayName": prediction_model_display_name(
                    TGN_MODEL
                ),
                "predictionModelAlgorithm": prediction_model_algorithm_name(
                    TGN_MODEL
                ),
            },
        )

    pool_size = max(
        int(body.predictK) * 20,
        int(body.predictK) + 50,
        1000,
    )
    generated = lp_mod._build_canonical_candidate_pairs(
        pool_size=pool_size,
        version=prediction_timestep,
    )
    shared_pool = get_or_create_candidate_pool(
        base_db=base,
        storage_representation=storage_mode,
        timestep=prediction_timestep,
        generated_candidates=generated,
        pool_size=pool_size,
        force_new=False,
    )
    gt_pairs = {_canonical(s, t) for s, t in pos_train + pos_val + pos_test}
    candidate_pairs: List[Tuple[str, str]] = []
    seen: set[Tuple[str, str]] = set()
    for row in shared_pool:
        pair = _canonical(row["s"], row["t"])
        if pair in seen or pair in gt_pairs:
            continue
        seen.add(pair)
        candidate_pairs.append(pair)

    scored = _score_for_db(
        state,
        candidate_pairs,
        predicted_ts,
        base,
        use_runtime_map=False,
    )
    selected = _select(scored, int(body.predictK), float(body.probThreshold))
    triples = _rows_to_triples(selected)
    record_candidate_scores(
        base_db=base,
        storage_representation=storage_mode,
        timestep=prediction_timestep,
        prediction_model=TGN_MODEL,
        embedding_family=TGN_FAMILY,
        scored_candidates=_artifact_rows(scored, selected),
    )

    if storage_mode == SINGLE_DB_MULTI_EDGE:
        rels = _write_predicted_edges_single_db(
            dbname=base,
            family=TGN_FAMILY,
            triples=triples,
            level=prediction_timestep,
            ts=predicted_ts,
            storage_representation=storage_mode,
        )
        materialized_db = base
        materialized_in_place = True
    else:
        lp_mod._project_predicted_graph(
            body.outputGraphName,
            triples,
            original_edges_cutoff_ts=None,
            predicted_ts=predicted_ts,
        )
        materialized_db = lp_mod._materialize_prediction_db(
            gname=body.outputGraphName,
            embedding_suffix=TGN_FAMILY,
            embedding_family=TGN_FAMILY,
            prediction_model=TGN_MODEL,
            predicted_level=prediction_timestep,
            predicted_triples=triples,
            predicted_ts=predicted_ts,
        )
        rels = len(triples)
        materialized_in_place = False

    update_family_evaluation_artifact(
        base_db=base,
        storage_representation=storage_mode,
        prediction_model=TGN_MODEL,
        embedding_family=TGN_FAMILY,
        metrics={
            "train": mtr,
            "validation": mval,
            "test": mts,
            "predictionModel": TGN_MODEL,
            "predictionModelDisplayName": prediction_model_display_name(
                TGN_MODEL
            ),
        },
        eval_pairs=diag_pairs,
        materialized_pairs=[(s, t) for s, t, _ in triples],
        ground_truth_pairs=pos_test,
    )
    elapsed = time.perf_counter() - started
    _cleanup_tgn_split_artifacts()

    return {
        "ok": True,
        "train": mtr,
        "validation": mval,
        "test": mts,
        "model": {
            "algorithm": prediction_model_algorithm_name(TGN_MODEL),
            "predictionModel": TGN_MODEL,
            "predictionModelDisplayName": prediction_model_display_name(
                TGN_MODEL
            ),
            "embeddingProperty": "TGN",
            "embeddingFamily": TGN_FAMILY,
            "features": "TemporalGraphNetwork(memory+messages+time_encoding+temporal_attention)",
        },
        "predicted": {
            "storageRepresentation": storage_mode,
            "materializedDatabase": materialized_db,
            "materializedInPlace": materialized_in_place,
            "embeddingFamily": TGN_FAMILY,
            "predictionModel": TGN_MODEL,
            "predictionModelDisplayName": prediction_model_display_name(
                TGN_MODEL
            ),
            "predictedLevel": prediction_timestep,
            "predictedTimestamp": predicted_ts,
            "keptAboveThreshold": len(triples),
            "threshold": float(body.probThreshold),
            "materialized": {"relationshipsWritten": rels},
            "diagnosticPredictions": {
                "pairs": len(diag_pairs),
                "triples": len(diag),
            },
            "materializedPredictions": {
                "pairs": len(triples),
                "triples": len(triples),
            },
            "timings": {"tgn_seconds": round(elapsed, 6)},
        },
    }



@router.post("/iterate")
def iterate_tgn_direct(body: TGNDirectIterateBody) -> Dict[str, Any]:
    storage_mode = normalize_storage_representation(body.storageRepresentation)
    if storage_mode not in {SINGLE_DB_MULTI_EDGE, SEPARATE_DBS}:
        raise HTTPException(
            status_code=400,
            detail="Unsupported storage representation for TGN iteration.",
        )

    base = current_database()
    target_db = (
        base
        if storage_mode == SINGLE_DB_MULTI_EDGE
        else _tgn_predicted_db_name(base)
    )

    if storage_mode == SEPARATE_DBS:
        rows = run_data(
            "SHOW DATABASE $db YIELD name RETURN name",
            {"db": target_db},
            db="system",
        )
        if not rows:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Predicted database '{target_db}' was not found. "
                    "Run the initial TGN prediction first."
                ),
            )

    prev_level, prev_ts = _current_family_level_and_ts(
        target_db,
        TGN_FAMILY,
    )
    new_level = prev_level + 1
    new_ts = _prediction_timestamp_for_level(base, new_level)
    retraining_interval = max(0, int(body.retrainingInterval))
    state_name = "initial" if retraining_interval == 0 else "latest"
    saved_state = load_lp_learning_state(
        base_db=base,
        storage_representation=storage_mode,
        embedding_family=TGN_FAMILY,
        prediction_model=TGN_MODEL,
        state_name=state_name,
    )
    has_state = bool(saved_state and saved_state.get("model") is not None)
    should_retrain = not has_state or retraining_interval > 0

    if should_retrain:
        with _with_db_bound(split_mod, target_db), _with_db_bound(
            lp_mod,
            target_db,
        ):
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
                    storageRepresentation=storage_mode,
                    includePredictedFamily=TGN_FAMILY,
                    includePredictedModel=TGN_MODEL,
                    includePredictedBeforeLevel=new_level,
                    keepTemporarySplitRelationships=True,
                )
            )
            try:
                state, training_meta, data_parts = _train_state_for_current_split(
                    target_db,
                    body,
                )
            except Exception:
                _cleanup_tgn_split_artifacts()
                raise
            finally:
                _cleanup_tgn_split_artifacts()

        saved = save_lp_learning_state(
            base_db=base,
            storage_representation=storage_mode,
            embedding_family=TGN_FAMILY,
            embedding_property="TGN",
            model=state,
            trained_at_timestep=new_level,
            prediction_model=TGN_MODEL,
            state_name="latest",
            embeddings={},
            metadata={
                "source": "twitter_research_tgn_iterative",
                **training_meta,
                "predictK": int(body.predictK),
                "probThreshold": float(body.probThreshold),
                "retrainingInterval": retraining_interval,
                "predictionModel": TGN_MODEL,
                "predictionModelDisplayName": prediction_model_display_name(
                    TGN_MODEL
                ),
                "predictionModelAlgorithm": prediction_model_algorithm_name(
                    TGN_MODEL
                ),
            },
        )
        npos_tr = int(training_meta.get("trainPairs", 0) or 0)
        nneg_tr = 0
        training_policy = {
            "retrainingInterval": retraining_interval,
            "didRetrainModel": True,
            "savedLearningState": saved,
        }
    else:
        state = saved_state["model"]
        metadata = saved_state.get("metadata") or {}
        npos_tr = int(metadata.get("trainPairs", 0) or 0)
        nneg_tr = int(metadata.get("negTrainPairs", 0) or 0)
        split_result = {
            "reused": True,
            "reason": "tgn_learning_state_reused",
        }
        training_policy = {
            "retrainingInterval": retraining_interval,
            "didRetrainModel": False,
            "loadedStateName": state_name,
        }

    pool_size = max(
        int(body.predictK) * 20,
        int(body.predictK) + 50,
        1000,
    )
    generated = lp_mod._build_canonical_candidate_pairs(
        pool_size=pool_size * 2,
        version=new_level,
    )
    shared_pool = get_or_create_candidate_pool(
        base_db=base,
        storage_representation=storage_mode,
        timestep=new_level,
        generated_candidates=generated,
        pool_size=pool_size,
        force_new=False,
    )
    already_predicted = _load_tgn_timeline_pairs(target_db)
    observed_pairs = _load_observed_pairs(base)
    candidate_pairs: List[Tuple[str, str]] = []
    seen: set[Tuple[str, str]] = set()
    for row in shared_pool:
        pair = _canonical(row["s"], row["t"])
        if pair in seen or pair in already_predicted or pair in observed_pairs:
            continue
        seen.add(pair)
        candidate_pairs.append(pair)

    scored = _score_for_db(
        state,
        candidate_pairs,
        new_ts,
        target_db,
        use_runtime_map=False,
    )
    selected = _select(scored, int(body.predictK), float(body.probThreshold))
    triples = _rows_to_triples(selected)
    record_candidate_scores(
        base_db=base,
        storage_representation=storage_mode,
        timestep=new_level,
        prediction_model=TGN_MODEL,
        embedding_family=TGN_FAMILY,
        scored_candidates=_artifact_rows(scored, selected),
    )
    rels = _write_predicted_edges_single_db(
        dbname=target_db,
        family=TGN_FAMILY,
        triples=triples,
        level=new_level,
        ts=new_ts,
        storage_representation=storage_mode,
    )
    count_above = sum(
        1
        for row in scored
        if float(row.get("probability", 0.0) or 0.0)
        >= float(body.probThreshold)
    )
    split_payload = (
        split_result
        if isinstance(split_result, dict)
        else split_result.model_dump()
        if hasattr(split_result, "model_dump")
        else {}
    )

    return {
        "ok": True,
        "baseDatabase": base,
        "storageRepresentation": storage_mode,
        "targetPredictedDb": (
            None if storage_mode == SINGLE_DB_MULTI_EDGE else target_db
        ),
        "materializedInPlace": storage_mode == SINGLE_DB_MULTI_EDGE,
        "embeddingFamily": TGN_FAMILY,
        "predictionModel": TGN_MODEL,
        "predictionModelDisplayName": prediction_model_display_name(TGN_MODEL),
        "embeddingProperty": "TGN",
        "splitResult": split_payload,
        "prevPredictedLevel": prev_level,
        "prevTimestamp": prev_ts,
        "newPredictedLevel": new_level,
        "newTimestamp": new_ts,
        "trainPairs": npos_tr,
        "negTrainPairs": nneg_tr,
        "embeddingInfo": {
            "directModel": True,
            "embeddingFamily": TGN_FAMILY,
            "tgn": True,
        },
        "trainingPolicy": training_policy,
        "prediction": {
            "requestedK": int(body.predictK),
            "candidateMultiplier": int(body.candidateMultiplier),
            "probThreshold": float(body.probThreshold),
            "totalCandidatesScored": len(scored),
            "candidatesAboveThreshold": count_above,
            "selectedNewPairs": len(triples),
            "relationshipsWritten": rels,
        },
    }

