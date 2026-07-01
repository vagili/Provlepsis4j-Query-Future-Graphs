from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Tuple
import re
import math
import random
import numpy as np
import sys
import time

from ..db import run, run_data, current_database
from ..storage_repr import (
    SEPARATE_DBS,
    SINGLE_DB_MULTI_EDGE,
    normalize_storage_representation,
    predicted_rel_type,
)
from ..prediction_artifacts import (
    get_or_create_candidate_pool,
    record_candidate_scores,
    update_family_evaluation_artifact,
    load_evaluation_artifacts,
)
from ..lp_model_state import (
    save_lp_learning_state,
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

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        accuracy_score,
        precision_recall_fscore_support,
    )
except Exception:
    LogisticRegression = None 
    make_pipeline = None
    StandardScaler = None

router = APIRouter()

# ------------------------ DEBUG HELPER ------------------------

def _dbg(tag: str, **fields: Any) -> None:

    try:
        db = current_database()
    except Exception:
        db = "<unknown>"
    msg = " ".join(f"{k}={v!r}" for k, v in fields.items())
    print(f"[LP DEBUG] {tag} db={db} {msg}", file=sys.stderr, flush=True)


# ------------------------ GLOBAL STATE ------------------------

_ACTIVE_CACHE_DB: Optional[str] = None
_PREDICTED_EDGES: Dict[str, List[Tuple[str, str]]] = {}  # e.g. {"FastRP": [...], "Node2Vec": [...]}
_EVAL_PREDICTED_EDGES: Dict[str, List[Tuple[str, str]]] = {} 
_CANON_POOL: List[Tuple[str, str]] = []
_CANON_SIG: Tuple[int, int] = (0, 0)  
_EMBED_KEYS = ("Node2Vec", "FastRP", "GraphSAGE", "HashGNN", "TGN")
_POOL_BY_KEY: Dict[str, List[Tuple[str, str]]] = {}    
_POOL_VERSION: int = 1 
_METRICS_BY_KEY: Dict[str, Dict[str, Any]] = {}  
_TEST_PAIRS_BY_GRAPH: Dict[str, List[Tuple[str, str]]] = {}

_DEBUG_INITIAL_BATCH: Dict[str, List[Tuple[str, str]]] = {}

# ------------------------
# Request/Response models
# ------------------------
class ResetSingleDbFamilyPredictionsBody(BaseModel):
    embeddingFamily: str

    predictionModel: str = Field(
        default="logistic_regression",
        description=(
            "Prediction model belonging to the timeline "
            "that should be reset."
        ),
    )

    deleteAllLevels: bool = Field(
        True,
        description=(
            "If true, delete all predicted levels for this "
            "embedding-family/model timeline. "
            "If false, delete only predicted level 1."
        ),
    )

class LPBody(BaseModel):
    # Name of the node embedding property
    embeddingProperty: str = Field(..., description="Name of the node embedding property")
    # Graph names already created by your split
    trainGraphName: str = "trainGraph"
    valGraphName: Optional[str] = None
    testGraphName: Optional[str] = "testGraph"
    # Negatives per positive (1.0 => 1:1)
    negativeRatio: float = Field(default=1.0, ge=0.1, le=10.0)
    # Top-K prediction settings
    predictK: int = Field(default=100, ge=1)
    candidateMultiplier: int = Field(default=20, ge=2, le=200)
    # Probability threshold — keep only predicted edges with prob >= this
    probThreshold: float = Field(default=0.50, ge=0.0, le=1.0)
    evaluationThreshold: float = Field(default=0.5, ge=0.0, le=1.0)
    # Output in-memory projection containing original + predicted edges
    outputGraphName: str = "predictedGraph"
    embeddingFamily: Optional[str] = None
    storageRepresentation: str = SEPARATE_DBS
    force_new_canonical_pool: bool = Field(
        False,
        description= ("If true, rebuild the negative candidate pool for this run"
        ),
    )
    predictionModel: str = Field(
        default="logistic_regression",
        description="Prediction model: logistic_regression, lasso_regression, or tgn_direct",
    )

class LPMetrics(BaseModel):
    count_pos: int
    count_neg: int
    auc: Optional[float] = None
    ap: Optional[float] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None

class LPResponse(BaseModel):
    ok: bool
    train: LPMetrics
    validation: Optional[LPMetrics] = None
    test: Optional[LPMetrics] = None
    model: Dict[str, Any]  
    predicted: Optional[Dict[str, Any]] = None  

# ------------------------
# Helpers
# ------------------------

_PROP_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

def _graph_snapshot_ts(gname: str, time_prop: str = "firstTimestamp") -> Optional[float]:
    rows = run_data(
        f"""
        CALL gds.graph.relationships.stream($g)
        YIELD sourceNodeId, targetNodeId
        WITH gds.util.asNode(sourceNodeId) AS sN, gds.util.asNode(targetNodeId) AS tN
        WITH elementId(sN) AS a, elementId(tN) AS b
        WITH CASE WHEN a < b THEN a ELSE b END AS s,
             CASE WHEN a < b THEN b ELSE a END AS t
        WITH DISTINCT s, t
        CALL {{
          WITH s, t
          MATCH (x)-[r]-(y)
          WHERE elementId(x) = s
            AND elementId(y) = t
            AND NOT type(r) STARTS WITH 'SPLIT_'
            AND coalesce(r.edgeOrigin, 'observed') = 'observed'
          RETURN coalesce(max(toFloat(r.`{time_prop}`)), max(toFloat(r.timestamp)), 0.0) AS ts
        }}
        RETURN coalesce(max(ts), 0.0) AS cutoff
        """,
        {"g": gname},
    )

    if not rows:
        return None

    return float(rows[0].get("cutoff", 0.0) or 0.0)

def _set_debug_initial_batch(emb_key: str, pairs: List[Tuple[str, str]]) -> None:

    sample = pairs[:100]
    _DEBUG_INITIAL_BATCH[emb_key] = list(sample)
    _dbg(
        "_set_debug_initial_batch",
        emb_key=emb_key,
        count=len(sample),
        edges=sample,
    )

def _make_lp_classifier(prediction_model: str):

    _assert_sklearn()
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

def _get_debug_initial_batch(emb_key: str) -> List[Tuple[str, str]]:

    batch = _DEBUG_INITIAL_BATCH.get(emb_key) or []
    _dbg(
        "_get_debug_initial_batch",
        emb_key=emb_key,
        count=len(batch),
    )
    return list(batch)



def _embedding_db_suffix(prop: str) -> str:
    s = (prop or "").lower()
    if "fastrp" in s:
        return "FastRP"
    if "node2vec" in s or "n2v" in s:
        return "Node2Vec"
    if "graphsage" in s or "sage" in s:
        return "GraphSAGE"
    if "hashgnn" in s or "hgnn" in s or ("hash" in s and "gnn" in s):
        return "HashGNN"
    if "tgn" in s:
        return "TGN"
    
    fallback = re.sub(r"[^A-Za-z0-9]+", "_", prop).strip("_")
    return fallback or "Embedding"

def _assert_sklearn():
    if LogisticRegression is None:
        _dbg("_assert_sklearn.missing")
        raise HTTPException(
            status_code=500,
            detail=(
                "scikit-learn is not installed in the backend image. "
                "Add it to requirements.txt (e.g., scikit-learn==1.4.*) and rebuild."
            ),
        )

def _sanitize_prop(name: str) -> str:
    if not _PROP_RE.match(name):
        _dbg("_sanitize_prop.invalid", name=name)
        raise HTTPException(status_code=400, detail=f"Invalid embedding property name: {name}")
    return name

def _stream_unique_pairs_from_gds(gname: str) -> List[Tuple[str, str]]:

    if not gname:
        _dbg("_stream_unique_pairs_from_gds.empty_name")
        return []
    _dbg("_stream_unique_pairs_from_gds.start", graph=gname)
    cypher = """
    CALL gds.graph.relationships.stream($g)
    YIELD sourceNodeId, targetNodeId
    WITH gds.util.asNode(sourceNodeId) AS sN, gds.util.asNode(targetNodeId) AS tN
    WITH elementId(sN) AS s, elementId(tN) AS t
    WITH CASE WHEN s < t THEN s ELSE t END AS s, CASE WHEN s < t THEN t ELSE s END AS t
    RETURN DISTINCT s, t
    """
    rows = run_data(cypher, {"g": gname})
    _dbg("_stream_unique_pairs_from_gds.done", graph=gname, pairs=len(rows))
    return [(r["s"], r["t"]) for r in rows]


def _sample_negatives(k: int, as_of_ts: Optional[float] = None) -> List[Tuple[str, str]]:
    if k <= 0:
        _dbg("_sample_negatives.skip", k=k)
        return []

    oversample = max(k * 3, 50)
    _dbg("_sample_negatives.start", k=k, oversample=oversample, as_of_ts=as_of_ts)

    cypher = """
    MATCH (n)
    WITH collect(elementId(n)) AS ids, count(n) AS N

    CALL {
      WITH ids, N
      UNWIND range(1, $oversample) AS _
      WITH ids, N,
           ids[toInteger(rand() * N)] AS a,
           ids[toInteger(rand() * N)] AS b
      WITH a, b
      WHERE a < b
      RETURN DISTINCT a AS s, b AS t
      LIMIT $oversample
    }

    WITH s, t
    WHERE NOT EXISTS {
      MATCH (sN)-[r]-(tN)
      WHERE elementId(sN) = s
        AND elementId(tN) = t
        AND coalesce(r.edgeOrigin, 'observed') = 'observed'
        AND (
          $asOfTs IS NULL
          OR coalesce(r.firstTimestamp, r.timestamp, 0) <= $asOfTs
        )
    }

    RETURN DISTINCT s, t
    LIMIT $k
    """

    rows = run_data(
        cypher,
        {
            "k": k,
            "oversample": oversample,
            "asOfTs": as_of_ts,
        },
    )

    _dbg("_sample_negatives.done", returned=len(rows))
    return [(r["s"], r["t"]) for r in rows]

def _db_seed(dbname: str, version: int = 1) -> int:
    seed = abs(hash((dbname, "lp_pool_v1", version))) % (2**31 - 1)
    _dbg("_db_seed", dbname=dbname, version=version, seed=seed)
    return seed


def _database_max_timestamp(dbname: Optional[str] = None) -> int:
    rows = run_data(
        """
        MATCH ()-[r]-()
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

def _existing_edge_pairs_limit(limit: int = 0) -> set[Tuple[str, str]]:

    _dbg("_existing_edge_pairs_limit.noop", limit=limit)
    return set()

def _build_canonical_candidate_pairs(pool_size: int, version: int = 1) -> List[Tuple[str, str]]:

    base = _ensure_cache_for_current_db()
    seed = _db_seed(base, version)
    rng = random.Random(seed)
    _dbg("_build_canonical_candidate_pairs.start", base_db=base, pool_size=pool_size, version=version)

    ids_rows = run_data("MATCH (n) RETURN elementId(n) AS eid ORDER BY eid", db=base)
    ids = [r["eid"] for r in ids_rows]
    N = len(ids)
    _dbg("_build_canonical_candidate_pairs.nodes", base_db=base, N=N)
    if N < 2:
        _dbg("_build_canonical_candidate_pairs.too_few_nodes", base_db=base)
        return []

    want = max(1, int(pool_size))
    target_raw = max(want * 3, want + 100)

    seen_idx: set[Tuple[int, int]] = set()
    raw_pairs: List[Tuple[str, str]] = []

    while len(raw_pairs) < target_raw:
        i = rng.randrange(N)
        j = rng.randrange(N)
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        key = (a, b)
        if key in seen_idx:
            continue
        seen_idx.add(key)
        raw_pairs.append((ids[a], ids[b]))

    _dbg("_build_canonical_candidate_pairs.raw_pairs", base_db=base, generated=len(raw_pairs))

    rows = run_data(
        """
        UNWIND $pairs AS P
        WITH P.s AS s, P.t AS t
        WHERE NOT EXISTS {
            MATCH (sN)-[r]-(tN)
            WHERE elementId(sN) = s
            AND elementId(tN) = t
            AND coalesce(r.edgeOrigin, 'observed') = 'observed'
        }
        RETURN s, t
        LIMIT $limit
        """,
        {"pairs": [{"s": s, "t": t} for (s, t) in raw_pairs], "limit": want},
        db=base,
    )
    _dbg("_build_canonical_candidate_pairs.filtered", base_db=base, kept=len(rows))
    return [(r["s"], r["t"]) for r in rows]

def _load_single_db_timeline_pairs(
    *,
    dbname: str,
    embedding_family: str,
    prediction_model: str,
) -> set[tuple[str, str]]:
    rows = run_data(
        """
        MATCH (a)-[r]-(b)
        WHERE coalesce(r.edgeOrigin, '') = 'predicted'
          AND coalesce(r.embeddingFamily, '') = $family
          AND coalesce(
                r.predictionModel,
                'logistic_regression'
              ) = $predictionModel

        WITH elementId(a) AS aId,
             elementId(b) AS bId

        RETURN DISTINCT
          CASE WHEN aId < bId THEN aId ELSE bId END AS s,
          CASE WHEN aId < bId THEN bId ELSE aId END AS t
        """,
        {
            "family": embedding_family,
            "predictionModel": normalize_prediction_model(
                prediction_model
            ),
        },
        db=dbname,
    )

    return {
        _pair_key(row["s"], row["t"])
        for row in rows
    }

def _get_canonical_pool(
    min_needed: int,
    max_needed: int,
    version: int = 1,
) -> List[Tuple[str, str]]:

    base = _ensure_cache_for_current_db()
    global _CANON_POOL, _CANON_SIG

    sig = (max_needed, version)
    _dbg(
        "_get_canonical_pool.enter",
        base_db=base,
        min_needed=min_needed,
        max_needed=max_needed,
        sig=sig,
        current_sig=_CANON_SIG,
        current_len=len(_CANON_POOL),
    )

    if _CANON_POOL and _CANON_SIG == sig and len(_CANON_POOL) >= min_needed:
        _dbg("_get_canonical_pool.cache_hit", base_db=base, pool_len=len(_CANON_POOL))
        return _CANON_POOL

    _dbg("_get_canonical_pool.rebuild", base_db=base, old_sig=_CANON_SIG, new_sig=sig)
    pool = _build_canonical_candidate_pairs(pool_size=max_needed, version=version)
    _CANON_POOL = pool
    _CANON_SIG = sig
    _dbg("_get_canonical_pool.built", base_db=base, pool_len=len(pool), sig=_CANON_SIG)
    return pool


def _ensure_pool_for_key(emb_key: str, ensure_remaining: int, version: int = 1) -> List[Tuple[str, str]]:

    base = _ensure_cache_for_current_db()
    _dbg("_ensure_pool_for_key.enter", base_db=base, emb_key=emb_key, ensure_remaining=ensure_remaining)

    max_needed = max(ensure_remaining * 3, ensure_remaining + 200)
    canon = _get_canonical_pool(min_needed=ensure_remaining, max_needed=max_needed, version=version)

    cur = _POOL_BY_KEY.get(emb_key) or []
    if not cur:
        _POOL_BY_KEY[emb_key] = list(canon)
        _dbg("_ensure_pool_for_key.init", base_db=base, emb_key=emb_key, pool_len=len(_POOL_BY_KEY[emb_key]))
        return _POOL_BY_KEY[emb_key]

    have = set(cur)
    appended = False
    for p in canon:
        if p not in have:
            cur.append(p)
            appended = True

    if appended:
        _POOL_BY_KEY[emb_key] = cur
        _dbg("_ensure_pool_for_key.extend", base_db=base, emb_key=emb_key, pool_len=len(cur))

    return cur

def _ensure_cache_for_current_db() -> str:

    global _ACTIVE_CACHE_DB, _CANON_POOL, _CANON_SIG, _POOL_BY_KEY, _PREDICTED_EDGES, _EVAL_PREDICTED_EDGES, _METRICS_BY_KEY, _TEST_PAIRS_BY_GRAPH

    db = current_database()
    if db != _ACTIVE_CACHE_DB:
        _dbg("_ensure_cache_for_current_db.switch", old=_ACTIVE_CACHE_DB, new=db)
        _ACTIVE_CACHE_DB = db
        _CANON_POOL = []
        _CANON_SIG = (0, 0)
        _POOL_BY_KEY = {}
        _PREDICTED_EDGES.clear()
        _EVAL_PREDICTED_EDGES.clear()
        _METRICS_BY_KEY = {}
        _TEST_PAIRS_BY_GRAPH = {}
    return db


def _predict_from_per_alg_pool(
    emb_key: str,
    emb: Dict[str, np.ndarray],
    model: LogisticRegression,
    k: int,
    threshold: float,
    candidate_multiplier: int,
    avoid_set: Optional[set[Tuple[str, str]]] = None,
) -> List[Tuple[str, str, float]]:

    want = max(1, int(k))
    thr = float(threshold)
    avoid = set(avoid_set or set())
    _dbg(
        "_predict_from_per_alg_pool.enter",
        emb_key=emb_key,
        want=want,
        thr=thr,
        candidate_multiplier=candidate_multiplier,
        avoid_size=len(avoid),
    )

    remaining_target = max(want * candidate_multiplier, want + 50)
    pool = _ensure_pool_for_key(emb_key, ensure_remaining=remaining_target, version=_POOL_VERSION)
    _dbg("_predict_from_per_alg_pool.pool", emb_key=emb_key, pool_len=len(pool))

    scored_keep: List[Tuple[str, str, float]] = []
    scored_buffer: List[Tuple[str, str, float]] = []

    for s, t in pool:
        key = _pair_key(s, t)
        if key in avoid:
            continue
        es = emb.get(s); et = emb.get(t)
        if es is None or et is None:
            continue
        p = float(model.predict_proba((es * et).reshape(1, -1))[0, 1])
        if p >= thr:
            scored_keep.append((key[0], key[1], p))
            if len(scored_keep) >= want:
                break
        else:
            scored_buffer.append((key[0], key[1], p))

    _dbg("_predict_from_per_alg_pool.after_scan", emb_key=emb_key,
         keep=len(scored_keep), buffer=len(scored_buffer))

    if len(scored_keep) < want:
        if len(scored_buffer) < want:
            _ensure_pool_for_key(emb_key, ensure_remaining=remaining_target * 2, version=_POOL_VERSION)
        scored_buffer.sort(key=lambda x: x[2], reverse=True)
        need = want - len(scored_keep)
        scored_keep.extend(scored_buffer[:need])

    if scored_keep:
        keep_set = {_pair_key(s, t) for (s, t, _p) in scored_keep}
        before_len = len(_POOL_BY_KEY[emb_key])
        _POOL_BY_KEY[emb_key] = [
            pair for pair in _POOL_BY_KEY[emb_key]
            if _pair_key(*pair) not in keep_set
        ]
        after_len = len(_POOL_BY_KEY[emb_key])
        _dbg("_predict_from_per_alg_pool.trim_pool", emb_key=emb_key,
             removed=before_len - after_len, new_pool_len=after_len)

    _dbg("_predict_from_per_alg_pool.done", emb_key=emb_key, kept=len(scored_keep))
    return scored_keep

def _predict_with_test_priority(
    emb_key: str,
    emb: Dict[str, np.ndarray],
    model: LogisticRegression,
    k: int,
    threshold: float,
    candidate_multiplier: int,
    test_pairs: Optional[list[tuple[str, str]]] = None,
    avoid_set: Optional[set[tuple[str, str]]] = None,
) -> list[tuple[str, str, float]]:

    want = max(1, int(k))
    thr = float(threshold)
    avoid = set(avoid_set or set())

    _dbg(
        "_predict_with_test_priority.enter",
        emb_key=emb_key,
        want=want,
        thr=thr,
        test_pairs=len(test_pairs or []),
        avoid_size=len(avoid),
        candidate_multiplier=candidate_multiplier,
    )

    kept: list[tuple[str, str, float]] = []
    seen_test: set[tuple[str, str]] = set()

    # --- Stage 1: scan test-set edges in fixed order ---
    for s, t in (test_pairs or []):
        key = _pair_key(s, t)
        if key in avoid or key in seen_test:
            continue

        es = emb.get(key[0])
        et = emb.get(key[1])
        if es is None or et is None:
            continue

        p = float(model.predict_proba((es * et).reshape(1, -1))[0, 1])
        if p >= thr:
            kept.append((key[0], key[1], p))
            seen_test.add(key)
            if len(kept) >= want:
                _dbg("_predict_with_test_priority.test_done", emb_key=emb_key, kept=len(kept))
                return kept 

    _dbg("_predict_with_test_priority.after_test", emb_key=emb_key, kept=len(kept))

    # --- Stage 2: fill remainder from canonical negative pool ---
    if len(kept) < want:
        extra_avoid = avoid | {_pair_key(s, t) for (s, t, _p) in kept}
        need = want - len(kept)
        more = _predict_from_per_alg_pool(
            emb_key=emb_key,
            emb=emb,
            model=model,
            k=need,
            threshold=threshold,
            candidate_multiplier=candidate_multiplier,
            avoid_set=extra_avoid,
        )
        kept.extend(more)

    _dbg("_predict_with_test_priority.done", emb_key=emb_key, kept=len(kept))
    return kept


def _load_embeddings(prop: str) -> Dict[str, np.ndarray]:

    prop = _sanitize_prop(prop)
    _dbg("_load_embeddings.start", prop=prop)
    cypher = f"""
    MATCH (n)
    WHERE n.{prop} IS NOT NULL
    RETURN elementId(n) AS eid, n.{prop} AS emb
    """
    rows = run_data(cypher)
    embs: Dict[str, np.ndarray] = {}
    for r in rows:
        v = r["emb"]
        if isinstance(v, list):
            embs[r["eid"]] = np.array(v, dtype=float)
    _dbg("_load_embeddings.done", prop=prop, count=len(embs))
    return embs

def _hadamard(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a * b

def _build_set_features(
    pos_pairs: List[Tuple[str, str]],
    emb: Dict[str, np.ndarray],
    negative_ratio: float = 1.0,
    as_of_ts: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Build X (features) and y (labels) for a set.
    """
    _dbg("_build_set_features.start", pos_pairs=len(pos_pairs), negative_ratio=negative_ratio)
    X_pos: List[np.ndarray] = []
    for s, t in pos_pairs:
        es = emb.get(s)
        et = emb.get(t)
        if es is None or et is None:
            continue
        X_pos.append(_hadamard(es, et))
    cnt_pos = len(X_pos)

    want_neg = int(math.ceil(cnt_pos * max(0.0, min(10.0, negative_ratio))))
    neg_pairs = _sample_negatives(want_neg, as_of_ts=as_of_ts)
    X_neg: List[np.ndarray] = []
    for s, t in neg_pairs:
        es = emb.get(s)
        et = emb.get(t)
        if es is None or et is None:
            continue
        X_neg.append(_hadamard(es, et))
    cnt_neg = len(X_neg)

    if X_pos or X_neg:
        X = np.vstack([X_pos, X_neg]) if X_neg else np.vstack(X_pos)
        y = np.hstack([np.ones(cnt_pos, dtype=int), np.zeros(cnt_neg, dtype=int)]) if X_neg else np.ones(cnt_pos, dtype=int)
    else:
        X = np.zeros((0, 1))
        y = np.zeros((0,), dtype=int)

    _dbg("_build_set_features.done", cnt_pos=cnt_pos, cnt_neg=cnt_neg, X_shape=X.shape)
    return X, y, cnt_pos, cnt_neg

def _eval_scores(y_true: np.ndarray, y_prob: np.ndarray) -> LPMetrics:
    y_hat = (y_prob >= 0.5).astype(int)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = None
    try:
        ap = average_precision_score(y_true, y_prob)
    except Exception:
        ap = None
    try:
        acc = accuracy_score(y_true, y_hat)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_hat, average="binary", zero_division=0
        )
    except Exception:
        acc = p = r = f1 = None

    met = LPMetrics(
        count_pos=int((y_true == 1).sum()),
        count_neg=int((y_true == 0).sum()),
        auc=auc, ap=ap, accuracy=acc, precision=p, recall=r, f1=f1,
    )
    _dbg(
        "_eval_scores",
        count_pos=met.count_pos,
        count_neg=met.count_neg,
        auc=met.auc,
        ap=met.ap,
        acc=met.accuracy,
        precision=met.precision,
        recall=met.recall,
        f1=met.f1,
    )
    return met

def _predict_topk_edges(
    emb: Dict[str, np.ndarray],
    model: LogisticRegression,
    k: int,
    candidate_multiplier: int,
) -> List[Tuple[str, str, float]]:

    pool = max(k * candidate_multiplier, k + 5)
    _dbg("_predict_topk_edges", k=k, pool=pool)
    cand = _sample_negatives(pool)
    scored: List[Tuple[str, str, float]] = []
    for s, t in cand:
        es = emb.get(s)
        et = emb.get(t)
        if es is None or et is None:
            continue
        x = _hadamard(es, et).reshape(1, -1)
        p = float(model.predict_proba(x)[0, 1])
        scored.append((s, t, p))
    scored.sort(key=lambda x: x[2], reverse=True)
    _dbg("_predict_topk_edges.done", returned=len(scored[:k]))
    return scored[:k]


def _predict_at_least_k_above_threshold(
    emb: Dict[str, np.ndarray],
    model: LogisticRegression,
    k: int,
    threshold: float,
    start_factor: int = 20,
    max_factor: int = 200,
    growth: float = 2.0,
    max_rounds: int = 5,
    avoid_set: Optional[set[tuple[str, str]]] = None, 
) -> List[Tuple[str, str, float]]:

    want = max(1, int(k))
    thr = float(threshold)
    avoid = set(avoid_set or set())
    _dbg(
        "_predict_at_least_k_above_threshold.enter",
        want=want,
        thr=thr,
        start_factor=start_factor,
        max_factor=max_factor,
        avoid_size=len(avoid),
    )

    max_needed = min(max(want * max(start_factor, 1), want + 5), want * max_factor)
    pool = _get_canonical_pool(min_needed=want, max_needed=max_needed, version=1)
    _dbg("_predict_at_least_k_above_threshold.pool", pool_len=len(pool))

    kept: List[Tuple[str, str, float]] = []
    seen_batch: set[tuple[str, str]] = set()

    def add_if_new(s: str, t: str, p: float) -> bool:
        key = _pair_key(s, t)
        if key in avoid or key in seen_batch:
            return False
        kept.append((key[0], key[1], p))
        seen_batch.add(key)
        return True

    for s, t in pool:
        es = emb.get(s); et = emb.get(t)
        if es is None or et is None:
            continue
        p = float(model.predict_proba((es * et).reshape(1, -1))[0, 1])
        if p >= thr and add_if_new(s, t, p):
            if len(kept) >= want:
                _dbg("_predict_at_least_k_above_threshold.threshold_done", kept=len(kept))
                return kept

    if len(kept) < want:
        scored: List[Tuple[str, str, float]] = []
        for s, t in pool:
            key = _pair_key(s, t)
            if key in avoid or key in seen_batch:
                continue
            es = emb.get(s); et = emb.get(t)
            if es is None or et is None:
                continue
            p = float(model.predict_proba((es * et).reshape(1, -1))[0, 1])
            scored.append((key[0], key[1], p))
        scored.sort(key=lambda x: x[2], reverse=True)
        needed = want - len(kept)
        kept.extend(scored[:needed])

    _dbg("_predict_at_least_k_above_threshold.done", kept=len(kept))
    return kept 


def _project_predicted_graph(
    graph_name: str,
    predicted_triples: list[tuple[str, str, float]],
    original_edges_cutoff_ts: Optional[float] = None,
    predicted_ts: Optional[int] = None,
):

    _dbg(
        "_project_predicted_graph.start",
        graph_name=graph_name,
        predicted=len(predicted_triples),
        original_edges_cutoff_ts=original_edges_cutoff_ts,
        predicted_ts=predicted_ts,
    )

    try:
        run("CALL gds.graph.drop($name, false)", {"name": graph_name})
    except Exception:
        pass


    if predicted_ts is None:
        rows_ts = run_data(
            """
            MATCH ()-[r]-()
            WHERE NOT type(r) STARTS WITH 'SPLIT_'
            WITH CASE
                WHEN r.lastTimestamp IS NOT NULL THEN toInteger(r.lastTimestamp)
                WHEN r.timestamp IS NOT NULL THEN toInteger(r.timestamp)
                ELSE 0
            END AS ts
            RETURN coalesce(max(ts), 0) AS max_ts
            """
        )
        max_ts = int(rows_ts[0]["max_ts"] or 0) if rows_ts else 0
        predicted_ts = max_ts + 1

    node_q = """
    MATCH (n)
    RETURN id(n) AS id
    """

    rel_q = """
    // ------------------------------------------------------------
    // (A) ORIGINAL stored edges (exclude SPLIT_*), emit both directions
    // ------------------------------------------------------------
    MATCH (a)-[r]-(b)
    WHERE NOT type(r) STARTS WITH 'SPLIT_'
      AND coalesce(r.edgeOrigin, 'observed') = 'observed'
      AND (
        $originalEdgesCutoffTs IS NULL
        OR coalesce(r.firstTimestamp, r.timestamp, 0) <= $originalEdgesCutoffTs
      )

    WITH
      id(a) AS aId,
      id(b) AS bId,
      type(r) AS typ,
      coalesce(r.probability, 1.0) AS probability,
      coalesce(r.predicted, 0) AS predicted,
      coalesce(r.firstTimestamp, r.timestamp, 0) AS firstTimestamp,
      coalesce(r.lastTimestamp,  r.timestamp, 0) AS lastTimestamp,
      CASE
        WHEN r.timestamps IS NOT NULL
             AND valueType(r.timestamps) STARTS WITH 'LIST'
             AND size(r.timestamps) > 0
          THEN r.timestamps
        ELSE [coalesce(r.timestamp, 0)]
      END AS timestamps,
      coalesce(
        r.interactionCount,
        CASE
          WHEN r.timestamps IS NOT NULL
               AND valueType(r.timestamps) STARTS WITH 'LIST'
            THEN size(r.timestamps)
          ELSE 1
        END
      ) AS interactionCount

    WITH
      CASE WHEN aId < bId THEN aId ELSE bId END AS s,
      CASE WHEN aId < bId THEN bId ELSE aId END AS t,
      typ, probability, predicted, firstTimestamp, lastTimestamp, timestamps, interactionCount

    WITH DISTINCT
      s, t, typ, probability, predicted, firstTimestamp, lastTimestamp, timestamps, interactionCount

    UNWIND [[s, t], [t, s]] AS dir
    RETURN
      dir[0] AS source,
      dir[1] AS target,
      typ AS type,
      {
        probability: probability,
        predicted: predicted,
        firstTimestamp: firstTimestamp,
        lastTimestamp: lastTimestamp,
        timestamp: lastTimestamp,
        timestamps: timestamps,
        interactionCount: interactionCount
      } AS properties

    UNION

    // ------------------------------------------------------------
    // (B) NEW predicted pairs, emit both directions
    // ------------------------------------------------------------
    UNWIND $predicted AS P
    MATCH (sNode) WHERE elementId(sNode) = P.s
    MATCH (tNode) WHERE elementId(tNode) = P.t
    WITH id(sNode) AS s, id(tNode) AS t, toFloat(P.p) AS p
    UNWIND [[s, t], [t, s]] AS dir
    RETURN
      dir[0] AS source,
      dir[1] AS target,
      'PREDICTED' AS type,
      {
        probability: p,
        predicted: 1,
        firstTimestamp: $predTs,
        lastTimestamp: $predTs,
        timestamp: $predTs,
        timestamps: [$predTs],
        interactionCount: 1
      } AS properties
    """

    config = {
        "parameters": {
            "predicted": [{"s": s, "t": t, "p": float(p)} for (s, t, p) in predicted_triples],
            "originalEdgesCutoffTs": original_edges_cutoff_ts,
            "predTs": int(predicted_ts),
        }
    }

    rows = run_data(
        """
        CALL gds.graph.project.cypher($name, $nodeQuery, $relQuery, $config)
        YIELD graphName, nodeCount, relationshipCount
        RETURN graphName, nodeCount, relationshipCount
        """,
        {
            "name": graph_name,
            "nodeQuery": node_q,
            "relQuery": rel_q,
            "config": config,
        },
    )

    if rows:
        _dbg(
            "_project_predicted_graph.done",
            graphName=rows[0]["graphName"],
            nodeCount=rows[0]["nodeCount"],
            relationshipCount=rows[0]["relationshipCount"],
            predicted_ts=predicted_ts,
        )
    else:
        _dbg("_project_predicted_graph.empty_result")

    return rows[0] if rows else {
        "graphName": graph_name,
        "nodeCount": 0,
        "relationshipCount": 0,
    }

def _get_emb_key_from_prop(prop: str) -> str:
    return _embedding_db_suffix(prop)

def _get_predicted_set_for_key(emb_key: str) -> set[tuple[str, str]]:
    lst = _PREDICTED_EDGES.get(emb_key) or []
    s = {_pair_key(s, t) for (s, t) in lst}
    _dbg("_get_predicted_set_for_key", emb_key=emb_key, size=len(s))
    return s

def _append_memory_for_key(emb_key: str, new_pairs: list[tuple[str, str]]) -> None:
    prev = _PREDICTED_EDGES.get(emb_key) or []
    prev_set = {_pair_key(s, t) for (s, t) in prev}
    added = 0
    for s, t in new_pairs:
        k = _pair_key(s, t)
        if k not in prev_set:
            prev.append((k[0], k[1]))
            prev_set.add(k)
            added += 1
    _PREDICTED_EDGES[emb_key] = prev
    _dbg("_append_memory_for_key", emb_key=emb_key, added=added, total=len(prev))

def _clear_prediction_memory_for_family(emb_key: str) -> None:

    _PREDICTED_EDGES.pop(emb_key, None)
    _POOL_BY_KEY.pop(emb_key, None)


def _delete_single_db_timeline_predictions(
    *,
    dbname: str,
    embedding_family: str,
    prediction_model: str,
    only_level: Optional[int] = None,
    batch_size: int = 10000,
) -> Dict[str, Any]:
    prediction_model = normalize_prediction_model(
        prediction_model
    )

    total_deleted = 0

    while True:
        rows = run_data(
            """
            MATCH ()-[r]-()
            WHERE coalesce(r.edgeOrigin, '') = 'predicted'
              AND coalesce(r.embeddingFamily, '') = $family
              AND coalesce(
                    r.predictionModel,
                    'logistic_regression'
                  ) = $predictionModel
              AND (
                    $onlyLevel IS NULL
                    OR toInteger(
                         coalesce(
                           r.predicted,
                           r.predicted_timestep,
                           0
                         )
                       ) = $onlyLevel
                  )
            WITH r
            LIMIT $batchSize
            DELETE r
            RETURN count(*) AS deleted
            """,
            {
                "family": embedding_family,
                "predictionModel": prediction_model,
                "onlyLevel": only_level,
                "batchSize": int(batch_size),
            },
            db=dbname,
        )

        deleted = (
            int(rows[0].get("deleted", 0) or 0)
            if rows
            else 0
        )

        total_deleted += deleted

        if deleted == 0 or deleted < batch_size:
            break

    return {
        "database": dbname,
        "embeddingFamily": embedding_family,
        "predictionModel": prediction_model,
        "onlyLevel": only_level,
        "relationshipsDeleted": total_deleted,
    }

def _materialize_prediction_db(
    gname: str = "predictedGraph",
    node_key: str = "eid",
    embedding_suffix: Optional[str] = None,
    predicted_triples: Optional[list[tuple[str, str, float]]] = None,
    predicted_ts: Optional[int] = None,
    embedding_family: Optional[str] = None,
    prediction_model: str = "logistic_regression",
    predicted_level: int = 1,
) -> str:
    pred_list = [{"s": s, "t": t, "p": float(p)} for (s, t, p) in (predicted_triples or [])]
    base_db = current_database()

    family_for_db = embedding_family or embedding_suffix or "Embedding"

    prediction_model = normalize_prediction_model(prediction_model)
    target_db = predicted_db_name(
        base_db,
        family_for_db,
        prediction_model,
    )
    embedding_family = _embedding_db_suffix(str(family_for_db))
    _dbg("_materialize_prediction_db.start", gname=gname, base_db=base_db, target_db=target_db,
         predicted=len(pred_list))

    if predicted_ts is None:
        max_ts = _database_max_timestamp(base_db)
        predicted_ts = max_ts + 1
        _dbg("_materialize_prediction_db.max_ts", max_ts=max_ts, predicted_ts=predicted_ts)

    _recreate_database(target_db)

    run(
        f"CREATE CONSTRAINT node_{node_key}_uniq IF NOT EXISTS "
        f"FOR (n:Entity) REQUIRE n.{node_key} IS UNIQUE",
        db=target_db,
    )

    node_rows = run_data(
        """
        CALL gds.graph.relationships.stream($g)
        YIELD sourceNodeId, targetNodeId
        WITH gds.util.asNode(sourceNodeId) AS sN,
             gds.util.asNode(targetNodeId) AS tN
        WITH collect(sN) + collect(tN) AS nodes
        UNWIND nodes AS n
        WITH DISTINCT n
        RETURN elementId(n)      AS eid,
               labels(n)         AS labels,
               properties(n)     AS props
        """,
        {"g": gname},
    )
    _dbg("_materialize_prediction_db.nodes", target_db=target_db, count=len(node_rows))

    if node_rows:
        run(
            """
            UNWIND $batch AS row
            CALL apoc.merge.node(row.labels, {eid: row.eid}, row.props, row.props) YIELD node
            RETURN count(*) AS upserted
            """,
            {"batch": node_rows},
            db=target_db,
        )

    rel_rows = run_data(
    """
    CALL gds.graph.relationships.stream($g)
    YIELD sourceNodeId, targetNodeId, relationshipType
    WITH gds.util.asNode(sourceNodeId) AS sN,
         gds.util.asNode(targetNodeId) AS tN,
         relationshipType AS type
    WITH elementId(sN) AS a, elementId(tN) AS b, type
    WITH CASE WHEN a < b THEN a ELSE b END AS sid,
         CASE WHEN a < b THEN b ELSE a END AS tid,
         type

    CALL {
      WITH sid, tid, type
      OPTIONAL MATCH (s)-[r]-(t)
      WHERE elementId(s) = sid
        AND elementId(t) = tid
        AND type(r) = type
        AND NOT type(r) STARTS WITH 'SPLIT_'
      RETURN
        CASE
          WHEN r IS NULL THEN []
            WHEN r.timestamps IS NOT NULL
                AND valueType(r.timestamps) STARTS WITH 'LIST'
                AND size(r.timestamps) > 0
            THEN [x IN r.timestamps | toFloat(x)]
          ELSE [toFloat(coalesce(r.timestamp, 0))]
        END AS tsList,
        coalesce(r.firstTimestamp, coalesce(r.timestamp, 0.0)) AS firstTs,
        coalesce(r.lastTimestamp,  coalesce(r.timestamp, 0.0)) AS lastTs,
        coalesce(
          r.interactionCount,
          CASE
            WHEN r.timestamps IS NOT NULL AND valueType(r.timestamps) STARTS WITH 'LIST' THEN size(r.timestamps)
            ELSE 1
          END
        ) AS cnt
    }

    WITH sid, tid, type, tsList, firstTs, lastTs, cnt,
         [p IN $pred WHERE p.s = sid AND p.t = tid | toFloat(p.p)] AS probs

    WITH sid, tid, type,
         CASE
           WHEN type <> 'PREDICTED' THEN 1.0
           ELSE CASE WHEN size(probs) > 0 THEN probs[0] ELSE 0.0 END
         END AS probability,
         CASE WHEN type = 'PREDICTED' THEN $predictedLevel ELSE 0 END AS predicted,
         CASE WHEN type = 'PREDICTED' THEN $predictedLevel ELSE 0 END AS predicted_timestep,
         CASE WHEN type = 'PREDICTED' THEN [$predTs] ELSE tsList END AS timestamps,
         CASE WHEN type = 'PREDICTED' THEN $predTs ELSE firstTs END AS firstTimestamp,
         CASE WHEN type = 'PREDICTED' THEN $predTs ELSE lastTs END AS lastTimestamp,
         CASE WHEN type = 'PREDICTED' THEN 1 ELSE cnt END AS interactionCount,
         CASE WHEN type = 'PREDICTED' THEN 'predicted' ELSE 'observed' END AS edgeOrigin,
         CASE WHEN type = 'PREDICTED' THEN $embeddingFamily ELSE null END AS embeddingFamily,
         CASE WHEN type = 'PREDICTED' THEN $predictionModel ELSE null END AS predictionModel,
         CASE WHEN type = 'PREDICTED' THEN $predictionModelDisplayName ELSE null END AS predictionModelDisplayName

    RETURN
      sid,
      tid,
      type,
      predicted,
      predicted_timestep,
      probability,
      timestamps,
      firstTimestamp,
      lastTimestamp,
      lastTimestamp AS timestamp,
      interactionCount,
      edgeOrigin,
      embeddingFamily,
      predictionModel,
      predictionModelDisplayName
    """,
    {
    "g": gname,
    "predTs": int(predicted_ts),
    "pred": pred_list,
    "predictedLevel": int(predicted_level),
    "embeddingFamily": embedding_family,
    "predictionModel": prediction_model,
    "predictionModelDisplayName": prediction_model_display_name(prediction_model),
    },
    )
    
    _dbg("_materialize_prediction_db.rels", target_db=target_db, count=len(rel_rows))

    if rel_rows:
        run(
            f"""
            UNWIND $batch AS row
            MATCH (s {{ {node_key}: row.sid }}),
                (t {{ {node_key}: row.tid }})
            CALL apoc.merge.relationship(
            s,
            row.type,
            {{}},
            {{
                predicted: row.predicted,
                predicted_timestep: row.predicted_timestep,
                probability: row.probability,
                timestamp: row.timestamp,
                firstTimestamp: row.firstTimestamp,
                lastTimestamp: row.lastTimestamp,
                timestamps: row.timestamps,
                interactionCount: row.interactionCount,
                edgeOrigin: row.edgeOrigin,
                embeddingFamily: row.embeddingFamily,
                predictionModel: row.predictionModel,
                predictionModelDisplayName: row.predictionModelDisplayName
            }},
            t
            ) YIELD rel
            RETURN count(*) AS upserted
            """,
            {"batch": rel_rows},
            db=target_db,
        )

    _dbg("_materialize_prediction_db.done", target_db=target_db)
    return target_db

def _recreate_database(dbname: str, timeout_s: float = 30.0) -> None:
    _dbg("_recreate_database.start", dbname=dbname, timeout_s=timeout_s)
    run("DROP DATABASE $db IF EXISTS", {"db": dbname}, db="system")
    run("CREATE DATABASE $db", {"db": dbname}, db="system")

    import time
    start = time.time()
    while True:
        rows = run_data(
            "SHOW DATABASE $db YIELD name, currentStatus RETURN currentStatus",
            {"db": dbname},
            db="system",
        )
        status = (rows[0]["currentStatus"].lower() if rows else None)
        if status == "online":
            _dbg("_recreate_database.online", dbname=dbname, elapsed=time.time() - start)
            break
        if time.time() - start > timeout_s:
            _dbg("_recreate_database.timeout", dbname=dbname, status=status)
            raise HTTPException(
                status_code=503,
                detail=f"Database '{dbname}' did not come ONLINE (status={status})"
            )
        time.sleep(0.5)

def _gds_graph_exists(name: str) -> bool:
    try:
        rows = run_data("CALL gds.graph.exists($name) YIELD exists RETURN exists", {"name": name})
        exists = bool(rows and rows[0]["exists"])
        _dbg("_gds_graph_exists.exists", name=name, exists=exists)
        return exists
    except Exception:
        rows = run_data("CALL gds.graph.list($name) YIELD graphName RETURN graphName", {"name": name})
        exists = bool(rows)
        _dbg("_gds_graph_exists.list", name=name, exists=exists)
        return exists

# ---- helpers for heatmap ----
def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)

def _get_test_pairs(graph_name: str = "testGraph") -> list[tuple[str, str]]:

    base = _ensure_cache_for_current_db()
    global _TEST_PAIRS_BY_GRAPH

    cached = _TEST_PAIRS_BY_GRAPH.get(graph_name)
    if cached is not None:
        _dbg("_get_test_pairs.cached_hit", base_db=base, graph=graph_name, count=len(cached))
        return cached

    if not _gds_graph_exists(graph_name):

        _dbg("_get_test_pairs.no_graph", base_db=base, graph=graph_name)
        return []

    # Use the existing streamer
    raw = _stream_unique_pairs_from_gds(graph_name)
    canon = sorted({_pair_key(s, t) for (s, t) in raw})
    _TEST_PAIRS_BY_GRAPH[graph_name] = canon
    _dbg("_get_test_pairs.built", base_db=base, graph=graph_name, count=len(canon))
    return canon


def _load_predicted_pairs_from_db(dbname: str) -> list[tuple[str, str]]:

    _dbg("_load_predicted_pairs_from_db.start", dbname=dbname)
    try:
        rows = run_data(
            """
            MATCH (x)-[r]-(y)
            WHERE coalesce(r.edgeOrigin, '') = 'predicted'
            OR toUpper(type(r)) STARTS WITH 'PREDICTED'
            WITH
            coalesce(x.id, x.eid, elementId(x)) AS a,
            coalesce(y.id, y.eid, elementId(y)) AS b
            WITH CASE WHEN a < b THEN a ELSE b END AS s,
                CASE WHEN a < b THEN b ELSE a END AS t
            RETURN DISTINCT s, t
            """,
            db=dbname,
        )
        _dbg("_load_predicted_pairs_from_db.done", dbname=dbname, count=len(rows))
        return [(r["s"], r["t"]) for r in rows]
    except Exception as e:
        _dbg("_load_predicted_pairs_from_db.error", dbname=dbname, error=str(e))
        return []

def _recognize_all_test_edges_above_threshold(
    emb: Dict[str, np.ndarray],
    model: LogisticRegression,
    threshold: float,
    test_pairs: Optional[list[tuple[str, str]]] = None,
) -> list[tuple[str, str, float]]:

    thr = float(threshold)
    kept: list[tuple[str, str, float]] = []

    for s, t in (test_pairs or []):
        key = _pair_key(s, t)

        es = emb.get(key[0])
        et = emb.get(key[1])
        if es is None or et is None:
            continue

        p = float(model.predict_proba((es * et).reshape(1, -1))[0, 1])
        if p >= thr:
            kept.append((key[0], key[1], p))

    _dbg(
        "_recognize_all_test_edges_above_threshold.done",
        threshold=thr,
        test_pairs=len(test_pairs or []),
        kept=len(kept),
    )
    return kept

def _dbg_embedding_stats(tag: str, emb: Dict[str, np.ndarray]) -> None:
    if not emb:
        _dbg(tag, count=0)
        return

    try:
        arr = np.vstack(list(emb.values()))
        norms = np.linalg.norm(arr, axis=1)
        per_dim_std = arr.std(axis=0)

        _dbg(
            tag,
            count=int(arr.shape[0]),
            dim=int(arr.shape[1]),
            norm_min=float(norms.min()),
            norm_max=float(norms.max()),
            norm_mean=float(norms.mean()),
            mean_dim_std=float(per_dim_std.mean()),
            max_dim_std=float(per_dim_std.max()),
            min_dim_std=float(per_dim_std.min()),
        )
    except Exception as e:
        _dbg(tag, error=str(e))

# ------------------------
# Core logic
# ------------------------

def _write_predictions_single_db_multi_edge(
    *,
    embedding_family: str,
    prediction_model: str,
    triples: List[Tuple[str, str, float]],
    level: int,
    predicted_ts: int,
) -> Dict[str, Any]:

    if not triples:
        return {
            "database": current_database(),
            "relationshipsWritten": 0,
            "predictedLevel": int(level),
            "predictedTimestamp": int(predicted_ts),
        }

    rel_type = predicted_rel_type(
        embedding_family,
        level=level,
        prediction_model=prediction_model,
    )

    rows = run_data(
        f"""
        UNWIND $rows AS row
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
            r.predictedTimestamp = $predictedTs,
            r.timestamp = $predictedTs,
            r.firstTimestamp = $predictedTs,
            r.lastTimestamp = $predictedTs,
            r.timestamps = [$predictedTs],
            r.interactionCount = 1,
            r.storageRepresentation = 'single_db_multi_edge'

        RETURN count(r) AS written
        """,
        {
            "rows": [{"s": s, "t": t, "p": float(p)} for s, t, p in triples],
            "family": embedding_family,
            "predictionModel": normalize_prediction_model(prediction_model),
            "predictionModelDisplayName": prediction_model_display_name(prediction_model),
            "level": int(level),
            "predictedTs": int(predicted_ts),
        },
    )

    written = int(rows[0]["written"]) if rows else 0

    return {
        "database": current_database(),
        "relationshipType": rel_type,
        "relationshipsWritten": written,
        "predictedLevel": int(level),
        "predictedTimestamp": int(predicted_ts),
        "predictionModel": normalize_prediction_model(prediction_model),
        "predictionModelDisplayName": prediction_model_display_name(prediction_model),
    }

def _run_lp_logic(body: LPBody) -> Dict[str, Any]:

    base_db = _ensure_cache_for_current_db()

    if body.force_new_canonical_pool:
        global _CANON_POOL, _CANON_SIG, _POOL_BY_KEY
        _dbg("_run_lp_logic.force_new_canonical_pool", base_db=base_db)
        _CANON_POOL = []
        _CANON_SIG = (0, 0)
        _POOL_BY_KEY = {}

    _dbg("_run_lp_logic.enter", body=body.dict())
    _assert_sklearn()

    prop = _sanitize_prop(body.embeddingProperty)

    storage_mode = normalize_storage_representation(body.storageRepresentation)
    prediction_model = normalize_prediction_model(body.predictionModel)

    embedding_family = (
        body.embeddingFamily
        or _embedding_db_suffix(prop)
    )

    embedding_family, prediction_model = validate_timeline(
        embedding_family,
        prediction_model,
    )

    timeline_id = timeline_key(
        embedding_family,
        prediction_model,
    )

    emb = _load_embeddings(prop)
    _dbg_embedding_stats(f"{prop}.embedding_stats", emb)

    if not emb:
        _dbg("_run_lp_logic.no_embeddings", prop=prop)
        raise HTTPException(
            status_code=400,
            detail=f"No nodes found with embedding property '{prop}'."
        )

    lp_stage_t0 = time.perf_counter()

    # ------------------------
    # Positive pairs
    # ------------------------
    pos_train = _stream_unique_pairs_from_gds(body.trainGraphName)

    pos_val = []
    if body.valGraphName and _gds_graph_exists(body.valGraphName):
        pos_val = _stream_unique_pairs_from_gds(body.valGraphName)

    pos_test = []
    if body.testGraphName and _gds_graph_exists(body.testGraphName):
        pos_test = _get_test_pairs(body.testGraphName)

    train_cutoff = _graph_snapshot_ts(
        body.trainGraphName,
        time_prop="firstTimestamp"
    )

    val_cutoff = (
        _graph_snapshot_ts(body.valGraphName, time_prop="firstTimestamp")
        if body.valGraphName
        else train_cutoff
    )

    test_cutoff = (
        _graph_snapshot_ts(body.testGraphName, time_prop="firstTimestamp")
        if body.testGraphName
        else train_cutoff
    )

    observed_graph_max_ts = _observed_database_max_timestamp(base_db)
    initial_predicted_ts = observed_graph_max_ts + 1

    _dbg(
        "_run_lp_logic.pos_pairs",
        train=len(pos_train),
        val=len(pos_val),
        test=len(pos_test),
        storageRepresentation=storage_mode,
        embeddingFamily=embedding_family,
        observedGraphMaxTimestamp=observed_graph_max_ts,
        initialPredictedTimestamp=initial_predicted_ts,
    )

    # ------------------------
    # Training
    # ------------------------
    Xtr, ytr, npos_tr, nneg_tr = _build_set_features(
        pos_train,
        emb,
        body.negativeRatio,
        as_of_ts=train_cutoff,
    )

    if Xtr.shape[0] == 0:
        _dbg("_run_lp_logic.no_training_pairs")
        raise HTTPException(
            status_code=400,
            detail="No usable training pairs (check embeddings exist for endpoints)."
        )

    prediction_model = normalize_prediction_model(getattr(body, "predictionModel", "logistic_regression"))
    model = _make_lp_classifier(prediction_model)
    model.fit(Xtr, ytr)

    prob_tr = model.predict_proba(Xtr)[:, 1]
    mtr = _eval_scores(ytr, prob_tr)
    mtr.count_pos = npos_tr
    mtr.count_neg = nneg_tr

    _dbg(
        f"{prop}.train_prob_stats",
        min=float(prob_tr.min()),
        max=float(prob_tr.max()),
        mean=float(prob_tr.mean()),
        std=float(prob_tr.std()),
    )

    # ------------------------
    # Validation evaluation
    # ------------------------
    mval = None

    if pos_val:
        Xv, yv, npos_v, nneg_v = _build_set_features(
            pos_val,
            emb,
            body.negativeRatio,
            as_of_ts=val_cutoff,
        )

        if Xv.shape[0] > 0:
            prob_v = model.predict_proba(Xv)[:, 1]
            mval = _eval_scores(yv, prob_v)
            mval.count_pos = npos_v
            mval.count_neg = nneg_v

            _dbg(
                f"{prop}.validation_prob_stats",
                min=float(prob_v.min()),
                max=float(prob_v.max()),
                mean=float(prob_v.mean()),
                std=float(prob_v.std()),
            )

    # ------------------------
    # Test evaluation
    # ------------------------
    mts = None

    if pos_test:
        Xte, yte, npos_te, nneg_te = _build_set_features(
            pos_test,
            emb,
            body.negativeRatio,
            as_of_ts=test_cutoff,
        )

        if Xte.shape[0] > 0:
            prob_te = model.predict_proba(Xte)[:, 1]
            mts = _eval_scores(yte, prob_te)
            mts.count_pos = npos_te
            mts.count_neg = nneg_te

            _dbg(
                f"{prop}.test_prob_stats",
                min=float(prob_te.min()),
                max=float(prob_te.max()),
                mean=float(prob_te.mean()),
                std=float(prob_te.std()),
            )

    # ------------------------
    # Remember metrics for this embedding family in this DB
    # ------------------------
    _METRICS_BY_KEY[timeline_id] = {
        "train": mtr.model_dump(),
        "validation": mval.model_dump() if mval else None,
        "test": mts.model_dump() if mts else None,
    }

    # ------------------------
    # Diagnostic prediction phase
    # ------------------------
    avoid_set_initial = _get_predicted_set_for_key(
        timeline_id
    )

    _dbg(
        "_run_lp_logic.predict_phase",
        timeline_id=timeline_id,
        embeddingFamily=embedding_family,
        storageRepresentation=storage_mode,
        avoid_size=len(avoid_set_initial),
        predictK=body.predictK,
        probThreshold=body.probThreshold,
        evaluationThreshold=body.evaluationThreshold,
        candidateMultiplier=body.candidateMultiplier,
        test_pairs=len(pos_test),
    )

    diag_filtered = _recognize_all_test_edges_above_threshold(
        emb=emb,
        model=model,
        threshold=body.evaluationThreshold,
        test_pairs=pos_test,
    )

    diag_triples = [(s, t, p) for (s, t, p) in diag_filtered]
    diag_pairs = [(s, t) for (s, t, _p) in diag_filtered]

    _dbg(
        "_run_lp_logic.diag_predicted",
        timeline_id=timeline_id,
        triples=len(diag_triples),
        pairs=len(diag_pairs),
    )

    _set_debug_initial_batch(
        timeline_id,
        diag_pairs,
    )

    _EVAL_PREDICTED_EDGES[timeline_id] = list(
        diag_pairs
    )


    if storage_mode == SINGLE_DB_MULTI_EDGE:

        _dbg(
            "_run_lp_logic.single_db_multi_edge.enter",
            base_db=base_db,
            timeline_id=timeline_id,
            embeddingFamily=embedding_family,
            predictionModel=prediction_model,
        )

        deleted_timeline = _delete_single_db_timeline_predictions(
            dbname=base_db,
            embedding_family=embedding_family,
            prediction_model=prediction_model,
        )

        _clear_prediction_memory_for_family(
            timeline_id
        )

        _dbg(
            "_run_lp_logic.single_db_multi_edge.reset",
            timeline_id=timeline_id,
            deletedRelationships=deleted_timeline.get(
                "relationshipsDeleted",
                0,
            ),
        )

        prediction_timestep = 1
        next_level = 1

        initial_predicted_ts = (
            int(observed_graph_max_ts) + 1
        )


        for state_name in ("initial", "latest"):
            save_lp_learning_state(
                base_db=base_db,
                storage_representation=storage_mode,
                embedding_family=embedding_family,
                embedding_property=prop,
                model=model,
                trained_at_timestep=prediction_timestep,
                prediction_model=prediction_model,
                state_name=state_name,
                embeddings=emb,
                metadata={
                    "source": "initial_lp",
                    "trainPairs": npos_tr,
                    "valPairs": len(pos_val),
                    "testPairs": len(pos_test),
                    "predictK": int(body.predictK),
                    "probThreshold": float(body.probThreshold),
                    "evaluationThreshold": float(body.evaluationThreshold),
                    "predictionModel": prediction_model,
                    "predictionModelDisplayName": prediction_model_display_name(prediction_model),
                },
            )

        pool_size = max(
            int(body.predictK) * 20,
            int(body.predictK) + 50,
            1000,
        )

        generated_candidates = _build_canonical_candidate_pairs(
            pool_size=pool_size,
            version=1,
        )

        shared_pool = get_or_create_candidate_pool(
            base_db=base_db,
            storage_representation=storage_mode,
            timestep=prediction_timestep,
            generated_candidates=generated_candidates,
            pool_size=pool_size,
            force_new=False,
        )

        already_predicted = _load_single_db_timeline_pairs(
            dbname=base_db,
            embedding_family=embedding_family,
            prediction_model=prediction_model,
        )

        gt_pairs = {
            _pair_key(s, t)
            for s, t in pos_train
        } | {
            _pair_key(s, t)
            for s, t in pos_val
        } | {
            _pair_key(s, t)
            for s, t in pos_test
        }

        candidate_pairs: List[Tuple[str, str]] = []
        seen_candidate_pairs: set[Tuple[str, str]] = set()

        for row in shared_pool:
            pair = _pair_key(
                str(row["s"]),
                str(row["t"]),
            )

            if pair in seen_candidate_pairs:
                continue

            if pair in gt_pairs:
                continue

            if pair in already_predicted:
                continue

            seen_candidate_pairs.add(pair)
            candidate_pairs.append(pair)

        _dbg(
            "_run_lp_logic.single_db_candidates_filtered",
            timeline_id=timeline_id,
            sharedPool=len(shared_pool),
            observedOrGroundTruth=len(gt_pairs),
            alreadyPredictedByTimeline=len(already_predicted),
            candidatesRemaining=len(candidate_pairs),
        )

        scored_candidates = _score_candidate_pairs_for_lp(
            candidate_pairs=candidate_pairs,
            emb=emb,
            model=model,
        )

        selected_rows = _select_top_prediction_rows_lp(
            scored_candidates=scored_candidates,
            predict_k=int(body.predictK),
            prob_threshold=float(body.probThreshold),
        )

        new_triples_for_db = _rows_to_triples_lp(selected_rows)

        scored_artifact_rows = _artifact_rows_from_scored_candidates(
            scored_candidates=scored_candidates,
            selected_rows=selected_rows,
        )

        record_candidate_scores(
            base_db=base_db,
            storage_representation=storage_mode,
            timestep=prediction_timestep,
            prediction_model=prediction_model,
            embedding_family=embedding_family,
            scored_candidates=scored_artifact_rows,
        )

        new_pairs_for_db = [
            (s, t)
            for s, t, _p in new_triples_for_db
        ]

        _append_memory_for_key(
            timeline_id,
            new_pairs_for_db,
        )

        link_prediction_seconds = time.perf_counter() - lp_stage_t0
        predicted_db_creation_seconds = 0.0
        predicted_graph_projection_seconds = 0.0

        materialize_db_stage_t0 = time.perf_counter()

        materialized = _write_predictions_single_db_multi_edge(
            embedding_family=embedding_family,
            prediction_model=prediction_model,
            triples=new_triples_for_db,
            level=next_level,
            predicted_ts=initial_predicted_ts,
        )

        materialize_prediction_db_seconds = time.perf_counter() - materialize_db_stage_t0

        update_family_evaluation_artifact(
            base_db=base_db,
            storage_representation=storage_mode,
            prediction_model=prediction_model,
            embedding_family=embedding_family,
            metrics={
                "train": mtr.model_dump(),
                "validation": mval.model_dump() if mval else None,
                "test": mts.model_dump() if mts else None,
                "predictionModel": prediction_model,
                "predictionModelDisplayName": prediction_model_display_name(prediction_model),
            },
            eval_pairs=diag_pairs,
            materialized_pairs=new_pairs_for_db,
            ground_truth_pairs=pos_test,
        )

        proj_info = {
            "graphName": None,
            "storageRepresentation": SINGLE_DB_MULTI_EDGE,
            "materializedDatabase": base_db,
            "materializedInPlace": True,
            "embeddingFamily": embedding_family,
            "predictionModel": prediction_model,
            "predictionModelDisplayName": prediction_model_display_name(prediction_model),
            "predictedLevel": next_level,
            "predictedTimestamp": initial_predicted_ts,
            "materialized": materialized,
            "keptAboveThreshold": len(new_triples_for_db),
            "threshold": float(body.probThreshold),
            "diagnosticPredictions": {
                "pairs": len(diag_pairs),
                "triples": len(diag_triples),
            },
            "materializedPredictions": {
                "pairs": len(new_pairs_for_db),
                "triples": len(new_triples_for_db),
            },
            "timings": {
                "link_prediction_seconds": round(link_prediction_seconds, 6),
                "predicted_db_creation_seconds": round(predicted_db_creation_seconds, 6),
                "predicted_graph_projection_seconds": round(predicted_graph_projection_seconds, 6),
                "materialize_prediction_db_seconds": round(materialize_prediction_db_seconds, 6),
            },
            "cache": {
                "db": base_db,
                "currentKey": timeline_id,
                "counts": {
                    key: len(pairs)
                    for key, pairs in _PREDICTED_EDGES.items()
                },
                "edgesForCurrentKey": _PREDICTED_EDGES.get(
                    timeline_id,
                    [],
                ),
            },
        }

        return {
            "ok": True,
            "train": mtr.model_dump(),
            "validation": mval.model_dump() if mval else None,
            "test": mts.model_dump() if mts else None,
            "model": {
                "algorithm": prediction_model_algorithm_name(prediction_model),
                "predictionModel": prediction_model,
                "predictionModelDisplayName": prediction_model_display_name(prediction_model),
                "embeddingProperty": prop,
                "embeddingFamily": embedding_family,
                "features": "Hadamard",
            },
            "predicted": proj_info,
        }

    # ======================================================================
    # separate current DB and predicted DB materialization
    # ======================================================================
    _clear_prediction_memory_for_family(
        timeline_id
    )

    gt_pairs = {
        _pair_key(s, t)
        for (s, t) in pos_train
    } | {
        _pair_key(s, t)
        for (s, t) in pos_val
    } | {
        _pair_key(s, t)
        for (s, t) in pos_test
    }


    avoid_for_db = (
        _get_predicted_set_for_key(timeline_id)
        | gt_pairs
    )

    _dbg(
        "_run_lp_logic.materialize_phase",
        timeline_id=timeline_id,
        storageRepresentation=storage_mode,
        avoid_for_db_size=len(avoid_for_db),
        gt_pairs=len(gt_pairs),
    )

    prediction_timestep = 1

    pool_size = max(
        int(body.predictK) * 20,
        int(body.predictK) + 50,
        1000,
    )

    generated_candidates = _build_canonical_candidate_pairs(
        pool_size=pool_size,
        version=1,
    )

    shared_pool = get_or_create_candidate_pool(
        base_db=base_db,
        storage_representation=storage_mode,
        timestep=prediction_timestep,
        generated_candidates=generated_candidates,
        pool_size=pool_size,
        force_new=False,
    )

    candidate_pairs: List[Tuple[str, str]] = []
    seen_candidate_pairs: set[Tuple[str, str]] = set()

    for row in shared_pool:
        pair = _pair_key(
            str(row["s"]),
            str(row["t"]),
        )

        if pair in seen_candidate_pairs:
            continue

        if pair in avoid_for_db:
            continue

        seen_candidate_pairs.add(pair)
        candidate_pairs.append(pair)

    _dbg(
        "_run_lp_logic.separate_db_candidates_filtered",
        timeline_id=timeline_id,
        sharedPool=len(shared_pool),
        avoidedForTimeline=len(avoid_for_db),
        candidatesRemaining=len(candidate_pairs),
    )

    scored_candidates = _score_candidate_pairs_for_lp(
        candidate_pairs=candidate_pairs,
        emb=emb,
        model=model,
    )

    selected_rows = _select_top_prediction_rows_lp(
        scored_candidates=scored_candidates,
        predict_k=int(body.predictK),
        prob_threshold=float(body.probThreshold),
    )

    new_triples_for_db = _rows_to_triples_lp(selected_rows)

    scored_artifact_rows = _artifact_rows_from_scored_candidates(
        scored_candidates=scored_candidates,
        selected_rows=selected_rows,
    )

    record_candidate_scores(
        base_db=base_db,
        storage_representation=storage_mode,
        timestep=prediction_timestep,
        prediction_model=prediction_model,
        embedding_family=embedding_family,
        scored_candidates=scored_artifact_rows,
    )

    new_pairs_for_db = [
        (s, t)
        for (s, t, _p) in new_triples_for_db
    ]

    _dbg(
        "_run_lp_logic.materialize_new",
        timeline_id=timeline_id,
        triples=len(new_triples_for_db),
        pairs=len(new_pairs_for_db),
    )

    _append_memory_for_key(
        timeline_id,
        new_pairs_for_db,
    )

    link_prediction_seconds = time.perf_counter() - lp_stage_t0

    # ---- C) PROJECT & MATERIALISE USING ONLY THE NEW EDGES ----
    predicted_db_stage_t0 = time.perf_counter()

    project_stage_t0 = time.perf_counter()

    proj_info = _project_predicted_graph(
        body.outputGraphName,
        new_triples_for_db,
        original_edges_cutoff_ts=None,
        predicted_ts=initial_predicted_ts,
    )

    proj_info["keptAboveThreshold"] = len(new_pairs_for_db)
    proj_info["threshold"] = float(body.probThreshold)

    # Expose diagnostic vs materialised sizes for debugging / UI.
    proj_info["diagnosticPredictions"] = {
        "pairs": len(diag_pairs),
        "triples": len(diag_triples),
    }

    proj_info["materializedPredictions"] = {
        "pairs": len(new_pairs_for_db),
        "triples": len(new_triples_for_db),
    }

    predicted_graph_projection_seconds = time.perf_counter() - project_stage_t0

    base_db = _ensure_cache_for_current_db()

    suffix = embedding_family

    proj_info["cache"] = {
        "db": base_db,
        "currentKey": timeline_id,
        "counts": {
            key: len(pairs)
            for key, pairs in _PREDICTED_EDGES.items()
        },
        "edgesForCurrentKey": _PREDICTED_EDGES.get(
            timeline_id,
            [],
        ),
    }

    _dbg("_run_lp_logic.cache_state", cache=proj_info["cache"])

    materialize_db_stage_t0 = time.perf_counter()

    target_db = _materialize_prediction_db(
        gname=body.outputGraphName,
        embedding_suffix=suffix,
        embedding_family=embedding_family,
        prediction_model=prediction_model,
        predicted_level=1,
        predicted_triples=new_triples_for_db,
        predicted_ts=initial_predicted_ts,
    )

    materialize_prediction_db_seconds = time.perf_counter() - materialize_db_stage_t0
    predicted_db_creation_seconds = time.perf_counter() - predicted_db_stage_t0

    proj_info["storageRepresentation"] = SEPARATE_DBS
    proj_info["materializedDatabase"] = target_db
    proj_info["materializedInPlace"] = False

    proj_info["timings"] = {
        "link_prediction_seconds": round(link_prediction_seconds, 6),
        "predicted_db_creation_seconds": round(predicted_db_creation_seconds, 6),
        "predicted_graph_projection_seconds": round(predicted_graph_projection_seconds, 6),
        "materialize_prediction_db_seconds": round(materialize_prediction_db_seconds, 6),
    }

    _dbg("_run_lp_logic.materialized", materialized_db=target_db)
    update_family_evaluation_artifact(
        base_db=base_db,
        storage_representation=storage_mode,
        prediction_model=prediction_model,
        embedding_family=embedding_family,
        metrics={
            "train": mtr.model_dump(),
            "validation": mval.model_dump() if mval else None,
            "test": mts.model_dump() if mts else None,
            "predictionModel": prediction_model,
            "predictionModelDisplayName": prediction_model_display_name(prediction_model),
        },
        eval_pairs=diag_pairs,
        materialized_pairs=new_pairs_for_db,
        ground_truth_pairs=pos_test,
    )

    for state_name in ("initial", "latest"):
        save_lp_learning_state(
            base_db=base_db,
            storage_representation=storage_mode,
            embedding_family=embedding_family,
            prediction_model=prediction_model,
            embedding_property=prop,
            model=model,
            trained_at_timestep=1,
            state_name=state_name,
            embeddings=emb,
            metadata={
                "source": "initial_lp",
                "predictionModel": prediction_model,
                "predictionModelDisplayName": prediction_model_display_name(prediction_model),
                "predictionModelAlgorithm": prediction_model_algorithm_name(prediction_model),
                "predictK": int(body.predictK),
                "probThreshold": float(body.probThreshold),
                "evaluationThreshold": float(body.evaluationThreshold),
                "materializedDatabase": target_db,
                "materializedPairs": len(new_pairs_for_db),
            },
        )


    return {
        "ok": True,
        "train": mtr.model_dump(),
        "validation": mval.model_dump() if mval else None,
        "test": mts.model_dump() if mts else None,
        "model": {
            "algorithm": prediction_model_algorithm_name(prediction_model),
            "predictionModel": prediction_model,
            "predictionModelDisplayName": prediction_model_display_name(prediction_model),
            "embeddingProperty": prop,
            "embeddingFamily": embedding_family,
            "features": "Hadamard",
        },
        "predicted": proj_info,
    }

# --- helpers to list predicted variants for the current DB ---
def _score_candidate_pairs_for_lp(
    *,
    candidate_pairs: List[Tuple[str, str]],
    emb: Dict[str, np.ndarray],
    model: LogisticRegression,
) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []

    for s, t in candidate_pairs:
        key = _pair_key(s, t)

        es = emb.get(key[0])
        et = emb.get(key[1])

        if es is None or et is None:
            continue

        p = float(model.predict_proba((es * et).reshape(1, -1))[0, 1])

        scored.append({
            "s": key[0],
            "t": key[1],
            "probability": p,
        })

    scored.sort(
        key=lambda row: float(row.get("probability", 0.0) or 0.0),
        reverse=True,
    )

    return scored


def _select_top_prediction_rows_lp(
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


def _rows_to_triples_lp(rows: List[Dict[str, Any]]) -> List[Tuple[str, str, float]]:
    return [
        (
            str(row["s"]),
            str(row["t"]),
            float(row.get("probability", 0.0) or 0.0),
        )
        for row in rows
    ]

def _artifact_rows_from_scored_candidates(
    scored_candidates: List[Dict[str, Any]],
    selected_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    selected_keys = {
        _pair_key(str(row["s"]), str(row["t"]))
        for row in selected_rows
    }

    out: List[Dict[str, Any]] = []

    for rank, row in enumerate(scored_candidates, start=1):
        key = _pair_key(str(row["s"]), str(row["t"]))
        out.append(
            {
                "s": key[0],
                "t": key[1],
                "pairKey": f"{key[0]}|{key[1]}",
                "probability": float(row.get("probability", 0.0) or 0.0),
                "rank": int(rank),
                "selected": bool(key in selected_keys),
            }
        )

    return out

def _generate_initial_candidate_pairs_for_explainability(
    *,
    dbname: str,
    emb_prop: str,
    embedding_family: str,
    timestep: int,
    candidate_limit: int,
    storage_representation: str,
) -> List[Tuple[str, str]]:

    prop = _sanitize_prop(emb_prop)

    rows = run_data(
        f"""
        MATCH (n)
        WHERE n.{prop} IS NOT NULL
        WITH collect(elementId(n)) AS ids, count(n) AS N

        CALL {{
            WITH ids, N
            UNWIND range(1, $sampleSize) AS _
            WITH ids, N,
                 ids[toInteger(rand() * N)] AS a,
                 ids[toInteger(rand() * N)] AS b
            WHERE a <> b
            WITH
                CASE WHEN a < b THEN a ELSE b END AS s,
                CASE WHEN a < b THEN b ELSE a END AS t
            RETURN DISTINCT s, t
            LIMIT $sampleSize
        }}

        WITH s, t

        WHERE NOT EXISTS {{
            MATCH (sn)-[r]-(tn)
            WHERE elementId(sn) = s
              AND elementId(tn) = t
              AND coalesce(r.edgeOrigin, 'observed') = 'observed'
        }}

        AND NOT EXISTS {{
            MATCH (sn)-[r]-(tn)
            WHERE elementId(sn) = s
              AND elementId(tn) = t
              AND coalesce(r.edgeOrigin, '') = 'predicted'
              AND coalesce(r.embeddingFamily, '') = 'Consensus'
              AND coalesce(r.consensusPromoted, false) = true
              AND toInteger(coalesce(r.effectiveFromTimestep, 999999999)) <= $timestep
        }}

        AND NOT EXISTS {{
            MATCH (sn)-[r]-(tn)
            WHERE elementId(sn) = s
              AND elementId(tn) = t
              AND coalesce(r.edgeOrigin, '') = 'predicted'
              AND coalesce(r.embeddingFamily, '') = $family
        }}

        RETURN DISTINCT s, t
        LIMIT $candidateLimit
        """,
        {
            "sampleSize": int(candidate_limit) * 4,
            "candidateLimit": int(candidate_limit),
            "family": embedding_family,
            "timestep": int(timestep),
        },
        db=dbname,
    )

    return [(r["s"], r["t"]) for r in rows]

def _observed_database_max_timestamp(dbname: Optional[str] = None) -> int:
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

def _db_exists(dbname: str) -> bool:
    try:
        rows = run_data(
            "SHOW DATABASE $db YIELD name, currentStatus RETURN currentStatus",
            {"db": dbname},
            db="system",
        )
        exists = bool(rows)
        _dbg("_db_exists", dbname=dbname, exists=exists)
        return exists
    except Exception as e:
        _dbg("_db_exists.error", dbname=dbname, error=str(e))
        return False

def _load_predicted_pairs_single_db(dbname: str, family: str) -> list[tuple[str, str]]:

    try:
        rows = run_data(
            """
            MATCH (x)-[r]-(y)
            WHERE coalesce(r.edgeOrigin, '') = 'predicted'
              AND coalesce(r.embeddingFamily, '') = $family
            WITH elementId(x) AS a, elementId(y) AS b
            WITH
              CASE WHEN a < b THEN a ELSE b END AS s,
              CASE WHEN a < b THEN b ELSE a END AS t
            RETURN DISTINCT s, t
            """,
            {"family": family},
            db=dbname,
        )
        return [(r["s"], r["t"]) for r in rows]
    except Exception as e:
        _dbg("_load_predicted_pairs_single_db.error", dbname=dbname, family=family, error=str(e))
        return []


def _predicted_candidates_single_db_for(base_db: str) -> list[dict]:

    rows = run_data(
        """
        MATCH ()-[r]-()
        WHERE coalesce(r.edgeOrigin, '') = 'predicted'
        WITH
          coalesce(r.embeddingFamily, '') AS family,
          coalesce(r.predictionModel, 'logistic_regression') AS predictionModel,
          count(r) AS cnt,
          coalesce(max(toInteger(coalesce(r.predicted_timestep, r.predicted, 0))), 0) AS maxPredicted
        WHERE family <> ''
        RETURN family, predictionModel, cnt, maxPredicted
        """,
        db=base_db,
    )

    counts = {
        (
            r["family"],
            normalize_prediction_model(r.get("predictionModel", "logistic_regression")),
        ): {
            "count": int(r.get("cnt", 0) or 0),
            "maxPredicted": int(r.get("maxPredicted", 0) or 0),
        }
        for r in rows
    }

    out = []

    for family, model_key in valid_timeline_combinations():
        info = counts.get(
            (family, model_key),
            {
                "count": 0,
                "maxPredicted": 0,
            },
        )

        count = int(info["count"])
        max_predicted = int(info["maxPredicted"])

        out.append(
            {
                "name": base_db,
                "type": family,
                "family": family,
                "predictionModel": model_key,
                "predictionModelDisplayName": (
                    prediction_model_display_name(model_key)
                ),
                "exists": count > 0,
                "count": count,
                "maxPredicted": max_predicted,
                "storageRepresentation": SINGLE_DB_MULTI_EDGE,
            }
        )

    _dbg("_predicted_candidates_single_db_for", base=base_db, candidates=out)
    return out


@router.get("/predicted/dbs-single/{base}")
def lp_predicted_dbs_single_db_for(base: str):
    return {
        "base": base,
        "storageRepresentation": SINGLE_DB_MULTI_EDGE,
        "candidates": _predicted_candidates_single_db_for(base),
    }


@router.get(
    "/predicted/series-single-db/"
    "{dbname}/{embeddingFamily}/{predictionModel}"
)
def lp_predicted_series_single_db(
    dbname: str,
    embeddingFamily: str,
    predictionModel: str,
):
    prediction_model = normalize_prediction_model(predictionModel)

    try:
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
              CASE
                WHEN elementId(a) < elementId(b)
                THEN coalesce(a.id, a.eid, elementId(a))
                ELSE coalesce(b.id, b.eid, elementId(b))
              END AS s,
              CASE
                WHEN elementId(a) < elementId(b)
                THEN coalesce(b.id, b.eid, elementId(b))
                ELSE coalesce(a.id, a.eid, elementId(a))
              END AS t,
              type(r) AS relationshipType,
              toFloat(coalesce(r.probability, 0.0)) AS probability,
              toInteger(
                coalesce(
                  r.lastTimestamp,
                  r.timestamp,
                  r.predictedTimestamp,
                  0
                )
              ) AS timestamp,
              toInteger(
                coalesce(
                  r.predicted_timestep,
                  r.predicted,
                  0
                )
              ) AS predicted_timestep

            RETURN DISTINCT
              s,
              t,
              relationshipType,
              probability,
              timestamp,
              predicted_timestep

            ORDER BY
              predicted_timestep ASC,
              timestamp ASC,
              probability DESC,
              s ASC,
              t ASC
            """,
            {
                "family": embeddingFamily,
                "predictionModel": prediction_model,
            },
            db=dbname,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Failed to load single-DB predicted series from "
                f"database '{dbname}': {exc}"
            ),
        ) from exc

    return {
        "db": dbname,
        "embeddingFamily": embeddingFamily,
        "predictionModel": prediction_model,
        "storageRepresentation": SINGLE_DB_MULTI_EDGE,
        "count": len(rows),
        "points": [
            {
                "s": row["s"],
                "t": row["t"],
                "relationshipType": row.get("relationshipType"),
                "probability": float(row.get("probability", 0.0) or 0.0),
                "timestamp": int(row.get("timestamp", 0) or 0),
                "predicted_timestep": int(
                    row.get("predicted_timestep", 0) or 0
                ),
                "predictionModel": prediction_model,
                "predictionModelDisplayName": prediction_model_display_name(
                    prediction_model
                ),
            }
            for row in rows
        ],
    }


def _predicted_candidates_for(
    base_db: str,
) -> list[dict]:
    out: list[dict] = []

    for family, prediction_model in valid_timeline_combinations():
        name = predicted_db_name(
            base_db,
            family,
            prediction_model,
        )

        out.append(
            {
                "name": name,
                "type": family,
                "family": family,
                "predictionModel": prediction_model,
                "predictionModelDisplayName": (
                    prediction_model_display_name(
                        prediction_model
                    )
                ),
                "exists": _db_exists(name),
                "storageRepresentation": SEPARATE_DBS,
            }
        )

    return out

# ------------------------
# Routes
# ------------------------

@router.post("/run")
def lp_run(body: LPBody):
    _dbg("lp_run.endpoint", body=body.dict())
    if normalize_prediction_model(body.predictionModel) == "tgn_direct":
        from . import tgn_direct
        return tgn_direct.tgn_direct_run(
            tgn_direct.TGNDirectBody(
                trainGraphName=body.trainGraphName,
                valGraphName=body.valGraphName,
                testGraphName=body.testGraphName,
                storageRepresentation=body.storageRepresentation,
                negativeRatio=body.negativeRatio,
                predictK=body.predictK,
                candidateMultiplier=body.candidateMultiplier,
                probThreshold=body.probThreshold,
                evaluationThreshold=body.evaluationThreshold,
                outputGraphName=body.outputGraphName,
                force_new_canonical_pool=body.force_new_canonical_pool,
            )
        )
    return _run_lp_logic(body)

@router.post("/train")
def lp_train(body: LPBody):
    _dbg("lp_train.endpoint", body=body.dict())
    if normalize_prediction_model(body.predictionModel) == "tgn_direct":
        return lp_run(body)
    return _run_lp_logic(body)

@router.get("/predicted/dbs")
def lp_predicted_dbs():
    base = current_database()
    _dbg("lp_predicted_dbs.endpoint", base=base)
    return {"base": base, "candidates": _predicted_candidates_for(base)}

@router.get("/predicted/dbs/{base}")
def lp_predicted_dbs_for(base: str):
    _dbg("lp_predicted_dbs_for.endpoint", base=base)
    return {"base": base, "candidates": _predicted_candidates_for(base)}

@router.get("/predicted/edges")
def lp_predicted_edges(
    predictionModel: str = "logistic_regression",
):
    base = _ensure_cache_for_current_db()

    model_key = normalize_prediction_model(
        predictionModel
    )

    families = (
        ["TGN"]
        if model_key == "tgn_direct"
        else list(EMBEDDING_FAMILIES)
    )

    out: dict[str, list[dict[str, str]]] = {}

    for family in families:
        current_timeline_id = timeline_key(
            family,
            model_key,
        )

        pairs = (
            _PREDICTED_EDGES.get(
                current_timeline_id
            )
            or []
        )

        if not pairs:
            dbname = predicted_db_name(
                base,
                family,
                model_key,
            )

            if _db_exists(dbname):
                pairs = _load_predicted_pairs_from_db(
                    dbname
                )

        if pairs:
            out[family] = [
                {
                    "s": s,
                    "t": t,
                }
                for s, t in pairs
            ]

    return {
        "base": base,
        "predictionModel": model_key,
        "edges": out,
    }

@router.get("/test-edges")
def lp_test_edges(
    predictionModel: str = "logistic_regression",
    storageRepresentation: str = SEPARATE_DBS,
):
    base = _ensure_cache_for_current_db()
    storage_mode = normalize_storage_representation(storageRepresentation)
    model_key = normalize_prediction_model(predictionModel)

    payload = load_evaluation_artifacts(
        base_db=base,
        storage_representation=storage_mode,
        prediction_model=model_key,
    )

    pairs = payload.get("groundTruth") or []

    if pairs:
        return {
            "base": base,
            "storageRepresentation": storage_mode,
            "predictionModel": model_key,
            "edges": [{"s": p["s"], "t": p["t"]} for p in pairs],
        }

    fallback_pairs = _get_test_pairs("testGraph")

    return {
        "base": base,
        "storageRepresentation": storage_mode,
        "predictionModel": model_key,
        "edges": [{"s": s, "t": t} for s, t in fallback_pairs],
    }


@router.get("/predicted/edges/by-db")
def lp_predicted_edges_by_db(
    predictionModel: str = "logistic_regression",
):
    base = current_database()
    model_key = normalize_prediction_model(predictionModel)
    out: dict[str, list[dict[str, str]]] = {}

    for candidate in _predicted_candidates_for(base):
        if not candidate.get("exists"):
            continue
        candidate_model = normalize_prediction_model(
            candidate.get("predictionModel", "logistic_regression")
        )
        if candidate_model != model_key:
            continue
        pairs = _load_predicted_pairs_from_db(candidate["name"])
        if pairs:
            family = candidate.get("family") or candidate.get("type")
            out[str(family)] = [
                {"s": s, "t": t}
                for s, t in pairs
            ]

    return {
        "base": base,
        "predictionModel": model_key,
        "edges": out,
    }


@router.get("/metrics")
def lp_metrics(
    predictionModel: str = "logistic_regression",
    storageRepresentation: str = SEPARATE_DBS,
):
    base = _ensure_cache_for_current_db()
    storage_mode = normalize_storage_representation(storageRepresentation)
    model_key = normalize_prediction_model(predictionModel)

    payload = load_evaluation_artifacts(
        base_db=base,
        storage_representation=storage_mode,
        prediction_model=model_key,
    )

    metrics = {}

    for fam, fam_payload in (payload.get("families") or {}).items():
        metrics[fam] = fam_payload.get("metrics") or {}

    return {
        "base": base,
        "storageRepresentation": storage_mode,
        "predictionModel": model_key,
        "metrics": metrics,
    }

@router.get(
    "/predicted/timestamps-single-db/"
    "{dbname}/{embeddingFamily}/{predictionModel}"
)
def lp_predicted_timestamps_single_db(
    dbname: str,
    embeddingFamily: str,
    predictionModel: str,
):
    prediction_model = normalize_prediction_model(predictionModel)

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
            "family": embeddingFamily,
            "predictionModel": prediction_model,
        },
        db=dbname,
    )

    row = rows[0] if rows else {
        "maxPredicted": 0,
        "maxTimestamp": 0,
    }

    return {
        "db": dbname,
        "embeddingFamily": embeddingFamily,
        "predictionModel": prediction_model,
        "predictionModelDisplayName": prediction_model_display_name(
            prediction_model
        ),
        "maxPredicted": int(row.get("maxPredicted", 0) or 0),
        "maxTimestamp": int(row.get("maxTimestamp", 0) or 0),
    }


@router.get("/predicted/timestamps/{dbname}")
def lp_predicted_timestamps(dbname: str):

    try:
        rows = run_data(
            """
            MATCH ()-[r]-()
            WITH
              toInteger(coalesce(r.predicted, 0)) AS lvl,
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
              END AS ts
            RETURN
              coalesce(max(lvl), 0) AS maxPredicted,
              coalesce(max(ts), 0)  AS maxTimestamp
            """,
            db=dbname,
        )
    except Exception as e:
        _dbg("lp_predicted_timestamps.error", dbname=dbname, error=str(e))
        raise HTTPException(
            status_code=400,
            detail=f"Failed to inspect predicted DB '{dbname}': {e}",
        )

    row = rows[0] if rows else {"maxPredicted": 0, "maxTimestamp": 0}
    return {
        "db": dbname,
        "maxPredicted": int(row.get("maxPredicted", 0) or 0),
        "maxTimestamp": int(row.get("maxTimestamp", 0) or 0),
    }

@router.get("/evaluation/predicted-edges")
def lp_evaluation_predicted_edges(
    predictionModel: str = "logistic_regression",
    storageRepresentation: str = SEPARATE_DBS,
):
    base = _ensure_cache_for_current_db()
    storage_mode = normalize_storage_representation(storageRepresentation)
    model_key = normalize_prediction_model(predictionModel)

    payload = load_evaluation_artifacts(
        base_db=base,
        storage_representation=storage_mode,
        prediction_model=model_key,
    )

    out: dict[str, list[dict[str, str]]] = {}

    for fam, fam_payload in (payload.get("families") or {}).items():
        pairs = fam_payload.get("evalPairs") or []
        if pairs:
            out[fam] = [{"s": p["s"], "t": p["t"]} for p in pairs]

    return {
        "base": base,
        "storageRepresentation": storage_mode,
        "predictionModel": model_key,
        "edges": out,
    }

@router.get("/predicted/series/{dbname}")
def lp_predicted_series(dbname: str):
    _dbg("lp_predicted_series.endpoint", dbname=dbname)

    try:
        rows = run_data(
            """
            MATCH (a)-[r]-(b)
            WHERE coalesce(r.edgeOrigin, '') = 'predicted'
            OR toUpper(type(r)) STARTS WITH 'PREDICTED'

            WITH
            CASE
                WHEN elementId(a) < elementId(b)
                THEN coalesce(a.id, a.eid, elementId(a))
                ELSE coalesce(b.id, b.eid, elementId(b))
            END AS s,
            CASE
                WHEN elementId(a) < elementId(b)
                THEN coalesce(b.id, b.eid, elementId(b))
                ELSE coalesce(a.id, a.eid, elementId(a))
            END AS t,
            type(r) AS relationshipType,
            coalesce(r.embeddingFamily, '') AS embeddingFamily,
            coalesce(r.predictionModel, '') AS predictionModel,
            coalesce(r.predictionModelDisplayName, '') AS predictionModelDisplayName,
            toFloat(coalesce(r.probability, 0.0)) AS probability,
            toInteger(coalesce(r.lastTimestamp, r.timestamp, r.predictedTimestamp, 0)) AS timestamp,
            toInteger(coalesce(r.predicted_timestep, r.predicted, 0)) AS predicted_timestep

            RETURN DISTINCT
            s,
            t,
            relationshipType,
            embeddingFamily,
            predictionModel,
            predictionModelDisplayName,
            probability,
            timestamp,
            predicted_timestep

            ORDER BY predicted_timestep ASC, timestamp ASC, probability DESC, s ASC, t ASC
            """,
            db=dbname,
        )
    except Exception as e:
        _dbg("lp_predicted_series.error", dbname=dbname, error=str(e))
        raise HTTPException(
            status_code=400,
            detail=f"Failed to load predicted probability series from DB '{dbname}': {e}",
        )

    return {
        "db": dbname,
        "count": len(rows),
        "points": [
            {
                "s": r["s"],
                "t": r["t"],
                "relationshipType": r.get("relationshipType"),
                "embeddingFamily": r.get("embeddingFamily"),
                "predictionModel": normalize_prediction_model(
                    r.get("predictionModel", "logistic_regression")
                ),
                "predictionModelDisplayName": r.get("predictionModelDisplayName"),
                "probability": float(r.get("probability", 0.0) or 0.0),
                "timestamp": int(r.get("timestamp", 0) or 0),
                "predicted_timestep": int(r.get("predicted_timestep", 0) or 0),
            }
            for r in rows
        ],
    }

@router.post("/predicted/reset-family-single-db")
def lp_reset_family_predictions_single_db(
    body: ResetSingleDbFamilyPredictionsBody,
):
    base = current_database()

    try:
        family, prediction_model = validate_timeline(
            body.embeddingFamily,
            body.predictionModel,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    timeline_id = timeline_key(
        family,
        prediction_model,
    )

    only_level = (
        None
        if body.deleteAllLevels
        else 1
    )

    deleted_info = (
        _delete_single_db_timeline_predictions(
            dbname=base,
            embedding_family=family,
            prediction_model=prediction_model,
            only_level=only_level,
        )
    )

    _clear_prediction_memory_for_family(
        timeline_id
    )

    return {
        "ok": True,
        "storageRepresentation": SINGLE_DB_MULTI_EDGE,
        "timelineId": timeline_id,
        **deleted_info,
    }