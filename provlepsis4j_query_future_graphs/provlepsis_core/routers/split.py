# app/routers/split.py
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Tuple, List
from uuid import uuid4
import re
import random

from ..db import run, run_data
from neo4j.exceptions import Neo4jError

router = APIRouter()

# =========================
# Feature helpers (optional)
# =========================
try:
    from .feature import get_projection_props
    _FEATURES_OK = True
except Exception:
    get_projection_props = None
    _FEATURES_OK = False

_FEATURE_PROPS_CACHE: Dict[str, List[str]] = {}
_FEATURE_PROPS_META: Dict[str, Dict[str, Any]] = {}


def _current_db_name() -> str:
    try:
        row = run_data("CALL db.info() YIELD name RETURN name")[0]
        return row["name"] or "neo4j"
    except Exception:
        return "neo4j"


def clear_feature_cache(db: Optional[str] = None):
    """
    Clear cached feature projection list.
    If db is None, clear all.
    """
    if db:
        _FEATURE_PROPS_CACHE.pop(db, None)
        _FEATURE_PROPS_META.pop(db, None)
    else:
        _FEATURE_PROPS_CACHE.clear()
        _FEATURE_PROPS_META.clear()


# ---- property discovery & filtering ----

def _all_node_labels() -> List[str]:
    rows = run_data("CALL db.labels() YIELD label RETURN collect(label) AS labels")
    return rows[0]["labels"] if rows else []


def _autodiscover_node_props(labels: List[str]) -> List[str]:
    """
    Heuristic discovery: union of keys() across a sample of nodes for the given labels.
    """
    if not labels:
        return []
    label_filter = " OR ".join([f"n:`{lbl}`" for lbl in labels])
    rows = run_data(f"""
        MATCH (n)
        WHERE {label_filter}
        WITH n LIMIT 2000
        WITH apoc.coll.toSet(apoc.coll.flatten(collect(keys(n)))) AS ks
        UNWIND ks AS k
        WITH DISTINCT k
        WHERE NOT k IN ['_id','_tmp','_ts']   // ignore obviously internal-ish keys if present
        RETURN collect(k) AS props
    """)
    return rows[0].get("props", []) if rows else []


def ensure_feature_props_loaded(force: bool = False) -> List[str]:
    """
    1) Try feature.get_projection_props()
    2) If empty/unavailable, auto-discover from the data
    Cache is per-DB and auto-invalidates if label set changes.
    """
    db = _current_db_name()

    # Quick label snapshot to detect schema changes
    labels = _all_node_labels()
    label_sig = tuple(sorted(labels))

    if not force:
        cached = _FEATURE_PROPS_CACHE.get(db)
        meta = _FEATURE_PROPS_META.get(db)
        if cached is not None and meta and meta.get("labels") == label_sig:
            return cached

    base: List[str] = []

    if _FEATURES_OK and get_projection_props:
        try:
            base = list(sorted(set(get_projection_props() or [])))
        except Exception:
            base = []

    if not base:
        base = _autodiscover_node_props(labels)

    # Save (or overwrite) cache for this DB
    _FEATURE_PROPS_CACHE[db] = base
    _FEATURE_PROPS_META[db] = {"labels": label_sig}
    return base


_ALLOWED_SCALARS = (int, float, bool, str)


def _filter_supported_node_props(props: List[str]) -> List[str]:
    """
    Keep props whose sampled non-null values look GDS-acceptable.
    If filtering would drop everything, fall back to the original list so we still project.
    """
    if not props:
        return []

    safe: List[str] = []
    for p in props:
        rows = run_data(
            "WITH $k AS k MATCH (n) WHERE n[k] IS NOT NULL RETURN n[k] AS v LIMIT 500",
            {"k": p},
        )

        # If we saw no non-null samples, keep it (features can be sparse)
        if not rows:
            safe.append(p)
            continue

        ok = True
        for r in rows:
            v = r.get("v")
            if v is None:
                continue
            if isinstance(v, _ALLOWED_SCALARS):
                continue
            if isinstance(v, list):
                # allow numeric arrays (embeddings, etc.)
                if all((x is None) or isinstance(x, (int, float)) for x in v):
                    continue
            ok = False
            break

        if ok:
            safe.append(p)

    return safe if safe else list(props)


# ----- Embedding property exclusions -----
# Exact property names we should NOT project as node features
_EMBED_PROP_NAMES = {
    "FastRP", "Node2Vec", "GraphSAGE", "HashGNN", "TGN",
    # common variants / lowercase
    "fastrp", "node2vec", "graphsage", "hashgnn", "tgn",
    "embedding_n2v_128", "embedding_fastrp_256", "embedding_sage_64", "embedding_hash_128",
}

# Conservative prefixes to exclude
_EMBED_PROP_PREFIXES = {
    "fastrp_", "node2vec_", "graphsage_", "hashgnn_", "tgn_",
    "emb_", "embedding_",
}


def _drop_embedding_props(props: List[str]) -> List[str]:
    """Remove properties that look like embedding outputs."""
    cleaned = []
    for p in props:
        if p in _EMBED_PROP_NAMES:
            continue
        if any(p.startswith(pref) for pref in _EMBED_PROP_PREFIXES):
            continue
        cleaned.append(p)
    return cleaned


# =========================
# Split cleanup safety
# =========================
def _delete_all_split_rels():
    """
    Safety reset in case a previous split run crashed/interrupted and left SPLIT_* rels behind.
    """
    try:
        run("MATCH ()-[r]-() WHERE type(r) STARTS WITH 'SPLIT_' DELETE r")
    except Exception:
        pass


# =========================
# GDS projection utilities
# =========================
def _drop_in_memory_gds_graph(name: str):
    try:
        run("CALL gds.graph.drop($name, false)", {"name": name})
    except Exception:
        pass


def _project_with_native(
    graph_name: str,
    rel_types: List[str],
    node_props: List[str]
) -> Dict[str, Any]:
    """
    Project using gds.graph.project with per-label nodeProjection
    so node properties are actually included.
    NOTE: This projects ALL labels; we prefer the temp-endpoint-label variant below
    to include only participating nodes.
    """
    labels = _all_node_labels()
    if not labels:
        return {"graphName": graph_name, "nodeCount": 0, "relationshipCount": 0, "note": "no labels in DB"}

    # Per-label nodeProjection including properties
    node_projection = {lbl: ({"properties": node_props} if node_props else {}) for lbl in labels}

    _drop_in_memory_gds_graph(graph_name)

    rows = run_data(
        """
        CALL gds.graph.project(
          $name,
          $nodeProjection,
          $relationshipTypes
        )
        YIELD graphName, nodeCount, relationshipCount
        RETURN graphName, nodeCount, relationshipCount
        """,
        {
            "name": graph_name,
            "nodeProjection": node_projection,
            "relationshipTypes": rel_types,
        },
    )
    return rows[0] if rows else {"graphName": graph_name, "nodeCount": 0, "relationshipCount": 0}


# ---------- temp-endpoint label projection (keeps only participating nodes) ----------
def _safe_label(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", s)


def _tag_split_endpoints(temp_label: str, reltype: str):
    temp_label = _safe_label(temp_label)
    run(f"""
      MATCH ()-[r:`{reltype}`]-()
      WITH startNode(r) AS a, endNode(r) AS b
      SET a:`{temp_label}`, b:`{temp_label}`
    """)


def _untag_split_endpoints(temp_label: str):
    temp_label = _safe_label(temp_label)
    run(f"MATCH (n:`{temp_label}`) REMOVE n:`{temp_label}`")


def _project_with_label_only(
    graph_name: str,
    endpoint_label: str,
    rel_types: List[str],
    node_props: List[str],
) -> Dict[str, Any]:
    """
    Project only nodes having `endpoint_label`, and only the provided rel_types.
    Node properties are included as in your native projection.
    """
    endpoint_label = _safe_label(endpoint_label)
    node_projection = {endpoint_label: ({"properties": node_props} if node_props else {})}
    _drop_in_memory_gds_graph(graph_name)
    rows = run_data(
        """
        CALL gds.graph.project(
          $name,
          $nodeProjection,
          $relationshipTypes
        )
        YIELD graphName, nodeCount, relationshipCount
        RETURN graphName, nodeCount, relationshipCount
        """,
        {
            "name": graph_name,
            "nodeProjection": node_projection,
            "relationshipTypes": rel_types,
        },
    )
    return rows[0] if rows else {"graphName": graph_name, "nodeCount": 0, "relationshipCount": 0}


# =========================
# Connected-train helpers 
# =========================

def _relationship_types_original() -> List[str]:
    rows = run_data("""
      CALL db.relationshipTypes() YIELD relationshipType
      RETURN relationshipType AS type
    """)
    return [r["type"] for r in rows if not str(r["type"]).startswith("SPLIT_")]


def _project_base_for_split(gname: str) -> Optional[str]:
    """
    Project the ORIGINAL graph (no SPLIT_ rels) as UNDIRECTED topology,
    so BFS/WCC see weak connectivity.
    """
    labels = _all_node_labels()
    if not labels:
        return None
    rel_types = _relationship_types_original()
    if not rel_types:
        return None

    # UNDIRECTED relationship projection to ensure BFS/WCC cover weak connectivity
    rel_proj = {rt: {"orientation": "UNDIRECTED"} for rt in rel_types}

    _drop_in_memory_gds_graph(gname)
    run_data("""
      CALL gds.graph.project($g, $nodeProj, $relProj)
      YIELD graphName
      RETURN graphName
    """, {
      "g": gname,
      "nodeProj": {lbl: {} for lbl in labels},  # topology only
      "relProj": rel_proj,
    })
    return gname


def _backbone_edges_with_bfs(gname: str) -> List[Tuple[int, int]]:
    """
    Return undirected node-id pairs (u,v) (u<v) that form a spanning forest
    across all WCCs in the base projection.
    """
    try:
        seeds = run_data("""
          CALL gds.wcc.stream($g)
          YIELD nodeId, componentId
          WITH componentId, collect(nodeId) AS ns
          RETURN ns[0] AS seed
        """, {"g": gname})

        backbone: List[Tuple[int, int]] = []
        for row in seeds:
            seed = int(row["seed"])
            rows = run_data("""
              CALL gds.bfs.stream($g, {sourceNode: $seed})
              YIELD nodeIds
              WITH nodeIds
              UNWIND range(1, size(nodeIds) - 1) AS i
              WITH nodeIds[i - 1] AS parentNodeId, nodeIds[i] AS nodeId
              WITH gds.util.asNode(nodeId) AS n,
                   gds.util.asNode(parentNodeId) AS p
              RETURN id(p) AS u, id(n) AS v
            """, {"g": gname, "seed": seed})

            for r in rows:
                u = int(r["u"])
                v = int(r["v"])
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                backbone.append((u, v))

        return list({e for e in backbone})
    except Neo4jError:
        return []


def _create_train_edges_from_node_ids(
    backbone_pairs: List[Tuple[int, int]],
    reltype: str,
    batch_size: int = 5000
) -> int:
    """
    Create mirrored TRAIN edges for the given undirected backbone pairs (u,v) of Neo4j node IDs.
    - Skips any pair that already has a SPLIT_* edge between the endpoints (no overlaps).
    - Uses MERGE to be idempotent.
    Returns the number of DISTINCT undirected pairs written for `reltype`.
    """
    if not backbone_pairs:
        return 0

    for i in range(0, len(backbone_pairs), batch_size):
        chunk = backbone_pairs[i:i + batch_size]
        run(f"""
          UNWIND $pairs AS p
          MATCH (a) WHERE id(a) = p[0]
          MATCH (b) WHERE id(b) = p[1]
          WITH a, b
          WHERE NOT EXISTS {{ MATCH (a)-[r]-(b) WHERE type(r) STARTS WITH 'SPLIT_' }}
          MERGE (a)-[:`{reltype}`]->(b)
          MERGE (b)-[:`{reltype}`]->(a)
        """, {"pairs": [[u, v] for (u, v) in chunk]})

    rows = run_data(f"""
      MATCH ()-[r:`{reltype}`]-()
      WITH id(startNode(r)) AS s, id(endNode(r)) AS t
      WITH CASE WHEN s < t THEN [s, t] ELSE [t, s] END AS p
      RETURN count(DISTINCT p) AS k
    """)
    return int(rows[0]["k"]) if rows else 0


# =========================
# Split materialization
# =========================
def _mk_run_suffix() -> str:
    # Relationship type suffix must be ascii letters/digits/underscore only.
    return uuid4().hex[:6].upper()


class _SplitTypes(Dict[str, str]):
    test: str
    val: str
    train: str


def _build_run_types(run_suffix: str) -> _SplitTypes:
    return {
        "test":  f"SPLIT_TEST_{run_suffix}",
        "val":   f"SPLIT_VAL_{run_suffix}",
        "train": f"SPLIT_TRAIN_{run_suffix}",
    }


def _materialize_temp_split_relationships(
    test_holdout: float,
    val_holdout: Optional[float],
    types: _SplitTypes,
    batch_size: int = 5000
) -> Dict[str, int]:
    """
    Random split (fallback). Kept for completeness behind ensureConnected flag.
    """
    t = max(0.0, min(1.0, float(test_holdout)))
    v = max(0.0, min(1.0, float(val_holdout or 0.0)))

    # Count P over UNDIRECTED pairs, ignoring any SPLIT_* rels
    row = run_data("""
        MATCH (a)-[r]-(b)
        WHERE NOT type(r) STARTS WITH 'SPLIT_'
        WITH
          CASE WHEN id(a) < id(b) THEN id(a) ELSE id(b) END AS s,
          CASE WHEN id(a) < id(b) THEN id(b) ELSE id(a) END AS t,
          type(r) AS typ
        RETURN count(DISTINCT [s,t,typ]) AS P
    """)[0]
    P = int(row["P"]) if row and row.get("P") is not None else 0

    from math import floor
    k_test = int(floor(P * t))
    k_val = int(floor(P * v))
    k_val = max(0, min(P - k_test, k_val))
    k_train = max(0, P - k_test - k_val)

    def _create_pairs(limit_k: int, reltype: str, exclude_types: List[str], batch_size: int = 5000) -> int:
        if limit_k <= 0:
            return 0

        cypher = f"""
        CALL {{
          WITH $k AS k, $ex AS ex
          MATCH (a)-[r]-(b)
          WHERE NOT type(r) STARTS WITH 'SPLIT_'
            AND ALL(x IN ex WHERE NOT EXISTS {{
              MATCH (a)-[r2]-(b) WHERE type(r2) = x
            }})
          WITH
            CASE WHEN id(a) < id(b) THEN id(a) ELSE id(b) END AS s,
            CASE WHEN id(a) < id(b) THEN id(b) ELSE id(a) END AS t,
            type(r) AS typ
          WITH DISTINCT s,t,typ
          ORDER BY rand()
          LIMIT $k
          RETURN s,t,typ
        }}
        CALL {{
          WITH s,t,typ
          MATCH (sN) WHERE id(sN)=s
          MATCH (tN) WHERE id(tN)=t
          CREATE (sN)-[:`{reltype}`]->(tN)
          CREATE (tN)-[:`{reltype}`]->(sN)
        }} IN TRANSACTIONS OF $batch ROWS
        RETURN 1
        """
        run(cypher, {"k": limit_k, "ex": exclude_types, "batch": batch_size})

        rows = run_data(f"""
          MATCH ()-[r:`{reltype}`]-()
          WITH id(startNode(r)) AS s, id(endNode(r)) AS t
          WITH CASE WHEN s < t THEN [s,t] ELSE [t,s] END AS p
          RETURN count(DISTINCT p) AS k
        """)
        return int(rows[0]["k"]) if rows else 0

    c_test = _create_pairs(k_test, types["test"], exclude_types=[], batch_size=batch_size)
    exclude = [types["test"]]
    c_val = 0
    if k_val > 0:
        c_val = _create_pairs(k_val, types["val"], exclude_types=exclude, batch_size=batch_size)
        exclude.append(types["val"])
    c_train = _create_pairs(k_train, types["train"], exclude_types=exclude, batch_size=batch_size)

    return {"P": P, "k_test": c_test, "k_val": c_val, "k_train": c_train}


def _materialize_temp_split_relationships_connected(
    test_holdout: float,
    val_holdout: Optional[float],
    types: _SplitTypes,
    run_suffix: str,
    batch_size: int = 5000
) -> Dict[str, int]:
    """
    Initial connected-ish split:
    - Build backbone/spanning forest into SPLIT_TRAIN_<RUN>
    - Randomly assign remaining edges to TEST/VAL/TRAIN
    """
    t = max(0.0, min(1.0, float(test_holdout)))
    v = max(0.0, min(1.0, float(val_holdout or 0.0)))

    row = run_data("""
        MATCH (a)-[r]-(b)
        WHERE NOT type(r) STARTS WITH 'SPLIT_'
        WITH CASE WHEN id(a) < id(b) THEN [id(a), id(b)] ELSE [id(b), id(a)] END AS p
        RETURN count(DISTINCT p) AS P
    """)[0]
    P = int(row["P"]) if row and row.get("P") is not None else 0

    from math import floor
    k_test = int(floor(P * t))
    k_val = int(floor(P * v))
    k_val = max(0, min(P - k_test, k_val))
    k_train_quota = max(0, P - k_test - k_val)

    base_g = f"_SPLIT_BASE_{run_suffix}"
    g = _project_base_for_split(base_g)
    backbone_pairs: List[Tuple[int, int]] = []
    try:
        if g:
            backbone_pairs = _backbone_edges_with_bfs(g)
    finally:
        if g:
            _drop_in_memory_gds_graph(g)

    c_backbone = _create_train_edges_from_node_ids(backbone_pairs, types["train"], batch_size=batch_size)

    def _create_pairs_excluding_backbone(limit_k: int, reltype: str, exclude_types: List[str]) -> int:
        if limit_k <= 0:
            return 0

        cypher = f"""
        CALL {{
          WITH $k AS k, $ex AS ex
          MATCH (a)-[r]-(b)
          WHERE NOT type(r) STARTS WITH 'SPLIT_'
            AND ALL(x IN ex WHERE NOT EXISTS {{
              MATCH (a)-[r2]-(b) WHERE type(r2) = x
            }})
            AND NOT EXISTS {{
              MATCH (a)-[rT:`{types["train"]}`]-(b)
            }}
          WITH
            CASE WHEN id(a) < id(b) THEN id(a) ELSE id(b) END AS s,
            CASE WHEN id(a) < id(b) THEN id(b) ELSE id(a) END AS t,
            type(r) AS typ
          WITH DISTINCT s,t,typ
          ORDER BY rand()
          LIMIT $k
          RETURN s,t,typ
        }}
        CALL {{
          WITH s,t,typ
          MATCH (sN) WHERE id(sN)=s
          MATCH (tN) WHERE id(tN)=t
          CREATE (sN)-[:`{reltype}`]->(tN)
          CREATE (tN)-[:`{reltype}`]->(sN)
        }} IN TRANSACTIONS OF $batch ROWS
        RETURN 1
        """
        run(cypher, {"k": limit_k, "ex": exclude_types, "batch": batch_size})
        rows = run_data(f"""
          MATCH ()-[r:`{reltype}`]-()
          WITH id(startNode(r)) AS s, id(endNode(r)) AS t
          WITH CASE WHEN s < t THEN [s,t] ELSE [t,s] END AS p
          RETURN count(DISTINCT p) AS k
        """)
        return int(rows[0]["k"]) if rows else 0

    c_test = _create_pairs_excluding_backbone(k_test, types["test"], exclude_types=[])
    exclude = [types["test"]]
    c_val = 0
    if k_val > 0:
        c_val = _create_pairs_excluding_backbone(k_val, types["val"], exclude_types=exclude)
        exclude.append(types["val"])
    remaining_train = max(0, k_train_quota - c_backbone)
    c_train_extra = _create_pairs_excluding_backbone(remaining_train, types["train"], exclude_types=exclude)

    return {
        "P": P,
        "k_backbone": c_backbone,
        "k_test": c_test,
        "k_val": c_val,
        "k_train": c_backbone + c_train_extra,
    }


def _cleanup_temp_types(types: _SplitTypes):
    # Remove only the temp relationships created for this run.
    for t in [types["test"], types["val"], types["train"]]:
        try:
            run(f"MATCH ()-[r:`{t}`]-() DELETE r")
        except Exception:
            pass

def _all_undirected_pairs_original() -> List[Tuple[int, int]]:
    """
    All original undirected pairs, ignoring temporary SPLIT_* rels.
    """
    rows = run_data("""
      MATCH (a)-[r]-(b)
      WHERE NOT type(r) STARTS WITH 'SPLIT_'
      WITH
        CASE WHEN id(a) < id(b) THEN id(a) ELSE id(b) END AS u,
        CASE WHEN id(a) < id(b) THEN id(b) ELSE id(a) END AS v
      RETURN DISTINCT u, v
    """)
    return [(int(r["u"]), int(r["v"])) for r in rows]


def _wcc_map_for_projected_graph(gname: str) -> Tuple[int, Dict[int, int], Dict[int, List[int]]]:
    """
    Returns:
      (componentCount,
       { neo4jNodeId -> componentId },
       { componentId -> [neo4jNodeId] })
    """
    rows = run_data("""
      CALL gds.wcc.stream($g)
      YIELD nodeId, componentId
      WITH gds.util.asNode(nodeId) AS n, componentId
      RETURN id(n) AS nid, componentId
    """, {"g": gname})

    comp_map: Dict[int, int] = {}
    groups: Dict[int, List[int]] = {}

    for r in rows:
        nid = int(r["nid"])
        cid = int(r["componentId"])
        comp_map[nid] = cid
        groups.setdefault(cid, []).append(nid)

    return len(groups), comp_map, groups


def _stitch_train_components(
    train_graph_name: str,
    train_reltype: str,
    batch_size: int = 5000
) -> int:

    comp_count, comp_map, groups = _wcc_map_for_projected_graph(train_graph_name)
    if comp_count <= 1:
        return 0

    # Largest component becomes the hub
    main_cid = max(groups.keys(), key=lambda cid: len(groups[cid]))
    main_rep = groups[main_cid][0]

    bridges: List[Tuple[int, int]] = []
    for cid, nodes in groups.items():
        if cid == main_cid:
            continue
        rep = nodes[0]
        u, v = main_rep, rep
        if u == v:
            continue
        bridges.append((u, v))

    if not bridges:
        return 0

    for i in range(0, len(bridges), batch_size):
        chunk = bridges[i:i + batch_size]
        run(f"""
          UNWIND $pairs AS p
          MATCH (a) WHERE id(a) = p[0]
          MATCH (b) WHERE id(b) = p[1]
          MERGE (a)-[:`{train_reltype}`]->(b)
          MERGE (b)-[:`{train_reltype}`]->(a)
        """, {"pairs": [[u, v] for (u, v) in chunk]})

    return len(bridges)


def _tag_all_original_nodes(temp_label: str):
    temp_label = _safe_label(temp_label)
    run(f"""
      MATCH (n)
      SET n:`{temp_label}`
    """)

def _create_split_edges_from_node_ids(
    pairs: List[Tuple[int, int]],
    reltype: str,
    batch_size: int = 5000
) -> int:
    if not pairs:
        return 0

    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i + batch_size]
        run(f"""
          UNWIND $pairs AS p
          MATCH (a) WHERE id(a) = p[0]
          MATCH (b) WHERE id(b) = p[1]
          MERGE (a)-[:`{reltype}`]->(b)
          MERGE (b)-[:`{reltype}`]->(a)
        """, {"pairs": [[u, v] for (u, v) in chunk]})

    rows = run_data(f"""
      MATCH ()-[r:`{reltype}`]-()
      WITH id(startNode(r)) AS s, id(endNode(r)) AS t
      WITH CASE WHEN s < t THEN [s, t] ELSE [t, s] END AS p
      RETURN count(DISTINCT p) AS k
    """)
    return int(rows[0]["k"]) if rows else 0


def _materialize_temporal_split_relationships(
    test_holdout: float,
    val_holdout: Optional[float],
    types: _SplitTypes,
    time_prop: str = "firstTimestamp",
    batch_size: int = 5000,
    explicit_test_timestamps: Optional[List[float]] = None,
    explicit_val_timestamps: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Temporal split by timestamp-occurrence slicing.

    Important semantics:
    - We do NOT assign a whole edge to exactly one set anymore.
    - We split the *timestamps of each pair* across train/val/test.
    - Therefore the same pair can appear in multiple sets, each with different
      temporal information.

    Example:
      pair (a,b) with timestamps [1,2,3,4,5]
      and holdout bands {4}=val, {5}=test
      =>
      train gets [1,2,3]
      val   gets [4]
      test  gets [5]

    If explicit_test_timestamps / explicit_val_timestamps are provided,
    they override the percentage-based timestamp-band selection.
    """
    from math import ceil

    rows = run_data("""
        MATCH (a)-[r]-(b)
        WHERE NOT type(r) STARTS WITH 'SPLIT_'
        WITH
          CASE WHEN id(a) < id(b) THEN id(a) ELSE id(b) END AS u,
          CASE WHEN id(a) < id(b) THEN id(b) ELSE id(a) END AS v,
          CASE
            WHEN r.timestamps IS NOT NULL
                 AND valueType(r.timestamps) STARTS WITH 'LIST'
                 AND size(r.timestamps) > 0
              THEN [x IN r.timestamps | toFloat(x)]
            ELSE [toFloat(coalesce(r.lastTimestamp, r.firstTimestamp, r.timestamp, 0.0))]
          END AS rel_ts
        RETURN u, v, rel_ts
    """)

    pair_map: Dict[Tuple[int, int], set[float]] = {}
    for r in rows:
        u = int(r["u"])
        v = int(r["v"])
        key = (u, v)
        pair_map.setdefault(key, set())

        for ts in (r.get("rel_ts") or []):
            if ts is None:
                continue
            try:
                pair_map[key].add(float(ts))
            except Exception:
                continue

    P = len(pair_map)
    if P == 0:
        return {
            "P": 0,
            "k_train": 0,
            "k_val": 0,
            "k_test": 0,
            "uniqueTimestampCount": 0,
            "testTimestamps": [],
            "valTimestamps": [],
        }

    uniq_ts = sorted({ts for ts_set in pair_map.values() for ts in ts_set})
    uniq_ts_set = set(uniq_ts)

    # -------------------------------------------------
    # 1) decide which timestamp bands belong to test/val
    # -------------------------------------------------
    if explicit_test_timestamps is not None:
        test_times = {
            float(x) for x in explicit_test_timestamps
            if float(x) in uniq_ts_set
        }
    else:
        n_test = int(ceil(len(uniq_ts) * max(0.0, min(1.0, float(test_holdout)))))
        n_test = min(n_test, len(uniq_ts))
        test_times = set(uniq_ts[-n_test:]) if n_test > 0 else set()

    if explicit_val_timestamps is not None:
        val_times = {
            float(x) for x in explicit_val_timestamps
            if float(x) in uniq_ts_set
        }
        val_times = val_times - test_times
    else:
        n_test = len(test_times)
        n_val = int(ceil(len(uniq_ts) * max(0.0, min(1.0, float(val_holdout or 0.0)))))
        n_val = min(n_val, max(0, len(uniq_ts) - n_test))

        if n_val > 0:
            end = len(uniq_ts) - n_test if n_test > 0 else len(uniq_ts)
            start = max(0, end - n_val)
            val_times = set(uniq_ts[start:end])
        else:
            val_times = set()

    # -------------------------------------------------
    # 2) slice each pair's timestamp list across sets
    # -------------------------------------------------
    train_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []

    for (u, v), ts_set in sorted(pair_map.items()):
        ts_sorted = sorted(ts_set)

        train_ts = [ts for ts in ts_sorted if ts not in test_times and ts not in val_times]
        val_ts = [ts for ts in ts_sorted if ts in val_times]
        test_ts = [ts for ts in ts_sorted if ts in test_times]

        if train_ts:
            train_rows.append({
                "u": u,
                "v": v,
                "timestamps": train_ts,
                "firstTimestamp": train_ts[0],
                "lastTimestamp": train_ts[-1],
                "interactionCount": len(train_ts),
            })

        if val_ts:
            val_rows.append({
                "u": u,
                "v": v,
                "timestamps": val_ts,
                "firstTimestamp": val_ts[0],
                "lastTimestamp": val_ts[-1],
                "interactionCount": len(val_ts),
            })

        if test_ts:
            test_rows.append({
                "u": u,
                "v": v,
                "timestamps": test_ts,
                "firstTimestamp": test_ts[0],
                "lastTimestamp": test_ts[-1],
                "interactionCount": len(test_ts),
            })

    c_train = _create_temporal_split_edges_from_rows(train_rows, types["train"], batch_size=batch_size)
    c_val   = _create_temporal_split_edges_from_rows(val_rows,   types["val"],   batch_size=batch_size)
    c_test  = _create_temporal_split_edges_from_rows(test_rows,  types["test"],  batch_size=batch_size)

    return {
        "P": P,
        "k_train": c_train,
        "k_val": c_val,
        "k_test": c_test,
        "uniqueTimestampCount": len(uniq_ts),
        "testTimestamps": sorted(test_times),
        "valTimestamps": sorted(val_times),
    }

# =========================
# Connectivity helper
# =========================
def _train_connectivity(name: str) -> Tuple[Optional[bool], Optional[int]]:
    try:
        rows = run_data("CALL gds.wcc.stats($g) YIELD componentCount RETURN componentCount", {"g": name})
        if rows and "componentCount" in rows[0]:
            comp = int(rows[0]["componentCount"])
            return (comp == 1, comp)
    except Exception:
        pass
    return (None, None)


# =========================
# Request model
# =========================
class ExecuteSplitBody(BaseModel):
    trainGraphName: str = "trainGraph"
    testGraphName: str = "testGraph"
    valGraphName: Optional[str] = "valGraph"
    testHoldout: float = Field(default=0.10, ge=0.0, le=1.0)
    valHoldout: float = Field(default=0.10, ge=0.0, le=1.0)
    ensureConnected: bool = True
    includeEmbeddingProps: bool = False
    reFreshFeatureCache: bool = True
    temporalMode: bool = False
    timeProperty: str = "firstTimestamp"
    explicitTestTimestamps: Optional[List[float]] = None
    explicitValTimestamps: Optional[List[float]] = None   

def _create_temporal_split_edges_from_rows(
    rows: List[Dict[str, Any]],
    reltype: str,
    batch_size: int = 5000,
) -> int:
    """
    Create mirrored SPLIT_* relationships carrying sliced temporal info.

    Each row must contain:
      {
        "u": <neo4j node id>,
        "v": <neo4j node id>,
        "timestamps": [...],
        "firstTimestamp": ...,
        "lastTimestamp": ...,
        "interactionCount": ...
      }
    """
    if not rows:
        return 0

    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        run(f"""
          UNWIND $rows AS row
          MATCH (a) WHERE id(a) = row.u
          MATCH (b) WHERE id(b) = row.v

          MERGE (a)-[r:`{reltype}`]->(b)
          SET r.timestamps = row.timestamps,
              r.firstTimestamp = row.firstTimestamp,
              r.lastTimestamp = row.lastTimestamp,
              r.timestamp = row.lastTimestamp,
              r.interactionCount = row.interactionCount

          MERGE (b)-[r2:`{reltype}`]->(a)
          SET r2.timestamps = row.timestamps,
              r2.firstTimestamp = row.firstTimestamp,
              r2.lastTimestamp = row.lastTimestamp,
              r2.timestamp = row.lastTimestamp,
              r2.interactionCount = row.interactionCount
        """, {"rows": chunk})

    return len(rows)

# =========================
# Route
# =========================
@router.post("/execute")
def execute_master_split(body: ExecuteSplitBody):
    _delete_all_split_rels()

    run_suffix = _mk_run_suffix()
    types = _build_run_types(run_suffix)
    bridges_added = 0

    # Load & filter node properties for projection
    raw_props = ensure_feature_props_loaded(force=body.reFreshFeatureCache)
    props = _filter_supported_node_props(raw_props)
    # exclude embedding-like props unless explicitly allowed
    if not body.includeEmbeddingProps:
        props = _drop_embedding_props(props)

    # Optional VAL creation
    want_val = bool(body.valGraphName) and (body.valHoldout > 0.0)

    # Create unique, safe endpoint labels per slice
    train_ep_label = f"__SPLIT_TRAIN_EP_{run_suffix}__"
    test_ep_label = f"__SPLIT_TEST_EP_{run_suffix}__"
    val_ep_label = f"__SPLIT_VAL_EP_{run_suffix}__"

    try:
        # 1) Materialize temporary split relationships
        if body.temporalMode:
            counts = _materialize_temporal_split_relationships(
                test_holdout=body.testHoldout,
                val_holdout=body.valHoldout if want_val else 0.0,
                types=types,
                time_prop=body.timeProperty, 
                explicit_test_timestamps=body.explicitTestTimestamps,
                explicit_val_timestamps=body.explicitValTimestamps if want_val else [],
            )
        else:
            if body.ensureConnected:
                counts = _materialize_temp_split_relationships_connected(
                    test_holdout=body.testHoldout,
                    val_holdout=body.valHoldout if want_val else 0.0,
                    types=types,
                    run_suffix=run_suffix
                )
            else:
                counts = _materialize_temp_split_relationships(
                    test_holdout=body.testHoldout,
                    val_holdout=body.valHoldout if want_val else 0.0,
                    types=types
                )

        # 2) Tag all ORIGINAL nodes so they appear as nodes in the TRAIN projection
        _tag_all_original_nodes(train_ep_label)

        # 3) Also tag TRAIN endpoints
        _tag_split_endpoints(train_ep_label, types["train"])

        # 4) Initial TRAIN projection
        train_info = _project_with_label_only(
            body.trainGraphName, train_ep_label, [types["train"]], props
        )

        # 5) Final bridge
        if body.ensureConnected and not body.temporalMode:
            bridges_added = _stitch_train_components(
                body.trainGraphName,
                types["train"]
            )

            if bridges_added > 0:
                _tag_split_endpoints(train_ep_label, types["train"])
                train_info = _project_with_label_only(
                    body.trainGraphName, train_ep_label, [types["train"]], props
                )

        # TEST
        _tag_split_endpoints(test_ep_label, types["test"])
        test_info = _project_with_label_only(
            body.testGraphName, test_ep_label, [types["test"]], props
        )

        # VAL (optional)
        val_info = None
        if want_val:
            _tag_split_endpoints(val_ep_label, types["val"])
            val_info = _project_with_label_only(
                body.valGraphName or "valGraph", val_ep_label, [types["val"]], props
            )

        # 6) Connectivity on final train graph
        if body.temporalMode:
            connected, components = (None, None)
        else:
            connected, components = _train_connectivity(body.trainGraphName)

        return {
            "ok": True,
            "counts": counts,  
            "train": train_info,
            "test": test_info,
            "validation": val_info,
            "testHoldout": body.testHoldout,
            "valHoldout": body.valHoldout if want_val else 0.0,
            "trainConnected": connected,
            "trainComponents": components,
            "featurePropsRequested": raw_props,
            "featurePropsUsed": props,
            "runSuffix": run_suffix,
            "temporalMode": body.temporalMode,
            "timeProperty": body.timeProperty
        }

    finally:
        # Always clean up the temporary relationships we created for this run
        _cleanup_temp_types(types)

        # Remove temporary endpoint labels
        for lbl in (train_ep_label, test_ep_label, val_ep_label):
            try:
                _untag_split_endpoints(lbl)
            except Exception:
                pass