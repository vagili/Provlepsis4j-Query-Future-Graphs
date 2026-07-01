from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from neo4j.graph import Node as NeoNode, Relationship as NeoRel, Path as NeoPath
import re
from ..db import run_data, current_database
from ..lp_model_state import normalize_prediction_model
from ..prediction_timeline import predicted_db_name, valid_timeline_combinations
from ..storage_repr import (
    SEPARATE_DBS,
    SINGLE_DB_MULTI_EDGE,
    normalize_storage_representation,
    is_single_db_multi_edge,
)
router = APIRouter()

USE_CLAUSE_PREFIX = False

class PresentBody(BaseModel):
    cypher: str
    params: Dict[str, Any] = {}

class FutureBody(BaseModel):
    cypher: str
    threshold: float = 0.0
    params: Dict[str, Any] = {}
    futureDb: Optional[str] = None
    predictionFamily: str = "Node2Vec"
    predictionModel: str = "logistic_regression"

class BothBody(BaseModel):
    cypher: str
    params: Dict[str, Any] = {}
    futureDb: Optional[str] = None
    asOfTimestamp: Optional[int] = None
    storageRepresentation: str = "separate_dbs"
    predictionFamily: Optional[str] = None
    predictionModel: Optional[str] = "logistic_regression"
    predictionUntilTimestep: Optional[int] = None

def _predicted_db_name(
    base: str,
    family: str = "Node2Vec",
    prediction_model: str = "logistic_regression",
) -> str:
    return predicted_db_name(base, family, prediction_model)


def _default_future_db(
    family: str = "Node2Vec",
    prediction_model: str = "logistic_regression",
) -> str:
    return predicted_db_name(
        current_database(),
        family,
        prediction_model,
    )


def _with_use_clause(dbname: str, cypher: str) -> str:
    q = cypher.lstrip()
    if q.lower().startswith("use "):
        return cypher
    return f"USE {dbname}\n{cypher}"

def _exec(cypher: str, params: Dict[str, Any], db: Optional[str] = None) -> List[Dict[str, Any]]:
    try:
        if db and USE_CLAUSE_PREFIX:
            cypher = _with_use_clause(db, cypher)
            rows = run_data(cypher, params)
        else:
            rows = run_data(cypher, params, db=db)
        return rows or []
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def _reject_gds_calls(cypher: str):
    if cypher.strip().lower().startswith(("call gds.", "yield gds.")):
        raise HTTPException(
            status_code=400,
            detail="This endpoint runs against databases only. Remove GDS calls (use pure Cypher)."
        )

from typing import Tuple

def _contains_shortest_path(cypher: str) -> bool:

    return bool(re.search(r"shortestpath\s*\(", cypher, flags=re.IGNORECASE))

def _contains_simple_path_return_p(cypher: str) -> bool:

    path_match = re.search(r"\bmatch\s+p\s*=", cypher, flags=re.IGNORECASE)

    ret_match = re.search(
        r"\breturn\s+(?:distinct\s+)?p(?:\s+limit\b|\s*$)",
        cypher,
        flags=re.IGNORECASE,
    )

    return bool(path_match and ret_match)

def _rewrite_shortest_path_cypher(cypher: str) -> str:

    # Locate first RETURN
    m = re.search(r"\bRETURN\b", cypher, flags=re.IGNORECASE)

    limit_skip_clause = ""
    if m:
        # Part before RETURN
        prefix = cypher[:m.start()]

        # Everything after RETURN (e.g. " p LIMIT 25")
        suffix = cypher[m.end():]

        # Preserve (optional) SKIP / LIMIT in the same order
        clauses = []
        for kw in ("skip", "limit"):
            km = re.search(rf"\b{kw}\b\s+([^\s;]+)", suffix, flags=re.IGNORECASE)
            if km:
                expr = km.group(1).strip()
                if expr:
                    clauses.append(f"{kw.upper()} {expr}")

        if clauses:
            limit_skip_clause = "\n" + " ".join(clauses)
    else:
        prefix = cypher

    # Ensure trailing semicolon doesn't break anything
    prefix = prefix.rstrip().rstrip(";")

    # Rebuild query to expose nodes/relationships with elementIds
    rewritten = f"""
{prefix}
WITH p
WITH p, nodes(p) AS ns, relationships(p) AS rs
RETURN
  [n IN ns | {{
    elementId: elementId(n),
    labels: labels(n),
    properties: properties(n)
  }}] AS nodes,
  [r IN rs | {{
    elementId: elementId(r),
    type: type(r),
    startNodeElementId: elementId(startNode(r)),
    endNodeElementId: elementId(endNode(r)),
    properties: properties(r)
  }}] AS relationships{limit_skip_clause}
"""
    return rewritten.strip()

def _edge_visible_at_asof(edge: Dict[str, Any], as_of: Optional[int]) -> bool:
    if as_of is None:
        return True

    try:
        first_ts = edge.get("firstTimestamp", edge.get("timestamp"))
        if first_ts is None:
            return True
        return float(first_ts) <= float(as_of)
    except Exception:
        return True

def _filter_single_db_present_future(
    graph: Dict[str, Any],
    prediction_family: Optional[str],
    prediction_model: Optional[str] = "logistic_regression",
    prediction_until_timestep: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    family = prediction_family or ""
    requested_model = normalize_prediction_model(prediction_model)
    all_nodes = graph.get("nodes", []) or []
    all_edges = graph.get("edges", []) or []
    present_edges: List[Dict[str, Any]] = []
    future_edges: List[Dict[str, Any]] = []

    for edge in all_edges:
        origin = str(
            edge.get("edgeOrigin") or edge.get("origin") or ""
        ).lower()
        try:
            predicted = int(
                edge.get("predicted_timestep")
                or edge.get("predicted")
                or 0
            )
        except Exception:
            predicted = 0
        is_predicted = origin == "predicted" or predicted > 0

        if not is_predicted:
            present_edges.append(edge)
            future_edges.append(edge)
            continue

        edge_family = str(edge.get("embeddingFamily") or "")
        if family and edge_family != family:
            continue

        edge_model = normalize_prediction_model(
            edge.get("predictionModel", "logistic_regression")
        )
        if edge_model != requested_model:
            continue

        if prediction_until_timestep is not None:
            try:
                edge_level = int(
                    edge.get("predicted_timestep")
                    or edge.get("predicted")
                    or 0
                )
            except Exception:
                edge_level = 0
            if edge_level > int(prediction_until_timestep):
                continue

        future_edges.append(edge)

    def nodes_for(edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not edges:
            return all_nodes
        ids = {
            str(edge.get("source"))
            for edge in edges
            if edge.get("source") is not None
        } | {
            str(edge.get("target"))
            for edge in edges
            if edge.get("target") is not None
        }
        return [node for node in all_nodes if str(node.get("id")) in ids]

    return (
        {"nodes": nodes_for(present_edges), "edges": present_edges},
        {"nodes": nodes_for(future_edges), "edges": future_edges},
    )


def _filter_graph_asof(graph: Dict[str, Any], as_of: Optional[int]) -> Dict[str, Any]:
    if as_of is None:
        return graph

    kept_edges = [e for e in graph.get("edges", []) if _edge_visible_at_asof(e, as_of)]
    node_ids = {e["source"] for e in kept_edges} | {e["target"] for e in kept_edges}
    kept_nodes = [n for n in graph.get("nodes", []) if n.get("id") in node_ids]

    return {"nodes": kept_nodes, "edges": kept_edges}

def _filter_graph_prediction_until(
    graph: Dict[str, Any],
    prediction_until_timestep: Optional[int],
) -> Dict[str, Any]:

    if prediction_until_timestep is None:
        return graph

    until = int(prediction_until_timestep)

    kept_edges: List[Dict[str, Any]] = []

    for e in graph.get("edges", []) or []:
        try:
            lvl = int(e.get("predicted_timestep") or e.get("predicted") or 0)
        except Exception:
            lvl = 0

        origin = str(e.get("edgeOrigin") or e.get("origin") or "").lower()
        is_predicted = origin == "predicted" or lvl > 0

        if not is_predicted:
            kept_edges.append(e)
            continue

        if lvl <= until:
            kept_edges.append(e)

    if not kept_edges:
        return {
            "nodes": graph.get("nodes", []) or [],
            "edges": [],
        }

    node_ids = {
        str(e.get("source"))
        for e in kept_edges
        if e.get("source") is not None
    } | {
        str(e.get("target"))
        for e in kept_edges
        if e.get("target") is not None
    }

    kept_nodes = [
        n
        for n in (graph.get("nodes", []) or [])
        if str(n.get("id")) in node_ids
    ]

    return {
        "nodes": kept_nodes,
        "edges": kept_edges,
    }

def _shortest_path_graph(rows: List[Dict[str, Any]]) -> Dict[str, Any]:

    nodes_by_id: Dict[str, Dict[str, Any]] = {}
    edges_out: List[Dict[str, Any]] = []

    def add_node(n: Dict[str, Any]):
        nid = str(
            n.get("elementId")
            or n.get("element_id")
            or n.get("id", "")
        )
        if not nid:
            return

        labels = n.get("labels") or []
        props = n.get("properties") or {}

        node_id_prop = props.get("id") if isinstance(props, dict) else None

        if nid not in nodes_by_id:
            node: Dict[str, Any] = {
                "id": nid,
                "labels": labels,
            }

            if node_id_prop is not None:
                node["nodeId"] = node_id_prop

            if isinstance(props, dict):
                for k, v in props.items():
                    if k not in ("id", "labels", "nodeId"):
                        node[k] = v

            nodes_by_id[nid] = node

    def add_rel(r: Dict[str, Any]):
        rid = str(
            r.get("elementId")
            or r.get("element_id")
            or r.get("id")
            or ""
        )

        sid = str(
            r.get("startNodeElementId")
            or r.get("start")
            or r.get("startNode")
            or ""
        )

        tid = str(
            r.get("endNodeElementId")
            or r.get("end")
            or r.get("endNode")
            or ""
        )

        if not sid or not tid:
            return

        rtype = r.get("type") or ""

        if not rid:
            rid = f"{sid}->{tid}:{rtype}:{len(edges_out)}"

        props = r.get("properties") or {}

        add_node({"elementId": sid, "labels": [], "properties": {}})
        add_node({"elementId": tid, "labels": [], "properties": {}})

        edge = {
            "id": rid,
            "type": rtype,
            "source": sid,
            "target": tid,
        }

        if isinstance(props, dict):
            for k, v in props.items():
                if k not in ("id", "labels"):
                    edge[k] = v

        edges_out.append(edge)

    for row in rows or []:
        if not isinstance(row, dict):
            continue

        ns = row.get("nodes") or []
        rs = row.get("relationships") or []

        if isinstance(ns, list):
            for n in ns:
                if isinstance(n, dict):
                    add_node(n)

        if isinstance(rs, list):
            for r in rs:
                if isinstance(r, dict):
                    add_rel(r)

    return {
        "nodes": list(nodes_by_id.values()),
        "edges": edges_out,
    }

def _graphify(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    nodes_by_id: Dict[str, Dict[str, Any]] = {}
    edges_out: List[Dict[str, Any]] = []
    seen_edge_ids: set[str] = set()

    def is_rel_like_dict(x: Any) -> bool:
        if not isinstance(x, dict):
            return False

        has_type = bool(x.get("type"))

        has_start = (
            x.get("startNodeElementId") is not None
            or x.get("start") is not None
            or x.get("startNode") is not None
        )

        has_end = (
            x.get("endNodeElementId") is not None
            or x.get("end") is not None
            or x.get("endNode") is not None
        )

        return has_type and has_start and has_end

    def node_id_from(x: Any) -> str:
        if x is None:
            return ""

        if isinstance(x, NeoNode):
            return str(getattr(x, "element_id", None) or getattr(x, "id", ""))

        if isinstance(x, dict):

            if is_rel_like_dict(x):
                return ""

            if "elementId" in x and x["elementId"] is not None:
                return str(x["elementId"])

            if "element_id" in x and x["element_id"] is not None:
                return str(x["element_id"])

            if "identity" in x and x["identity"] is not None:
                return str(x["identity"])

            if "id" in x and x["id"] is not None:
                return str(x["id"])

        return ""

    def endpoint_id_from(x: Any) -> str:
        if x is None:
            return ""

        if isinstance(x, NeoNode):
            return str(getattr(x, "element_id", None) or getattr(x, "id", ""))

        if isinstance(x, dict):
            if "elementId" in x and x["elementId"] is not None:
                return str(x["elementId"])

            if "element_id" in x and x["element_id"] is not None:
                return str(x["element_id"])

            if "identity" in x and x["identity"] is not None:
                return str(x["identity"])

            if "id" in x and x["id"] is not None:
                return str(x["id"])

        return str(x)

    def add_node(x: Any):
        if x is None:
            return

        if isinstance(x, NeoNode):
            nid = str(getattr(x, "element_id", None) or getattr(x, "id", ""))
            labels = list(getattr(x, "labels", []))
            props = dict(getattr(x, "_properties", {}) or {})

        elif isinstance(x, dict):
            if is_rel_like_dict(x):
                return

            nid = node_id_from(x)
            labels = x.get("labels") if isinstance(x.get("labels"), list) else []

            if "properties" in x and isinstance(x["properties"], dict):
                props = dict(x["properties"])
            else:
                props = {
                    k: v
                    for k, v in x.items()
                    if k not in (
                        "labels",
                        "type",
                        "start",
                        "end",
                        "startNode",
                        "endNode",
                        "startNodeElementId",
                        "endNodeElementId",
                        "identity",
                        "elementId",
                        "element_id",
                        "segments",
                        "nodes",
                        "relationships",
                    )
                }

        else:
            return

        if not nid:
            return

        node_id_prop = None
        if isinstance(props, dict) and "id" in props:
            node_id_prop = props["id"]
        elif isinstance(x, dict) and "id" in x and not is_rel_like_dict(x):
            node_id_prop = x["id"]

        if nid in nodes_by_id:
            existing = nodes_by_id[nid]

            if node_id_prop is not None:
                existing["nodeId"] = node_id_prop

            if labels and not existing.get("labels"):
                existing["labels"] = labels

            for k, v in props.items():
                if k not in ("id", "labels", "nodeId"):
                    existing.setdefault(k, v)

            return

        node_obj: Dict[str, Any] = {
            "id": nid,
            "labels": labels,
        }

        if node_id_prop is not None:
            node_obj["nodeId"] = node_id_prop

        for k, v in props.items():
            if k not in ("id", "labels", "nodeId"):
                node_obj[k] = v

        nodes_by_id[nid] = node_obj

    def add_rel(r: Any, src: Any = None, trg: Any = None):
        if r is None:
            return

        if isinstance(r, NeoRel):
            if src is None:
                src = r.start_node
            if trg is None:
                trg = r.end_node

            sid = endpoint_id_from(src)
            tid = endpoint_id_from(trg)

            if not sid or not tid:
                return

            rtype = getattr(r, "type", "")
            raw_id = getattr(r, "element_id", None) or getattr(r, "id", None)
            rid = str(raw_id) if raw_id is not None else f"{sid}->{tid}:{rtype}:{len(edges_out)}"
            props = dict(getattr(r, "_properties", {}) or {})

        elif isinstance(r, dict):
            if not is_rel_like_dict(r):
                return

            raw_start = (
                src
                if src is not None
                else r.get("startNodeElementId")
                or r.get("start")
                or r.get("startNode")
            )

            raw_end = (
                trg
                if trg is not None
                else r.get("endNodeElementId")
                or r.get("end")
                or r.get("endNode")
            )

            sid = endpoint_id_from(raw_start)
            tid = endpoint_id_from(raw_end)

            if not sid or not tid:
                return

            rtype = r.get("type", "") or ""

            raw_id = (
                r.get("elementId")
                or r.get("element_id")
                or r.get("identity")
                or r.get("id")
            )

            rid = str(raw_id) if raw_id is not None else f"{sid}->{tid}:{rtype}:{len(edges_out)}"

            if isinstance(r.get("properties"), dict):
                props = dict(r.get("properties") or {})
            else:
                props = {
                    k: v
                    for k, v in r.items()
                    if k not in (
                        "labels",
                        "type",
                        "start",
                        "end",
                        "startNode",
                        "endNode",
                        "startNodeElementId",
                        "endNodeElementId",
                        "identity",
                        "elementId",
                        "element_id",
                        "id",
                        "segments",
                        "nodes",
                        "relationships",
                    )
                }


            if isinstance(raw_start, dict):
                add_node(raw_start)
            else:
                add_node({"elementId": sid, "labels": [], "properties": {}})

            if isinstance(raw_end, dict):
                add_node(raw_end)
            else:
                add_node({"elementId": tid, "labels": [], "properties": {}})

        else:
            return

        if rid in seen_edge_ids:
            return

        seen_edge_ids.add(rid)

        if isinstance(r, NeoRel):
            add_node(src)
            add_node(trg)

        edge = {
            "id": rid,
            "type": rtype,
            "source": sid,
            "target": tid,
        }

        for k, v in props.items():
            if k not in ("id", "labels"):
                edge[k] = v

        edges_out.append(edge)

    def consume_value(v: Any):
        if isinstance(v, NeoNode):
            add_node(v)
            return True

        if isinstance(v, NeoRel):
            add_rel(v)
            return True

        if isinstance(v, NeoPath):
            for i in range(len(v.relationships)):
                a = v.nodes[i]
                b = v.nodes[i + 1]
                r = v.relationships[i]
                add_rel(r, a, b)
            return True

        if isinstance(v, list):
            for x in v:
                consume_value(x)
            return True

        if isinstance(v, dict):
            if all(k in v for k in ("a", "b", "r")):
                add_node(v["a"])
                add_node(v["b"])
                add_rel(v["r"], v["a"], v["b"])
                return True

            if is_rel_like_dict(v):
                add_rel(v)
                return True

            if node_id_from(v):
                add_node(v)
                return True

            segs = v.get("segments")
            if isinstance(segs, list) and segs:
                for s in segs:
                    a = s.get("start")
                    b = s.get("end")
                    r = s.get("relationship")
                    if a and b and r:
                        add_node(a)
                        add_node(b)
                        add_rel(r, a, b)
                return True

            for x in v.values():
                consume_value(x)

            return True

        return False

    for rec in records or []:
        if not isinstance(rec, dict):
            continue

        for val in rec.values():
            if isinstance(val, NeoNode):
                add_node(val)
            elif isinstance(val, dict) and not is_rel_like_dict(val) and node_id_from(val):
                add_node(val)

        for val in rec.values():
            consume_value(val)

    return {
        "nodes": list(nodes_by_id.values()),
        "edges": edges_out,
    }

def _neo_graph_id(n: NeoNode) -> str:
    return str(getattr(n, "element_id", None) or getattr(n, "id"))


def _coerce_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return default
        return int(v)
    except Exception:
        return default


def _coerce_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _rel_to_edge(rel: NeoRel) -> Dict[str, Any]:
    predicted_level = _coerce_int(rel.get("predicted_timestep", rel.get("predicted", 0)), 0)
    ts = rel.get("lastTimestamp", rel.get("timestamp", 0))

    return {
        "id": str(getattr(rel, "element_id", None) or getattr(rel, "id")),
        "source": _neo_graph_id(rel.start_node),
        "target": _neo_graph_id(rel.end_node),
        "type": rel.type,
        "predicted": predicted_level,
        "predicted_timestep": predicted_level,
        "probability": _coerce_float(rel.get("probability"), 1.0 if predicted_level == 0 else 0.0),
        "timestamp": _coerce_int(ts, 0),
        "firstTimestamp": _coerce_int(rel.get("firstTimestamp", ts), 0),
        "lastTimestamp": _coerce_int(rel.get("lastTimestamp", ts), 0),
    }

@router.post("/present")
def query_present(body: PresentBody):
    base_db = current_database()
    _reject_gds_calls(body.cypher)
    params = dict(body.params)
    return {"records": _exec(body.cypher, params, db=base_db)}

@router.post("/future")
def query_future(body: FutureBody):
    _reject_gds_calls(body.cypher)
    params = dict(body.params)
    params.setdefault("thr", float(body.threshold))
    predicted_db = body.futureDb or _default_future_db(
        body.predictionFamily,
        body.predictionModel,
    )
    rows = _exec(body.cypher, params, db=predicted_db)
    return {"records": rows, "db": predicted_db}

@router.post("/both")
def query_both(body: BothBody):
    _reject_gds_calls(body.cypher)
    params = dict(body.params)
    
    base_db = current_database()
    predicted_db = body.futureDb or _default_future_db(
        body.predictionFamily or "Node2Vec",
        body.predictionModel or "logistic_regression",
    )
    if body.asOfTimestamp is not None:
        params.setdefault("t", int(body.asOfTimestamp))
    present_rows = _exec(body.cypher, params, db=base_db)
    future_rows  = _exec(body.cypher, params, db=predicted_db)

    return {
        "present": present_rows,
        "future": future_rows,             
        "predicted": future_rows, 
        "databases": {"present": base_db, "future": predicted_db, "predicted": predicted_db},
    }

# ---- graph mode variant ----
class ModeBody(BothBody):
    mode: str = "table"  # "table" | "graph"

_AUTO_FAMILY_PARAM = "__autoPredictionFamily"
_AUTO_MODEL_PARAM = "__autoPredictionModel"
_AUTO_UNTIL_PARAM = "__autoPredictionUntilTimestep"

def _has_manual_visibility_filter(cypher: str) -> bool:
    """
    If the user already wrote edgeOrigin / embeddingFamily logic,
    do not rewrite the query. We assume the user knows what they are doing.
    """
    return bool(
        re.search(
            r"\b(edgeOrigin|embeddingFamily|predictionModel|predicted_timestep|predicted)\b",
            cypher,
            flags=re.IGNORECASE,
        )
    )


def _relationship_visibility_expr(
    rel_var: str,
    include_predicted: bool,
) -> str:
    observed_expr = (
        f"coalesce({rel_var}.edgeOrigin, 'observed') = 'observed'"
    )
    if not include_predicted:
        return observed_expr

    predicted_expr = (
        f"coalesce({rel_var}.edgeOrigin, '') = 'predicted' "
        f"AND coalesce({rel_var}.embeddingFamily, '') = "
        f"${_AUTO_FAMILY_PARAM} "
        f"AND coalesce({rel_var}.predictionModel, "
        f"'logistic_regression') = ${_AUTO_MODEL_PARAM} "
        f"AND toInteger(coalesce({rel_var}.predicted_timestep, "
        f"{rel_var}.predicted, 0)) <= ${_AUTO_UNTIL_PARAM}"
    )
    return f"({observed_expr} OR ({predicted_expr}))"



def _extract_path_vars(cypher: str) -> List[str]:

    out: List[str] = []

    for m in re.finditer(
        r"\bMATCH\s+([A-Za-z_][A-Za-z0-9_]*)\s*=",
        cypher,
        flags=re.IGNORECASE,
    ):
        name = m.group(1)
        if name not in out:
            out.append(name)

    return out


def _extract_relationship_vars(cypher: str) -> List[Tuple[str, bool]]:

    out: List[Tuple[str, bool]] = []

    rel_pat = re.compile(
        r"\[\s*([A-Za-z_][A-Za-z0-9_]*)\s*([^\]]*)\]",
        flags=re.IGNORECASE,
    )

    for m in rel_pat.finditer(cypher):
        var_name = m.group(1)
        rest = m.group(2) or ""


        if var_name.upper() in {
            "N", "R", "X", "Y", "ROW", "ITEM"
        } and " IN " in rest.upper():
            continue

        is_list = "*" in rest
        item = (var_name, is_list)

        if item not in out:
            out.append(item)

    return out


def _build_visibility_predicate(cypher: str, include_predicted: bool) -> Optional[str]:

    predicates: List[str] = []

    path_vars = _extract_path_vars(cypher)
    rel_vars = _extract_relationship_vars(cypher)

    for p in path_vars:
        rel_expr = _relationship_visibility_expr("_autoRel", include_predicted)
        predicates.append(
            f"all(_autoRel IN relationships({p}) WHERE {rel_expr})"
        )

    for rel_var, is_list in rel_vars:
        if is_list:
            rel_expr = _relationship_visibility_expr("_autoRel", include_predicted)
            predicates.append(
                f"all(_autoRel IN {rel_var} WHERE {rel_expr})"
            )
        else:
            predicates.append(
                _relationship_visibility_expr(rel_var, include_predicted)
            )


    deduped: List[str] = []
    for p in predicates:
        if p not in deduped:
            deduped.append(p)

    if not deduped:
        return None

    return " AND ".join(f"({p})" for p in deduped)


def _find_last_match_before_clause(cypher: str) -> Optional[Tuple[int, int]]:

    match_iter = list(
        re.finditer(
            r"\b(?:OPTIONAL\s+)?MATCH\b",
            cypher,
            flags=re.IGNORECASE,
        )
    )

    if not match_iter:
        return None

    last_match = match_iter[-1]
    start = last_match.start()

    next_clause = re.search(
        r"\b(WITH|RETURN|CALL|UNWIND|CREATE|MERGE|SET|DELETE|REMOVE|UNION)\b",
        cypher[last_match.end():],
        flags=re.IGNORECASE,
    )

    if next_clause:
        end = last_match.end() + next_clause.start()
    else:
        end = len(cypher)

    return start, end


def _inject_visibility_predicate(cypher: str, predicate: str) -> str:

    cypher = cypher.rstrip().rstrip(";")

    loc = _find_last_match_before_clause(cypher)
    if loc is None:
        return cypher

    block_start, insert_pos = loc
    block = cypher[block_start:insert_pos]

    before = cypher[:insert_pos].rstrip()
    after = cypher[insert_pos:]

    if re.search(r"\bWHERE\b", block, flags=re.IGNORECASE):
        return f"{before}\n  AND ({predicate})\n{after.lstrip()}"

    return f"{before}\nWHERE {predicate}\n{after.lstrip()}"


def _rewrite_single_db_query_for_visibility(
    cypher: str,
    include_predicted: bool,
) -> Tuple[str, Dict[str, Any]]:

    meta = {
        "rewritten": False,
        "manual_visibility_filter": False,
        "reason": None,
    }

    if _has_manual_visibility_filter(cypher):
        meta["manual_visibility_filter"] = True
        meta["reason"] = "manual_visibility_filter_present"
        return cypher, meta

    if re.search(r"\bUNION\b", cypher, flags=re.IGNORECASE):
        meta["reason"] = "union_not_rewritten"
        return cypher, meta

    predicate = _build_visibility_predicate(
        cypher,
        include_predicted=include_predicted,
    )

    if not predicate:
        meta["reason"] = "no_relationship_or_path_variable_found"
        return cypher, meta

    rewritten = _inject_visibility_predicate(cypher, predicate)

    meta["rewritten"] = rewritten != cypher
    meta["reason"] = "auto_visibility_rewrite" if meta["rewritten"] else "unchanged"

    return rewritten, meta

@router.post("/both2")
def query_both2(body: ModeBody):
    _reject_gds_calls(body.cypher)

    params = dict(body.params)

    if body.asOfTimestamp is not None:
        params.setdefault("t", int(body.asOfTimestamp))
    prediction_until = body.predictionUntilTimestep
    if prediction_until is None:
        prediction_until = 10**9

    params[_AUTO_UNTIL_PARAM] = int(prediction_until)
    model = normalize_prediction_model(body.predictionModel)
    params[_AUTO_MODEL_PARAM] = model

    base_db = current_database()
    predicted_db = body.futureDb or _default_future_db(
        body.predictionFamily or "Node2Vec",
        body.predictionModel or "logistic_regression",
    )

    storage_mode = normalize_storage_representation(
        body.storageRepresentation or SEPARATE_DBS
    )
    single_db_mode = is_single_db_multi_edge(storage_mode)

    # ============================================================
    # SINGLE-DB MULTI-EDGE MODE: TABLE MODE
    # ============================================================
    if single_db_mode and body.mode != "graph":
        family = body.predictionFamily or "FastRP"
        model = normalize_prediction_model(body.predictionModel)

        prediction_until = body.predictionUntilTimestep
        if prediction_until is None:
            prediction_until = 10**9

        params[_AUTO_FAMILY_PARAM] = family
        params[_AUTO_MODEL_PARAM] = model
        params[_AUTO_UNTIL_PARAM] = int(prediction_until)

        present_cypher, present_meta = _rewrite_single_db_query_for_visibility(
            body.cypher,
            include_predicted=False,
        )

        future_cypher, future_meta = _rewrite_single_db_query_for_visibility(
            body.cypher,
            include_predicted=True,
        )


        if present_meta.get("manual_visibility_filter"):
            rows = _exec(body.cypher, params, db=base_db)

            return {
                "present": rows,
                "future": rows,
                "predicted": rows,
                "databases": {
                    "present": base_db,
                    "future": base_db,
                    "predicted": base_db,
                },
                "storageRepresentation": SINGLE_DB_MULTI_EDGE,
                "predictionFamily": family,
                "predictionModel": model,
                "predictionUntilTimestep": int(prediction_until),
                "autoRewrite": {
                    "applied": False,
                    "reason": "manual_visibility_filter_present",
                },
                "ok": True,
            }

        if not present_meta.get("rewritten") or not future_meta.get("rewritten"):
            rows = _exec(body.cypher, params, db=base_db)

            return {
                "present": rows,
                "future": rows,
                "predicted": rows,
                "databases": {
                    "present": base_db,
                    "future": base_db,
                    "predicted": base_db,
                },
                "storageRepresentation": SINGLE_DB_MULTI_EDGE,
                "predictionFamily": family,
                "predictionModel": model,
                "predictionUntilTimestep": int(prediction_until),
                "autoRewrite": {
                    "applied": False,
                    "present": present_meta,
                    "future": future_meta,
                },
                "warning": (
                    "This single-DB query could not be automatically rewritten. "
                    "Use an explicit relationship variable like [r], or return a path variable like p."
                ),
                "ok": True,
            }

        present_rows = _exec(present_cypher, params, db=base_db)
        future_rows = _exec(future_cypher, params, db=base_db)

        return {
            "present": present_rows,
            "future": future_rows,
            "predicted": future_rows,
            "databases": {
                "present": base_db,
                "future": base_db,
                "predicted": base_db,
            },
            "storageRepresentation": SINGLE_DB_MULTI_EDGE,
            "predictionFamily": family,
            "predictionModel": model,
            "predictionUntilTimestep": int(prediction_until),
            "autoRewrite": {
                "applied": True,
                "presentCypher": present_cypher,
                "futureCypher": future_cypher,
            },
            "ok": True,
        }

    # ============================================================
    # EXISTING TWO-DB MODE: TABLE MODE
    # ============================================================
    if body.mode != "graph":
        present_rows = _exec(body.cypher, params, db=base_db)
        future_rows = _exec(body.cypher, params, db=predicted_db)

        return {
            "present": present_rows,
            "future": future_rows,
            "predicted": future_rows,
            "databases": {
                "present": base_db,
                "future": predicted_db,
                "predicted": predicted_db,
            },
            "storageRepresentation": SEPARATE_DBS,
            "predictionUntilTimestep": int(prediction_until),
            "ok": True,
        }

    # ============================================================
    # SINGLE-DB MULTI-EDGE MODE: GRAPH MODE
    # ============================================================
    if single_db_mode:
        family = body.predictionFamily or "FastRP"
        model = normalize_prediction_model(body.predictionModel)

        prediction_until = body.predictionUntilTimestep
        if prediction_until is None:
            prediction_until = 10**9

        params[_AUTO_FAMILY_PARAM] = family
        params[_AUTO_MODEL_PARAM] = model
        params[_AUTO_UNTIL_PARAM] = int(prediction_until)

        present_cypher, present_meta = _rewrite_single_db_query_for_visibility(
            body.cypher,
            include_predicted=False,
        )

        future_cypher, future_meta = _rewrite_single_db_query_for_visibility(
            body.cypher,
            include_predicted=True,
        )


        if present_meta.get("manual_visibility_filter"):
            if _contains_shortest_path(body.cypher) or _contains_simple_path_return_p(body.cypher):
                path_cypher = _rewrite_shortest_path_cypher(body.cypher)
                rows = _exec(path_cypher, params, db=base_db)

                graph = _filter_graph_asof(
                    _shortest_path_graph(rows),
                    body.asOfTimestamp,
                )
            else:
                rows = _exec(body.cypher, params, db=base_db)

                graph = _filter_graph_asof(
                    _graphify(rows),
                    body.asOfTimestamp,
                )

            present_graph, future_graph = _filter_single_db_present_future(
                graph,
                family,
                model,
                prediction_until_timestep=prediction_until,
            )

            return {
                "present": present_graph,
                "future": future_graph,
                "predicted": future_graph,
                "databases": {
                    "present": base_db,
                    "future": base_db,
                    "predicted": base_db,
                },
                "storageRepresentation": SINGLE_DB_MULTI_EDGE,
                "predictionFamily": family,
                "predictionModel": model,
                "predictionUntilTimestep": int(prediction_until),
                "autoRewrite": {
                    "applied": False,
                    "reason": "manual_visibility_filter_present",
                },
                "ok": True,
            }

        if not present_meta.get("rewritten") or not future_meta.get("rewritten"):
            if _contains_shortest_path(body.cypher) or _contains_simple_path_return_p(body.cypher):
                path_cypher = _rewrite_shortest_path_cypher(body.cypher)
                rows = _exec(path_cypher, params, db=base_db)

                graph = _filter_graph_asof(
                    _shortest_path_graph(rows),
                    body.asOfTimestamp,
                )
            else:
                rows = _exec(body.cypher, params, db=base_db)

                graph = _filter_graph_asof(
                    _graphify(rows),
                    body.asOfTimestamp,
                )

            present_graph, future_graph = _filter_single_db_present_future(
                graph,
                family,
                model,
                prediction_until_timestep=prediction_until,
            )

            return {
                "present": present_graph,
                "future": future_graph,
                "predicted": future_graph,
                "databases": {
                    "present": base_db,
                    "future": base_db,
                    "predicted": base_db,
                },
                "storageRepresentation": SINGLE_DB_MULTI_EDGE,
                "predictionFamily": family,
                "predictionModel": model,
                "predictionUntilTimestep": int(prediction_until),
                "autoRewrite": {
                    "applied": False,
                    "present": present_meta,
                    "future": future_meta,
                },
                "warning": (
                    "This single-DB graph query could not be automatically rewritten. "
                    "Use a relationship variable like [r], or bind and return a path variable like p."
                ),
                "ok": True,
            }

        # Present graph: observed edges only.
        if _contains_shortest_path(present_cypher) or _contains_simple_path_return_p(present_cypher):
            present_path_cypher = _rewrite_shortest_path_cypher(present_cypher)
            present_rows = _exec(present_path_cypher, params, db=base_db)

            present_graph = _filter_graph_asof(
                _shortest_path_graph(present_rows),
                body.asOfTimestamp,
            )
        else:
            present_rows = _exec(present_cypher, params, db=base_db)

            present_graph = _filter_graph_asof(
                _graphify(present_rows),
                body.asOfTimestamp,
            )

        # Future graph: observed edges + selected prediction family.
        if _contains_shortest_path(future_cypher) or _contains_simple_path_return_p(future_cypher):
            future_path_cypher = _rewrite_shortest_path_cypher(future_cypher)
            future_rows = _exec(future_path_cypher, params, db=base_db)

            future_graph = _filter_graph_asof(
                _shortest_path_graph(future_rows),
                body.asOfTimestamp,
            )
        else:
            future_rows = _exec(future_cypher, params, db=base_db)

            future_graph = _filter_graph_asof(
                _graphify(future_rows),
                body.asOfTimestamp,
            )

        return {
            "present": present_graph,
            "future": future_graph,
            "predicted": future_graph,
            "databases": {
                "present": base_db,
                "future": base_db,
                "predicted": base_db,
            },
            "storageRepresentation": SINGLE_DB_MULTI_EDGE,
            "predictionFamily": family,
            "predictionModel": model,
            "predictionUntilTimestep": prediction_until,
            "autoRewrite": {
                "applied": True,
                "presentCypher": present_cypher,
                "futureCypher": future_cypher,
            },
            "ok": True,
        }

    # ============================================================
    # EXISTING TWO-DB MODE: GRAPH MODE
    # ============================================================


    if _contains_shortest_path(body.cypher) or _contains_simple_path_return_p(body.cypher):
        path_cypher = _rewrite_shortest_path_cypher(body.cypher)

        present_rows = _exec(path_cypher, params, db=base_db)
        future_rows = _exec(path_cypher, params, db=predicted_db)

        present_graph = _filter_graph_asof(
            _shortest_path_graph(present_rows),
            body.asOfTimestamp,
        )

        future_graph = _filter_graph_asof(
            _shortest_path_graph(future_rows),
            body.asOfTimestamp,
        )

        future_graph = _filter_graph_prediction_until(
            future_graph,
            int(prediction_until),
        )

        return {
            "present": present_graph,
            "future": future_graph,
            "predicted": future_graph,
            "databases": {
                "present": base_db,
                "future": predicted_db,
                "predicted": predicted_db,
            },
            "storageRepresentation": SEPARATE_DBS,
            "predictionUntilTimestep": int(prediction_until),
            "ok": True,
        }

    # 2) All other graph queries
    present_rows = _exec(body.cypher, params, db=base_db)
    future_rows = _exec(body.cypher, params, db=predicted_db)

    present_graph = _filter_graph_asof(
        _graphify(present_rows),
        body.asOfTimestamp,
    )

    future_graph = _filter_graph_asof(
        _graphify(future_rows),
        body.asOfTimestamp,
    )

    future_graph = _filter_graph_prediction_until(
        future_graph,
        int(prediction_until),
    )

    return {
        "present": present_graph,
        "future": future_graph,
        "predicted": future_graph,
        "databases": {
            "present": base_db,
            "future": predicted_db,
            "predicted": predicted_db,
        },
        "storageRepresentation": "separate_dbs",
        "ok": True,
        "predictionUntilTimestep": int(prediction_until),
    }

@router.get("/dbs")
def query_dbs():
    base = current_database()
    candidates = [
        {
            "family": family,
            "predictionModel": model,
            "name": predicted_db_name(base, family, model),
        }
        for family, model in valid_timeline_combinations()
    ]
    default_name = predicted_db_name(
        base,
        "Node2Vec",
        "logistic_regression",
    )
    return {
        "present": base,
        "predicted": default_name,
        "future": default_name,
        "candidates": candidates,
    }
