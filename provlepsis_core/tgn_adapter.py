from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import math
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support, roc_auc_score

from .db import run_data

_TGN_ROOT = Path(__file__).resolve().parent / "tgn"
if str(_TGN_ROOT) not in sys.path:
    sys.path.insert(0, str(_TGN_ROOT))

from model.tgn import TGN  # type: ignore
from utils.utils import RandEdgeSampler, get_neighbor_finder  # type: ignore


@dataclass
class TGNEventData:
    sources: np.ndarray
    destinations: np.ndarray
    timestamps: np.ndarray
    edge_idxs: np.ndarray
    labels: np.ndarray
    n_interactions: int
    unique_nodes: set[int]


@dataclass
class TGNState:
    model: Any
    node_id_to_idx: Dict[str, int]
    idx_to_node_id: Dict[int, str]
    train_data: TGNEventData
    full_data: TGNEventData
    train_neighbor_finder: Any
    full_neighbor_finder: Any
    node_features: np.ndarray
    edge_features: np.ndarray
    num_neighbors: int
    batch_size: int
    device: str = "cpu"


def _canon_pair(a: Any, b: Any) -> Tuple[str, str]:
    sa, sb = str(a), str(b)
    return (sa, sb) if sa < sb else (sb, sa)


def _to_float_ts(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _split_events_query(split_prefix: str) -> str:
    return """
    MATCH (a)-[sr]-(b)
    WHERE type(sr) STARTS WITH $splitPrefix
    WITH
      CASE
        WHEN toString(coalesce(a.eid, elementId(a))) < toString(coalesce(b.eid, elementId(b)))
        THEN toString(coalesce(a.eid, elementId(a)))
        ELSE toString(coalesce(b.eid, elementId(b)))
      END AS s,
      CASE
        WHEN toString(coalesce(a.eid, elementId(a))) < toString(coalesce(b.eid, elementId(b)))
        THEN toString(coalesce(b.eid, elementId(b)))
        ELSE toString(coalesce(a.eid, elementId(a)))
      END AS t,
      CASE
        WHEN sr.timestamps IS NOT NULL
             AND valueType(sr.timestamps) STARTS WITH 'LIST'
             AND size(sr.timestamps) > 0
          THEN sr.timestamps
        ELSE [coalesce(sr.firstTimestamp, sr.timestamp, sr.lastTimestamp, 0)]
      END AS tsList
    WITH DISTINCT s, t, tsList
    UNWIND tsList AS ts
    RETURN s AS s,
           t AS t,
           toFloat(ts) AS ts,
           1.0 AS edgeFeature
    ORDER BY ts ASC, s ASC, t ASC
    """


def load_split_events(dbname: str, split_prefix: str, family: str = "TGN") -> List[Dict[str, Any]]:
    rows = run_data(
        _split_events_query(split_prefix),
        {"splitPrefix": split_prefix, "family": family},
        db=dbname,
    )
    events: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, float]] = set()
    for r in rows:
        s, t = _canon_pair(r.get("s"), r.get("t"))
        ts = _to_float_ts(r.get("ts"))
        key = (s, t, ts)
        if key in seen:
            continue
        seen.add(key)
        events.append({"s": s, "t": t, "ts": ts, "edgeFeature": float(r.get("edgeFeature") or 1.0)})
    events.sort(key=lambda x: (float(x["ts"]), x["s"], x["t"]))
    return events


def load_all_node_ids(dbname: str) -> List[str]:
    rows = run_data(
        """
        MATCH (n)
        RETURN toString(coalesce(n.eid, elementId(n))) AS eid
        ORDER BY eid
        """,
        db=dbname,
    )
    return [str(r["eid"]) for r in rows]


def build_node_mapping(dbname: str, events: Iterable[Dict[str, Any]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    ids = set(load_all_node_ids(dbname))
    for e in events:
        ids.add(str(e["s"]))
        ids.add(str(e["t"]))
    node_id_to_idx = {eid: i + 1 for i, eid in enumerate(sorted(ids))}
    idx_to_node_id = {v: k for k, v in node_id_to_idx.items()}
    return node_id_to_idx, idx_to_node_id


def make_tgn_data(events: List[Dict[str, Any]], node_id_to_idx: Dict[str, int], start_edge_idx: int = 1) -> Tuple[TGNEventData, int, np.ndarray]:
    sources: List[int] = []
    destinations: List[int] = []
    timestamps: List[float] = []
    edge_idxs: List[int] = []
    edge_feats: List[List[float]] = []

    for local_i, e in enumerate(sorted(events, key=lambda x: (float(x["ts"]), x["s"], x["t"])), start=start_edge_idx):
        s = str(e["s"])
        t = str(e["t"])
        if s not in node_id_to_idx or t not in node_id_to_idx:
            continue
        sources.append(node_id_to_idx[s])
        destinations.append(node_id_to_idx[t])
        timestamps.append(float(e["ts"]))
        edge_idxs.append(int(local_i))
        edge_feats.append([float(e.get("edgeFeature", 1.0) or 1.0)])

    arr_s = np.asarray(sources, dtype=np.int32)
    arr_d = np.asarray(destinations, dtype=np.int32)
    arr_ts = np.asarray(timestamps, dtype=np.float32)
    arr_eidx = np.asarray(edge_idxs, dtype=np.int32)
    labels = np.zeros(len(arr_s), dtype=np.float32)
    unique_nodes = set(arr_s.tolist()) | set(arr_d.tolist())
    data = TGNEventData(arr_s, arr_d, arr_ts, arr_eidx, labels, len(arr_s), unique_nodes)
    return data, start_edge_idx + len(edge_feats), np.asarray(edge_feats, dtype=np.float32)


def concat_data(parts: List[TGNEventData]) -> TGNEventData:
    non_empty = [p for p in parts if p is not None and p.n_interactions > 0]
    if not non_empty:
        empty_i = np.asarray([], dtype=np.int32)
        empty_f = np.asarray([], dtype=np.float32)
        return TGNEventData(empty_i, empty_i, empty_f, empty_i, empty_f, 0, set())
    s = np.concatenate([p.sources for p in non_empty])
    d = np.concatenate([p.destinations for p in non_empty])
    ts = np.concatenate([p.timestamps for p in non_empty])
    eidx = np.concatenate([p.edge_idxs for p in non_empty])
    labels = np.concatenate([p.labels for p in non_empty])
    order = np.argsort(ts, kind="mergesort")
    s, d, ts, eidx, labels = s[order], d[order], ts[order], eidx[order], labels[order]
    return TGNEventData(s, d, ts, eidx, labels, len(s), set(s.tolist()) | set(d.tolist()))


def compute_time_statistics(sources: np.ndarray, destinations: np.ndarray, timestamps: np.ndarray) -> Tuple[float, float, float, float]:
    last_timestamp_sources: Dict[int, float] = {}
    last_timestamp_dst: Dict[int, float] = {}
    all_timediffs_src: List[float] = []
    all_timediffs_dst: List[float] = []

    for k in range(len(sources)):
        src = int(sources[k])
        dst = int(destinations[k])
        ts = float(timestamps[k])
        all_timediffs_src.append(ts - last_timestamp_sources.get(src, 0.0))
        all_timediffs_dst.append(ts - last_timestamp_dst.get(dst, 0.0))
        last_timestamp_sources[src] = ts
        last_timestamp_dst[dst] = ts

    src_arr = np.asarray(all_timediffs_src or [0.0], dtype=np.float32)
    dst_arr = np.asarray(all_timediffs_dst or [0.0], dtype=np.float32)
    return (
        float(np.mean(src_arr)),
        float(np.std(src_arr) if np.std(src_arr) > 0 else 1.0),
        float(np.mean(dst_arr)),
        float(np.std(dst_arr) if np.std(dst_arr) > 0 else 1.0),
    )


def _eval_scores(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    y_hat = (y_prob >= 0.5).astype(int)
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = None
    try:
        ap = float(average_precision_score(y_true, y_prob))
    except Exception:
        ap = None
    try:
        accuracy = float(accuracy_score(y_true, y_hat))
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_hat, average="binary", zero_division=0)
        precision, recall, f1 = float(precision), float(recall), float(f1)
    except Exception:
        accuracy = precision = recall = f1 = None
    return {
        "count_pos": int((y_true == 1).sum()),
        "count_neg": int((y_true == 0).sum()),
        "auc": auc,
        "ap": ap,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def train_tgn(
    *,
    dbname: str,
    family: str = "TGN",
    n_epoch: int = 5,
    batch_size: int = 200,
    n_neighbors: int = 10,
    n_heads: int = 2,
    n_layers: int = 1,
    learning_rate: float = 1e-4,
    dropout: float = 0.1,
    node_dim: int = 100,
    memory_dim: int = 100,
    message_dim: int = 100,
    use_memory: bool = True,
    device_string: Optional[str] = None,
) -> Tuple[TGNState, Dict[str, Any], Dict[str, TGNEventData]]:
    train_events = load_split_events(dbname, "SPLIT_TRAIN_", family=family)
    val_events = load_split_events(dbname, "SPLIT_VAL_", family=family)
    test_events = load_split_events(dbname, "SPLIT_TEST_", family=family)
    all_events = train_events + val_events + test_events

    if not train_events:
        raise ValueError("No SPLIT_TRAIN_ temporal events found for TGN training.")

    node_id_to_idx, idx_to_node_id = build_node_mapping(dbname, all_events)
    max_node_idx = max(node_id_to_idx.values() or [0])

    train_data, next_edge_idx, train_edge_feats = make_tgn_data(train_events, node_id_to_idx, start_edge_idx=1)
    val_data, next_edge_idx, val_edge_feats = make_tgn_data(val_events, node_id_to_idx, start_edge_idx=next_edge_idx)
    test_data, next_edge_idx, test_edge_feats = make_tgn_data(test_events, node_id_to_idx, start_edge_idx=next_edge_idx)
    full_data = concat_data([train_data, val_data, test_data])

    edge_features = np.vstack([
        np.zeros((1, 1), dtype=np.float32),
        train_edge_feats if train_edge_feats.size else np.zeros((0, 1), dtype=np.float32),
        val_edge_feats if val_edge_feats.size else np.zeros((0, 1), dtype=np.float32),
        test_edge_feats if test_edge_feats.size else np.zeros((0, 1), dtype=np.float32),
    ]).astype(np.float32)

    node_features = np.zeros((max_node_idx + 1, int(node_dim)), dtype=np.float32)

    train_ngh_finder = get_neighbor_finder(train_data, uniform=False, max_node_idx=max_node_idx)
    full_ngh_finder = get_neighbor_finder(full_data, uniform=False, max_node_idx=max_node_idx)

    train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)

    device_string = device_string or ("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_string)

    mean_src, std_src, mean_dst, std_dst = compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

    model = TGN(
        neighbor_finder=train_ngh_finder,
        node_features=node_features,
        edge_features=edge_features,
        device=device,
        n_layers=int(n_layers),
        n_heads=int(n_heads),
        dropout=float(dropout),
        use_memory=bool(use_memory),
        message_dimension=int(message_dim),
        memory_dimension=int(memory_dim),
        memory_update_at_start=True,
        embedding_module_type="graph_attention",
        message_function="identity",
        aggregator_type="last",
        memory_updater_type="gru",
        n_neighbors=int(n_neighbors),
        mean_time_shift_src=mean_src,
        std_time_shift_src=std_src,
        mean_time_shift_dst=mean_dst,
        std_time_shift_dst=std_dst,
    ).to(device)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    num_instance = int(train_data.n_interactions)
    num_batch = max(1, math.ceil(num_instance / int(batch_size)))
    train_losses: List[float] = []

    for _epoch in range(int(n_epoch)):
        if use_memory:
            model.memory.__init_memory__()
        model.set_neighbor_finder(train_ngh_finder)
        model.train()
        epoch_losses: List[float] = []

        for batch_idx in range(num_batch):
            start_idx = batch_idx * int(batch_size)
            end_idx = min(num_instance, start_idx + int(batch_size))
            if start_idx >= end_idx:
                continue

            src = train_data.sources[start_idx:end_idx]
            dst = train_data.destinations[start_idx:end_idx]
            ts = train_data.timestamps[start_idx:end_idx]
            eidx = train_data.edge_idxs[start_idx:end_idx]
            size = len(src)
            _, neg = train_rand_sampler.sample(size)

            pos_label = torch.ones(size, dtype=torch.float, device=device)
            neg_label = torch.zeros(size, dtype=torch.float, device=device)

            optimizer.zero_grad()
            pos_prob, neg_prob = model.compute_edge_probabilities(src, dst, neg, ts, eidx, int(n_neighbors))
            loss = criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
            if use_memory:
                model.memory.detach_memory()

        if epoch_losses:
            train_losses.append(float(np.mean(epoch_losses)))

    model.eval()

    if use_memory:
        model.memory.__init_memory__()
        model.set_neighbor_finder(train_ngh_finder)
        with torch.no_grad():
            for batch_idx in range(num_batch):
                start_idx = batch_idx * int(batch_size)
                end_idx = min(num_instance, start_idx + int(batch_size))
                if start_idx >= end_idx:
                    continue
                src = train_data.sources[start_idx:end_idx]
                dst = train_data.destinations[start_idx:end_idx]
                ts = train_data.timestamps[start_idx:end_idx]
                eidx = train_data.edge_idxs[start_idx:end_idx]
                _, neg = train_rand_sampler.sample(len(src))
                model.compute_edge_probabilities(src, dst, neg, ts, eidx, int(n_neighbors))

    state = TGNState(
        model=model,
        node_id_to_idx=node_id_to_idx,
        idx_to_node_id=idx_to_node_id,
        train_data=train_data,
        full_data=full_data,
        train_neighbor_finder=train_ngh_finder,
        full_neighbor_finder=full_ngh_finder,
        node_features=node_features,
        edge_features=edge_features,
        num_neighbors=int(n_neighbors),
        batch_size=int(batch_size),
        device=str(device),
    )

    metrics = {
        "trainLoss": float(train_losses[-1]) if train_losses else None,
        "trainPairs": int(train_data.n_interactions),
        "validationPairs": int(val_data.n_interactions),
        "testPairs": int(test_data.n_interactions),
        "epochs": int(n_epoch),
        "batchSize": int(batch_size),
        "numNeighbors": int(n_neighbors),
        "nodeDimension": int(node_dim),
        "memoryDimension": int(memory_dim),
        "messageDimension": int(message_dim),
        "learningRate": float(learning_rate),
        "nHeads": int(n_heads),
        "nLayers": int(n_layers),
        "dropout": float(dropout),
        "usesMemory": bool(use_memory),
        "architecture": "tgn_graph_attention_memory_gru",
    }

    return state, metrics, {"train": train_data, "validation": val_data, "test": test_data}



def load_runtime_to_stable_id_map(dbname: str) -> Dict[str, str]:

    rows = run_data(
        """
        MATCH (n)
        RETURN elementId(n) AS runtimeId,
               coalesce(n.eid, elementId(n)) AS stableId
        """,
        db=dbname,
    )
    return {str(r["runtimeId"]): str(r["stableId"]) for r in rows}

def _backup_memory(model: Any):
    if getattr(model, "use_memory", False) and getattr(model, "memory", None) is not None:
        return model.memory.backup_memory()
    return None


def _restore_memory(model: Any, backup: Any) -> None:
    if backup is not None and getattr(model, "use_memory", False) and getattr(model, "memory", None) is not None:
        model.memory.restore_memory(backup)


def score_pairs(
    state: TGNState,
    pairs: List[Tuple[str, str]],
    timestamp: float,
    runtime_to_stable: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    if not pairs:
        return []

    runtime_to_stable = runtime_to_stable or {}
    model = state.model
    device = torch.device(state.device if torch.cuda.is_available() and str(state.device).startswith("cuda") else "cpu")
    model = model.to(device)
    model.device = device
    model.set_neighbor_finder(state.train_neighbor_finder)
    model.eval()

    rows: List[Dict[str, Any]] = []
    batch_size = max(1, int(state.batch_size))
    backup = _backup_memory(model)

    try:
        with torch.no_grad():
            for start in range(0, len(pairs), batch_size):
                batch = pairs[start:start + batch_size]
                src_idx: List[int] = []
                dst_idx: List[int] = []
                actual_pairs: List[Tuple[str, str]] = []
                for s_raw, t_raw in batch:
                    raw_s, raw_t = _canon_pair(s_raw, t_raw)
                    stable_s = runtime_to_stable.get(raw_s, raw_s)
                    stable_t = runtime_to_stable.get(raw_t, raw_t)
                    stable_s, stable_t = _canon_pair(stable_s, stable_t)
                    if stable_s not in state.node_id_to_idx or stable_t not in state.node_id_to_idx:
                        continue
                    src_idx.append(state.node_id_to_idx[stable_s])
                    dst_idx.append(state.node_id_to_idx[stable_t])

                    actual_pairs.append((raw_s, raw_t))

                if not src_idx:
                    continue

                src = np.asarray(src_idx, dtype=np.int32)
                dst = np.asarray(dst_idx, dtype=np.int32)
                ts = np.full(len(src), float(timestamp), dtype=np.float32)
                eidx = np.zeros(len(src), dtype=np.int32)
                neg = dst.copy()


                batch_backup = _backup_memory(model)
                pos_prob, _ = model.compute_edge_probabilities(src, dst, neg, ts, eidx, int(state.num_neighbors))
                probs = pos_prob.detach().cpu().numpy().reshape(-1)
                _restore_memory(model, batch_backup)

                for (s, t), p in zip(actual_pairs, probs):
                    rows.append({"s": s, "t": t, "probability": float(p)})
    finally:
        _restore_memory(model, backup)

    rows.sort(key=lambda r: float(r.get("probability", 0.0) or 0.0), reverse=True)
    return rows


def evaluate_pairs(state: TGNState, pos_pairs: List[Tuple[str, str]], timestamp: float, negative_pairs: List[Tuple[str, str]]) -> Optional[Dict[str, Any]]:
    if not pos_pairs:
        return None
    pos_scored = score_pairs(state, pos_pairs, timestamp)
    neg_scored = score_pairs(state, negative_pairs, timestamp)
    if not pos_scored or not neg_scored:
        return None
    y_true = np.concatenate([np.ones(len(pos_scored), dtype=int), np.zeros(len(neg_scored), dtype=int)])
    y_prob = np.asarray([r["probability"] for r in pos_scored] + [r["probability"] for r in neg_scored], dtype=float)
    return _eval_scores(y_true, y_prob)
