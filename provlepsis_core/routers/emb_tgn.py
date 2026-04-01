from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Tuple
import random
import re

import numpy as np

from ..db import run_data

router = APIRouter()

_PROP_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class TGNWriteBody(BaseModel):
    graphName: str = "trainGraph"
    writeProperty: str = "TGN"
    embeddingDimension: int = Field(default=128, ge=2, le=2048)
    memoryDimension: Optional[int] = Field(default=None, ge=2, le=2048)
    timeDimension: int = Field(default=64, ge=2, le=512)
    batchSize: int = Field(default=200, ge=1, le=100000)
    epochs: int = Field(default=5, ge=1, le=500)
    learningRate: float = Field(default=1e-4, gt=0.0, le=1.0)
    numNeighbors: int = Field(default=10, ge=1, le=1000)
    randomSeed: int = 42
    device: str = "auto"
    asOfTimestamp: Optional[int] = None

def _sanitize_prop(name: str) -> str:
    if not _PROP_RE.match(str(name or "")):
        raise HTTPException(status_code=400, detail=f"Invalid writeProperty: {name}")
    return str(name)


def _assert_tgn_deps():
    try:
        import torch
        from torch.nn import Linear
        from torch_geometric.nn import TGNMemory, TransformerConv
        from torch_geometric.nn.models.tgn import (
            IdentityMessage,
            LastAggregator,
            LastNeighborLoader,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "TGN dependencies are missing. Install PyTorch + PyTorch Geometric "
                f"in the backend environment before using TGN. Import error: {e}"
            ),
        )
    return (
        torch,
        Linear,
        TGNMemory,
        TransformerConv,
        IdentityMessage,
        LastAggregator,
        LastNeighborLoader,
    )


def _pick_device(torch, requested: str):
    req = (requested or "auto").strip().lower()
    if req == "cpu":
        return torch.device("cpu")
    if req == "cuda":
        if not torch.cuda.is_available():
            raise HTTPException(
                status_code=400,
                detail="device='cuda' was requested but CUDA is not available.",
            )
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_seed(torch, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _graph_node_ids(graph_name: str, dbname: Optional[str]) -> List[str]:
    rows = run_data(
        """
        CALL gds.degree.stream($g) YIELD nodeId
        WITH DISTINCT gds.util.asNode(nodeId) AS n
        RETURN elementId(n) AS eid
        ORDER BY eid
        """,
        {"g": graph_name},
        db=dbname,
    )
    return [r["eid"] for r in rows if r.get("eid") is not None]

def _infer_graph_snapshot_ts(graph_name: str, dbname: Optional[str], time_prop: str = "firstTimestamp") -> Optional[int]:
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
          RETURN coalesce(max(toInteger(r.`{time_prop}`)), max(toInteger(r.timestamp)), 0) AS ts
        }}
        RETURN coalesce(max(ts), 0) AS cutoff
        """,
        {"g": graph_name},
        db=dbname,
    )
    if not rows:
        return None
    return int(rows[0].get("cutoff", 0) or 0)

def _graph_temporal_edges(
    graph_name: str,
    dbname: Optional[str],
    as_of_ts: Optional[int] = None,
) -> List[Tuple[str, str, int]]:
    rows = run_data(
        """
        CALL gds.graph.relationships.stream($g)
        YIELD sourceNodeId, targetNodeId
        WITH gds.util.asNode(sourceNodeId) AS sN, gds.util.asNode(targetNodeId) AS tN
        WITH elementId(sN) AS a, elementId(tN) AS b
        WITH CASE WHEN a < b THEN a ELSE b END AS s,
             CASE WHEN a < b THEN b ELSE a END AS t
        WITH DISTINCT s, t
        CALL {
          WITH s, t
          MATCH (x)-[r]-(y)
          WHERE elementId(x) = s
            AND elementId(y) = t
            AND NOT type(r) STARTS WITH 'SPLIT_'
            WITH CASE
                WHEN r.timestamps IS NOT NULL
                        AND valueType(r.timestamps) STARTS WITH 'LIST'
                        AND size(r.timestamps) > 0
                    THEN [ts IN r.timestamps | toInteger(ts)]
                ELSE [toInteger(coalesce(r.lastTimestamp, r.timestamp, 0))]
                END AS ts_list
          UNWIND ts_list AS ts
          WITH ts
          WHERE $asOfTs IS NULL OR ts <= $asOfTs
          RETURN ts
        }
        RETURN s, t, ts
        ORDER BY ts ASC, s ASC, t ASC
        """,
        {"g": graph_name, "asOfTs": as_of_ts},
        db=dbname,
    )
    return [(r["s"], r["t"], int(r.get("ts") or 0)) for r in rows]

def _write_embeddings(
    dbname: Optional[str], prop: str, rows: List[Dict[str, Any]], batch_size: int = 1000
) -> int:
    total = 0
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        out = run_data(
            f"""
            UNWIND $rows AS row
            MATCH (n) WHERE elementId(n) = row.eid
            SET n.{prop} = row.emb
            RETURN count(*) AS c
            """,
            {"rows": chunk},
            db=dbname,
        )
        total += int(out[0]["c"]) if out else 0
    return total


class GraphAttentionEmbedding:
    def __init__(self, torch, TransformerConv, in_channels: int, out_channels: int, msg_dim: int, time_enc):
        self._torch = torch
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(
            in_channels,
            out_channels // 2,
            heads=2,
            dropout=0.1,
            edge_dim=edge_dim,
        )

    def parameters(self):
        return self.conv.parameters()

    def train(self):
        self.conv.train()

    def eval(self):
        self.conv.eval()

    def to(self, device):
        self.conv = self.conv.to(device)
        return self

    def __call__(self, x, last_update, edge_index, t, msg):
        if edge_index.numel() == 0 or t.numel() == 0 or msg.numel() == 0:
            return x
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = self._torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class LinkPredictor:
    def __init__(self, torch, Linear, in_channels: int):
        self._torch = torch
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def parameters(self):
        return (
            list(self.lin_src.parameters())
            + list(self.lin_dst.parameters())
            + list(self.lin_final.parameters())
        )

    def train(self):
        self.lin_src.train()
        self.lin_dst.train()
        self.lin_final.train()

    def eval(self):
        self.lin_src.eval()
        self.lin_dst.eval()
        self.lin_final.eval()

    def to(self, device):
        self.lin_src = self.lin_src.to(device)
        self.lin_dst = self.lin_dst.to(device)
        self.lin_final = self.lin_final.to(device)
        return self

    def __call__(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h)


def _negative_dst(torch, src, dst, num_nodes: int):
    neg = torch.randint(0, num_nodes, dst.size(), device=dst.device)
    clash = (neg == src) | (neg == dst)
    tries = 0
    while bool(clash.any()) and tries < 5:
        neg[clash] = torch.randint(
            0, num_nodes, (int(clash.sum().item()),), device=dst.device
        )
        clash = (neg == src) | (neg == dst)
        tries += 1
    return neg


def _fit_and_encode_tgn(
    dbname: Optional[str],
    graph_name: str,
    write_property: str,
    embedding_dim: int,
    memory_dim: int,
    time_dim: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    num_neighbors: int,
    random_seed: int,
    requested_device: str,
    as_of_ts: Optional[int] = None,
) -> Dict[str, Any]:
    (
        torch,
        Linear,
        TGNMemory,
        TransformerConv,
        IdentityMessage,
        LastAggregator,
        LastNeighborLoader,
    ) = _assert_tgn_deps()

    device = _pick_device(torch, requested_device)
    _set_seed(torch, random_seed)

    node_ids = _graph_node_ids(graph_name, dbname)
    if not node_ids:
        raise HTTPException(status_code=400, detail=f"Graph '{graph_name}' has no nodes.")

    node_to_idx = {eid: i for i, eid in enumerate(node_ids)}
    if as_of_ts is None:
        as_of_ts = _infer_graph_snapshot_ts(graph_name, dbname, time_prop="firstTimestamp")

    temporal_edges = _graph_temporal_edges(graph_name, dbname, as_of_ts=as_of_ts)

    if not temporal_edges:
        zero_rows = [{"eid": eid, "emb": [0.0] * embedding_dim} for eid in node_ids]
        wrote = _write_embeddings(dbname, write_property, zero_rows)
        return {
            "ok": True,
            "graphName": graph_name,
            "writeProperty": write_property,
            "nodePropertiesWritten": wrote,
            "numNodes": len(node_ids),
            "numTemporalEdges": 0,
            "device": str(device),
            "lossHistory": [],
            "note": "Graph had no relationships; wrote zero vectors for all nodes.",
        }

    events: List[Tuple[int, int, float]] = []
    for s_eid, t_eid, ts in temporal_edges:
        s = node_to_idx[s_eid]
        t = node_to_idx[t_eid]
        events.append((s, t, ts))
        events.append((t, s, ts))  

    events.sort(key=lambda x: (x[2], x[0], x[1]))

    src = torch.as_tensor([e[0] for e in events], dtype=torch.long, device=device)
    dst = torch.as_tensor([e[1] for e in events], dtype=torch.long, device=device)
    t = torch.as_tensor([e[2] for e in events], dtype=torch.long, device=device)
    msg = torch.zeros((len(events), 1), dtype=torch.float32, device=device)


    memory = TGNMemory(
        len(node_ids),
        msg.size(-1),
        memory_dim,
        time_dim,
        message_module=IdentityMessage(msg.size(-1), memory_dim, time_dim),
        aggregator_module=LastAggregator(),
    ).to(device)

    gnn = GraphAttentionEmbedding(
        torch=torch,
        TransformerConv=TransformerConv,
        in_channels=memory_dim,
        out_channels=embedding_dim,
        msg_dim=msg.size(-1),
        time_enc=memory.time_enc,
    ).to(device)

    link_pred = LinkPredictor(torch=torch, Linear=Linear, in_channels=embedding_dim).to(device)

    optimizer = torch.optim.Adam(
        set(memory.parameters()) | set(gnn.parameters()) | set(link_pred.parameters()),
        lr=learning_rate,
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    assoc = torch.empty(len(node_ids), dtype=torch.long, device=device)
    neighbor_loader = LastNeighborLoader(len(node_ids), size=num_neighbors, device=device)

    loss_history: List[float] = []
    num_events = int(src.size(0))

    for _epoch in range(epochs):
        memory.train()
        gnn.train()
        link_pred.train()
        memory.reset_state()
        neighbor_loader.reset_state()

        total_loss = 0.0

        for start in range(0, num_events, batch_size):
            end = min(start + batch_size, num_events)
            bsrc = src[start:end]
            bdst = dst[start:end]
            bt = t[start:end]
            bmsg = msg[start:end]
            bneg = _negative_dst(torch, bsrc, bdst, len(node_ids))

            optimizer.zero_grad()

            query_nodes = torch.cat([bsrc, bdst, bneg]).unique()
            n_id, edge_index, e_id = neighbor_loader(query_nodes)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)

            z, last_update = memory(n_id)
            z = gnn(z, last_update, edge_index, t[e_id], msg[e_id]) if e_id.numel() > 0 else z

            pos_out = link_pred(z[assoc[bsrc]], z[assoc[bdst]])
            neg_out = link_pred(z[assoc[bsrc]], z[assoc[bneg]])

            loss = criterion(pos_out, torch.ones_like(pos_out))
            loss = loss + criterion(neg_out, torch.zeros_like(neg_out))

            memory.update_state(
                bsrc.long(),
                bdst.long(),
                bt.long(),
                bmsg.float(),
            )
            neighbor_loader.insert(bsrc.long(), bdst.long())


            loss.backward()
            optimizer.step()
            memory.detach()

            total_loss += loss.detach().item() * int(end - start)

        loss_history.append(total_loss / max(1, num_events))

    with torch.no_grad():
        memory.eval()
        gnn.eval()
        link_pred.eval()
        memory.reset_state()
        neighbor_loader.reset_state()

        for start in range(0, num_events, batch_size):
            end = min(start + batch_size, num_events)
            memory.update_state(
                src[start:end].long(),
                dst[start:end].long(),
                t[start:end].long(),
                msg[start:end].float(),
            )
            neighbor_loader.insert(src[start:end].long(), dst[start:end].long())


        all_nodes = torch.arange(len(node_ids), dtype=torch.long, device=device)
        n_id, edge_index, e_id = neighbor_loader(all_nodes)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, t[e_id], msg[e_id]) if e_id.numel() > 0 else z
        final_emb = z[assoc[all_nodes]].detach().cpu().numpy()

    rows = [
        {"eid": eid, "emb": [float(x) for x in final_emb[i].tolist()]}
        for i, eid in enumerate(node_ids)
    ]
    wrote = _write_embeddings(dbname, write_property, rows)

    return {
        "ok": True,
        "graphName": graph_name,
        "writeProperty": write_property,
        "nodePropertiesWritten": wrote,
        "numNodes": len(node_ids),
        "numTemporalEdges": len(temporal_edges),
        "numDirectedEvents": len(events),
        "embeddingDimension": embedding_dim,
        "memoryDimension": memory_dim,
        "timeDimension": time_dim,
        "batchSize": batch_size,
        "epochs": epochs,
        "learningRate": learning_rate,
        "numNeighbors": num_neighbors,
        "randomSeed": random_seed,
        "device": str(device),
        "lossHistory": loss_history,
    }


def tgn_write_for_db(body: TGNWriteBody, dbname: Optional[str] = None) -> Dict[str, Any]:
    prop = _sanitize_prop(body.writeProperty)
    memory_dim = int(body.memoryDimension or body.embeddingDimension)

    return _fit_and_encode_tgn(
        dbname=dbname,
        graph_name=body.graphName,
        write_property=prop,
        embedding_dim=int(body.embeddingDimension),
        memory_dim=memory_dim,
        time_dim=int(body.timeDimension),
        batch_size=int(body.batchSize),
        epochs=int(body.epochs),
        learning_rate=float(body.learningRate),
        num_neighbors=int(body.numNeighbors),
        random_seed=int(body.randomSeed),
        requested_device=body.device,
        as_of_ts=body.asOfTimestamp,
    )


@router.post("/write")
def tgn_write(body: TGNWriteBody):
    return tgn_write_for_db(body, dbname=None)
