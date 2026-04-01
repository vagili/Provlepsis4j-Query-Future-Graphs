from pathlib import Path
import asyncio
import io
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import time
import base64

from provlepsis_core import db
from provlepsis_core.routers import (
    config as config_router,
    databases as db_router,
    gds_store,
    query as query_router,
    load_graph as load_graph_router,
    split as split_router,
    emb_fastrp,
    emb_node2vec,
    emb_graphsage,
    emb_hashgnn,
    emb_tgn,
    lp as lp_router,
    lp_iter as lp_iter_router,
)

# ---------------------------
# Streamlit setup + styling
# ---------------------------

LOGO_PATH = Path(__file__).parent / "provlepsis4j.png"
st.set_page_config(
    page_title="Provlepsis4j",
    page_icon=str(LOGO_PATH),
    layout="wide",
)

def inject_css():
    st.markdown(
        """
        <style>
          .stApp {
            background: radial-gradient(1200px 800px at 20% 5%, rgba(46, 204, 113, 0.20), rgba(255,255,255,0) 55%),
                        radial-gradient(900px 650px at 85% 15%, rgba(241, 196, 15, 0.18), rgba(255,255,255,0) 55%),
                        radial-gradient(900px 650px at 80% 80%, rgba(231, 76, 60, 0.14), rgba(255,255,255,0) 55%),
                        linear-gradient(180deg, #0b1220 0%, #0b1220 35%, #0f172a 100%);
            color: #f8fafc;
          }

          section.main > div {
            padding-top: 1rem;
          }

          section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
            border-right: 1px solid rgba(255,255,255,0.10);
          }

          section[data-testid="stSidebar"] * {
            color: #f8fafc !important;
          }

          .prov-card {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 18px;
            padding: 16px 16px 10px 16px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.20);
          }

          .prov-title {
            font-weight: 800;
            letter-spacing: 0.2px;
            color: #f8fafc !important;
          }

          .muted {
            color: rgba(255,255,255,0.70) !important;
            font-size: 0.92rem;
          }

          div[data-baseweb="textarea"] textarea,
          div[data-baseweb="input"] input,
          div[data-baseweb="select"] > div {
            background: rgba(255,255,255,0.04) !important;
            color: #f8fafc !important;
          }

          div[data-baseweb="textarea"] textarea {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
            font-size: 0.90rem;
          }

          .stButton > button {
            background: rgba(255,255,255,0.06);
            color: #f8fafc;
            border: 1px solid rgba(255,255,255,0.14);
            border-radius: 10px;
          }

          .stCheckbox label,
          .stRadio label,
          .stSelectbox label,
          .stNumberInput label,
          .stTextInput label,
          .stFileUploader label {
            color: #f8fafc !important;
          }

          .prov-gradient-wrap {
            margin: 0 0 0.75rem 0;
            padding: 0.10rem 0 0.35rem 0;
          }

          .prov-gradient-row {
            display: flex;
            align-items: center;
            gap: 10px;
          }

          .prov-gradient-label {
            font-size: 0.85rem;
            color: rgba(255,255,255,0.75);
            font-weight: 600;
            white-space: nowrap;
          }

          .prov-gradient-bar {
            flex: 1;
            height: 14px;
            border-radius: 999px;
            background: linear-gradient(90deg, #22c55e 0%, #facc15 50%, #ef4444 100%);
            border: 1px solid rgba(255,255,255,0.16);
            box-shadow: inset 0 1px 2px rgba(0,0,0,0.18);
          }

          footer { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <style>
        /* Hide the built-in dialog title text */
        [data-testid="stDialog"] h2 {
            display: none !important;
        }

        /* Tighten the top spacing a bit after hiding the native title */
        [data-testid="stDialog"] [role="dialog"] {
            padding-top: 0.35rem !important;
        }

        .prov-dialog-header {
            text-align: center;
            margin-top: 0.15rem;
            margin-bottom: 1rem;
        }

        .prov-dialog-title {
            color: #f8fafc;
            font-weight: 800;
            font-size: 1.35rem;
            margin-top: 0.35rem;
            letter-spacing: 0.2px;
        }

        .prov-dialog-subtitle {
            color: rgba(248,250,252,0.78);
            font-size: 0.95rem;
            margin-top: 0.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_css()


# ---------------------------
# Helpers
# ---------------------------

@dataclass
class _PseudoUpload:
    file: io.BytesIO


ALL_EMB_KEYS = ["Node2Vec", "FastRP", "GraphSAGE", "HashGNN", "TGN"]

EMB_WRITE_PROP = {
    "FastRP": "FastRP",
    "Node2Vec": "Node2Vec",
    "GraphSAGE": "GraphSAGE",
    "HashGNN": "HashGNN",
    "TGN": "TGN",
}


def _as_pseudo_upload(uploaded_file) -> _PseudoUpload:
    return _PseudoUpload(file=io.BytesIO(uploaded_file.getvalue()))


def _is_configured() -> bool:
    try:
        _ = db.current_database()
        _ = db.run_value("RETURN 1", default=None)
        return True
    except Exception:
        return False


def _safe_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs), None
    except Exception as e:
        detail = getattr(e, "detail", None)
        msg = detail if isinstance(detail, str) else str(e)
        return None, msg


def _to_df(x):
    if x is None:
        return pd.DataFrame()
    if isinstance(x, list):
        return pd.DataFrame.from_records(x)
    if isinstance(x, dict):
        try:
            return pd.DataFrame(x)
        except Exception:
            return pd.DataFrame([x])
    return pd.DataFrame([{"value": x}])


def _toast_ok(msg: str):
    st.toast(msg, icon="✅")


def _queue_toast(msg: str):
    st.session_state["_pending_toast"] = msg


def _flush_pending_toast():
    msg = st.session_state.pop("_pending_toast", None)
    if msg:
        _toast_ok(msg)


def _split_signature(*, base_db: str, temporal_mode: bool, test_holdout: float, val_holdout: float) -> Dict[str, Any]:
    return {
        "base_db": str(base_db),
        "temporal_mode": bool(temporal_mode),
        "test_holdout": round(float(test_holdout), 6),
        "val_holdout": round(float(val_holdout), 6),
        "time_property": "firstTimestamp",
    }


def _get_session_split_state() -> Optional[Dict[str, Any]]:
    state = st.session_state.get("_session_split_state")
    return state if isinstance(state, dict) else None


def _reset_session_split_state():
    st.session_state.pop("_session_split_state", None)


def _get_session_split_state_for_db(base_db: str) -> Optional[Dict[str, Any]]:
    state = _get_session_split_state()
    if not state:
        return None
    if str(state.get("base_db")) != str(base_db):
        return None
    if not bool(state.get("done")):
        return None
    return state


def _remember_session_split_state(*, base_db: str, split_seconds: Optional[float], temporal_mode: bool, test_holdout: float, val_holdout: float) -> Dict[str, Any]:
    state = _split_signature(
        base_db=base_db,
        temporal_mode=temporal_mode,
        test_holdout=test_holdout,
        val_holdout=val_holdout,
    )
    state["done"] = True
    state["split_seconds"] = split_seconds
    st.session_state["_session_split_state"] = state
    return state


def _split_graphs_ready(*, want_val: bool) -> bool:
    needed = {"trainGraph", "testGraph"}
    if want_val:
        needed.add("valGraph")

    try:
        rows = db.run_data("CALL gds.graph.list() YIELD graphName RETURN collect(graphName) AS names")
        names = set((rows[0] or {}).get("names") or []) if rows else set()
        return needed.issubset(names)
    except Exception:
        return False


def _should_run_split_for_session(*, base_db: str, temporal_mode: bool, test_holdout: float, val_holdout: float) -> Tuple[bool, str, Dict[str, Any]]:
    desired_sig = _split_signature(
        base_db=base_db,
        temporal_mode=temporal_mode,
        test_holdout=test_holdout,
        val_holdout=val_holdout,
    )
    session_sig = _get_session_split_state_for_db(base_db)

    if session_sig:
        effective_sig = _split_signature(
            base_db=base_db,
            temporal_mode=bool(session_sig.get("temporal_mode")),
            test_holdout=float(session_sig.get("test_holdout", 0.0)),
            val_holdout=float(session_sig.get("val_holdout", 0.0)),
        )
        want_val = float(effective_sig["val_holdout"]) > 0.0

        if not _split_graphs_ready(want_val=want_val):
            return True, "missing_graphs", effective_sig

        if desired_sig != effective_sig:
            return False, "locked_to_first_split", effective_sig

        return False, "reuse_session", effective_sig

    return True, "first_run", desired_sig


_flush_pending_toast()

VIOLIN_FILL = {
    "Node2Vec": "rgba(106, 174, 229, 0.82)",
    "FastRP": "rgba(242, 159, 103, 0.82)",
    "GraphSAGE": "rgba(121, 201, 158, 0.82)",
    "HashGNN": "rgba(169, 139, 234, 0.82)",
    "TGN": "rgba(243, 197, 107, 0.82)",
}

VIOLIN_LINE = {
    "Node2Vec": "#5b8fdc",
    "FastRP": "#df874c",
    "GraphSAGE": "#58b980",
    "HashGNN": "#8d68dc",
    "TGN": "#d7aa46",
}


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _lerp_channel(a: int, b: int, t: float) -> int:
    return int(round(a + (b - a) * t))


def _lerp_rgb(a: Tuple[int, int, int], b: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    return (
        _lerp_channel(a[0], b[0], t),
        _lerp_channel(a[1], b[1], t),
        _lerp_channel(a[2], b[2], t),
    )


def _predicted_level_color(level: int, max_level: int) -> str:
    green = (34, 197, 94)
    yellow = (250, 204, 21)
    red = (239, 68, 68)

    if max_level <= 1:
        return _rgb_to_hex(green)

    ratio = (max(1, level) - 1) / max(1, max_level - 1)

    if ratio <= 0.5:
        local_t = ratio / 0.5
        return _rgb_to_hex(_lerp_rgb(green, yellow, local_t))

    local_t = (ratio - 0.5) / 0.5
    return _rgb_to_hex(_lerp_rgb(yellow, red, local_t))


def _render_prediction_gradient_bar(max_level: int) -> str:
    return f"""
    <div class="prov-gradient-wrap">
      <div class="prov-gradient-row">
        <span class="prov-gradient-label">Present</span>
        <div class="prov-gradient-bar"></div>
        <span class="prov-gradient-label">Future</span>
      </div>
    </div>
    """


def _common_prediction_overlap(
    series_by_key: Dict[str, pd.DataFrame],
    visible_keys: List[str],
) -> Optional[Tuple[int, int]]:
    mins: List[int] = []
    maxs: List[int] = []

    for fam in visible_keys:
        df = series_by_key.get(fam)
        if df is None or df.empty or "predicted_timestep" not in df.columns:
            continue

        vals = pd.to_numeric(df["predicted_timestep"], errors="coerce").dropna().astype(int)
        vals = vals[vals > 0]
        if vals.empty:
            continue

        mins.append(int(vals.min()))
        maxs.append(int(vals.max()))

    if not mins or not maxs:
        return None

    overlap_start = max(mins)
    overlap_end = min(maxs)

    if overlap_start > overlap_end:
        return None

    return overlap_start, overlap_end


def _apply_prediction_range(
    series_by_key: Dict[str, pd.DataFrame],
    visible_keys: List[str],
    predicted_range: Optional[Tuple[int, int]],
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}

    for fam in visible_keys:
        df = series_by_key.get(fam)
        if df is None:
            out[fam] = pd.DataFrame()
            continue

        df2 = df.copy()

        if predicted_range is not None and not df2.empty and "predicted_timestep" in df2.columns:
            lo, hi = predicted_range
            pts = pd.to_numeric(df2["predicted_timestep"], errors="coerce").fillna(0).astype(int)
            df2 = df2[(pts >= lo) & (pts <= hi)].copy()

        out[fam] = df2

    return out

def _canon_pair_key(s: Any, t: Any) -> str:
    a = str(s or "")
    b = str(t or "")
    return f"{a}|{b}" if a < b else f"{b}|{a}"


def _pairs_payload_to_set(pairs: List[Dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for p in pairs or []:
        out.add(_canon_pair_key(p.get("s"), p.get("t")))
    return out


def _compute_overlap_pct(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    k = min(len(a), len(b))
    if k <= 0:
        return 0.0

    common = 0
    if len(a) <= len(b):
        for x in a:
            if x in b:
                common += 1
    else:
        for x in b:
            if x in a:
                common += 1

    return (common / k) * 100.0


def _compute_gt_coverage_pct(recognized: set[str], gt: set[str]) -> float:
    if not gt:
        return 0.0
    return (len(recognized & gt) / len(gt)) * 100.0


def _pick_metrics_block(metrics_obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not metrics_obj:
        return None
    return metrics_obj.get("test") or metrics_obj.get("validation") or metrics_obj.get("train") or metrics_obj


def _fmt_metric(x: Any) -> str:
    try:
        if x is None:
            return "—"
        return f"{float(x):.3f}"
    except Exception:
        return "—"

def _graph_to_plotly(graph: Dict[str, Any], title: str, future: bool = False) -> go.Figure:
    if graph is None:
        graph = {"nodes": [], "edges": []}
    if isinstance(graph, list):
        if len(graph) == 1 and isinstance(graph[0], dict):
            graph = graph[0]
        else:
            graph = {"nodes": graph, "edges": []}

    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []

    G = nx.Graph()

    for n in nodes:
        nid = str(n.get("id"))
        if not nid:
            continue
        label = n.get("nodeId")
        if label is None:
            label = n.get("id")
        G.add_node(nid, label=str(label))

    for e in edges:
        s = str(e.get("source"))
        t = str(e.get("target"))
        if not s or not t:
            continue

        try:
            predicted_lvl = int(e.get("predicted_timestep") or e.get("predicted") or 0)
        except Exception:
            predicted_lvl = 0

        try:
            probability = float(e.get("probability", 1.0 if predicted_lvl == 0 else 0.0) or 0.0)
        except Exception:
            probability = 1.0 if predicted_lvl == 0 else 0.0

        try:
            timestamp = int(e.get("timestamp") or e.get("lastTimestamp") or 0)
        except Exception:
            timestamp = 0

        G.add_edge(
            s,
            t,
            predicted=predicted_lvl,
            probability=probability,
            timestamp=timestamp,
            etype=str(e.get("type") or ""),
        )

    fig = go.Figure()

    if len(G.nodes) == 0:
        fig.update_layout(
            title={"text": title, "font": {"size": 14, "color": "#64748b"}},
            height=520,
            margin=dict(l=10, r=10, t=35, b=10),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#334155"),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return fig

    pos = nx.spring_layout(G, seed=7, k=0.9 / math.sqrt(max(len(G.nodes), 1)))

    max_predicted_level = max(
        [int(data.get("predicted", 0)) for _, _, data in G.edges(data=True) if int(data.get("predicted", 0)) > 0] or [0]
    )

    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]

        predicted_lvl = int(data.get("predicted", 0) or 0)
        probability = float(data.get("probability", 0.0) or 0.0)
        timestamp = int(data.get("timestamp", 0) or 0)
        etype = str(data.get("etype", "") or "")

        is_pred = future and predicted_lvl > 0

        if is_pred:
            color = _predicted_level_color(predicted_lvl, max_predicted_level)
            width = 3.0
            hovertemplate = (
                "<b>Predicted edge</b><br>"
                "probability=%{customdata[0]:.4f}<br>"
                "predicted_timestep=%{customdata[1]}<br>"
                "timestamp=%{customdata[2]}<br>"
                "type=%{customdata[3]}"
                "<extra></extra>"
            )
        else:
            color = "#94a3b8"
            width = 1.7
            hovertemplate = (
                "<b>Existing edge</b><br>"
                "probability=%{customdata[0]:.4f}<br>"
                "timestamp=%{customdata[2]}<br>"
                "type=%{customdata[3]}"
                "<extra></extra>"
            )

        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(width=width, color=color),
                customdata=[[probability, predicted_lvl, timestamp, etype]],
                hovertemplate=hovertemplate,
                showlegend=False,
            )
        )

    node_x, node_y, node_text = [], [], []
    for nid in G.nodes():
        x, y = pos[nid]
        node_x.append(x)
        node_y.append(y)
        node_text.append(G.nodes[nid].get("label", nid))

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            hovertext=node_text,
            hoverinfo="text",
            marker=dict(
                size=16,
                color="#5b8fdc",
                line=dict(width=1.2, color="#ffffff"),
            ),
            textfont=dict(
                color="#ffffff",
                size=13,
            ),
            showlegend=False,
        )
    )

    fig.update_layout(
        title={"text": title, "font": {"size": 14, "color": "#64748b"}},
        height=520,
        margin=dict(l=10, r=10, t=35, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc"),
        showlegend=False,
        xaxis=dict(visible=False, showgrid=False, zeroline=False),
        yaxis=dict(visible=False, showgrid=False, zeroline=False),
        hoverlabel=dict(bgcolor="#0f172a", font_color="#f8fafc"),    
        )
    return fig

def _db_candidates_for_base(base: str) -> List[Dict[str, Any]]:
    out, err = _safe_call(lp_router.lp_predicted_dbs_for, base)
    if err or not out:
        return []
    return out.get("candidates") or []


def _predicted_db_dropdown(base: str) -> Tuple[Optional[str], List[str]]:
    cand = _db_candidates_for_base(base)
    existing = [c["name"] for c in cand if c.get("exists")]
    if not existing:
        return None, []

    preferred = None
    for name in existing:
        if name.endswith("-Node2Vec"):
            preferred = name
            break
    selected = st.session_state.get("future_db") or preferred or existing[0]
    if selected not in existing:
        selected = existing[0]
    return selected, existing


def _refresh_predicted_timesteps(pred_db: Optional[str]) -> Dict[str, int]:
    if not pred_db:
        return {"maxPredicted": 0, "maxTimestamp": 0}
    out, err = _safe_call(lp_router.lp_predicted_timestamps, pred_db)
    if err or not out:
        return {"maxPredicted": 0, "maxTimestamp": 0}
    return {
        "maxPredicted": int(out.get("maxPredicted", 0) or 0),
        "maxTimestamp": int(out.get("maxTimestamp", 0) or 0),
    }


# ---------------------------
# Plot helpers
# ---------------------------

def _series_payload_to_df(points: List[Dict[str, Any]], family: str) -> pd.DataFrame:
    cols = ["family", "s", "t", "probability", "timestamp", "predicted_timestep"]
    if not points:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame.from_records(points)

    if "probability" not in df.columns:
        df["probability"] = 0.0
    if "timestamp" not in df.columns:
        df["timestamp"] = 0
    if "predicted_timestep" not in df.columns:
        df["predicted_timestep"] = 0
    if "s" not in df.columns:
        df["s"] = ""
    if "t" not in df.columns:
        df["t"] = ""

    df["probability"] = pd.to_numeric(df["probability"], errors="coerce").fillna(0.0)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0).astype(int)
    df["predicted_timestep"] = pd.to_numeric(df["predicted_timestep"], errors="coerce").fillna(0).astype(int)
    df["family"] = family

    return df[cols].sort_values(["timestamp", "probability"], ascending=[True, False]).reset_index(drop=True)


def _refresh_plot_payload(base: str):
    cand = _db_candidates_for_base(base)

    series_by_key: Dict[str, pd.DataFrame] = {}
    errs: Dict[str, str] = {}
    exists_by_type: Dict[str, bool] = {}

    for c in cand or []:
        fam = c.get("type")
        dbname = c.get("name")
        exists = bool(c.get("exists"))

        if fam:
            exists_by_type[fam] = exists

        if not fam or not dbname or not exists:
            continue

        raw, err = _safe_call(lp_router.lp_predicted_series, dbname)
        if err:
            errs[fam] = err
            series_by_key[fam] = _series_payload_to_df([], fam)
            continue

        points = (raw or {}).get("points") or []
        series_by_key[fam] = _series_payload_to_df(points, fam)

    st.session_state["plots_series_by_key"] = series_by_key
    st.session_state["plots_errs"] = errs
    st.session_state["plots_exists_by_type"] = exists_by_type
    st.session_state["plots_loaded_for_db"] = base


def _build_probability_vs_timestamp_figure_for_family(
    df: pd.DataFrame,
    family: str,
) -> go.Figure:
    fig = go.Figure()

    if df is None or df.empty:
        fig.update_layout(
            height=420,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f8fafc"),
            xaxis_title="Timestamp",
            yaxis_title="Probability",
            yaxis=dict(
                range=[0.0, 1.0],
                showgrid=True,
                gridcolor="rgba(255,255,255,0.10)",
                zeroline=False,
            ),
            xaxis=dict(showgrid=False),
            showlegend=False,
            hoverlabel=dict(bgcolor="#0f172a", font_color="#f8fafc"),
        )
        return fig

    df2 = df.copy()
    df2["timestamp"] = pd.to_numeric(df2["timestamp"], errors="coerce").fillna(0).astype(int)
    df2["probability"] = pd.to_numeric(df2["probability"], errors="coerce").fillna(0.0)

    unique_timestamps = sorted(df2["timestamp"].dropna().unique().tolist())
    ts_labels = [str(ts) for ts in unique_timestamps]
    n_ts = len(unique_timestamps)

    pad_left = [f"__pad_l{i}__" for i in range(2, 0, -1)]
    pad_right = [f"__pad_r{i}__" for i in range(1, 3)]
    category_array = pad_left + ts_labels + pad_right

    fig.add_trace(
        go.Scatter(
            x=category_array,
            y=[0.0] * len(category_array),
            mode="markers",
            marker=dict(size=0.1, opacity=0),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    if n_ts == 1:
        violin_width = 0.42
        figure_height = 420
    elif n_ts == 2:
        violin_width = 0.34
        figure_height = 450
    elif n_ts == 3:
        violin_width = 0.42
        figure_height = 500
    else:
        violin_width = 0.55
        figure_height = 520

    for ts in unique_timestamps:
        ts_label = str(ts)
        ts_df = df2[df2["timestamp"] == ts]
        if ts_df.empty:
            continue

        fig.add_trace(
            go.Violin(
                x=[ts_label] * len(ts_df),
                y=ts_df["probability"],
                name=ts_label,
                points=False,
                fillcolor=VIOLIN_FILL.get(family, "rgba(148,163,184,0.82)"),
                line=dict(color=VIOLIN_LINE.get(family, "#64748b"), width=1.7),
                box=dict(
                    visible=True,
                    fillcolor="white",
                    line=dict(color="black", width=1.8),
                ),
                meanline=dict(visible=False),
                scalemode="width",
                width=violin_width,
                hovertemplate=(
                    f"<b>{family}</b><br>"
                    "timestamp=%{x}<br>"
                    "probability=%{y:.4f}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        height=figure_height,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc"),
        xaxis_title="Timestamp",
        yaxis_title="Probability",
        yaxis=dict(
            range=[0.0, 1.0],
            showgrid=True,
            gridcolor="rgba(255,255,255,0.10)",
            zeroline=False,
        ),
        xaxis=dict(
            showgrid=False,
            type="category",
            categoryorder="array",
            categoryarray=category_array,
            tickmode="array",
            tickvals=ts_labels,
            ticktext=ts_labels,
        ),
        showlegend=False,
        violinmode="group",
        hoverlabel=dict(bgcolor="#0f172a", font_color="#f8fafc"),
    )

    return fig

def _build_probability_summary_df(
    series_by_key: Dict[str, pd.DataFrame],
    visible_keys: List[str],
    ) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for fam in visible_keys:
        df = series_by_key.get(fam)
        if df is None or df.empty:
            continue

        rows.append({
            "Embedding": fam,
            "Predicted edges": int(len(df)),
            "Min prediction timestamp": int(df["timestamp"].min()),
            "Max prediction timestamp": int(df["timestamp"].max()),
            "Mean prediction probability": float(df["probability"].mean()),
            "Max prediction probability": float(df["probability"].max()),
        })

    return pd.DataFrame(rows)


# ---------------------------
# Admin evaluation refresh
# ---------------------------

def _refresh_eval_payload(base: str):
    pred_raw, err1 = _safe_call(lp_router.lp_evaluation_predicted_edges)
    met_raw, err2 = _safe_call(lp_router.lp_metrics)
    gt_raw, err3 = _safe_call(lp_router.lp_test_edges)

    if err1:
        st.session_state["eval_pred_err"] = err1
        st.session_state["eval_pred_sets"] = {}
    else:
        edges = (pred_raw or {}).get("edges") or {}
        st.session_state["eval_pred_sets"] = {k: _pairs_payload_to_set(v) for k, v in edges.items()}
        st.session_state["eval_pred_err"] = None

    if err2:
        st.session_state["eval_met_err"] = err2
        st.session_state["eval_metrics"] = {}
    else:
        st.session_state["eval_metrics"] = (met_raw or {}).get("metrics") or {}
        st.session_state["eval_met_err"] = None

    if err3:
        st.session_state["eval_gt_err"] = err3
        st.session_state["eval_gt_set"] = set()
    else:
        st.session_state["eval_gt_set"] = _pairs_payload_to_set((gt_raw or {}).get("edges") or [])
        st.session_state["eval_gt_err"] = None

    st.session_state["eval_loaded_for_db"] = base


# ---------------------------
# Admin workflow helpers
# ---------------------------

def _prepare_hidden_fullgraph_projection() -> Tuple[Optional[Any], Optional[str]]:
    _safe_call(gds_store.drop_projected_graph, {"name": "fullGraph"})
    out, err = _safe_call(gds_store.project_store_graph, {"name": "fullGraph"})
    return out, err


def _run_embedding_for_family(fam: str, params: Dict[str, Any]) -> Tuple[Optional[Any], Optional[str], str]:
    write_prop = EMB_WRITE_PROP[fam]

    if fam == "FastRP":
        body = emb_fastrp.FastRPWriteBody(
            graphName="trainGraph",
            embeddingDimension=int(params["embeddingDimension"]),
            writeProperty=write_prop,
            propertyRatio=float(params["propertyRatio"]),
        )
        out, err = _safe_call(emb_fastrp.fastrp_write, body)
        if err:
            return None, err, write_prop
        if isinstance(out, dict) and not out.get("ok", True):
            return out, out.get("error", "FastRP failed"), write_prop
        return out, None, write_prop

    if fam == "Node2Vec":
        body = emb_node2vec.Node2VecBody(
            embeddingDimension=int(params["embeddingDimension"]),
            walkLength=int(params["walkLength"]),
            walksPerNode=int(params["walksPerNode"]),
            returnFactor=float(params["returnFactor"]),
            inOutFactor=float(params["inOutFactor"]),
            writeProperty=write_prop,
        )
        out, err = _safe_call(emb_node2vec.node2vec_write, body)
        return out, err, write_prop

    if fam == "GraphSAGE":
        body = emb_graphsage.GraphSAGETrainWriteBody(
            graphName="trainGraph",
            modelName="sage_model",
            writeProperty=write_prop,
            embeddingDimension=int(params["embeddingDimension"]),
            epochs=int(params["epochs"]),
            learningRate=float(params["learningRate"]),
        )
        out, err = _safe_call(emb_graphsage.graphsage_train_write, body)
        return out, err, write_prop

    if fam == "HashGNN":
        payload = {
            "outputDimension": int(params["outputDimension"]),
            "iterations": int(params["iterations"]),
            "embeddingDensity": int(params["embeddingDensity"]),
            "writeProperty": write_prop,
        }
        out, err = _safe_call(emb_hashgnn.hashgnn_write, payload)
        return out, err, write_prop

    if fam == "TGN":
        body = emb_tgn.TGNWriteBody(
            graphName="trainGraph",
            writeProperty=write_prop,
            embeddingDimension=int(params["embeddingDimension"]),
            memoryDimension=int(params["memoryDimension"]),
            timeDimension=int(params["timeDimension"]),
            epochs=int(params["epochs"]),
            batchSize=int(params["batchSize"]),
            learningRate=float(params["learningRate"]),
            numNeighbors=int(params["numNeighbors"]),
        )
        out, err = _safe_call(emb_tgn.tgn_write, body)
        return out, err, write_prop

    return None, f"Unsupported embedding family '{fam}'.", write_prop


def _run_lp_for_property(embedding_property: str, predict_k: int, prob_threshold: float) -> Tuple[Optional[Any], Optional[str]]:
    body = lp_router.LPBody(
        embeddingProperty=embedding_property,
        trainGraphName="trainGraph",
        valGraphName="valGraph",
        testGraphName="testGraph",
        negativeRatio=1.0,
        predictK=int(predict_k),
        candidateMultiplier=20,
        probThreshold=float(prob_threshold),
        outputGraphName="predictedGraph",
        force_new_canonical_pool=False,
    )
    return _safe_call(lp_router.lp_run, body)


# ---------------------------
# Load dataset modal
# ---------------------------

@st.dialog("Load Dataset", width="large")
def _load_dataset_dialog():
    edges = st.file_uploader(
        "Edges CSV (source,target[,type][,timestamp])",
        type=["csv"],
        key="dialog_edges",
    )
    features = st.file_uploader(
        "Features CSV (node id + props) — optional",
        type=["csv"],
        key="dialog_features",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        is_temporal = st.checkbox("Temporal graph import", value=False, key="dialog_is_temporal")
    with c2:
        timestamp_col = st.text_input(
            "Timestamp column",
            value="timestamp",
            disabled=not is_temporal,
            key="dialog_timestamp_col",
        )
    with c3:
        dataset_name = st.text_input("Dataset DB name (optional)", value="", key="dialog_dataset_name")

    if st.button("Load Dataset", use_container_width=True, key="dialog_load_dataset_btn"):
        if not edges:
            st.error("Please upload an edges CSV.")
            return

        pseudo_edges = _as_pseudo_upload(edges)
        pseudo_feats = _as_pseudo_upload(features) if features else None

        async def _run():
            return await load_graph_router.load_graph(
                edges=pseudo_edges,
                features=pseudo_feats,
                isTemporal=is_temporal,
                timestampColumn=timestamp_col if is_temporal else None,
                dataset_name=dataset_name if dataset_name.strip() else None,
                purge_all_stringish=True,
            )

        try:
            out = asyncio.run(_run())
            st.session_state["admin_load_response"] = out
            _reset_session_split_state()
            _queue_toast(f"Loaded dataset into DB: {out.get('database')}")
            st.rerun()
        except Exception as e:
            msg = getattr(e, "detail", None) or str(e)
            st.error(msg)


# ---------------------------
# Connection dialog
# ---------------------------

@st.dialog(" ", width="large")
def _connection_dialog():
    st.markdown("<div class='prov-dialog-header'>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 3, 1])
    with c2:
        logo_b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode()
        st.markdown(
            f"""
            <div style="display: flex; justify-content: center; width: 100%;">
                <img src="data:image/png;base64,{logo_b64}" style="width: 300px;" />
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="prov-dialog-title">Connection Configuration</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)


    default_uri = st.session_state.get("neo4j_uri") or "bolt://localhost:7687"
    default_user = st.session_state.get("neo4j_user") or "neo4j"
    default_pwd = st.session_state.get("neo4j_password") or ""
    default_db = st.session_state.get("neo4j_db") or "neo4j"

    uri = st.text_input("NEO4J_URI", value=default_uri, key="dialog_neo4j_uri")
    user = st.text_input("NEO4J_USER", value=default_user, key="dialog_neo4j_user")
    password = st.text_input("NEO4J_PASSWORD", value=default_pwd, type="password", key="dialog_neo4j_password")
    database_name = st.text_input("Default database", value=default_db, key="dialog_neo4j_db")

    if st.button("Connect", use_container_width=True, key="connect_btn_dialog"):
        st.session_state["neo4j_uri"] = uri
        st.session_state["neo4j_user"] = user
        st.session_state["neo4j_password"] = password
        st.session_state["neo4j_db"] = database_name

        cfg = config_router.Neo4jConfig(uri=uri, user=user, password=password, database=database_name)
        _, err = _safe_call(config_router.set_neo4j, cfg)
        if err:
            st.error(err)
        else:
            active_db = database_name
            try:
                active_db = db.current_database()
            except Exception:
                pass
            _reset_session_split_state()
            _queue_toast(f"Connected. Active DB: {active_db}")
            st.rerun()


# ---------------------------
# Sidebar: Mode + Navigation
# ---------------------------

with st.sidebar:
    with st.sidebar:
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.image(str(LOGO_PATH), width=120)

        mode = st.radio(
            "Swap Views",
            ["Administrator", "User"],
            index=0,
            horizontal=False,
        )

    st.divider()

    configured = _is_configured()

    if configured and mode == "Administrator":
        st.markdown("<div class='prov-title'>Database Selection</div>", unsafe_allow_html=True)
        db_data, db_err = _safe_call(db_router.get_databases)
        if db_err:
            st.error(db_err)
        else:
            current_db = db_data.get("current")
            db_names = [r["name"] for r in (db_data.get("databases") or []) if r.get("name")]

            st.caption(f"Current Database: {current_db}")
            selected_admin_db = st.selectbox(
                "Available databases",
                options=db_names,
                index=db_names.index(current_db) if current_db in db_names else 0,
                key="admin_db_select",
            )

            c_use, c_drop = st.columns(2)
            with c_use:
                if st.button("Use", use_container_width=True, key="admin_use_db_btn"):
                    body = db_router.UseDbBody(name=selected_admin_db)
                    _, err = _safe_call(db_router.use_database, body)
                    if err:
                        st.error(err)
                    else:
                        _reset_session_split_state()
                        _queue_toast(f"Active DB set to {selected_admin_db}")
                        st.rerun()

            with c_drop:
                protected = selected_admin_db.lower() in {"system", "neo4j"}
                if st.button("Drop", use_container_width=True, key="admin_drop_db_btn", disabled=protected):
                    body = db_router.DropDbBody(name=selected_admin_db)
                    _, err = _safe_call(db_router.drop_database, body)
                    if err:
                        st.error(err)
                    else:
                        _reset_session_split_state()
                        _queue_toast(f"Dropped database {selected_admin_db}")
                        st.rerun()

            if protected:
                st.caption("")

        st.divider()

        st.markdown("<div class='prov-title'>Dataset Loading</div>", unsafe_allow_html=True)
        if st.button("Load Dataset", use_container_width=True, key="open_load_dataset_dialog_btn"):
            _load_dataset_dialog()

        st.divider()

        st.markdown("<div class='prov-title'>Link Prediction</div>", unsafe_allow_html=True)
        admin_temporal_split = st.checkbox(
            "Temporal split",
            value=False,
            key="admin_temporal_split",
        )
        admin_family = st.selectbox(
            "Embedding",
            ["Node2Vec", "FastRP", "GraphSAGE", "HashGNN", "TGN"],
            index=0,
            key="admin_lp_family",
        )
        admin_test_holdout = st.number_input(
            "Test Set",
            min_value=0.0,
            max_value=0.5,
            value=0.10,
            step=0.01,
            format="%.2f",
            key="admin_test_holdout",
        )
        admin_val_holdout = st.number_input(
            "Validation Set",
            min_value=0.0,
            max_value=0.5,
            value=0.10,
            step=0.01,
            format="%.2f",
            key="admin_val_holdout",
        )
        admin_predict_k = st.number_input(
            "Number of predicted edges",
            min_value=1,
            max_value=50000,
            value=100,
            step=10,
            key="admin_predict_k",
        )
        admin_prob_threshold = st.number_input(
            "Edge existence probability threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.50,
            step=0.01,
            format="%.2f",
            key="admin_prob_threshold",
        )
        admin_advanced = st.checkbox("Advanced Embedding Parameters", value=False, key="admin_advanced")

        emb_params: Dict[str, Any] = {}

        if admin_family == "FastRP":
            emb_params = {
                "embeddingDimension": 128,
                "propertyRatio": 0.50,
            }
            if admin_advanced:
                emb_params["embeddingDimension"] = st.number_input("Embedding dimension", 2, 2048, 128, 1, key="adv_fastrp_dim")
                emb_params["propertyRatio"] = st.number_input(
                    "propertyRatio",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.50,
                    step=0.05,
                    format="%.2f",
                    key="adv_fastrp_ratio",
                )

        elif admin_family == "Node2Vec":
            emb_params = {
                "embeddingDimension": 128,
                "walkLength": 80,
                "walksPerNode": 10,
                "returnFactor": 1.0,
                "inOutFactor": 1.0,
            }
            if admin_advanced:
                emb_params["embeddingDimension"] = st.number_input("Embedding dimension", 2, 2048, 128, 1, key="adv_n2v_dim")
                emb_params["walkLength"] = st.number_input("walkLength", 1, 200, 80, 1, key="adv_n2v_wl")
                emb_params["walksPerNode"] = st.number_input("walksPerNode", 1, 200, 10, 1, key="adv_n2v_wpn")
                emb_params["returnFactor"] = st.number_input("returnFactor", 0.01, 10.0, 1.0, 0.05, key="adv_n2v_rf")
                emb_params["inOutFactor"] = st.number_input("inOutFactor", 0.01, 10.0, 1.0, 0.05, key="adv_n2v_iof")

        elif admin_family == "GraphSAGE":
            emb_params = {
                "embeddingDimension": 128,
                "epochs": 10,
                "learningRate": 0.01,
            }
            if admin_advanced:
                emb_params["embeddingDimension"] = st.number_input("Embedding dimension", 2, 2048, 128, 1, key="adv_sage_dim")
                emb_params["epochs"] = st.number_input("epochs", 1, 200, 10, 1, key="adv_sage_epochs")
                emb_params["learningRate"] = st.number_input("learningRate", 0.0001, 1.0, 0.01, 0.0005, key="adv_sage_lr")

        elif admin_family == "HashGNN":
            emb_params = {
                "outputDimension": 128,
                "iterations": 10,
                "embeddingDensity": 10,
            }
            if admin_advanced:
                emb_params["outputDimension"] = st.number_input("Output dimension", 2, 2048, 128, 1, key="adv_hash_dim")
                emb_params["iterations"] = st.number_input("iterations", 1, 200, 10, 1, key="adv_hash_iters")
                emb_params["embeddingDensity"] = st.number_input("embeddingDensity", 1, 200, 10, 1, key="adv_hash_dens")

        elif admin_family == "TGN":
            emb_params = {
                "embeddingDimension": 128,
                "memoryDimension": 128,
                "timeDimension": 64,
                "epochs": 5,
                "batchSize": 200,
                "learningRate": 0.0001,
                "numNeighbors": 10,
            }
            if admin_advanced:
                emb_params["embeddingDimension"] = st.number_input("Embedding dimension", 2, 2048, 128, 1, key="adv_tgn_dim")
                emb_params["memoryDimension"] = st.number_input("Memory dimension", 2, 2048, 128, 1, key="adv_tgn_mem")
                emb_params["timeDimension"] = st.number_input("Time dimension", 2, 512, 64, 1, key="adv_tgn_time")
                emb_params["epochs"] = st.number_input("epochs", 1, 500, 5, 1, key="adv_tgn_epochs")
                emb_params["batchSize"] = st.number_input("batchSize", 1, 100000, 200, 1, key="adv_tgn_batch")
                emb_params["learningRate"] = st.number_input(
                    "learningRate",
                    min_value=0.00001,
                    max_value=1.0,
                    value=0.0001,
                    step=0.00001,
                    format="%.5f",
                    key="adv_tgn_lr",
                )
                emb_params["numNeighbors"] = st.number_input("numNeighbors", 1, 1000, 10, 1, key="adv_tgn_neighbors")

        current_admin_db = db.current_database()
        split_run_now, split_reason, effective_split_sig = _should_run_split_for_session(
            base_db=current_admin_db,
            temporal_mode=bool(admin_temporal_split),
            test_holdout=float(admin_test_holdout),
            val_holdout=float(admin_val_holdout),
        )
        if split_run_now:
            if split_reason == "missing_graphs":
                st.caption("")

        else:
            if split_reason == "locked_to_first_split":
                st.caption("")

        if st.button("Predict Edges", use_container_width=True, key="admin_train_predict_btn"):
            base_db_now = db.current_database()
            split_run_now, split_reason, effective_split_sig = _should_run_split_for_session(
                base_db=base_db_now,
                temporal_mode=bool(admin_temporal_split),
                test_holdout=float(admin_test_holdout),
                val_holdout=float(admin_val_holdout),
            )
            effective_temporal_split = bool(effective_split_sig["temporal_mode"])
            effective_test_holdout = float(effective_split_sig["test_holdout"])
            effective_val_holdout = float(effective_split_sig["val_holdout"])
            split_seconds: Optional[float] = None
            split_info = _get_session_split_state_for_db(base_db_now)

            _, prep_err = _prepare_hidden_fullgraph_projection()
            if prep_err:
                st.error(f"Hidden fullGraph projection failed: {prep_err}")
            else:
                split_ok = True
                if split_run_now:
                    split_t0 = time.perf_counter()
                    split_body = split_router.ExecuteSplitBody(
                        trainGraphName="trainGraph",
                        testGraphName="testGraph",
                        valGraphName="valGraph",
                        testHoldout=effective_test_holdout,
                        valHoldout=effective_val_holdout,
                        ensureConnected=True,
                        includeEmbeddingProps=False,
                        reFreshFeatureCache=True,
                        temporalMode=effective_temporal_split,
                    )
                    _, split_err = _safe_call(split_router.execute_master_split, split_body)
                    if split_err:
                        split_ok = False
                        st.error(split_err)
                    else:
                        split_seconds = time.perf_counter() - split_t0
                        split_info = _remember_session_split_state(
                            base_db=base_db_now,
                            split_seconds=split_seconds,
                            temporal_mode=effective_temporal_split,
                            test_holdout=effective_test_holdout,
                            val_holdout=effective_val_holdout,
                        )

                if split_ok:
                    embedding_t0 = time.perf_counter()
                    emb_out, emb_err, emb_prop = _run_embedding_for_family(admin_family, emb_params)
                    embedding_seconds = time.perf_counter() - embedding_t0

                    if emb_err:
                        st.error(emb_err)
                    else:
                        lp_out, lp_err = _run_lp_for_property(
                            embedding_property=emb_prop,
                            predict_k=int(admin_predict_k),
                            prob_threshold=float(admin_prob_threshold),
                        )
                        if lp_err:
                            st.error(lp_err)
                        else:
                            st.session_state["admin_embedding_response"] = emb_out
                            st.session_state["admin_lp_response"] = lp_out
                            _refresh_eval_payload(base_db_now)

                            if split_run_now:
                                if split_reason == "missing_graphs":
                                    _toast_ok(f"{admin_family} embeddings created. Link prediction completed")
                                else:
                                    _toast_ok(f"{admin_family} embeddings created. Link prediction completed")
                            else:
                                _toast_ok(f"{admin_family} embeddings created. Link prediction completed")

configured = _is_configured()

if not configured:
    _connection_dialog()
    st.info("Enter the Neo4j connection settings in the startup dialog to begin.")
    st.stop()

base_db = db.current_database()


# ---------------------------
# USER VIEW
# ---------------------------

if mode == "User":
    with st.sidebar:
        st.divider()
        st.markdown("<div class='prov-title'>Prediction Controls</div>", unsafe_allow_html=True)

        future_db, existing_pred = _predicted_db_dropdown(base_db)
        if existing_pred:
            st.session_state["future_db"] = st.selectbox(
                "Future DB",
                existing_pred,
                index=existing_pred.index(future_db) if future_db in existing_pred else 0,
            )
            if st.session_state.get("_last_future_db_toasted") != st.session_state["future_db"]:
                _toast_ok(f"Future DB selected: {st.session_state['future_db']}")
                st.session_state["_last_future_db_toasted"] = st.session_state["future_db"]
        else:
            st.info("No predicted DB found for this base database. Create one from Admin → Train & Predict.")
            st.session_state["future_db"] = None

        ts_info = _refresh_predicted_timesteps(st.session_state.get("future_db"))
        st.text_input("Current predicted timesteps", value=str(ts_info["maxPredicted"]), disabled=True)

        st.write("")
        st.markdown("<div class='prov-title'>Predicted Graph Extension</div>", unsafe_allow_html=True)
        emb_family = st.selectbox("Embedding", ["FastRP", "Node2Vec", "GraphSAGE", "HashGNN", "TGN"], index=0)
        predict_k = st.number_input("Number of predicted edges (K)", min_value=1, value=100, step=10)
        prob_thr = st.number_input(
            "Edge existence probability threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.50,
            step=0.01,
            format="%.2f",
        )
        add_steps = st.number_input("Additional predicted timesteps", min_value=1, value=1, step=1)

        if st.button("Extend Predicted Graph", use_container_width=True):
            if not st.session_state.get("future_db"):
                st.error("No predicted DB selected.")
            else:
                body = lp_iter_router.IterateBody(
                    embeddingFamily=emb_family,
                    embeddingProperty=EMB_WRITE_PROP.get(emb_family, "FastRP"),
                    predictK=int(predict_k),
                    candidateMultiplier=20,
                    probThreshold=float(prob_thr),
                    negativeRatio=1.0,
                )
                ok = True
                last = None
                for _ in range(int(add_steps)):
                    last, err = _safe_call(lp_iter_router.iterate_prediction, body)
                    if err:
                        ok = False
                        st.error(err)
                        break
                if ok:
                    _toast_ok("Predicted graph extended.")
                    st.session_state["last_iterate"] = last

    qcol, btncol = st.columns([7, 1.2])
    with qcol:
        st.markdown("<div class='prov-card'><div class='prov-title'>Cypher Query</div>", unsafe_allow_html=True)
        if "cypher" not in st.session_state:
            st.session_state["cypher"] = "MATCH p = ()-[]-()\nRETURN p\nLIMIT 500"

        cypher = st.text_area(
            label="",
            height=130,
            key="cypher",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with btncol:
        st.write("")
        st.write("")
        if "run_mode" not in st.session_state:
            st.session_state["run_mode"] = "Graph"

        run_mode = st.selectbox(
            "Mode",
            ["Graph", "Table"],
            key="run_mode",
        )
        run_btn = st.button("Run on both", use_container_width=True)

    if run_btn:
        body = query_router.ModeBody(
            cypher=cypher,
            params={},
            futureDb=st.session_state.get("future_db"),
            mode="graph" if run_mode == "Graph" else "table",
            asOfTimestamp=None,
        )
        out, err = _safe_call(query_router.query_both2, body)
        if err:
            st.error(err)
        else:
            st.session_state["last_query_result"] = out

    res = st.session_state.get("last_query_result")

    st.write("")
    if res:
        if run_mode == "Graph" and st.session_state.get("future_db"):
            fut_meta = _refresh_predicted_timesteps(st.session_state.get("future_db"))
            if int(fut_meta.get("maxPredicted", 0) or 0) > 0:
                st.markdown(
                    _render_prediction_gradient_bar(fut_meta["maxPredicted"]),
                    unsafe_allow_html=True,
                )

        left, right = st.columns(2)

        with left:
            st.markdown(
                "<div class='prov-card'><div class='prov-title'>Database: Present</div><div class='muted'>"
                + str(res.get("databases", {}).get("present", base_db))
                + "</div></div>",
                unsafe_allow_html=True,
            )
            st.write("")
            if run_mode == "Graph":
                g = res.get("present") or {"nodes": [], "edges": []}
                st.plotly_chart(_graph_to_plotly(g, "Present graph", future=False), use_container_width=True)
            else:
                rows = res.get("present") or []
                st.dataframe(_to_df(rows), use_container_width=True, height=520)

        with right:
            st.markdown(
                "<div class='prov-card'><div class='prov-title'>Database: Future</div><div class='muted'>"
                + str(res.get("databases", {}).get("future", st.session_state.get("future_db")))
                + "</div></div>",
                unsafe_allow_html=True,
            )
            st.write("")
            if run_mode == "Graph":
                g = res.get("future") or {"nodes": [], "edges": []}
                st.plotly_chart(_graph_to_plotly(g, "Future graph", future=True), use_container_width=True)
            else:
                rows = res.get("future") or []
                st.dataframe(_to_df(rows), use_container_width=True, height=520)
    else:
        st.info("Run a query to see results.")


# ---------------------------
# ADMINISTRATOR VIEW
# ---------------------------

else:
    if "admin_content_mode" not in st.session_state:
        st.session_state["admin_content_mode"] = "Heatmap"

    st.markdown("<div class='prov-title' style='font-size:1.2rem;'>Administrator</div>", unsafe_allow_html=True)
    mode_c1, mode_c2, _ = st.columns([1, 1, 6])
    with mode_c1:
        if st.button(
            "Heatmap",
            use_container_width=True,
            key="admin_content_heatmap_btn",
            type="primary" if st.session_state.get("admin_content_mode") == "Heatmap" else "secondary",
        ):
            st.session_state["admin_content_mode"] = "Heatmap"
    with mode_c2:
        if st.button(
            "Plots",
            use_container_width=True,
            key="admin_content_plots_btn",
            type="primary" if st.session_state.get("admin_content_mode") == "Plots" else "secondary",
        ):
            st.session_state["admin_content_mode"] = "Plots"

    st.write("")
    admin_content_mode = st.session_state.get("admin_content_mode", "Heatmap")

    if admin_content_mode == "Plots":
        if st.session_state.get("plots_loaded_for_db") != base_db:
            _refresh_plot_payload(base_db)
            exists_by_type_init = st.session_state.get("plots_exists_by_type") or {}
            st.session_state["plots_selected"] = {
                k for k in ALL_EMB_KEYS if exists_by_type_init.get(k, False)
            }

        series_by_key: Dict[str, pd.DataFrame] = st.session_state.get("plots_series_by_key") or {}

        left, right = st.columns([1.15, 3.0])

        with left:
            st.markdown("<div class='prov-card'><div class='prov-title'>Plot Components</div>", unsafe_allow_html=True)

            show_prob_plot = st.checkbox(
                "Probability Distribution Plot",
                value=st.session_state.get("plots_show_prob_plot", True),
                key="plots_show_prob_plot",
            )

            show_summary = st.checkbox(
                "Summary table",
                value=st.session_state.get("plots_show_summary", True),
                key="plots_show_summary",
            )

            st.write("")
            st.markdown("<div class='prov-title' style='font-size:1.0rem;'>Embeddings</div>", unsafe_allow_html=True)

            exists_by_type = st.session_state.get("plots_exists_by_type") or {}
            selected_now: set[str] = set(st.session_state.get("plots_selected") or set())
            next_selected: set[str] = set()

            for k in ALL_EMB_KEYS:
                has_pred = exists_by_type.get(k, False)
                checked = st.checkbox(
                    k,
                    value=(k in selected_now) and has_pred,
                    disabled=not has_pred,
                    key=f"plots_sel_{k}",
                )
                if checked and has_pred:
                    next_selected.add(k)

            st.session_state["plots_selected"] = next_selected
            st.session_state["plots_active_prediction_range"] = None
            st.write("")
            if st.button("Refresh components", use_container_width=True, key="plots_refresh"):
                _refresh_plot_payload(base_db)
                _toast_ok("Prediction comparison refreshed.")

            plot_errs = st.session_state.get("plots_errs") or {}
            if plot_errs:
                for fam, err in plot_errs.items():
                    st.warning(f"{fam}: {err}")

            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            visible_keys = [k for k in ALL_EMB_KEYS if k in (st.session_state.get("plots_selected") or set())]
            filtered_series = {k: series_by_key.get(k, pd.DataFrame()) for k in visible_keys}
            if not visible_keys:
                st.info("Select at least one embedding family that has a predicted DB.")
            else:
                if show_summary:
                    summary_df = _build_probability_summary_df(filtered_series, visible_keys)
                    if not summary_df.empty:
                        st.markdown(
                            "<div class='prov-card'><div class='prov-title'>Selected Embeddings Summary</div></div>",
                            unsafe_allow_html=True,
                        )
                        st.write("")
                        st.dataframe(summary_df, use_container_width=True, hide_index=True)
                        st.write("")

                if show_prob_plot:
                    for fam in visible_keys:
                        df_one = filtered_series.get(fam)
                        if df_one is None or df_one.empty:
                            continue

                        st.markdown(
                            f"<div class='prov-card'><div class='prov-title'>{fam} Probability Distribution by Timestamp</div></div>",
                            unsafe_allow_html=True,
                        )
                        st.write("")

                        fig = _build_probability_vs_timestamp_figure_for_family(df_one, fam)
                        st.plotly_chart(fig, use_container_width=True)

                        st.write("")

    else:
        if "eval_loaded_for_db" not in st.session_state:
            st.session_state["eval_loaded_for_db"] = None

        if st.session_state.get("eval_loaded_for_db") != base_db:
            _refresh_eval_payload(base_db)

        left, right = st.columns([1.15, 3.0])

        with left:
            st.markdown("<div class='prov-title' style='font-size:1.0rem;'>Evaluation</div>", unsafe_allow_html=True)

            show_metrics = st.checkbox(
                "Performance Metrics",
                value=st.session_state.get("eval_show_metrics", True),
                key="eval_show_metrics",
            )
            show_heatmap = st.checkbox(
                "Prediction Comparison",
                value=st.session_state.get("eval_show_heatmap", True),
                key="eval_show_heatmap",
            )

            st.write("")
            st.markdown("<div class='prov-title' style='font-size:1.0rem;'>Prediction Comparison</div>", unsafe_allow_html=True)

            cand = _db_candidates_for_base(base_db)
            exists_by_type = {c.get("type"): bool(c.get("exists")) for c in (cand or [])}

            if "eval_selected" not in st.session_state:
                st.session_state["eval_selected"] = {k for k in ALL_EMB_KEYS if exists_by_type.get(k, False)}

            selected: set[str] = set(st.session_state.get("eval_selected") or set())
            next_selected: set[str] = set()

            for k in ALL_EMB_KEYS:
                has_pred = exists_by_type.get(k, False)
                v = st.checkbox(
                    k,
                    value=(k in selected) and has_pred,
                    disabled=not has_pred,
                    key=f"eval_sel_{k}",
                )
                if v and has_pred:
                    next_selected.add(k)

            st.session_state["eval_selected"] = next_selected

            st.write("")
            show_gt = st.checkbox("Ground Truth", value=st.session_state.get("eval_show_gt", True), key="eval_show_gt")

            st.write("")
            if st.button("Refresh components", use_container_width=True, key="eval_refresh"):
                _refresh_eval_payload(base_db)
                _toast_ok("Evaluation refreshed.")

            if st.session_state.get("eval_pred_err"):
                st.warning(f"Predicted edges: {st.session_state['eval_pred_err']}")
            if st.session_state.get("eval_met_err"):
                st.warning(f"Metrics: {st.session_state['eval_met_err']}")
            if st.session_state.get("eval_gt_err"):
                st.warning(f"Ground truth: {st.session_state['eval_gt_err']}")

        with right:
            pred_sets: Dict[str, set[str]] = st.session_state.get("eval_pred_sets") or {}
            metrics_all: Dict[str, Any] = st.session_state.get("eval_metrics") or {}
            gt_set: set[str] = st.session_state.get("eval_gt_set") or set()

            visible_keys = [k for k in ALL_EMB_KEYS if k in (st.session_state.get("eval_selected") or set())]

            if show_metrics and visible_keys:
                st.markdown(
                    "<div class='prov-card'><div class='prov-title'>Performance Metrics</div>",
                    unsafe_allow_html=True
                )
                st.write("")

                cols = st.columns(min(4, max(1, len(visible_keys))))
                for i, k in enumerate(visible_keys):
                    with cols[i % len(cols)]:
                        m = _pick_metrics_block(metrics_all.get(k) or {})
                        if not m:
                            st.info(f"No metrics cached for {k}. Run Train & Predict first.")
                            continue

                        st.markdown(
                            f"""
                            <div style='background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12);
                                        border-radius: 16px; padding: 12px; margin-bottom: 10px;'>
                            <div style='font-weight: 750; font-size: 1.02rem; margin-bottom: 6px;'>{k}</div>
                            <div class='muted' style='display:flex; justify-content:space-between;'><span>Accuracy</span><span style='font-weight:700;'>{_fmt_metric(m.get('accuracy'))}</span></div>
                            <div class='muted' style='display:flex; justify-content:space-between;'><span>Precision</span><span style='font-weight:700;'>{_fmt_metric(m.get('precision'))}</span></div>
                            <div class='muted' style='display:flex; justify-content:space-between;'><span>Recall</span><span style='font-weight:700;'>{_fmt_metric(m.get('recall'))}</span></div>
                            <div class='muted' style='display:flex; justify-content:space-between;'><span>F1</span><span style='font-weight:700;'>{_fmt_metric(m.get('f1'))}</span></div>
                            <div class='muted' style='display:flex; justify-content:space-between;'><span>AUC</span><span style='font-weight:700;'>{_fmt_metric(m.get('auc'))}</span></div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                st.markdown("</div>", unsafe_allow_html=True)
                st.write("")

            if show_heatmap:
                keys = list(visible_keys)
                if show_gt and visible_keys:
                    keys = keys + ["Ground Truth"]

                if not visible_keys:
                    st.info("Select at least one embedding that has a predicted DB.")
                else:
                    def _set_for_key(k: str) -> set[str]:
                        if k == "Ground Truth":
                            return gt_set
                        return pred_sets.get(k) or set()

                    z: List[List[float]] = []
                    for row_k in keys:
                        row_set = _set_for_key(row_k)
                        row_vals: List[float] = []

                        for col_k in keys:
                            col_set = _set_for_key(col_k)

                            if row_k == "Ground Truth" and col_k == "Ground Truth":
                                row_vals.append(100.0 if gt_set else 0.0)
                            elif row_k == "Ground Truth":
                                row_vals.append(_compute_gt_coverage_pct(col_set, gt_set))
                            elif col_k == "Ground Truth":
                                row_vals.append(_compute_gt_coverage_pct(row_set, gt_set))
                            else:
                                row_vals.append(_compute_overlap_pct(row_set, col_set))

                        z.append(row_vals)

                    fig = go.Figure(
                        data=go.Heatmap(
                            z=z,
                            x=keys,
                            y=keys,
                            zmin=0,
                            zmax=100,
                            colorscale=[[0.0, "rgb(255,255,255)"], [1.0, "rgb(255,0,0)"]],
                            text=[[f"{v:.2f}" for v in row] for row in z],
                            texttemplate="%{text}",
                            hovertemplate="%{y} vs %{x}<br>%{z:.2f}%<extra></extra>",
                        )
                    )
                    fig.update_layout(
                        title="Prediction Comparison",
                        height=520,
                        margin=dict(l=30, r=20, t=45, b=30),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig, use_container_width=True)
