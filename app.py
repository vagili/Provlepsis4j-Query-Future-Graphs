from pathlib import Path
import asyncio
import io
import math
import json
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
    tgn_direct as tgn_direct_router,
)

from provlepsis_core.storage_repr import (
    SEPARATE_DBS,
    SINGLE_DB_MULTI_EDGE,
    normalize_storage_representation,
    is_single_db_multi_edge,
)

from provlepsis_core.lp_model_state import (
    normalize_prediction_model,
    prediction_model_display_name,
)

from provlepsis_core.prediction_timeline import (
    EMBEDDING_FAMILIES,
    predicted_db_name,
)

from provlepsis_core.admin_extension_config import (
    load_admin_extension_config,
    save_admin_extension_config,
)

# ---------------------------
# Streamlit setup + styling
# ---------------------------

LOGO_PATH = Path(__file__).parent / "Provlepsis4j.png"
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

        .metric-cards-row {
            display: flex;
            flex-wrap: nowrap;
            justify-content: flex-start;
            align-items: stretch;
            gap: 12px;

            width: 100%;
            margin: 0;
            padding: 0;
            overflow-x: auto;
        }

        .metric-card {
            flex: 1 1 0;
            width: auto;
            min-width: 170px;
            box-sizing: border-box;

            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 14px;

            padding: 13px 16px;
            margin: 0;
        }

        .metric-card-title {
            color: #f8fafc;
            font-size: 1.16rem;
            font-weight: 800;
            line-height: 1.25;
            margin-bottom: 8px;
        }

        .metric-row {
            display: grid;
            grid-template-columns: 1fr max-content;
            column-gap: 12px;
            align-items: center;

            width: 100%;
            font-size: 1.08rem;
            line-height: 1.45;
            white-space: nowrap;
        }

        .metric-value {
            min-width: 48px;
            text-align: right;

            color: #f8fafc;
            font-size: 1.09rem;
            font-weight: 800;
        }

        .prov-field-label,
        div[data-testid="stSelectbox"] label,
        div[data-testid="stSelectbox"] label p {
            color: #f8fafc !important;
            font-family: "Source Sans Pro", Arial, sans-serif !important;
            font-size: 15px !important;
            font-weight: 400 !important;
            line-height: 1.25 !important;
            letter-spacing: normal !important;
        }

        .prov-field-label {
            margin: 0 0 0.35rem 0 !important;
        }

        section.main div[data-testid="stCheckbox"] label,
        section.main div[data-testid="stCheckbox"] label p {
            font-size: 1.2rem !important;
            font-weight: 600 !important;
            line-height: 1.35 !important;
        }

        section.main div[data-testid="stSelectbox"] label,
        section.main div[data-testid="stSelectbox"] label p,
        section.main .prov-field-label {
            font-size: 1.3rem !important;
            font-weight: 700 !important;
            line-height: 1.35 !important;
        }

        section.main div[data-baseweb="select"] > div,
        section.main div[data-baseweb="select"] span {
            font-size: 1.2rem !important;
            font-weight: 600 !important;
        }

        section.main .stButton > button,
        section.main .stButton > button p {
            font-size: 1.3rem !important;
            font-weight: 700 !important;
        }

        .control-panel-title {
            color: #f8fafc !important;
            font-size: 1.22rem !important;
            font-weight: 800 !important;
            line-height: 1.30 !important;
            margin-bottom: 0.30rem;
        }

        div[data-testid="stSelectbox"] label,
        div[data-testid="stSelectbox"] label p {
            margin: 0 !important;
            padding: 0 !important;
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

          .prov-sidebar-logo {
            display: flex;
            justify-content: center;
            align-items: center;
            width: 100%;
            margin: 0.35rem 0 0.85rem 0;
        }

        .prov-sidebar-logo img {
            display: block;
            width: 130px;
            height: auto;
            transform: translateX(-4px);
        }

          .prov-gradient-bar {
            flex: 1;
            height: 14px;
            border-radius: 999px;
            background: linear-gradient(90deg, #22c55e 0%, #facc15 50%, #ef4444 100%);
            border: 1px solid rgba(255,255,255,0.16);
            box-shadow: inset 0 1px 2px rgba(0,0,0,0.18);
          }

          div[data-testid="stMetricValue"] > div {
            font-size: 2.8rem !important;
            font-weight: 800 !important;
            color: #f8fafc !important;
            letter-spacing: -0.5px;
          }

          div[data-testid="stMetricLabel"] > div {
            font-size: 0.95rem !important;
            color: rgba(255, 255, 255, 0.75) !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
          }

        section.main div[data-testid="stTextArea"] textarea,
        section.main div[data-baseweb="textarea"] textarea {
            font-family: "Source Sans Pro", Arial, sans-serif !important;
            font-size: 1.4rem !important;
            line-height: 1.50 !important;
            font-weight: 500 !important;
        }

          footer { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <style>
        [data-testid="stDialog"] h2 {
            display: none !important;
        }

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

STORAGE_REPRESENTATION_OPTIONS = {
    "Separate databases": SEPARATE_DBS,
    "Single database": SINGLE_DB_MULTI_EDGE,
}

PREDICTION_MODEL_OPTIONS = {
    "Logistic regression": "logistic_regression",
    "LASSO regression": "lasso_regression",
    "TGN": "tgn_direct",
}

TGN_DIRECT_DEFAULT_PARAMS: Dict[str, Any] = {
    "nEpoch": 5,
    "batchSize": 200,
    "numNeighbors": 10,
    "nodeDimension": 100,
    "memoryDimension": 100,
    "messageDimension": 100,
    "learningRate": 0.0001,
    "nHeads": 2,
    "nLayers": 1,
    "dropout": 0.10,
    "useMemory": True,
}

if "storage_representation" not in st.session_state:
    st.session_state["storage_representation"] = SEPARATE_DBS

if "predicted_edge_inclusion" not in st.session_state:
    st.session_state["predicted_edge_inclusion"] = True

def _admin_storage_representation_changed() -> None:
    label = st.session_state.get(
        "admin_storage_representation_label",
        "Separate databases",
    )

    st.session_state["storage_representation"] = (
        STORAGE_REPRESENTATION_OPTIONS[label]
    )

def _selected_storage_representation() -> str:
    return normalize_storage_representation(
        st.session_state.get("storage_representation", SEPARATE_DBS)
    )


def _prediction_model_label_to_value(label: str) -> str:
    return PREDICTION_MODEL_OPTIONS.get(label, "logistic_regression")


def _prediction_model_value_to_label(value: str) -> str:
    value = normalize_prediction_model(value)

    for label, raw in PREDICTION_MODEL_OPTIONS.items():
        if normalize_prediction_model(raw) == value:
            return label

    return "Logistic regression"

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


def _split_state_path(base_db: str) -> Path:
    safe_db = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_"
        for ch in str(base_db or "default")
    )
    return Path("provlepsis_runtime") / safe_db / "split_state.json"


def _get_session_split_state(
    base_db: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    target_db = str(base_db or "")
    if not target_db:
        try:
            target_db = db.current_database()
        except Exception:
            return None

    state = st.session_state.get("_session_split_state")
    if isinstance(state, dict) and str(state.get("base_db")) == target_db:
        return state

    path = _split_state_path(target_db)
    if not path.exists():
        return None

    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if not isinstance(loaded, dict) or str(loaded.get("base_db")) != target_db:
        return None

    st.session_state["_session_split_state"] = loaded
    return loaded


def _reset_session_split_state(
    *,
    base_db: Optional[str] = None,
    delete_persisted: bool = False,
):
    st.session_state.pop("_session_split_state", None)
    if not delete_persisted:
        return

    target_db = str(base_db or "")
    if not target_db:
        try:
            target_db = db.current_database()
        except Exception:
            return

    path = _split_state_path(target_db)
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def _get_session_split_state_for_db(base_db: str) -> Optional[Dict[str, Any]]:
    state = _get_session_split_state(base_db)
    if not state:
        return None
    if str(state.get("base_db")) != str(base_db):
        return None
    if not bool(state.get("done")):
        return None
    return state


def _remember_session_split_state(
    *,
    base_db: str,
    split_seconds: Optional[float],
    temporal_mode: bool,
    test_holdout: float,
    val_holdout: float,
) -> Dict[str, Any]:
    state = _split_signature(
        base_db=base_db,
        temporal_mode=temporal_mode,
        test_holdout=test_holdout,
        val_holdout=val_holdout,
    )
    state["done"] = True
    state["split_seconds"] = split_seconds
    st.session_state["_session_split_state"] = state
    path = _split_state_path(base_db)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(state, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(path)
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

PREDICTION_MODEL_ORDER = [
    "logistic_regression",
    "lasso_regression",
    "tgn_direct",
]

PREDICTION_MODEL_VIOLIN_FILL = {
    "logistic_regression": "rgba(91, 143, 220, 0.82)",
    "lasso_regression": "rgba(223, 135, 76, 0.82)",
    "tgn_direct": "rgba(141, 104, 220, 0.82)",
}

PREDICTION_MODEL_VIOLIN_LINE = {
    "logistic_regression": "#5b8fdc",
    "lasso_regression": "#df874c",
    "tgn_direct": "#8d68dc",
}


def _plot_prediction_model_key(value: Any) -> str:
    try:
        model = normalize_prediction_model(str(value or "logistic_regression"))
    except Exception:
        model = "logistic_regression"

    if model not in set(PREDICTION_MODEL_ORDER):
        model = "logistic_regression"

    return model


def _plot_prediction_model_label(model: str) -> str:
    model = _plot_prediction_model_key(model)

    if model == "logistic_regression":
        return "Logistic regression"
    if model == "lasso_regression":
        return "LASSO regression"
    if model == "tgn_direct":
        return "TGN"

    return prediction_model_display_name(model)

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

def _compute_jaccard_pct(a: set[str], b: set[str]) -> float:

    if not a and not b:
        return 100.0

    if not a or not b:
        return 0.0

    intersection_size = len(a & b)
    union_size = len(a | b)

    if union_size <= 0:
        return 0.0

    return (intersection_size / union_size) * 100.0

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


    layout_graph = nx.Graph()
    node_labels: Dict[str, str] = {}

    for n in nodes:
        nid = str(n.get("id"))
        if not nid:
            continue

        label = n.get("nodeId")
        if label is None:
            label = n.get("id")

        layout_graph.add_node(nid)
        node_labels[nid] = str(label)


    normalized_edges: List[Dict[str, Any]] = []

    for idx, e in enumerate(edges):
        s = str(e.get("source"))
        t = str(e.get("target"))

        if not s or not t or s == "None" or t == "None":
            continue

        if s not in layout_graph:
            layout_graph.add_node(s)
            node_labels.setdefault(s, s)

        if t not in layout_graph:
            layout_graph.add_node(t)
            node_labels.setdefault(t, t)

        layout_graph.add_edge(s, t)

        try:
            predicted_lvl = int(e.get("predicted_timestep") or e.get("predicted") or 0)
        except Exception:
            predicted_lvl = 0

        is_predicted = predicted_lvl > 0 or str(e.get("edgeOrigin") or "").lower() == "predicted"

        try:
            probability = float(
                e.get(
                    "probability",
                    0.0 if is_predicted else 1.0,
                )
            )
        except Exception:
            probability = 0.0 if is_predicted else 1.0

        try:
            timestamp = int(e.get("timestamp") or e.get("lastTimestamp") or e.get("firstTimestamp") or 0)
        except Exception:
            timestamp = 0

        normalized_edges.append(
            {
                "id": str(e.get("id") or f"{s}->{t}:{idx}"),
                "source": s,
                "target": t,
                "type": str(e.get("type") or ""),
                "predicted": predicted_lvl,
                "isPredicted": is_predicted,
                "probability": probability,
                "timestamp": timestamp,
                "embeddingFamily": str(e.get("embeddingFamily") or ""),
            }
        )

    fig = go.Figure()

    if len(layout_graph.nodes) == 0:
        fig.update_layout(
            title={"text": title, "font": {"size": 14, "color": "#94a3b8"}},
            height=520,
            margin=dict(l=10, r=10, t=35, b=10),
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font=dict(color="#cbd5e1"),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )

        fig.add_annotation(
            text="No matching results in this graph state",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=16, color="#94a3b8"),
        )

        return fig

    pos = nx.spring_layout(
        layout_graph,
        seed=7,
        k=0.9 / math.sqrt(max(len(layout_graph.nodes), 1)),
    )

    max_predicted_level = max(
        [
            max(1, int(e.get("predicted", 0) or 0))
            for e in normalized_edges
            if bool(e.get("isPredicted", False))
        ] or [1]
    )


    pair_to_edges: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    for e in normalized_edges:
        s = e["source"]
        t = e["target"]
        key = (s, t) if s <= t else (t, s)
        pair_to_edges.setdefault(key, []).append(e)

    for pair_key, pair_edges in pair_to_edges.items():
        total_parallel = len(pair_edges)

        for parallel_idx, data in enumerate(pair_edges):
            u = data["source"]
            v = data["target"]

            x0, y0 = pos[u]
            x1, y1 = pos[v]

            dx = x1 - x0
            dy = y1 - y0
            length = math.sqrt(dx * dx + dy * dy) or 1.0


            nxp = -dy / length
            nyp = dx / length


            offset = (parallel_idx - (total_parallel - 1) / 2.0) * 0.06

            mx = (x0 + x1) / 2.0 + nxp * offset
            my = (y0 + y1) / 2.0 + nyp * offset

            predicted_lvl = int(data.get("predicted", 0) or 0)
            probability = float(data.get("probability", 0.0) or 0.0)

            is_pred = future and bool(data.get("isPredicted", False))

            display_timestep = max(1, predicted_lvl) if is_pred else 0

            if is_pred:
                color = _predicted_level_color(display_timestep, max_predicted_level)
                width = 3.0
            else:
                color = "#94a3b8"
                width = 1.7

            hovertemplate = (
                "<b>probability:</b> %{customdata[0]:.4f}<br>"
                "<b>timestep:</b> %{customdata[1]:.0f}"
                "<extra></extra>"
            )


            curve_steps = 24
            xs = []
            ys = []

            for step in range(curve_steps + 1):
                t_curve = step / curve_steps

                bx = (
                    ((1 - t_curve) ** 2) * x0
                    + 2 * (1 - t_curve) * t_curve * mx
                    + (t_curve ** 2) * x1
                )
                by = (
                    ((1 - t_curve) ** 2) * y0
                    + 2 * (1 - t_curve) * t_curve * my
                    + (t_curve ** 2) * y1
                )

                xs.append(bx)
                ys.append(by)

            custom = [[probability, display_timestep]] * len(xs)

            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(width=width, color=color),
                    customdata=custom,
                    hovertemplate=hovertemplate,
                    showlegend=False,
                )
            )

    node_x, node_y, node_text = [], [], []

    for nid in layout_graph.nodes():
        x, y = pos[nid]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node_labels.get(nid, nid))

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
        hoverlabel=dict(
            bgcolor="#0f172a",
            bordercolor="rgba(255,255,255,0.40)",
            align="left",
            font=dict(
                size=22,
                color="#f8fafc",
                family="Source Sans Pro, Arial, sans-serif",
            ),
        ),
    )

    return fig

def _db_candidates_for_base(base: str) -> List[Dict[str, Any]]:
    storage_mode = _selected_storage_representation()

    if is_single_db_multi_edge(storage_mode):
        out, err = _safe_call(lp_router.lp_predicted_dbs_single_db_for, base)
    else:
        out, err = _safe_call(lp_router.lp_predicted_dbs_for, base)

    if err or not out:
        return []

    return out.get("candidates") or []


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
    cols = [
        "family",
        "s",
        "t",
        "probability",
        "timestamp",
        "predicted_timestep",
        "predictionModel",
        "predictionModelDisplayName",
    ]

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

    if "predictionModel" not in df.columns:
        df["predictionModel"] = "logistic_regression"

    df["probability"] = pd.to_numeric(df["probability"], errors="coerce").fillna(0.0)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0).astype(int)
    df["predicted_timestep"] = pd.to_numeric(df["predicted_timestep"], errors="coerce").fillna(0).astype(int)
    df["family"] = family

    df["predictionModel"] = df["predictionModel"].apply(_plot_prediction_model_key)
    df["predictionModelDisplayName"] = df["predictionModel"].apply(_plot_prediction_model_label)

    return df[cols].sort_values(
        ["timestamp", "predictionModel", "probability"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def _refresh_plot_payload(
    base: str,
    prediction_model: Optional[str] = None,
):
    storage_mode = _selected_storage_representation()
    selected_model = normalize_prediction_model(
        prediction_model
        or st.session_state.get(
            "plots_prediction_model",
            "logistic_regression",
        )
    )
    candidates = _db_candidates_for_base(base)
    series_by_key: Dict[str, pd.DataFrame] = {}
    errors: Dict[str, str] = {}
    exists_by_type: Dict[str, bool] = {}
    families = ["TGN"] if selected_model == "tgn_direct" else list(EMBEDDING_FAMILIES)

    if is_single_db_multi_edge(storage_mode):
        for family in families:
            has_predictions = any(
                bool(candidate.get("exists"))
                and (candidate.get("family") or candidate.get("type")) == family
                and normalize_prediction_model(
                    candidate.get("predictionModel", "logistic_regression")
                ) == selected_model
                for candidate in candidates or []
            )
            exists_by_type[family] = has_predictions
            if not has_predictions:
                continue

            raw, error = _safe_call(
                lp_router.lp_predicted_series_single_db,
                base,
                family,
                selected_model,
            )
            if error:
                errors[family] = error
                series_by_key[family] = _series_payload_to_df([], family)
                continue
            points = (raw or {}).get("points") or []
            series_by_key[family] = _series_payload_to_df(points, family)
    else:
        for candidate in candidates or []:
            family = candidate.get("family") or candidate.get("type")
            if not family or family not in families:
                continue
            candidate_model = normalize_prediction_model(
                candidate.get("predictionModel", "logistic_regression")
            )
            if candidate_model != selected_model:
                continue
            exists = bool(candidate.get("exists"))
            exists_by_type[family] = exists
            if not exists:
                continue
            raw, error = _safe_call(
                lp_router.lp_predicted_series,
                candidate.get("name"),
            )
            if error:
                errors[family] = error
                series_by_key[family] = _series_payload_to_df([], family)
                continue
            points = (raw or {}).get("points") or []
            series_by_key[family] = _series_payload_to_df(points, family)

    for family in families:
        exists_by_type.setdefault(family, False)

    st.session_state["plots_series_by_key"] = series_by_key
    st.session_state["plots_errs"] = errors
    st.session_state["plots_exists_by_type"] = exists_by_type
    st.session_state["plots_loaded_for_db"] = (
        f"{base}|{storage_mode}|{selected_model}"
    )



def _build_probability_vs_timestamp_figure_for_family(
    df: pd.DataFrame,
    family: str,
) -> go.Figure:
    fig = go.Figure()
    base_layout = dict(
        height=460,
        margin=dict(l=25, r=25, t=35, b=25),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc"),
        yaxis=dict(
            title=dict(
                text="Probability",
                font=dict(
                    size=20,
                    color="#f8fafc",
                    family="Source Sans Pro, Arial, sans-serif",
                ),
            ),
            tickfont=dict(
                size=16,
                color="#f8fafc",
                family="Source Sans Pro, Arial, sans-serif",
            ),
            range=[0.0, 1.0],
            showgrid=True,
            gridcolor="rgba(255,255,255,0.10)",
            zeroline=False,
        ),
        xaxis=dict(
            title=dict(
                text="Timestamp",
                font=dict(
                    size=20,
                    color="#f8fafc",
                    family="Source Sans Pro, Arial, sans-serif",
                ),
            ),
            tickfont=dict(
                size=17,
                color="#f8fafc",
                family="Source Sans Pro, Arial, sans-serif",
            ),
            showgrid=False,
        ),
        showlegend=False,
        hoverlabel=dict(bgcolor="#0f172a", font_color="#f8fafc"),
    )

    if df is None or df.empty:
        fig.update_layout(**base_layout)
        return fig

    df2 = df.copy()
    df2["timestamp"] = pd.to_numeric(
        df2["timestamp"],
        errors="coerce",
    ).fillna(0).astype(int)
    df2["probability"] = pd.to_numeric(
        df2["probability"],
        errors="coerce",
    ).fillna(0.0)
    unique_timestamps = sorted(df2["timestamp"].dropna().unique().tolist())
    timestamp_labels = [str(timestamp) for timestamp in unique_timestamps]
    count = len(unique_timestamps)
    pad_left = [f"__pad_l{i}__" for i in range(2, 0, -1)]
    pad_right = [f"__pad_r{i}__" for i in range(1, 3)]
    category_array = pad_left + timestamp_labels + pad_right

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

    if count == 1:
        violin_width = 0.42
        figure_height = 450
    elif count == 2:
        violin_width = 0.34
        figure_height = 470
    elif count == 3:
        violin_width = 0.42
        figure_height = 520
    else:
        violin_width = 0.55
        figure_height = 540

    for timestamp in unique_timestamps:
        timestamp_df = df2[df2["timestamp"] == timestamp]
        if timestamp_df.empty:
            continue
        fig.add_trace(
            go.Violin(
                x=[str(timestamp)] * len(timestamp_df),
                y=timestamp_df["probability"],
                name=family,
                showlegend=False,
                points=False,
                fillcolor=VIOLIN_FILL.get(
                    family,
                    "rgba(148,163,184,0.82)",
                ),
                line=dict(
                    color=VIOLIN_LINE.get(family, "#64748b"),
                    width=1.8,
                ),
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
                    "timestamp: %{x}<br>"
                    "probability: %{y:.4f}"
                    "<extra></extra>"
                ),
            )
        )

    base_layout["height"] = figure_height
    base_layout["xaxis"] = dict(
        base_layout["xaxis"],
        type="category",
        categoryorder="array",
        categoryarray=category_array,
        tickmode="array",
        tickvals=timestamp_labels,
        ticktext=timestamp_labels,
    )
    fig.update_layout(**base_layout, violinmode="group")
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
    prediction_model = st.session_state.get(
        "heatmap_prediction_model",
        st.session_state.get("admin_prediction_model", "logistic_regression"),
    )
    storage_mode = _selected_storage_representation()

    pred_raw, err1 = _safe_call(
        lp_router.lp_evaluation_predicted_edges,
        predictionModel=prediction_model,
        storageRepresentation=storage_mode,
    )
    met_raw, err2 = _safe_call(
        lp_router.lp_metrics,
        predictionModel=prediction_model,
        storageRepresentation=storage_mode,
    )
    gt_raw, err3 = _safe_call(
        lp_router.lp_test_edges,
        predictionModel=prediction_model,
        storageRepresentation=storage_mode,
    )

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

    st.session_state["eval_loaded_for_db"] = f"{base}|{storage_mode}|{prediction_model}"
    available_embeddings = {
        k
        for k in ALL_EMB_KEYS
        if st.session_state.get("eval_pred_sets", {}).get(k)
        or st.session_state.get("eval_metrics", {}).get(k)
    }

    if available_embeddings:
        current_selected = st.session_state.get("eval_selected")

        if not current_selected:
            st.session_state["eval_selected"] = set(available_embeddings)

            for family in available_embeddings:
                st.session_state[f"eval_sel_{family}"] = True

# ---------------------------
# Admin workflow helpers
# ---------------------------

def _gds_projection_exists(name: str) -> bool:
    if not name:
        return False

    try:
        rows = db.run_data(
            "CALL gds.graph.exists($name) YIELD exists RETURN exists",
            {"name": name},
        )
        return bool(rows and rows[0].get("exists"))
    except Exception:
        return False


def _prepare_hidden_fullgraph_projection() -> Tuple[Optional[Any], Optional[str]]:
    return {
        "ok": True,
        "skipped": True,
        "reason": " ",
    }, None


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


def _run_lp_for_property(
    embedding_property: str,
    predict_k: int,
    prob_threshold: float,
    embedding_family: str,
    prediction_model: str,
) -> Tuple[Optional[Any], Optional[str]]:
    body = lp_router.LPBody(
        embeddingProperty=embedding_property,
        embeddingFamily=embedding_family,
        predictionModel=prediction_model,
        storageRepresentation=_selected_storage_representation(),
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

    storage_label = st.selectbox(
        "Data storage representation",
        list(STORAGE_REPRESENTATION_OPTIONS.keys()),
        index=0,
        key="dialog_storage_representation_label",
    )

    storage_representation = STORAGE_REPRESENTATION_OPTIONS[storage_label]

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

        st.session_state["storage_representation"] = storage_representation

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
                storageRepresentation=storage_representation,
            )

        try:
            out = asyncio.run(_run())
            st.session_state["admin_load_response"] = out
            _reset_session_split_state(
                base_db=out.get("database"),
                delete_persisted=True,
            )
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
    sidebar_logo_b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode()

    st.markdown(
        f"""
        <div class="prov-sidebar-logo">
            <img
                src="data:image/png;base64,{sidebar_logo_b64}"
                alt="Provlepsis4j"
            />
        </div>
        """,
        unsafe_allow_html=True,
    )

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

        current_admin_db = db.current_database()
        saved_admin_config = load_admin_extension_config(current_admin_db)

        saved_admin_model_label = _prediction_model_value_to_label(
            saved_admin_config["predictionModel"]
        )

        admin_restore_keys = [
            "admin_storage_representation_label",
            "admin_prediction_model_label",
            "admin_predicted_edge_inclusion",
            "admin_predict_k",
            "admin_prob_threshold",
            "admin_additional_timesteps",
        ]

        if saved_admin_model_label != "TGN":
            admin_restore_keys.append("admin_lp_family")

        admin_needs_restore = (
            st.session_state.get("_admin_config_db") != current_admin_db
            or any(
                key not in st.session_state
                for key in admin_restore_keys
            )
        )

        if admin_needs_restore:
            saved_storage = normalize_storage_representation(
                saved_admin_config["storageRepresentation"]
            )
            saved_storage_label = next(
                label
                for label, value in STORAGE_REPRESENTATION_OPTIONS.items()
                if normalize_storage_representation(value) == saved_storage
            )
            st.session_state["_admin_config_db"] = current_admin_db
            st.session_state["storage_representation"] = saved_storage
            st.session_state["admin_storage_representation_label"] = saved_storage_label
            st.session_state["admin_prediction_model_label"] = saved_admin_model_label
            st.session_state["admin_lp_family"] = saved_admin_config["embeddingFamily"]
            st.session_state["admin_predicted_edge_inclusion"] = bool(
                saved_admin_config["predictedEdgeInclusion"]
            )
            st.session_state["predicted_edge_inclusion"] = bool(
                saved_admin_config["predictedEdgeInclusion"]
            )
            st.session_state["admin_predict_k"] = int(saved_admin_config["predictK"])
            st.session_state["admin_prob_threshold"] = float(
                saved_admin_config["probThreshold"]
            )
            st.session_state["admin_additional_timesteps"] = int(
                saved_admin_config["additionalTimesteps"]
            )

        st.markdown("<div class='prov-title'>Data Storage Representation</div>", unsafe_allow_html=True)

        storage_labels = list(STORAGE_REPRESENTATION_OPTIONS.keys())
        current_storage = _selected_storage_representation()

        current_storage_label = next(
            (
                label
                for label, value in STORAGE_REPRESENTATION_OPTIONS.items()
                if normalize_storage_representation(value) == current_storage
            ),
            storage_labels[0],
        )

        if "admin_storage_representation_label" not in st.session_state:
            st.session_state["admin_storage_representation_label"] = current_storage_label

        st.selectbox(
            "Representation",
            storage_labels,
            key="admin_storage_representation_label",
            on_change=_admin_storage_representation_changed,
        )

        st.divider()

        st.markdown("<div class='prov-title'>Dataset Loading</div>", unsafe_allow_html=True)
        if st.button("Load Dataset", use_container_width=True, key="open_load_dataset_dialog_btn"):
            _load_dataset_dialog()

        st.divider()

        st.markdown("<div class='prov-title'>Link Prediction</div>", unsafe_allow_html=True)

        admin_prediction_model_label = st.selectbox(
            "Prediction model",
            list(PREDICTION_MODEL_OPTIONS.keys()),
            index=0,
            key="admin_prediction_model_label",
        )
        admin_prediction_model = PREDICTION_MODEL_OPTIONS[admin_prediction_model_label]
        admin_prediction_model = normalize_prediction_model(admin_prediction_model)
        st.session_state["admin_prediction_model"] = admin_prediction_model
        admin_uses_tgn_direct = admin_prediction_model == "tgn_direct"

        if admin_uses_tgn_direct:
            admin_temporal_split = True
            admin_family = "TGN"
        else:
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

        admin_predicted_edge_inclusion = st.checkbox(
            "Predicted edge inclusion",
            value=bool(st.session_state.get("predicted_edge_inclusion", True)),
            key="admin_predicted_edge_inclusion",
        )

        st.session_state["predicted_edge_inclusion"] = bool(admin_predicted_edge_inclusion)

        admin_retraining_interval = 1 if st.session_state["predicted_edge_inclusion"] else 0
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
            value=float(saved_admin_config["probThreshold"]),
            step=0.01,
            format="%.2f",
            key="admin_prob_threshold",
        )
        admin_additional_timesteps = st.number_input(
            "Additional predicted timesteps",
            min_value=1,
            value=int(saved_admin_config["additionalTimesteps"]),
            step=1,
            key="admin_additional_timesteps",
        )

        admin_advanced = st.checkbox("Advanced parameters", value=False, key="admin_advanced")

        emb_params: Dict[str, Any] = {}
        tgn_params: Dict[str, Any] = dict(TGN_DIRECT_DEFAULT_PARAMS)
        tgn_params.update(saved_admin_config.get("tgnDirectParams") or {})

        if admin_uses_tgn_direct and admin_advanced:
            tgn_params["nEpoch"] = st.number_input(
                "Epochs",
                min_value=1,
                max_value=200,
                value=int(tgn_params["nEpoch"]),
                step=1,
                key="adv_tgn_direct_epochs",
            )

            tgn_params["messageDimension"] = st.number_input(
                "Message dimension",
                min_value=2,
                max_value=2048,
                value=int(tgn_params["messageDimension"]),
                step=1,
                key="adv_tgn_direct_message_dim",
            )

            tgn_params["batchSize"] = st.number_input(
                "Batch size",
                min_value=1,
                max_value=100000,
                value=int(tgn_params["batchSize"]),
                step=1,
                key="adv_tgn_direct_batch",
            )

            tgn_params["learningRate"] = st.number_input(
                "Learning rate",
                min_value=0.000001,
                max_value=1.0,
                value=float(tgn_params["learningRate"]),
                step=0.00001,
                format="%.6f",
                key="adv_tgn_direct_lr",
            )

            tgn_params["numNeighbors"] = st.number_input(
                "Number of neighbors",
                min_value=1,
                max_value=1000,
                value=int(tgn_params["numNeighbors"]),
                step=1,
                key="adv_tgn_direct_neighbors",
            )

            tgn_params["nHeads"] = st.number_input(
                "Attention heads",
                min_value=1,
                max_value=16,
                value=int(tgn_params["nHeads"]),
                step=1,
                key="adv_tgn_direct_heads",
            )

            tgn_params["nodeDimension"] = st.number_input(
                "Node dimension",
                min_value=2,
                max_value=2048,
                value=int(tgn_params["nodeDimension"]),
                step=1,
                key="adv_tgn_direct_node_dim",
            )

            tgn_params["nLayers"] = st.number_input(
                "TGN layers",
                min_value=1,
                max_value=8,
                value=int(tgn_params["nLayers"]),
                step=1,
                key="adv_tgn_direct_layers",
            )

            tgn_params["memoryDimension"] = st.number_input(
                "Memory dimension",
                min_value=2,
                max_value=2048,
                value=int(tgn_params["memoryDimension"]),
                step=1,
                key="adv_tgn_direct_memory_dim",
            )

            tgn_params["dropout"] = st.number_input(
                "Dropout",
                min_value=0.0,
                max_value=0.90,
                value=float(tgn_params["dropout"]),
                step=0.01,
                format="%.2f",
                key="adv_tgn_direct_dropout",
            )

            tgn_params["useMemory"] = st.checkbox(
                "Use memory",
                value=bool(tgn_params["useMemory"]),
                key="adv_tgn_direct_use_memory",
            )

        st.session_state["tgn_direct_params"] = dict(tgn_params)

        if not admin_uses_tgn_direct and admin_family == "FastRP":
            emb_params = {
                "embeddingDimension": 128,
                "propertyRatio": 0.50,
            }
            if admin_advanced:
                emb_params["embeddingDimension"] = st.number_input("Embedding dimension", 2, 2048, 128, 1, key="adv_fastrp_dim")
                emb_params["propertyRatio"] = st.number_input(
                    "Property ratio",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.50,
                    step=0.05,
                    format="%.2f",
                    key="adv_fastrp_ratio",
                )

        elif not admin_uses_tgn_direct and admin_family == "Node2Vec":
            emb_params = {
                "embeddingDimension": 128,
                "walkLength": 80,
                "walksPerNode": 10,
                "returnFactor": 1.0,
                "inOutFactor": 1.0,
            }
            if admin_advanced:
                emb_params["embeddingDimension"] = st.number_input("Embedding dimension", 2, 2048, 128, 1, key="adv_n2v_dim")
                emb_params["walkLength"] = st.number_input("Walk length", 1, 200, 80, 1, key="adv_n2v_wl")
                emb_params["walksPerNode"] = st.number_input("Walks per node", 1, 200, 10, 1, key="adv_n2v_wpn")
                emb_params["returnFactor"] = st.number_input("Return factor", 0.01, 10.0, 1.0, 0.05, key="adv_n2v_rf")
                emb_params["inOutFactor"] = st.number_input("In-out factor", 0.01, 10.0, 1.0, 0.05, key="adv_n2v_iof")

        elif not admin_uses_tgn_direct and admin_family == "GraphSAGE":
            emb_params = {
                "embeddingDimension": 128,
                "epochs": 10,
                "learningRate": 0.01,
            }
            if admin_advanced:
                emb_params["embeddingDimension"] = st.number_input("Embedding dimension", 2, 2048, 128, 1, key="adv_sage_dim")
                emb_params["epochs"] = st.number_input("Epochs", 1, 200, 10, 1, key="adv_sage_epochs")
                emb_params["learningRate"] = st.number_input("Learning rate", 0.0001, 1.0, 0.01, 0.0005, key="adv_sage_lr")

        elif not admin_uses_tgn_direct and admin_family == "HashGNN":
            emb_params = {
                "outputDimension": 128,
                "iterations": 10,
                "embeddingDensity": 10,
            }
            if admin_advanced:
                emb_params["outputDimension"] = st.number_input("Output dimension", 2, 2048, 128, 1, key="adv_hash_dim")
                emb_params["iterations"] = st.number_input("Iterations", 1, 200, 10, 1, key="adv_hash_iters")
                emb_params["embeddingDensity"] = st.number_input("Embedding density", 1, 200, 10, 1, key="adv_hash_dens")

        elif not admin_uses_tgn_direct and admin_family == "TGN":
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
                emb_params["epochs"] = st.number_input("Epochs", 1, 500, 5, 1, key="adv_tgn_epochs")
                emb_params["batchSize"] = st.number_input("Batch size", 1, 100000, 200, 1, key="adv_tgn_batch")
                emb_params["learningRate"] = st.number_input(
                    "Learning rate",
                    min_value=0.00001,
                    max_value=1.0,
                    value=0.0001,
                    step=0.00001,
                    format="%.5f",
                    key="adv_tgn_lr",
                )
                emb_params["numNeighbors"] = st.number_input("Number of neighbors", 1, 1000, 10, 1, key="adv_tgn_neighbors")

        current_admin_db = db.current_database()
        active_admin_family = "TGN" if admin_uses_tgn_direct else admin_family
        active_embedding_property = EMB_WRITE_PROP.get(
            active_admin_family,
            active_admin_family,
        )
        saved_admin_config = save_admin_extension_config(
            current_admin_db,
            {
                "storageRepresentation": _selected_storage_representation(),
                "predictionModel": admin_prediction_model,
                "embeddingFamily": active_admin_family,
                "embeddingProperty": active_embedding_property,
                "predictK": int(admin_predict_k),
                "candidateMultiplier": 20,
                "probThreshold": float(admin_prob_threshold),
                "negativeRatio": 1.0,
                "predictedEdgeInclusion": bool(admin_predicted_edge_inclusion),
                "retrainingInterval": 1 if admin_predicted_edge_inclusion else 0,
                "additionalTimesteps": int(admin_additional_timesteps),
                "tgnDirectParams": dict(tgn_params),
            },
        )
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
            if admin_uses_tgn_direct:
                _reset_session_split_state(
                    base_db=base_db_now,
                    delete_persisted=True,
                )

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

            if not split_run_now and not _gds_projection_exists("trainGraph"):
                split_run_now = True
                split_reason = "trainGraph_missing_after_cold_start"
                split_info = None

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
                        storageRepresentation=_selected_storage_representation(),
                        keepTemporarySplitRelationships=admin_uses_tgn_direct,
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
                    if admin_uses_tgn_direct:
                        emb_out = {"directModel": True, "embeddingFamily": "TGN"}
                        lp_body = tgn_direct_router.TGNDirectBody(
                            trainGraphName="trainGraph",
                            valGraphName="valGraph",
                            testGraphName="testGraph",
                            storageRepresentation=_selected_storage_representation(),
                            negativeRatio=1.0,
                            predictK=int(admin_predict_k),
                            candidateMultiplier=20,
                            probThreshold=float(admin_prob_threshold),
                            evaluationThreshold=0.5,
                            outputGraphName="predictedGraph",
                            force_new_canonical_pool=False,
                            **tgn_params,
                        )
                        lp_out, lp_err = _safe_call(tgn_direct_router.tgn_direct_run, lp_body)
                    else:
                        embedding_t0 = time.perf_counter()
                        emb_out, emb_err, emb_prop = _run_embedding_for_family(admin_family, emb_params)
                        embedding_seconds = time.perf_counter() - embedding_t0

                        if emb_err:
                            st.error(emb_err)
                            lp_out, lp_err = None, emb_err
                        else:
                            lp_out, lp_err = _run_lp_for_property(
                                embedding_property=emb_prop,
                                predict_k=int(admin_predict_k),
                                prob_threshold=float(admin_prob_threshold),
                                embedding_family=admin_family,
                                prediction_model=admin_prediction_model,
                            )

                    if lp_err:
                        st.error(lp_err)
                    else:
                        st.session_state["admin_embedding_response"] = emb_out
                        st.session_state["admin_lp_response"] = lp_out

                        active_family = "TGN" if admin_uses_tgn_direct else admin_family
                        active_model = normalize_prediction_model(admin_prediction_model)
                        active_model_label = _prediction_model_value_to_label(active_model)

                        st.session_state["heatmap_prediction_model"] = active_model
                        st.session_state["heatmap_prediction_model_label"] = active_model_label
                        st.session_state[f"eval_sel_{active_family}"] = True

                        eval_selected = set(st.session_state.get("eval_selected") or set())
                        eval_selected.add(active_family)
                        st.session_state["eval_selected"] = eval_selected

                        st.session_state["plots_prediction_model"] = active_model
                        st.session_state["plots_prediction_model_label"] = active_model_label
                        st.session_state[f"plot_sel_{active_family}"] = True

                        plots_selected = set(st.session_state.get("plots_selected") or set())
                        plots_selected.add(active_family)
                        st.session_state["plots_selected"] = plots_selected

                        _refresh_eval_payload(base_db_now)
                        _refresh_plot_payload(base_db_now, active_model)

                        st.session_state["_pending_query_prediction_family"] = active_family

                        storage_mode_now = _selected_storage_representation()

                        if is_single_db_multi_edge(storage_mode_now):
                            st.session_state["future_db"] = base_db_now
                        else:
                            predicted_info = (lp_out or {}).get("predicted") if isinstance(lp_out, dict) else {}
                            st.session_state["future_db"] = (
                                predicted_info.get("materializedDatabase")
                                or predicted_db_name(
                                    base_db_now,
                                    active_family,
                                    admin_prediction_model,
                                )
                            )

                        if admin_uses_tgn_direct:
                            _queue_toast("TGN prediction completed")
                        else:
                            _queue_toast(f"{active_family} embeddings created. Link prediction completed")

                        st.rerun()

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
    user_extension_config = load_admin_extension_config(base_db)
    saved_prediction_model = normalize_prediction_model(
        user_extension_config["predictionModel"]
    )
    saved_extension_family = user_extension_config["embeddingFamily"]
    st.session_state["storage_representation"] = user_extension_config[
        "storageRepresentation"
    ]

    with st.sidebar:
        st.divider()
        st.markdown("<div class='prov-title'>Query Controls</div>", unsafe_allow_html=True)

        candidates = _db_candidates_for_base(base_db)
        existing = [candidate for candidate in candidates if candidate.get("exists")]
        families_existing = sorted(
            {
                candidate.get("family") or candidate.get("type")
                for candidate in existing
                if normalize_prediction_model(
                    candidate.get("predictionModel", "logistic_regression")
                ) == saved_prediction_model
                and (candidate.get("family") or candidate.get("type"))
            }
        )

        if saved_prediction_model == "tgn_direct":
            families_existing = ["TGN"]
        elif not families_existing:
            families_existing = list(EMBEDDING_FAMILIES)

        pending_query_family = st.session_state.pop(
            "_pending_query_prediction_family",
            None,
        )
        if pending_query_family in families_existing:
            st.session_state["query_prediction_family"] = pending_query_family
        elif st.session_state.get("query_prediction_family") not in families_existing:
            st.session_state["query_prediction_family"] = families_existing[0]

        query_family = st.selectbox(
            "Embedding family to query",
            families_existing,
            key="query_prediction_family",
        )
        storage_mode = _selected_storage_representation()

        if is_single_db_multi_edge(storage_mode):
            st.session_state["future_db"] = base_db
            timestamp_raw, timestamp_error = _safe_call(
                lp_router.lp_predicted_timestamps_single_db,
                base_db,
                query_family,
                saved_prediction_model,
            )
            if timestamp_error:
                timestamp_info = {"maxPredicted": 0, "maxTimestamp": 0}
            else:
                timestamp_info = timestamp_raw or {
                    "maxPredicted": 0,
                    "maxTimestamp": 0,
                }
        else:
            matching = [
                candidate
                for candidate in existing
                if (
                    (candidate.get("family") or candidate.get("type"))
                    == query_family
                    and normalize_prediction_model(
                        candidate.get("predictionModel", "logistic_regression")
                    )
                    == saved_prediction_model
                )
            ]
            st.session_state["future_db"] = (
                matching[0]["name"] if matching else None
            )
            timestamp_info = _refresh_predicted_timesteps(
                st.session_state.get("future_db")
            )

        current_timestep = int(timestamp_info.get("maxPredicted", 0) or 0)
        st.text_input(
            "Current predicted timesteps",
            value=str(current_timestep),
            disabled=True,
        )
        st.session_state["query_family_max_predicted"] = current_timestep
        st.session_state["query_family_for_until"] = query_family

        if st.button("Extend Predicted Graph", use_container_width=True):
            extension_model = saved_prediction_model
            extension_family = saved_extension_family
            predict_k = int(user_extension_config["predictK"])
            probability_threshold = float(user_extension_config["probThreshold"])
            additional_timesteps = int(
                user_extension_config["additionalTimesteps"]
            )
            retraining_interval = int(user_extension_config["retrainingInterval"])

            if extension_model == "tgn_direct":
                saved_tgn_params = dict(TGN_DIRECT_DEFAULT_PARAMS)
                saved_tgn_params.update(
                    user_extension_config.get("tgnDirectParams") or {}
                )
                iteration_body = tgn_direct_router.TGNDirectIterateBody(
                    storageRepresentation=_selected_storage_representation(),
                    predictK=predict_k,
                    candidateMultiplier=20,
                    probThreshold=probability_threshold,
                    negativeRatio=1.0,
                    retrainingInterval=retraining_interval,
                    **saved_tgn_params,
                )
                iteration_function = tgn_direct_router.iterate_tgn_direct
            else:
                iteration_body = lp_iter_router.IterateBody(
                    embeddingFamily=extension_family,
                    embeddingProperty=(
                        user_extension_config.get("embeddingProperty")
                        or EMB_WRITE_PROP[extension_family]
                    ),
                    predictionModel=extension_model,
                    storageRepresentation=_selected_storage_representation(),
                    predictK=predict_k,
                    candidateMultiplier=20,
                    probThreshold=probability_threshold,
                    negativeRatio=1.0,
                    retrainingInterval=retraining_interval,
                )
                iteration_function = lp_iter_router.iterate_prediction

            last_result = None
            succeeded = True
            for _ in range(additional_timesteps):
                last_result, error = _safe_call(
                    iteration_function,
                    iteration_body,
                )
                if error:
                    succeeded = False
                    st.error(error)
                    break

            if succeeded:
                st.session_state["last_iterate"] = last_result
                st.session_state["_pending_query_prediction_family"] = extension_family
                if is_single_db_multi_edge(_selected_storage_representation()):
                    st.session_state["future_db"] = base_db
                else:
                    st.session_state["future_db"] = predicted_db_name(
                        base_db,
                        extension_family,
                        extension_model,
                    )
                _refresh_eval_payload(base_db)
                _refresh_plot_payload(base_db, extension_model)
                _queue_toast("Predicted graph extended.")
                st.rerun()

    qcol, btncol = st.columns([7, 1.2])
    with qcol:
        st.markdown(
            "<div class='prov-card'><div class='prov-title'>Cypher Query</div>",
            unsafe_allow_html=True,
        )
        if "cypher" not in st.session_state:
            st.session_state["cypher"] = "MATCH p = ()-[]-()\nRETURN p\nLIMIT 500"
        cypher = st.text_area(label="", height=130, key="cypher")
        st.markdown("</div>", unsafe_allow_html=True)

    with btncol:
        st.write("")
        st.write("")
        if "run_mode" not in st.session_state:
            st.session_state["run_mode"] = "Graph"
        run_mode = st.selectbox("Mode", ["Graph", "Table"], key="run_mode")
        max_until = max(
            0,
            int(st.session_state.get("query_family_max_predicted", 0) or 0),
        )
        horizon_key = (
            f"query_horizon_{_selected_storage_representation()}_"
            f"{saved_prediction_model}_{query_family}"
        )
        if (
            horizon_key not in st.session_state
            or int(st.session_state[horizon_key]) > max_until
            or int(st.session_state[horizon_key]) < 0
        ):
            st.session_state[horizon_key] = max_until
        query_until_timestep = st.number_input(
            "Query horizon",
            min_value=0,
            max_value=max_until,
            value=max_until,
            step=1,
            disabled=max_until <= 0,
            key=horizon_key,
        )
        run_btn = st.button("Run on both", use_container_width=True)

    if run_btn:
        query_body = query_router.ModeBody(
            cypher=cypher,
            params={},
            futureDb=st.session_state.get("future_db"),
            mode="graph" if run_mode == "Graph" else "table",
            asOfTimestamp=None,
            storageRepresentation=_selected_storage_representation(),
            predictionFamily=st.session_state.get("query_prediction_family"),
            predictionModel=saved_prediction_model,
            predictionUntilTimestep=int(query_until_timestep),
        )
        output, error = _safe_call(query_router.query_both2, query_body)
        if error:
            st.error(error)
        else:
            st.session_state["last_query_result"] = output

    result = st.session_state.get("last_query_result")
    st.write("")
    if result:
        if run_mode == "Graph" and st.session_state.get("future_db"):
            if is_single_db_multi_edge(_selected_storage_representation()):
                future_meta = {
                    "maxPredicted": int(
                        st.session_state.get("query_family_max_predicted", 0)
                        or 0
                    )
                }
            else:
                future_meta = _refresh_predicted_timesteps(
                    st.session_state.get("future_db")
                )
            if int(future_meta.get("maxPredicted", 0) or 0) > 0:
                st.markdown(
                    _render_prediction_gradient_bar(future_meta["maxPredicted"]),
                    unsafe_allow_html=True,
                )

        left, right = st.columns(2)
        with left:
            st.markdown(
                "<div class='prov-card'><div class='prov-title'>Database: Present</div></div>",
                unsafe_allow_html=True,
            )
            st.write("")
            if run_mode == "Graph":
                graph = result.get("present") or {"nodes": [], "edges": []}
                st.plotly_chart(
                    _graph_to_plotly(graph, "Present graph", future=False),
                    use_container_width=True,
                )
            else:
                rows = result.get("present") or []
                st.dataframe(_to_df(rows), use_container_width=True, height=520)

        with right:
            st.markdown(
                "<div class='prov-card'><div class='prov-title'>Database: Future</div></div>",
                unsafe_allow_html=True,
            )
            st.write("")
            if run_mode == "Graph":
                graph = result.get("future") or {"nodes": [], "edges": []}
                st.plotly_chart(
                    _graph_to_plotly(graph, "Future graph", future=True),
                    use_container_width=True,
                )
            else:
                rows = result.get("future") or []
                st.dataframe(_to_df(rows), use_container_width=True, height=520)
    else:
        st.info("Run a query to see results.")


# ---------------------------
# ADMINISTRATOR VIEW
# ---------------------------

else:
    if "admin_content_mode" not in st.session_state:
        st.session_state["admin_content_mode"] = "Heatmap"

    st.markdown(
        "<div class='prov-title' style='font-size:1.2rem;'>Administrator</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([2, 10], gap="small")

    with left:
        mode_c1, mode_c2 = st.columns(2, gap="small")

        with mode_c1:
            if st.button(
                "Heatmap",
                use_container_width=True,
                key="admin_content_heatmap_btn",
                type="secondary",
            ):
                st.session_state["admin_content_mode"] = "Heatmap"

        with mode_c2:
            if st.button(
                "Plots",
                use_container_width=True,
                key="admin_content_plots_btn",
                type="secondary",
            ):
                st.session_state["admin_content_mode"] = "Plots"

    admin_content_mode = st.session_state.get("admin_content_mode", "Heatmap")

    if admin_content_mode == "Plots":
        with left:
            st.markdown(
                "<div class='control-panel-title'>Plot Components</div>",
                unsafe_allow_html=True,
            )
            show_summary = st.checkbox(
                "Summary table",
                value=st.session_state.get("plots_show_summary", True),
                key="plots_show_summary",
            )
            show_prob_plot = st.checkbox(
                "Probability Distribution Plot",
                value=st.session_state.get("plots_show_prob_plot", True),
                key="plots_show_prob_plot",
            )
            st.write("")
            if "plots_prediction_model_label" not in st.session_state:
                st.session_state["plots_prediction_model_label"] = (
                    _prediction_model_value_to_label(
                        st.session_state.get(
                            "admin_prediction_model",
                            "logistic_regression",
                        )
                    )
                )
            plots_prediction_model_label = st.selectbox(
                "Prediction model",
                list(PREDICTION_MODEL_OPTIONS.keys()),
                key="plots_prediction_model_label",
            )
            plots_prediction_model = normalize_prediction_model(
                PREDICTION_MODEL_OPTIONS[plots_prediction_model_label]
            )
            st.session_state["plots_prediction_model"] = plots_prediction_model
            plot_cache_key = (
                f"{base_db}|{_selected_storage_representation()}|"
                f"{plots_prediction_model}"
            )
            if st.session_state.get("plots_loaded_for_db") != plot_cache_key:
                _refresh_plot_payload(base_db, plots_prediction_model)

            available_plot_families = (
                ["TGN"]
                if plots_prediction_model == "tgn_direct"
                else list(EMBEDDING_FAMILIES)
            )
            exists_by_type = st.session_state.get("plots_exists_by_type") or {}
            if st.session_state.get("plots_selected_model") != plots_prediction_model:
                st.session_state["plots_selected"] = {
                    family
                    for family in available_plot_families
                    if exists_by_type.get(family, False)
                }
                st.session_state["plots_selected_model"] = plots_prediction_model

            st.write("")
            st.markdown(
                "<div class='prov-field-label'>Embedding family</div>",
                unsafe_allow_html=True,
            )
            selected_now = set(st.session_state.get("plots_selected") or set())
            next_selected: set[str] = set()
            for family in available_plot_families:
                has_predictions = bool(exists_by_type.get(family, False))
                checked = st.checkbox(
                    family,
                    value=family in selected_now and has_predictions,
                    disabled=not has_predictions,
                    key=f"plots_sel_{plots_prediction_model}_{family}",
                )
                if checked and has_predictions:
                    next_selected.add(family)
            st.session_state["plots_selected"] = next_selected
            st.session_state["plots_active_prediction_range"] = None
            st.write("")
            if st.button(
                "Refresh components",
                use_container_width=True,
                key="plots_refresh",
            ):
                _refresh_plot_payload(base_db, plots_prediction_model)
                _toast_ok("Prediction comparison refreshed.")
            plot_errors = st.session_state.get("plots_errs") or {}
            for family, error in plot_errors.items():
                st.warning(f"{family}: {error}")

        series_by_key: Dict[str, pd.DataFrame] = (
            st.session_state.get("plots_series_by_key") or {}
        )
        with right:
            visible_keys = [
                family
                for family in available_plot_families
                if family in (st.session_state.get("plots_selected") or set())
            ]
            filtered_series = {
                family: series_by_key.get(family, pd.DataFrame())
                for family in visible_keys
            }
            if not visible_keys:
                st.info(
                    "Select at least one embedding family with predictions "
                    "for the selected model."
                )
            else:
                if show_summary:
                    summary_df = _build_probability_summary_df(
                        filtered_series,
                        visible_keys,
                    )
                    if not summary_df.empty:
                        st.markdown(
                            "<div class='prov-card'><div class='prov-title'>Selected Embeddings Summary</div></div>",
                            unsafe_allow_html=True,
                        )
                        st.write("")
                        st.dataframe(
                            summary_df,
                            use_container_width=True,
                            hide_index=True,
                        )
                        st.write("")
                if show_prob_plot:
                    for family in visible_keys:
                        family_df = filtered_series.get(family)
                        if family_df is None or family_df.empty:
                            continue
                        st.markdown(
                            f"<div class='prov-card'><div class='prov-title'>{family} Probability Distribution by Timestamp</div></div>",
                            unsafe_allow_html=True,
                        )
                        st.write("")
                        figure = _build_probability_vs_timestamp_figure_for_family(
                            family_df,
                            family,
                        )
                        st.plotly_chart(figure, use_container_width=True)
                        st.write("")

    else:
        if "eval_loaded_for_db" not in st.session_state:
            st.session_state["eval_loaded_for_db"] = None

        _eval_model = st.session_state.get(
            "heatmap_prediction_model",
            st.session_state.get("admin_prediction_model", "logistic_regression"),
        )
        _eval_key = f"{base_db}|{_selected_storage_representation()}|{_eval_model}"

        if st.session_state.get("eval_loaded_for_db") != _eval_key:
            _refresh_eval_payload(base_db)

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

            heatmap_prediction_model_label = st.selectbox(
                "Prediction model",
                list(PREDICTION_MODEL_OPTIONS.keys()),
                index=0,
                key="heatmap_prediction_model_label",
            )

            heatmap_prediction_model = normalize_prediction_model(
                PREDICTION_MODEL_OPTIONS[heatmap_prediction_model_label]
            )
            st.session_state["heatmap_prediction_model"] = heatmap_prediction_model
            current_heatmap_model = heatmap_prediction_model

            current_eval_key = f"{base_db}|{_selected_storage_representation()}|{current_heatmap_model}"
            if st.session_state.get("eval_loaded_for_db") != current_eval_key:
                _refresh_eval_payload(base_db)

            pred_sets_now: Dict[str, set[str]] = st.session_state.get("eval_pred_sets") or {}
            metrics_now: Dict[str, Any] = st.session_state.get("eval_metrics") or {}

            heatmap_keys = ["TGN"] if current_heatmap_model == "tgn_direct" else ALL_EMB_KEYS
            st.write("")
            st.markdown(
                "<div class='prov-field-label'>Embedding family</div>",
                unsafe_allow_html=True,
            )
            exists_by_type = {
                k: bool(pred_sets_now.get(k)) or bool(metrics_now.get(k))
                for k in heatmap_keys
            }

            heatmap_selection_key = (
                f"{base_db}|"
                f"{_selected_storage_representation()}|"
                f"{current_heatmap_model}"
            )

            existing_for_model = {
                k for k in heatmap_keys
                if exists_by_type.get(k, False)
            }

            if (
                st.session_state.get("_heatmap_selection_key")
                != heatmap_selection_key
            ):
                st.session_state["_heatmap_selection_key"] = heatmap_selection_key
                st.session_state["eval_selected"] = set(existing_for_model)

                for k in ALL_EMB_KEYS:
                    st.session_state[f"eval_sel_{k}"] = (
                        k in existing_for_model
                    )

            current_selected = set(
                st.session_state.get("eval_selected")
                or set(existing_for_model)
            )

            current_selected = {
                k for k in current_selected
                if k in heatmap_keys and exists_by_type.get(k, False)
            }

            if not current_selected and existing_for_model:
                current_selected = set(existing_for_model)

                for k in ALL_EMB_KEYS:
                    st.session_state[f"eval_sel_{k}"] = (
                        k in existing_for_model
                    )

            next_selected: set[str] = set()

            for k in heatmap_keys:
                has_pred = exists_by_type.get(k, False)

                if f"eval_sel_{k}" not in st.session_state:
                    st.session_state[f"eval_sel_{k}"] = (
                        k in current_selected
                    ) and has_pred

                v = st.checkbox(
                    k,
                    value=st.session_state.get(
                        f"eval_sel_{k}",
                        (k in current_selected) and has_pred,
                    ),
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

            visible_order = ["TGN"] if st.session_state.get("heatmap_prediction_model") == "tgn_direct" else ALL_EMB_KEYS
            visible_keys = [k for k in visible_order if k in (st.session_state.get("eval_selected") or set())]

            if show_metrics and visible_keys:
                st.markdown(
                    "<div class='prov-card'><div class='prov-title'>Performance Metrics</div>",
                    unsafe_allow_html=True
                )
                st.write("")

                cols = st.columns(len(visible_keys), gap="small")

                for i, k in enumerate(visible_keys):
                    with cols[i]:
                        m = _pick_metrics_block(metrics_all.get(k) or {})

                        if not m:
                            st.info(f"No metrics cached for {k}. Run Train & Predict first.")
                            continue

                        st.markdown(
                            (
                                f"<div class='metric-card'>"
                                f"<div class='metric-card-title'>{k}</div>"

                                f"<div class='metric-row muted'>"
                                f"<span>Accuracy</span>"
                                f"<span class='metric-value'>{_fmt_metric(m.get('accuracy'))}</span>"
                                f"</div>"

                                f"<div class='metric-row muted'>"
                                f"<span>Precision</span>"
                                f"<span class='metric-value'>{_fmt_metric(m.get('precision'))}</span>"
                                f"</div>"

                                f"<div class='metric-row muted'>"
                                f"<span>Recall</span>"
                                f"<span class='metric-value'>{_fmt_metric(m.get('recall'))}</span>"
                                f"</div>"

                                f"<div class='metric-row muted'>"
                                f"<span>F1</span>"
                                f"<span class='metric-value'>{_fmt_metric(m.get('f1'))}</span>"
                                f"</div>"

                                f"<div class='metric-row muted'>"
                                f"<span>AUC</span>"
                                f"<span class='metric-value'>{_fmt_metric(m.get('auc'))}</span>"
                                f"</div>"

                                f"</div>"
                            ),
                            unsafe_allow_html=True,
                        )

                st.markdown("</div>", unsafe_allow_html=True)
                st.write("")

            if show_heatmap:
                keys = list(visible_keys)
                if show_gt and visible_keys:
                    keys = keys + ["Ground Truth"]

                if not visible_keys:
                    st.info("Select at least one embedding that has a family of predictions.")
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

                            if row_k == col_k:
                                row_vals.append(100.0 if row_set else 0.0)
                            else:
                                row_vals.append(_compute_jaccard_pct(row_set, col_set))

                        z.append(row_vals)

                    fig = go.Figure(
                        data=go.Heatmap(
                            z=z,
                            x=keys,
                            y=keys,
                            zmin=0,
                            zmax=100,
                            colorscale=[[0.0, "rgb(255,255,255)"], [1.0, "rgb(255,0,0)"]],
                            hovertemplate="%{y} vs %{x}<br>%{z:.2f}%<extra></extra>",
                            colorbar=dict(
                                tickfont=dict(
                                    size=22,          
                                    color="#ffffff",   
                                    family="Source Sans Pro" 
                                )
                            )
                        )
                    )

                    for yi, row_name in enumerate(keys):
                        for xi, col_name in enumerate(keys):
                            fig.add_annotation(
                                x=col_name,
                                y=row_name,
                                text=f"<b>{z[yi][xi]:.2f}</b>",
                                showarrow=False,
                                font=dict(
                                    size=26,
                                    color="black",
                                    family="Source Sans Pro"
                                ),
                                xanchor="center",
                                yanchor="middle",
                            )

                    fig.update_layout(
                        title=dict(
                            text="Prediction Comparison",
                            font=dict(size=22, color="#ffffff")
                        ),
                        height=650,
                        margin=dict(l=30, r=20, t=55, b=30),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(
                            side="bottom",
                            categoryorder="array",
                            categoryarray=keys,
                            tickfont=dict(
                                size=22,
                                color="#ffffff",
                                family="Source Sans Pro",
                            ),
                            automargin=True,
                        ),
                        yaxis=dict(
                            autorange="reversed",
                            categoryorder="array",
                            categoryarray=keys,
                            tickfont=dict(
                                size=22,
                                color="#ffffff",
                                family="Source Sans Pro",
                            ),
                            automargin=True,
                        ),
                    )

                    st.plotly_chart(fig, use_container_width=True)
