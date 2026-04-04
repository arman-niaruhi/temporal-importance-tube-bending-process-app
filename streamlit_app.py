import streamlit as st
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from importance_3d_plot import BASE_FOLDERS, find_supported_csvs, load_importance_csv

# Page config
st.set_page_config(
    page_title="Importance Atlas",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #23160f;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6d584b;
        margin-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f3dfd2;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #8c2f39 !important;
        color: white !important;
    }
    .control-container {
        background-color: #f6efe6;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid #e0d5c8;
    }
    .stSelectbox, .stMultiselect {
        font-size: 0.9rem;
    }
    .stSelectbox > div > div {
        font-size: 0.9rem;
    }
    .stMultiselect > div > div {
        max-height: 200px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

# Constants
SENSOR_DATA_PATH = Path("data/machine_and_movement.csv")
ANNOT_TIMESTEPS = [150, 340, 820, 1280]
BENDING_ONLY_FOLDER = "a116ded8d403424489c23809a916fbd2"

EXPERIMENT_LABELS = {
    "8ecd70f401c845779e58e60c40729ff6": (
        "TCN-LSTM | No Feature Attention | MLP Attention | No Sampling"
    ),
    "25f4112799414b94ad1891d39122b9eb": (
        "LSTM | No Feature Attention | Bahdanau Attention | Resampled to 400 Windows"
    ),
    "a116ded8d403424489c23809a916fbd2": (
        "LSTM | Feature Attention | Bahdanau Attention | No Sampling | Bending Only"
    ),
}

def normalize_zero_one(values):
    values = np.asarray(values, dtype=float)
    v_min = float(np.nanmin(values))
    v_max = float(np.nanmax(values))
    if np.isclose(v_min, v_max):
        return np.zeros_like(values, dtype=float)
    return (values - v_min) / (v_max - v_min)

def load_reference_sensor_frame(folder=None):
    if not SENSOR_DATA_PATH.exists():
        return None, None, None

    df = pd.read_csv(SENSOR_DATA_PATH)
    if "Experiment_ID" not in df.columns:
        return None, None, None

    experiment_ids = sorted(df["Experiment_ID"].dropna().unique().tolist())
    if not experiment_ids:
        return None, None, None

    reference_experiment_id = (
        experiment_ids[-5] if len(experiment_ids) >= 5 else experiment_ids[-1]
    )
    ref = df[df["Experiment_ID"] == reference_experiment_id].copy()
    sensor_cols = [c for c in ref.columns if c not in {"Time_[s]", "Experiment_ID"}]
    sensor_values = ref[sensor_cols].astype(float)
    sensor_values = sensor_values.apply(normalize_zero_one, axis=0)
    x_values = (
        ref["Time_[s]"].to_numpy()
        if "Time_[s]" in ref.columns
        else np.arange(len(ref), dtype=float)
    )

    if folder == BENDING_ONLY_FOLDER and len(sensor_values) > 0:
        bend_start = max(0, min(ANNOT_TIMESTEPS[1], len(sensor_values) - 1))
        bend_end = max(bend_start + 1, min(ANNOT_TIMESTEPS[2], len(sensor_values)))
        sensor_values = sensor_values.iloc[bend_start:bend_end].reset_index(drop=True)
        x_values = np.asarray(x_values)[bend_start:bend_end]

    return x_values, sensor_values, reference_experiment_id

def find_video_files(folder):
    folder = Path(folder)
    videos_dir = folder / "videos"
    if not videos_dir.exists():
        return []

    preferred = []
    seen = set()
    for web_video in sorted(videos_dir.glob("*_web.mp4")):
        preferred.append(web_video)
        seen.add(web_video.name)

    for video in sorted(videos_dir.glob("*.mp4")):
        if video.name.endswith("_h264_test.mp4"):
            continue
        if video.name in seen:
            continue
        web_name = f"{video.stem}_web.mp4"
        if (videos_dir / web_name).exists():
            continue
        preferred.append(video)
    return preferred

def find_matching_video(folder, csv_path):
    video_files = find_video_files(folder)
    if not csv_path:
        return video_files[0] if video_files else None

    csv_stem = Path(csv_path).stem
    expected_names = [f"{csv_stem}_web.mp4", f"{csv_stem}.mp4"]
    for video_path in video_files:
        if video_path.name in expected_names:
            return video_path

    for video_path in video_files:
        if video_path.stem == csv_stem or video_path.stem == f"{csv_stem}_web":
            return video_path
    return video_files[0] if video_files else None

def build_importance_options(folder):
    csv_files = find_supported_csvs(folder) if folder else []
    attention_files = [p for p in csv_files if "attention" in p.name.lower()]
    window_files = [p for p in csv_files if "window_importance" in p.name.lower()]

    options = []
    if len(attention_files) >= 2:
        for path in attention_files:
            lower_name = path.name.lower()
            if "feat_00" in lower_name:
                label = "Attention Importance (Main Axis)"
            elif "feat_01" in lower_name:
                label = "Attention Importance (Secondary Axis)"
            else:
                label = f"Attention Importance ({path.stem})"
            options.append({"label": label, "value": str(path)})
    elif len(attention_files) == 1:
        options.append(
            {"label": "Attention Importance", "value": str(attention_files[0])}
        )

    for path in window_files:
        options.append({"label": "Window Importance", "value": str(path)})

    return options

def build_heatmap(x_values, y_values, z_values, x_label, title_kind, selected_sensors, folder):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.42, 0.58],
    )

    sensor_x, sensor_values, _ = load_reference_sensor_frame(folder=folder)
    if sensor_values is not None and selected_sensors:
        x_target = np.asarray(x_values, dtype=float)
        source_axis = np.linspace(0.0, 1.0, len(sensor_values), dtype=float)
        target_axis = np.linspace(0.0, 1.0, len(x_target), dtype=float)
        filtered_columns = [c for c in sensor_values.columns if c in selected_sensors]

        for col in filtered_columns:
            series = sensor_values[col].to_numpy(dtype=float)
            scaled_interp = np.interp(target_axis, source_axis, series)
            short_name = (
                col.replace("MACHINE_", "")
                .replace("Movement_[mm]", "Move")
                .replace("Max_Torque_[%]", "Torque")
                .replace("Angle_[°]", "Angle")
            )
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=scaled_interp,
                    mode="lines",
                    line=dict(width=1.6),
                    name=short_name,
                    showlegend=False,
                    hovertemplate=f"{short_name}<extra></extra>",
                ),
                row=1,
                col=1,
            )

    fig.add_trace(
        go.Heatmap(
            x=x_values,
            y=y_values,
            z=z_values,
            colorscale="Magma",
            colorbar=dict(title="Importance"),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=760,
        width=1650,
        autosize=False,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font=dict(color="#23160f", family="Arial, Helvetica, sans-serif"),
        title_font=dict(size=24, color="#23160f"),
    )
    fig.update_xaxes(title_text=x_label, showgrid=False, zeroline=False, row=2, col=1)
    fig.update_xaxes(showgrid=False, zeroline=False, row=1, col=1)
    fig.update_yaxes(title_text="Sensors", showgrid=False, zeroline=False, row=1, col=1)
    fig.update_yaxes(title_text="Angle Number", showgrid=False, zeroline=False, row=2, col=1)
    return fig

def build_surface(x_values, y_values, z_values, x_label, title_kind, selected_sensors, folder, camera=None):
    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=x_values,
            y=y_values,
            z=z_values,
            colorscale="Magma",
            colorbar=dict(title="Importance"),
            showscale=True,
            name="Importance",
        )
    )

    _, sensor_values, _ = load_reference_sensor_frame(folder=folder)
    if sensor_values is not None and selected_sensors:
        x_target = np.asarray(x_values, dtype=float)
        source_axis = np.linspace(0.0, 1.0, len(sensor_values), dtype=float)
        target_axis = np.linspace(0.0, 1.0, len(x_target), dtype=float)

        filtered_columns = [c for c in sensor_values.columns if c in selected_sensors]
        for idx, col in enumerate(filtered_columns):
            series = sensor_values[col].to_numpy(dtype=float)
            scaled_interp = np.interp(target_axis, source_axis, series)
            sensor_y = np.full_like(x_target, -(idx + 1), dtype=float)
            short_name = (
                col.replace("MACHINE_", "")
                .replace("Movement_[mm]", "Move")
                .replace("Max_Torque_[%]", "Torque")
                .replace("Angle_[°]", "Angle")
            )
            fig.add_trace(
                go.Scatter3d(
                    x=x_target,
                    y=sensor_y,
                    z=scaled_interp,
                    mode="lines",
                    line=dict(width=4),
                    name=short_name,
                    showlegend=False,
                    hovertemplate=f"{short_name}<extra></extra>",
                )
            )
    angle_ticks = np.linspace(
        float(np.min(y_values)),
        float(np.max(y_values)),
        num=min(6, len(y_values)),
    )
    tickvals = angle_ticks.tolist()
    ticktext = [f"Angle {int(round(v))}" for v in angle_ticks]

    scene = dict(
        xaxis_title=x_label,
        yaxis_title="Angle Number / Sensors",
        zaxis_title="Importance",
        xaxis=dict(
            showbackground=False,
            showline=False,
            zeroline=False,
            showgrid=False,
        ),
        yaxis=dict(
            tickvals=tickvals,
            ticktext=ticktext,
            showbackground=False,
            showline=False,
            zeroline=False,
            showgrid=False,
        ),
        zaxis=dict(
            showbackground=False,
            showline=False,
            zeroline=False,
            showgrid=False,
        ),
    )
    if camera is not None:
        scene["camera"] = camera

    fig.update_layout(
        scene=scene,
        height=980,
        autosize=True,
        margin=dict(l=10, r=10, t=50, b=10),
        showlegend=False,
        uirevision="importance_surface_view",
        paper_bgcolor="rgba(255,255,255,0)",
        font=dict(color="#23160f", family="Arial, Helvetica, sans-serif"),
        title_font=dict(size=24, color="#23160f"),
    )
    return fig

# Main app
def main():
    st.markdown('<h1 class="main-header">Temporal Importance Analysis for Tube Bending Process</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Interactive 3D importance surface and heatmap explorer with sensor overlays.</p>', unsafe_allow_html=True)

    # Controls at top
    st.markdown('<div class="control-container">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1.2])

    with col1:
        existing_folders = [folder for folder in BASE_FOLDERS if Path(folder).exists()]
        default_folder = existing_folders[0] if existing_folders else None

        folder_options = {EXPERIMENT_LABELS.get(f, f): f for f in existing_folders}
        selected_folder_label = st.selectbox(
            "Experiment",
            options=list(folder_options.keys()),
            index=0 if default_folder else None,
            key="folder_select"
        )
        folder = folder_options.get(selected_folder_label)

    with col2:
        importance_options = build_importance_options(folder) if folder else []
        if importance_options:
            importance_labels = [opt["label"] for opt in importance_options]
            selected_importance_label = st.selectbox(
                "Importance Type",
                options=importance_labels,
                index=0,
                key="importance_select"
            )
            selected_importance = next(
                opt["value"] for opt in importance_options
                if opt["label"] == selected_importance_label
            )
        else:
            selected_importance = None
            st.warning("No importance files found for this experiment.")

    with col3:
        _, sensor_values, _ = load_reference_sensor_frame()
        sensor_options = sensor_values.columns.tolist() if sensor_values is not None else []
        selected_sensors = st.multiselect(
            "Sensors",
            options=sensor_options,
            default=sensor_options,
            help="Select sensors to overlay on the plots",
            key="sensor_select"
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # Main content
    if not selected_importance:
        st.error("Please select an importance type.")
        return

    # Load data
    with st.spinner("Loading importance data..."):
        x_grid, y_grid, z_values, x_label, title_kind = load_importance_csv(selected_importance)
        x_values = x_grid[0]
        y_values = y_grid[:, 0]
        z_values = normalize_zero_one(z_values)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["3D Surface", "Heatmap", "Videos"])

    with tab1:
        st.subheader("3D Surface View")
        fig_surface = build_surface(
            x_values, y_values, z_values, x_label, title_kind,
            selected_sensors, Path(selected_importance).parent.name
        )
        st.plotly_chart(fig_surface, use_container_width=True)

    with tab2:
        st.subheader("Heatmap View")
        fig_heatmap = build_heatmap(
            x_values, y_values, z_values, x_label, title_kind,
            selected_sensors, Path(selected_importance).parent.name
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

    with tab3:
        st.subheader("Videos")
        video_path = find_matching_video(folder, selected_importance)
        if video_path and video_path.exists():
            st.video(str(video_path))
        else:
            st.warning("No matching video found for this importance type.")

if __name__ == "__main__":
    main()