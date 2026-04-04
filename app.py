from pathlib import Path
import os

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html
from flask import abort, send_from_directory
from plotly.subplots import make_subplots

from importance_3d_plot import BASE_FOLDERS, find_supported_csvs, load_importance_csv


SENSOR_DATA_PATH = Path("data/machine_and_movement.csv")
ANNOT_TIMESTEPS = [150, 340, 820, 1280]
BENDING_ONLY_FOLDER = "a116ded8d403424489c23809a916fbd2"

APP_BG = "linear-gradient(180deg, #f6efe6 0%, #fbf8f4 36%, #ffffff 100%)"
PANEL_BG = "rgba(255, 255, 255, 0.88)"
PANEL_BORDER = "1px solid rgba(91, 44, 10, 0.10)"
TEXT_DARK = "#23160f"
TEXT_MUTED = "#6d584b"
ACCENT = "#8c2f39"
ACCENT_SOFT = "#f3dfd2"
SHADOW = "0 18px 48px rgba(66, 31, 14, 0.10)"
FONT_FAMILY = "Arial, Helvetica, sans-serif"

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


def build_heatmap(
    x_values, y_values, z_values, x_label, title_kind, selected_sensors, folder
):
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
        font=dict(color=TEXT_DARK, family=FONT_FAMILY),
        title_font=dict(size=24, color=TEXT_DARK),
    )
    fig.update_xaxes(title_text=x_label, showgrid=False, zeroline=False, row=2, col=1)
    fig.update_xaxes(showgrid=False, zeroline=False, row=1, col=1)
    fig.update_yaxes(title_text="Sensors", showgrid=False, zeroline=False, row=1, col=1)
    fig.update_yaxes(title_text="Angle Number", showgrid=False, zeroline=False, row=2, col=1)
    return fig


def build_surface(
    x_values,
    y_values,
    z_values,
    x_label,
    title_kind,
    selected_sensors,
    folder,
    camera=None,
):
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
        font=dict(color=TEXT_DARK, family=FONT_FAMILY),
        title_font=dict(size=24, color=TEXT_DARK),
    )
    return fig


existing_folders = [folder for folder in BASE_FOLDERS if Path(folder).exists()]
default_folder = existing_folders[0] if existing_folders else None
default_options = build_importance_options(default_folder) if default_folder else []
default_file = default_options[0]["value"] if default_options else None
_, sensor_values, _ = load_reference_sensor_frame()
sensor_options = sensor_values.columns.tolist() if sensor_values is not None else []

app = dash.Dash(__name__)
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .sensor-dropdown .Select-control {
                min-height: 46px !important;
                border-radius: 14px !important;
                border: 1px solid rgba(91, 44, 10, 0.16) !important;
                background: rgba(255, 255, 255, 0.95) !important;
                box-shadow: none !important;
            }
            .sensor-dropdown .Select-menu-outer {
                border-radius: 16px !important;
                border: 1px solid rgba(91, 44, 10, 0.12) !important;
                overflow: hidden !important;
                box-shadow: 0 18px 40px rgba(66, 31, 14, 0.10) !important;
            }
            .sensor-dropdown .Select--multi .Select-value {
                background: linear-gradient(135deg, #f3dfd2 0%, #fbf1ea 100%) !important;
                border: 1px solid rgba(140, 47, 57, 0.18) !important;
                border-radius: 999px !important;
                color: #7b2c35 !important;
                margin-top: 6px !important;
                margin-left: 6px !important;
                padding: 2px 10px !important;
            }
            .sensor-dropdown .Select--multi .Select-value-label {
                color: #7b2c35 !important;
                font-weight: 700 !important;
                padding: 0 8px 0 6px !important;
            }
            .sensor-dropdown .Select--multi .Select-value-icon {
                border-right: 1px solid rgba(140, 47, 57, 0.16) !important;
                color: #7b2c35 !important;
                padding: 0 8px 0 6px !important;
            }
            .sensor-dropdown .is-focused:not(.is-open) > .Select-control {
                border-color: rgba(140, 47, 57, 0.40) !important;
                box-shadow: 0 0 0 3px rgba(140, 47, 57, 0.10) !important;
            }
            .sensor-dropdown .Select-placeholder,
            .sensor-dropdown .Select--single > .Select-control .Select-value {
                line-height: 44px !important;
            }
            .sensor-dropdown .Select-input {
                margin-left: 8px !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""


@app.server.route("/video/<folder>/<path:filename>")
def serve_video(folder, filename):
    if folder not in BASE_FOLDERS:
        abort(404)
    videos_dir = (Path.cwd() / folder / "videos").resolve()
    if not videos_dir.exists():
        abort(404)
    response = send_from_directory(str(videos_dir), filename, conditional=False)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


app.layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    "Temporal Importance Analysis for Tube Bending Process",
                    style={
                        "fontSize": "38px",
                        "fontWeight": "700",
                        "letterSpacing": "-0.03em",
                        "color": TEXT_DARK,
                        "lineHeight": "1.0",
                    },
                ),
                html.Div(
                    "Interactive 3D importance surface and heatmap explorer with sensor overlays.",
                    style={
                        "marginTop": "10px",
                        "fontSize": "15px",
                        "color": TEXT_MUTED,
                        "maxWidth": "760px",
                    },
                ),
            ],
            style={
                "padding": "26px 24px 8px 24px",
            },
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            "Folder",
                            style={
                                "fontWeight": "700",
                                "marginBottom": "8px",
                                "fontSize": "12px",
                                "textTransform": "uppercase",
                                "letterSpacing": "0.08em",
                                "color": TEXT_MUTED,
                            },
                        ),
                        dcc.Dropdown(
                            id="folder-dropdown",
                            options=[
                                {
                                    "label": EXPERIMENT_LABELS.get(f, f),
                                    "value": f,
                                }
                                for f in existing_folders
                            ],
                            value=default_folder,
                            clearable=False,
                        ),
                    ],
                    style={"flex": "1", "minWidth": "240px"},
                ),
                html.Div(
                    [
                        html.Div(
                            "Importance Type",
                            style={
                                "fontWeight": "700",
                                "marginBottom": "8px",
                                "fontSize": "12px",
                                "textTransform": "uppercase",
                                "letterSpacing": "0.08em",
                                "color": TEXT_MUTED,
                            },
                        ),
                        dcc.Dropdown(id="file-dropdown", clearable=False),
                    ],
                    style={"flex": "1", "minWidth": "280px"},
                ),
                html.Div(
                    [
                        html.Div(
                            "Sensors",
                            style={
                                "fontWeight": "700",
                                "marginBottom": "8px",
                                "fontSize": "12px",
                                "textTransform": "uppercase",
                                "letterSpacing": "0.08em",
                                "color": TEXT_MUTED,
                            },
                        ),
                        dcc.Dropdown(
                            id="sensor-checklist",
                            options=[{"label": s, "value": s} for s in sensor_options],
                            value=sensor_options,
                            multi=True,
                            placeholder="Select sensors",
                            className="sensor-dropdown",
                        ),
                    ],
                    style={"flex": "2", "minWidth": "420px"},
                ),
            ],
            style={
                "display": "flex",
                "gap": "16px",
                "alignItems": "flex-end",
                "padding": "20px 24px 22px 24px",
                "background": PANEL_BG,
                "backdropFilter": "blur(10px)",
                "border": PANEL_BORDER,
                "borderRadius": "22px",
                "boxShadow": SHADOW,
                "margin": "8px 24px 12px 24px",
                "position": "sticky",
                "top": "12px",
                "zIndex": "1000",
            },
        ),
        html.Div(
            [
                dcc.Tabs(
                    [
                        dcc.Tab(
                            label="3D Surface",
                            children=[
                                dcc.Graph(
                                    id="surface-graph",
                                    config={"displaylogo": False, "responsive": True},
                                    style={"height": "980px", "width": "100%"},
                                )
                            ],
                            style={
                                "padding": "14px 22px",
                                "backgroundColor": "transparent",
                                "border": "none",
                                "color": TEXT_MUTED,
                                "fontWeight": "700",
                            },
                            selected_style={
                                "padding": "14px 22px",
                                "backgroundColor": ACCENT_SOFT,
                                "border": "none",
                                "borderRadius": "14px",
                                "color": ACCENT,
                                "fontWeight": "700",
                            },
                        ),
                        dcc.Tab(
                            label="Heatmap",
                            children=[
                                html.Div(
                                    dcc.Graph(
                                        id="heatmap-graph",
                                        config={"displaylogo": False, "responsive": False},
                                        style={"height": "760px", "width": "1650px"},
                                    ),
                                    style={
                                        "display": "flex",
                                        "justifyContent": "center",
                                        "width": "100%",
                                    },
                                )
                            ],
                            style={
                                "padding": "14px 22px",
                                "backgroundColor": "transparent",
                                "border": "none",
                                "color": TEXT_MUTED,
                                "fontWeight": "700",
                            },
                            selected_style={
                                "padding": "14px 22px",
                                "backgroundColor": ACCENT_SOFT,
                                "border": "none",
                                "borderRadius": "14px",
                                "color": ACCENT,
                                "fontWeight": "700",
                            },
                        ),
                        dcc.Tab(
                            label="Videos",
                            children=[html.Div(id="videos-content", style={"padding": "16px 4px"})],
                            style={
                                "padding": "14px 22px",
                                "backgroundColor": "transparent",
                                "border": "none",
                                "color": TEXT_MUTED,
                                "fontWeight": "700",
                            },
                            selected_style={
                                "padding": "14px 22px",
                                "backgroundColor": ACCENT_SOFT,
                                "border": "none",
                                "borderRadius": "14px",
                                "color": ACCENT,
                                "fontWeight": "700",
                            },
                        ),
                    ],
                    colors={
                        "border": "transparent",
                        "primary": ACCENT,
                        "background": "transparent",
                    },
                    parent_style={
                        "background": "transparent",
                    },
                )
            ],
            style={
                "padding": "8px 24px 24px 24px",
            },
        ),
        dcc.Store(id="camera-store"),
    ],
    style={
        "minHeight": "100vh",
        "background": APP_BG,
        "fontFamily": FONT_FAMILY,
    },
)


@app.callback(
    Output("file-dropdown", "options"),
    Output("file-dropdown", "value"),
    Input("folder-dropdown", "value"),
)
def update_file_dropdown(folder):
    options = build_importance_options(folder)
    value = options[0]["value"] if options else None
    return options, value


@app.callback(
    Output("camera-store", "data"),
    Input("surface-graph", "relayoutData"),
    State("camera-store", "data"),
    prevent_initial_call=True,
)
def store_camera(relayout_data, current_camera):
    if not relayout_data:
        return current_camera
    camera = relayout_data.get("scene.camera")
    return camera if camera is not None else current_camera


@app.callback(
    Output("surface-graph", "figure"),
    Output("heatmap-graph", "figure"),
    Input("file-dropdown", "value"),
    Input("sensor-checklist", "value"),
    State("camera-store", "data"),
)
def update_graphs(csv_path, selected_sensors, camera):
    if not csv_path:
        return go.Figure(), go.Figure()

    x_grid, y_grid, z_values, x_label, title_kind = load_importance_csv(csv_path)
    x_values = x_grid[0]
    y_values = y_grid[:, 0]
    z_values = normalize_zero_one(z_values)

    surface = build_surface(
        x_values,
        y_values,
        z_values,
        x_label,
        title_kind,
        selected_sensors or [],
        Path(csv_path).parent.name,
        camera=camera,
    )
    heatmap = build_heatmap(
        x_values,
        y_values,
        z_values,
        x_label,
        title_kind,
        selected_sensors or [],
        Path(csv_path).parent.name,
    )
    return surface, heatmap


@app.callback(
    Output("videos-content", "children"),
    Input("folder-dropdown", "value"),
    Input("file-dropdown", "value"),
)
def update_videos(folder, csv_path):
    if not folder:
        return html.Div("No folder selected.", style={"color": TEXT_MUTED})

    video_path = find_matching_video(folder, csv_path)
    if video_path is None:
        return html.Div("No matching video found for this importance type.", style={"color": TEXT_MUTED})

    version = int(video_path.stat().st_mtime)
    return html.Div(
        [
            html.Video(
                src=f"/video/{folder}/{video_path.name}?v={version}",
                controls=True,
                preload="metadata",
                style={
                    "width": "82%",
                    "maxWidth": "960px",
                    "borderRadius": "16px",
                    "backgroundColor": "#000",
                    "display": "block",
                    "margin": "0 auto",
                },
            ),
        ],
        style={
            "background": PANEL_BG,
            "border": PANEL_BORDER,
            "boxShadow": SHADOW,
            "borderRadius": "20px",
            "padding": "18px",
            "marginBottom": "18px",
        },
    )


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8050)), debug=False)
