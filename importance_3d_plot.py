import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_FOLDERS = [
    "8ecd70f401c845779e58e60c40729ff6",
    "25f4112799414b94ad1891d39122b9eb",
    "a116ded8d403424489c23809a916fbd2",
]


def find_supported_csvs(path):
    path = Path(path)
    if path.is_file():
        return [path]

    patterns = ("*attention*.csv", "*window_importance*.csv")
    files = []
    for pattern in patterns:
        files.extend(sorted(path.glob(pattern)))
    return files


def parse_angle_label(raw_label, idx):
    s = str(raw_label)
    nums = re.findall(r"\d+", s)
    if nums:
        val = int(nums[-1])
        if s.lower().startswith("pred_"):
            return val + 1
        return val
    return idx + 1


def load_importance_csv(csv_path):
    csv_path = Path(csv_path)

    if "window_importance" in csv_path.name.lower():
        df = pd.read_csv(csv_path)
        if "angle" in df.columns:
            angle_labels = [
                parse_angle_label(raw_label, idx)
                for idx, raw_label in enumerate(df["angle"].tolist())
            ]
            values = df.drop(columns=["angle"])
        else:
            angle_labels = [idx + 1 for idx in range(len(df))]
            values = df

        values = values.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
        x_label = "Window Index"
        x_values = np.arange(values.shape[1], dtype=float)
        title_kind = "Window Importance"
    else:
        df = pd.read_csv(csv_path, index_col=0)
        df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
        angle_labels = [
            parse_angle_label(raw_label, idx)
            for idx, raw_label in enumerate(df.index.tolist())
        ]
        values = df
        x_label = "Time Step"
        x_values = np.arange(values.shape[1], dtype=float)
        title_kind = "Attention Importance"

    z = values.to_numpy(dtype=float)
    y_values = np.asarray(angle_labels, dtype=float)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    return x_grid, y_grid, z, x_label, title_kind


def build_3d_plot(csv_path, output_path=None, elev=28, azim=-128, show=False):
    csv_path = Path(csv_path)
    x_grid, y_grid, z, x_label, title_kind = load_importance_csv(csv_path)

    fig = plt.figure(figsize=(15, 10), dpi=160)
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        x_grid,
        y_grid,
        z,
        cmap="magma",
        linewidth=0,
        antialiased=True,
        alpha=0.96,
    )

    ax.set_title(f"{title_kind} over Time and Angle", fontsize=16, fontweight="bold", pad=18)
    ax.set_xlabel(x_label, fontsize=12, labelpad=12)
    ax.set_ylabel("Angle Number", fontsize=12, labelpad=12)
    ax.set_zlabel("Importance", fontsize=12, labelpad=10)
    ax.view_init(elev=elev, azim=azim)

    ax.xaxis.pane.set_facecolor((0.98, 0.98, 0.98, 1.0))
    ax.yaxis.pane.set_facecolor((0.98, 0.98, 0.98, 1.0))
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.grid(True, alpha=0.2)

    fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.08, label="Importance")
    fig.tight_layout()

    if output_path is None:
        output_path = csv_path.with_name(f"{csv_path.stem}_3d.png")
    else:
        output_path = Path(output_path)

    fig.savefig(output_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def main():
    for base_folder in BASE_FOLDERS:
        base_path = Path(base_folder)
        if not base_path.exists():
            continue

        csv_files = find_supported_csvs(base_path)
        for csv_file in csv_files:
            output_path = build_3d_plot(csv_path=csv_file)
            print(output_path)


if __name__ == "__main__":
    main()
