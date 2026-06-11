"""
Generate overview graphs comparing all models by parameter count vs metric.

For each metric (MAE, ME, SD, RMSE), produces one PNG and one interactive HTML:
  Left:  SBP (systolic blood pressure)
  Right: DBP (diastolic blood pressure)

Also produces a single inference-time graph:
  x-axis: trainable parameter count (log scale)
  y-axis: average inference time per sample (ms/sample)

Data sources:
  data/models/<model>/struct.txt           — trainable parameter count
  data/models/<model>/eval_results.json    — metric values + inference time

Output:
  images/plot_mae.png / images/plot_mae.html
  images/plot_me.png / images/plot_me.html
  images/plot_sd.png / images/plot_sd.html
  images/plot_rmse.png / images/plot_rmse.html
  images/plot_inference_time.png / images/plot_inference_time.html
  images/bar_mae.png
  images/bar_me.png
  images/bar_sd.png
  images/bar_rmse.png
  images/bar_inference_time.png

Usage:
    uv run python scripts/generate-overview.py
    uv run python scripts/generate-overview.py --models-dir data/models --output-dir images
    uv run python scripts/generate-overview.py --format html
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


_PALETTE = [cm.tab20(i) for i in range(20)]
PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"

EXCLUDE_MODELS: set[str] = {"naive"}

METRICS: list[tuple[str, str, bool]] = [
    ("mae",  "MAE (mmHg)",  False),
    ("me",   "ME (mmHg)",   True),
    ("sd",   "SD (mmHg)",   False),
    ("rmse", "RMSE (mmHg)", False),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate parameter-count vs metric overview graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--models-dir", type=Path, default=Path("data/models"),
        help="Root directory containing model subdirectories (default: data/models)",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("images"),
        help="Directory to write graph files (default: images)",
    )
    p.add_argument(
        "--format", choices=["png", "html", "both"], default="both",
        help="Output format to generate (default: both)",
    )
    return p.parse_args()


def _parse_param_count(struct_path: Path) -> int | None:
    text = struct_path.read_text(encoding="utf-8")
    m = re.search(r"Trainable params:\s*([\d,]+)", text)
    return int(m.group(1).replace(",", "")) if m else None


def load_model_data(models_dir: Path) -> list[dict]:
    """Return list of dicts with model name, param count, and eval metrics."""
    records: list[dict] = []

    for model_dir in sorted(models_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        if model_dir.name in EXCLUDE_MODELS:
            continue

        struct_path = model_dir / "struct.txt"
        if not struct_path.exists():
            continue

        n_params = _parse_param_count(struct_path)
        if n_params is None:
            print(f"  [warn] could not parse param count from {struct_path}")
            continue

        eval_path = model_dir / "eval_results.json"
        if not eval_path.exists():
            print(f"  [warn] no eval_results.json found for {model_dir.name}")
            continue
        with open(eval_path, encoding="utf-8") as f:
            eval_data = json.load(f)

        records.append({
            "model": model_dir.name,
            "n_params": n_params,
            "sbp": eval_data["sbp"],
            "dbp": eval_data["dbp"],
            "avg_ms_per_sample": eval_data.get("avg_ms_per_sample"),
        })

    return records


def _param_formatter(x: float, _pos=None) -> str:
    """Format parameter counts as 2, 15K, 440K, 2.18M, etc."""
    if x < 1_000:
        return f"{int(x)}"
    if x < 1_000_000:
        v = x / 1_000
        return f"{v:.0f}K" if v == int(v) else f"{v:.1f}K"
    v = x / 1_000_000
    return f"{v:.0f}M" if v == int(v) else f"{v:.2f}M"


def _plotly_tick_values(data: list[dict]) -> tuple[list[int], list[str]]:
    vals = sorted({rec["n_params"] for rec in data})
    return vals, [_param_formatter(v) for v in vals]


def _annotate(ax, x: float, y: float, label: str) -> None:
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(5, 4),
        textcoords="offset points",
        fontsize=7.5,
        clip_on=True,
    )


def _sorted_bar_records(data: list[dict], bp: str, metric: str) -> list[dict]:
    return sorted(data, key=lambda rec: rec[bp][metric])


def _sorted_inference_records(data: list[dict]) -> list[dict]:
    return sorted(
        (rec for rec in data if rec["avg_ms_per_sample"] is not None),
        key=lambda rec: rec["avg_ms_per_sample"],
    )


def _bar_colors(n: int) -> list:
    return [_PALETTE[i % len(_PALETTE)] for i in range(n)]


def _write_html(out_path: Path, title: str, figure_spec: dict) -> None:
    fig_json = json.dumps(figure_spec, ensure_ascii=False)
    html = (
        "<!doctype html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "  <meta charset=\"utf-8\">\n"
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"  <title>{title}</title>\n"
        f"  <script src=\"{PLOTLY_CDN}\"></script>\n"
        "  <style>\n"
        "    body { margin: 0; background: #f4f5fb; font-family: Segoe UI, Arial, sans-serif; }\n"
        "    #plot { width: 100vw; height: 100vh; }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        "  <div id=\"plot\"></div>\n"
        "  <script>\n"
        f"    const fig = {fig_json};\n"
        "    Plotly.newPlot('plot', fig.data, fig.layout, fig.config);\n"
        "  </script>\n"
        "</body>\n"
        "</html>\n"
    )
    out_path.write_text(html, encoding="utf-8")
    print(f"  Saved: {out_path}")


def _build_metric_html_figure(
    data: list[dict],
    metric: str,
    ylabel: str,
    zero_line: bool,
) -> dict:
    tickvals, ticktext = _plotly_tick_values(data)
    traces: list[dict] = []
    label_trace_indices: list[int] = []

    for i, rec in enumerate(data):
        color = mcolors.to_hex(_PALETTE[i % len(_PALETTE)])
        for bp, bp_label, xaxis, yaxis, showlegend in [
            ("sbp", "SBP", "x", "y", True),
            ("dbp", "DBP", "x2", "y2", False),
        ]:
            value = rec[bp][metric]
            traces.append({
                "type": "scatter",
                "mode": "markers",
                "name": rec["model"],
                "legendgroup": rec["model"],
                "showlegend": showlegend,
                "xaxis": xaxis,
                "yaxis": yaxis,
                "x": [rec["n_params"]],
                "y": [value],
                "marker": {
                    "size": 12,
                    "color": color,
                    "line": {"width": 0.6, "color": "#23314a"},
                },
                "hovertemplate": (
                    f"{rec['model']}<br>"
                    f"{bp_label} {metric.upper()}: {value:.4f}<br>"
                    f"Params: {rec['n_params']:,}<extra></extra>"
                ),
            })
            label_trace_indices.append(len(traces))
            traces.append({
                "type": "scatter",
                "mode": "text",
                "name": f"{rec['model']} label",
                "legendgroup": rec["model"],
                "showlegend": False,
                "visible": True,
                "xaxis": xaxis,
                "yaxis": yaxis,
                "x": [rec["n_params"]],
                "y": [value],
                "text": [rec["model"]],
                "textposition": "top right",
                "textfont": {"size": 11, "color": color},
                "hoverinfo": "skip",
            })

    shapes = []
    if zero_line:
        shapes.extend([
            {"type": "line", "xref": "paper", "yref": "y", "x0": 0.0, "x1": 0.46,
             "y0": 0, "y1": 0, "line": {"color": "gray", "width": 1, "dash": "dash"}},
            {"type": "line", "xref": "paper", "yref": "y2", "x0": 0.54, "x1": 1.0,
             "y0": 0, "y1": 0, "line": {"color": "gray", "width": 1, "dash": "dash"}},
        ])

    layout = {
        "title": {"text": f"Model Comparison: {metric.upper()} vs Parameter Count", "x": 0.36, "y": 0.98},
        "paper_bgcolor": "#f4f5fb",
        "plot_bgcolor": "#ffffff",
        "hovermode": "closest",
        "legend": {
            "orientation": "v",
            "yanchor": "top",
            "y": 1.0,
            "xanchor": "left",
            "x": 1.02,
            "groupclick": "togglegroup",
            "bgcolor": "rgba(255,255,255,0.90)",
            "bordercolor": "rgba(120,130,160,0.35)",
            "borderwidth": 1,
            "font": {"size": 11},
        },
        "margin": {"l": 80, "r": 280, "t": 120, "b": 80},
        "width": 1600,
        "height": 720,
        "xaxis": {
            "domain": [0.0, 0.40],
            "type": "log",
            "title": "Trainable Parameters (log scale)",
            "tickvals": tickvals,
            "ticktext": ticktext,
            "gridcolor": "rgba(120,130,160,0.20)",
            "zeroline": False,
        },
        "xaxis2": {
            "domain": [0.48, 0.88],
            "type": "log",
            "title": "Trainable Parameters (log scale)",
            "tickvals": tickvals,
            "ticktext": ticktext,
            "gridcolor": "rgba(120,130,160,0.20)",
            "zeroline": False,
        },
        "yaxis": {
            "title": ylabel,
            "gridcolor": "rgba(120,130,160,0.20)",
            "zeroline": False,
        },
        "yaxis2": {
            "title": ylabel,
            "gridcolor": "rgba(120,130,160,0.20)",
            "zeroline": False,
            "anchor": "x2",
        },
        "annotations": [
            {"text": f"SBP - {metric.upper()}", "xref": "paper", "yref": "paper",
             "x": 0.20, "y": 0.98, "showarrow": False, "font": {"size": 14}},
            {"text": f"DBP - {metric.upper()}", "xref": "paper", "yref": "paper",
             "x": 0.68, "y": 0.98, "showarrow": False, "font": {"size": 14}},
        ],
        "updatemenus": [{
            "type": "buttons",
            "direction": "right",
            "x": 1.0,
            "xanchor": "right",
            "y": 1.20,
            "yanchor": "top",
            "showactive": True,
            "buttons": [
                {
                    "label": "Labels Off",
                    "method": "restyle",
                    "args": [{"visible": False}, label_trace_indices],
                },
                {
                    "label": "Labels On",
                    "method": "restyle",
                    "args": [{"visible": True}, label_trace_indices],
                },
            ],
        }],
        "shapes": shapes,
    }
    config = {
        "responsive": True,
        "displaylogo": False,
        "toImageButtonOptions": {"format": "png", "filename": f"plot_{metric}"},
    }
    return {"data": traces, "layout": layout, "config": config}


def _build_inference_time_html_figure(data: list[dict]) -> dict:
    tickvals, ticktext = _plotly_tick_values(data)
    traces: list[dict] = []
    label_trace_indices: list[int] = []

    for i, rec in enumerate(data):
        color = mcolors.to_hex(_PALETTE[i % len(_PALETTE)])
        traces.append({
            "type": "scatter",
            "mode": "markers",
            "name": rec["model"],
            "legendgroup": rec["model"],
            "x": [rec["n_params"]],
            "y": [rec["avg_ms_per_sample"]],
            "marker": {
                "size": 12,
                "color": color,
                "line": {"width": 0.6, "color": "#23314a"},
            },
            "hovertemplate": (
                f"{rec['model']}<br>"
                f"Inference: {rec['avg_ms_per_sample']:.4f} ms/sample<br>"
                f"Params: {rec['n_params']:,}<extra></extra>"
            ),
        })
        label_trace_indices.append(len(traces))
        traces.append({
            "type": "scatter",
            "mode": "text",
            "name": f"{rec['model']} label",
            "legendgroup": rec["model"],
            "showlegend": False,
            "visible": True,
            "x": [rec["n_params"]],
            "y": [rec["avg_ms_per_sample"]],
            "text": [rec["model"]],
            "textposition": "top right",
            "textfont": {"size": 11, "color": color},
            "hoverinfo": "skip",
        })

    layout = {
        "title": {"text": "Model Comparison: Inference Time vs Parameter Count", "x": 0.42, "y": 0.98},
        "paper_bgcolor": "#f4f5fb",
        "plot_bgcolor": "#ffffff",
        "hovermode": "closest",
        "legend": {
            "orientation": "v",
            "yanchor": "top",
            "y": 1.0,
            "xanchor": "left",
            "x": 1.02,
            "groupclick": "togglegroup",
            "bgcolor": "rgba(255,255,255,0.90)",
            "bordercolor": "rgba(120,130,160,0.35)",
            "borderwidth": 1,
            "font": {"size": 11},
        },
        "margin": {"l": 80, "r": 280, "t": 120, "b": 80},
        "width": 1280,
        "height": 720,
        "xaxis": {
            "type": "log",
            "title": "Trainable Parameters (log scale)",
            "tickvals": tickvals,
            "ticktext": ticktext,
            "gridcolor": "rgba(120,130,160,0.20)",
            "zeroline": False,
        },
        "yaxis": {
            "title": "Inference Time (ms / sample)",
            "gridcolor": "rgba(120,130,160,0.20)",
            "zeroline": False,
        },
        "updatemenus": [{
            "type": "buttons",
            "direction": "right",
            "x": 0.0,
            "xanchor": "left",
            "y": 1.12,
            "yanchor": "top",
            "showactive": True,
            "buttons": [
                {
                    "label": "Labels Off",
                    "method": "restyle",
                    "args": [{"visible": False}, label_trace_indices],
                },
                {
                    "label": "Labels On",
                    "method": "restyle",
                    "args": [{"visible": True}, label_trace_indices],
                },
            ],
        }],
    }
    config = {
        "responsive": True,
        "displaylogo": False,
        "toImageButtonOptions": {"format": "png", "filename": "plot_inference_time"},
    }
    return {"data": traces, "layout": layout, "config": config}


def plot_metric_png(
    data: list[dict],
    metric: str,
    ylabel: str,
    zero_line: bool,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, bp, bp_label in zip(axes, ["sbp", "dbp"], ["SBP", "DBP"]):
        for i, rec in enumerate(data):
            x = rec["n_params"]
            y = rec[bp][metric]
            color = _PALETTE[i % len(_PALETTE)]
            ax.scatter(x, y, s=70, color=color, zorder=5, label=rec["model"])
            _annotate(ax, x, y, rec["model"])

        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(_param_formatter))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())

        ax.set_xlabel("Trainable Parameters (log scale)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"{bp_label} - {metric.upper()}", fontsize=11)
        ax.grid(True, which="major", linestyle="--", alpha=0.4)
        ax.grid(True, which="minor", linestyle=":",  alpha=0.2)

        if zero_line:
            ax.axhline(0, color="gray", linewidth=0.9, linestyle="--")

    fig.suptitle(
        f"Model Comparison: {metric.upper()} vs Parameter Count",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()

    out_path = output_dir / f"plot_{metric}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_metric_html(
    data: list[dict],
    metric: str,
    ylabel: str,
    zero_line: bool,
    output_dir: Path,
) -> None:
    fig = _build_metric_html_figure(data, metric, ylabel, zero_line)
    _write_html(output_dir / f"plot_{metric}.html", f"PLOT {metric.upper()} Overview", fig)


def plot_inference_time_png(data: list[dict], output_dir: Path) -> None:
    timed = [r for r in data if r["avg_ms_per_sample"] is not None]
    if not timed:
        print("  [warn] no inference time data - skipping inference_time.png")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, rec in enumerate(timed):
        x = rec["n_params"]
        y = rec["avg_ms_per_sample"]
        color = _PALETTE[i % len(_PALETTE)]
        ax.scatter(x, y, s=70, color=color, zorder=5, label=rec["model"])
        _annotate(ax, x, y, rec["model"])

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_param_formatter))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    ax.set_xlabel("Trainable Parameters (log scale)", fontsize=10)
    ax.set_ylabel("Inference Time (ms / sample)", fontsize=10)
    ax.set_title("Model Comparison: Inference Time vs Parameter Count", fontsize=11)
    ax.grid(True, which="major", linestyle="--", alpha=0.4)
    ax.grid(True, which="minor", linestyle=":",  alpha=0.2)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.7)

    fig.tight_layout()

    out_path = output_dir / "plot_inference_time.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_inference_time_html(data: list[dict], output_dir: Path) -> None:
    timed = [r for r in data if r["avg_ms_per_sample"] is not None]
    if not timed:
        print("  [warn] no inference time data - skipping inference_time.html")
        return
    fig = _build_inference_time_html_figure(timed)
    _write_html(output_dir / "plot_inference_time.html", "Inference Time Overview", fig)


def plot_metric_bar_png(
    data: list[dict],
    metric: str,
    ylabel: str,
    zero_line: bool,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    for ax, bp, bp_label in zip(axes, ["sbp", "dbp"], ["SBP", "DBP"]):
        rows = _sorted_bar_records(data, bp, metric)
        models = [rec["model"] for rec in rows]
        values = [rec[bp][metric] for rec in rows]
        colors = _bar_colors(len(rows))
        bars = ax.bar(models, values, color=colors, edgecolor="#23314a", linewidth=0.5)

        if zero_line:
            ax.axhline(0, color="gray", linewidth=0.9, linestyle="--")

        ax.set_title(f"{bp_label} - {metric.upper()}", fontsize=11)
        ax.set_xlabel("Models (sorted by value)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.tick_params(axis="x", rotation=55, labelsize=8)
        ax.set_axisbelow(True)

        y_min = min(0.0, min(values))
        y_max = max(0.0, max(values))
        margin = max((y_max - y_min) * 0.08, 0.1)
        ax.set_ylim(y_min - margin, y_max + margin)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + (0.01 * (y_max - y_min if y_max != y_min else 1.0)),
                f"{val:.2f}",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=7,
                rotation=90,
            )

    fig.suptitle(f"Model Ranking: {metric.upper()} by Model", fontsize=13)
    out_path = output_dir / f"bar_{metric}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_inference_time_bar_png(data: list[dict], output_dir: Path) -> None:
    rows = _sorted_inference_records(data)
    if not rows:
        print("  [warn] no inference time data - skipping bar_inference_time.png")
        return

    models = [rec["model"] for rec in rows]
    values = [rec["avg_ms_per_sample"] for rec in rows]
    colors = _bar_colors(len(rows))

    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    bars = ax.bar(models, values, color=colors, edgecolor="#23314a", linewidth=0.5)
    ax.set_title("Model Ranking: Inference Time by Model", fontsize=12)
    ax.set_xlabel("Models (sorted by value)", fontsize=10)
    ax.set_ylabel("Inference Time (ms / sample)", fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis="x", rotation=55, labelsize=8)
    ax.set_axisbelow(True)

    y_max = max(values)
    margin = max(y_max * 0.08, 0.02)
    ax.set_ylim(0, y_max + margin)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + margin * 0.15,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
        )

    out_path = output_dir / "bar_inference_time.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main() -> None:
    args = parse_args()

    data = load_model_data(args.models_dir)
    if not data:
        print("No model data found - nothing to plot.")
        return

    print(f"Loaded {len(data)} models.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    want_png = args.format in {"png", "both"}
    want_html = args.format in {"html", "both"}

    for metric, ylabel, zero_line in METRICS:
        if want_png:
            plot_metric_png(data, metric, ylabel, zero_line, args.output_dir)
            plot_metric_bar_png(data, metric, ylabel, zero_line, args.output_dir)
        if want_html:
            plot_metric_html(data, metric, ylabel, zero_line, args.output_dir)

    if want_png:
        plot_inference_time_png(data, args.output_dir)
        plot_inference_time_bar_png(data, args.output_dir)
    if want_html:
        plot_inference_time_html(data, args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
