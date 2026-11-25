"""Plot the two TCM series in Figure 5 for the 32 published workloads."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter


ROOT = Path(__file__).resolve().parent
TEXT_WIDTH = 522 * 72 / 72.27  # TeX points -> PDF points.
PANEL_WIDTH = 0.16 * TEXT_WIDTH
PANEL_HEIGHT = 110
LEGEND_HEIGHT = 18
CAPTION_HEIGHT = 14
SERIES = (
    ("matched", "Canonical (matched entries)", "s", ":", "#009E73"),
    ("cas", "CAS", "o", "-", "#D55E00"),
)


def cases(rows):
    result = [
        (f"fig_5_{i}_nq_{k}x{k}", [r for r in rows if r["group"] == "sample" and r["K"] == k],
         "S", "Output size $S$", f"Conv ${k} \\times {k}$")
        for i, k in enumerate((3, 5, 7, 11), 1)
    ]
    result.extend((
        ("fig_5_5_nq_in_f", [r for r in rows if r["group"] == "in_f"], "IF", "Input channels", "Conv $3 \\times 3$"),
        ("fig_5_6_nq_of", [r for r in rows if r["group"] == "of"], "OF", "Output channels", "Conv $3 \\times 3$"),
    ))
    return result


def draw(fig, case, index, origin=(0, 0)):
    _, rows, key, xlabel, _ = case
    width, height = fig.get_size_inches() * 72
    left, bottom = origin
    ax = fig.add_axes([(left + 25) / width, (bottom + 25) / height,
                       (PANEL_WIDTH - 28) / width, (PANEL_HEIGHT - 38) / height])
    x = [r[key] for r in rows]
    for variant, label, marker, linestyle, color in SERIES:
        values = [r[variant] for r in rows]
        line, = ax.plot(x, values, label=label, marker=marker, linestyle=linestyle,
                        color=color, linewidth=0.85, markersize=2.8,
                        markerfacecolor="white" if variant == "matched" else color,
                        markeredgewidth=0.65)
        assert np.array_equal(line.get_xdata(), x)
        assert np.array_equal(line.get_ydata(), values)
    upper = max(r[v] for r in rows for v, *_ in SERIES) * 1.07
    ticks = MaxNLocator(nbins=3, min_n_ticks=3).tick_values(0, upper)
    ax.set_ylim(0, ticks[-1])
    ax.set_yticks(ticks)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    formatter.set_useOffset(False)
    ax.yaxis.set_major_formatter(formatter)
    ax.set_xticks(x if key == "S" else [32, 128, 256])
    ax.margins(x=0.075)
    ax.set_xlabel(xlabel, labelpad=2, fontsize=7.2)
    if index == 0:
        ax.set_ylabel("TCM (count)", labelpad=2, fontsize=7.2)
    ax.tick_params(labelsize=7.2, length=2, width=0.6, pad=1.6)
    ax.yaxis.get_offset_text().set_fontsize(7.2)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)
    return ax


def add_legend(fig, anchor=(0.5, 0.5)):
    handles = [Line2D([], [], color=color, marker=marker, linestyle=style,
                      linewidth=0.9, markersize=3.5,
                      markerfacecolor="white" if key == "matched" else color,
                      markeredgewidth=0.7)
               for key, _, marker, style, color in SERIES]
    return fig.legend(handles, [entry[1] for entry in SERIES], loc="center",
                      bbox_to_anchor=anchor, ncol=len(SERIES), frameon=False, fontsize=8,
                      handlelength=2.1, handletextpad=0.5, columnspacing=1.8)


def check_text_bounds(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        text = [ax.xaxis.label, ax.yaxis.label, ax.yaxis.get_offset_text(),
                *ax.get_xticklabels(), *ax.get_yticklabels()]
        for item in text:
            if item.get_visible() and item.get_text():
                box = item.get_window_extent(renderer)
                assert box.x0 >= -0.5 and box.y0 >= -0.5, (item.get_text(), box)
                assert box.x1 <= fig.bbox.width + 0.5 and box.y1 <= fig.bbox.height + 0.5, (item.get_text(), box)


def save_assets(fig, output, name):
    check_text_bounds(fig)
    fig.savefig(output / (name + ".pdf"))
    fig.savefig(output / (name + ".svg"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "reference" / "results.json",
                        help="JSON rows from recompute.py (defaults to archived results)")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 7.2, "font.family": "DejaVu Sans",
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none", "svg.hashsalt": "prl-tcm-single-row"})
    data = args.input.read_bytes()
    rows = json.loads(data)
    groups = cases(rows)
    assert sum(len(case[1]) for case in groups) == len(rows) == 32
    for i, case in enumerate(groups):
        fig = plt.figure(figsize=(PANEL_WIDTH / 72, PANEL_HEIGHT / 72))
        draw(fig, case, i)
        save_assets(fig, args.output_dir, case[0])
        plt.close(fig)
    legend = plt.figure(figsize=(TEXT_WIDTH / 72, LEGEND_HEIGHT / 72))
    add_legend(legend)
    save_assets(legend, args.output_dir, "fig_5_legend")
    plt.close(legend)

    # Standalone figure preview only: the manuscript retains its LaTeX subcaptions.
    height = PANEL_HEIGHT + LEGEND_HEIGHT + CAPTION_HEIGHT
    preview = plt.figure(figsize=(TEXT_WIDTH / 72, height / 72))
    gap = (TEXT_WIDTH - 6 * PANEL_WIDTH) / 5
    for i, case in enumerate(groups):
        left = i * (PANEL_WIDTH + gap)
        draw(preview, case, i, (left, CAPTION_HEIGHT))
        preview.text((left + PANEL_WIDTH / 2) / TEXT_WIDTH, 5 / height,
                     f"({chr(97 + i)}) {case[4]}", ha="center", fontsize=7.2)
    add_legend(preview, (0.5, (height - LEGEND_HEIGHT / 2) / height))
    check_text_bounds(preview)
    preview.savefig(args.output_dir / "single_row_preview.png", dpi=240)
    plt.close(preview)
    report = {"data_sha256": hashlib.sha256(data).hexdigest(), "workloads": len(rows),
              "series": len(SERIES), "plotted_values": len(SERIES) * len(rows), "tick_and_axis_font_pt": 7.2,
              "legend_font_pt": 8, "panel_width_pdf_pt": PANEL_WIDTH,
              "panel_height_pdf_pt": PANEL_HEIGHT,
              "checked": ["all plotted coordinates match corrected data", "text stays inside exported panels"]}
    (args.output_dir / "layout_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
