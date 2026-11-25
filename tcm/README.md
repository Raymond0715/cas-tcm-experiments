# Figure 5 · Traversal Continuity Metric

Figure 5 compares two convolution schedules across 32 workloads. In the first row (8 input channels, 64 output channels, output side 14, kernel side 3), the canonical schedule counts 300,458 input discontinuities and 35,839 output-update discontinuities. CAS counts 61,831 and 37,503, respectively. The totals plotted in Figure 5(a) are 336,297 and 99,334. [Full counts and tile dimensions](reference/results.csv)

## Reproduce the counts

Run these commands from the `tcm/` directory. We verified the calculation on a CPU with Python 3.11.4, NumPy 1.25.1 and Numba 0.61.2. Matplotlib is required only for plotting.

```sh
python -m pip install numpy==1.25.1 numba==0.61.2
python recompute.py
python -c "import json; assert json.load(open('results.json')) == json.load(open('reference/results.json'))"
```

The calculator first checks five small workloads under both schedules. It compares each TCM count against an independently sorted trace, verifies that each multiply–accumulate occurs once, and compares the convolution outputs with a direct numerical reference. It then evaluates the full workloads and writes the per-operand counts and totals beside the script in `results.csv` and `results.json`. The final command exits silently when the regenerated data match every archived row. Run just the built-in checks with `python -c "from recompute import validate; print(validate())"`.

## Read the data

Each row represents a valid square convolution with unit stride, unit dilation, zero padding and row-major NCHW addresses. Both schedules clip the last tile to the valid domain. Input addresses and output-update addresses have independent residency states and unit event weights. Their discontinuity counts sum to each plotted total.

| Figure 5 panels | Rows (`group`) | Workloads |
| --- | --- | --- |
| (a)–(d) | `sample` | 8 input and 64 output channels. Each kernel side (3, 5, 7, 11) is evaluated at output sides 14, 28, 42 and 56. |
| (e) | `in_f` | Input channels range from 32 to 256 in steps of 32, at 64 output channels, output side 56 and kernel side 3. |
| (f) | `of` | Output channels range from 32 to 256 in steps of 32, at 8 input channels, output side 56 and kernel side 3. |

The two Figure 5 series use the following columns.

| Figure 5 series | Input | Output updates | Plotted total | Tile dimensions |
| --- | --- | --- | --- | --- |
| Canonical (matched entries) | `matched_input` | `matched_output` | `matched` | 64 output channels × `8K` output rows × `K` output columns |
| CAS | `cas_input` | `cas_output` | `cas` | 64 output channels × 8 input channels × `K` kernel rows × `K` kernel columns |

Here `K` is the kernel side. Both schedules have `512 × K²` nominal resident entries per operand. The `matched_ratio` column divides the canonical total by the CAS total and ranges from 1.18 to 6.11. This ratio compares address discontinuity counts. The archived `canonical_*`, `canonical` and `ratio` columns record an additional canonical tiling evaluated by the same script.

## Plot the counts

```sh
python -m pip install matplotlib==3.8.3
python plot_figure5.py --output-dir figure5
```

The plotter reads `reference/results.json` by default. To use a fresh run, add `--input results.json`. The output directory contains six panels, the shared legend and `single_row_preview.png`. The plots reproduce the two numerical series. The published panels include manual layout edits.
