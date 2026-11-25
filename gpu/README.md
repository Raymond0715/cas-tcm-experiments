# Paired GPU convolution experiment

Figure 6 compares two direct FP32 NCHW convolution kernels. The `canonical` kernel traverses kernel rows and columns and broadcasts inputs across output channels in dense convolution. The `reuse` kernel (CAS-GPU in the paper) traverses input rows and uses each loaded input vector to update every applicable output-row accumulator in its tile. Both kernels use four warps per block and the same per-output FMA order.

The 11 workloads in [cases.json](cases.json) run at both tile sizes in [configs.json](configs.json). The tile dimensions are output rows × output columns × output channels. All workloads use batch size one. The dense identifiers encode output side length and input channel count. The MobileNetV3-Large identifiers encode the bottleneck and operator.

Run the commands below from this `gpu/` directory.

## Reproduce the TCM values on a CPU

```sh
python -m pip install numpy==1.25.1 numba==0.61.2
python model.py
cmp model_results.json reference/model_results.json
```

This uses NumPy and Numba. Each block starts with independent, empty states for the input and output-update streams. For each step of the kernel loops, the model serializes valid operations in output row, output channel, then output column order. It counts and reports input and output TCM separately. Their sum gives the horizontal coordinate in Figure 6 after normalization to the canonical value for the same case and tile.

## Reproduce the timings on a CUDA GPU

The recorded run used an NVIDIA GeForce RTX 4070, Python 3.11.4, PyTorch 2.5.0+cu124 and Triton 3.5.0. On Linux, PyTorch 2.5.0 declares a dependency on Triton 3.1.0. To match the experiment, use a dedicated environment with the CUDA 12.4 build of PyTorch 2.5.0 and its dependencies installed first, then run `python -m pip install --no-deps triton==3.5.0`. The kernels invoke Triton directly. [Recorded environment and source hashes](reference/environment.json)

```sh
python run.py --mode check
python run.py --mode full
```

The first command validates the two schedules on five small cases at both tile sizes. It compares instrumented GPU input and output-update addresses with the model and checks both instrumented and production outputs against an FP64 CPU convolution rounded to FP32. The tolerances are `atol=2e-5` and `rtol=2e-4`. The saved [address traces](reference/captures/) are indexed by block, schedule step and consumer slot. Each NumPy archive has `input` and `output` arrays of element addresses, with `-1` for inactive slots. [checks.json](reference/checks.json) records the 20 checks.

The second command checks bitwise equality of each production pair and repeats the CPU reference checks on all 11 benchmark workloads. It runs 20 eager warmups per variant, captures 50 launches per CUDA Graph and warms up each graph with three replays. Each of 15 rounds times one replay per variant in randomized order using CUDA events. The reported median is taken over the 15 mean times per launch. The timing excludes compilation, graph capture, allocation and host–device copies. GPU clocks were unlocked.

Both commands print a new directory under `runs/`. The full run writes `results.json` there. For a shorter timing run after the checks, use `python run.py --mode full --case dense_s14_ci8 --config y4x8c4`.

## Read or plot Figure 6

```sh
python -m pip install matplotlib==3.8.3
python plot_figure6.py
```

The plotting script uses Matplotlib and reads the [recorded timings and TCM counts](reference/results.json) by default. It writes the two panels under `runs/figure6/`. To plot a new full run, pass its `results.json` with `--results` and select a destination with `--output-dir`.

In [paired_results.csv](reference/paired_results.csv), `canonical_us` and `reuse_us` are median times in microseconds. The CSV columns `tcm_ratio` and `speedup` are canonical divided by CAS-GPU. Figure 6 plots the reciprocals, with normalized TCM on the horizontal axis and normalized time on the vertical axis. All 20 dense and depthwise pairs lie below one on both axes. The two pointwise pairs have equal TCM and shorter measured CAS-GPU times.
