# Figure 7 · GPU instruction counts

The Nsight Compute collector measures total warp instructions (`smsp__inst_executed.sum`) and global-load warp instructions (`smsp__sass_inst_executed_op_global_ld.sum`) per kernel launch. Figure 7 uses `config=y4x8c4` in [the counter table](reference/figure_instruction_values.csv). These 22 rows cover 11 workloads and both variants. The same table also contains the companion 4 × 16 × 4 measurements.

## Check the published values on a CPU

From this directory, run the original summarization on a copy of the archived samples. It requires only Python's standard library.

```sh
mkdir -p summary
cp reference/manifest.json reference/measurements.json summary/
python collect_instructions.py --summarize summary
cmp summary/figure_instruction_values.csv reference/figure_instruction_values.csv
```

The CSV reports the median of seven measured launches after two diagnostic launches. For example, for the pointwise workload at the 4 × 8 × 4 tile, canonical and CAS-GPU execute 16,018,688 and 14,513,408 total warp instructions per launch. Both execute 2,007,040 global-load warp instructions. This collection used Nsight Compute 2026.4.0.0.

## Collect new counters

Run the [GPU timing experiment](../README.md) first. From the `gpu/` directory, replace `FULL_RUN_ID` below with the timestamp printed by `run.py --mode full`.

```sh
python profiling/collect_instructions.py --archive runs/FULL_RUN_ID
```

Run the profiler in the same Python environment and on the same GPU as that timing run, with `ncu` on the executable search path and access to hardware counters. To try one pair, append `--limit-pairs 1`. The collector checks the compiled kernel against the timing run, warms up each kernel with 20 launches before each profiled launch, and uses application replay with cache flushing and clock control disabled. Results appear under `profiling/instruction_collections/` in a timestamped directory.
