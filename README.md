# Continuity-Aware Schedule and Traversal Continuity Metric

Experimental code and reference data for Figures 5–7 of *Sequential Memory Access Characterization and Schedule Optimization for Deep Neural Network Inference*.

| Figure | Experiment | Where to start |
| --- | --- | --- |
| 5 | Compute input and output-update TCM for the canonical schedule and CAS across 32 convolution workloads. | [CPU calculator and reference counts](tcm/README.md) |
| 6 | Compare canonical and CAS-GPU kernels on 11 workloads at two output-tile sizes. Each pair has a modeled TCM and measured execution times. | [Triton kernels, timing scripts and reference results](gpu/README.md) |
| 7 | Compare total and global-load warp instructions per launch for both kernels on the 11 workloads with the 4 × 8 × 4 tile. | [Nsight Compute collector and counter samples](gpu/profiling/README.md) |

Figure 5 uses the paper's matched-capacity schedule model. Figure 6 follows the GPU kernel loops and initializes the input and output residency state independently in each thread block. In the GPU scripts and data files, `reuse` denotes CAS-GPU.

The reference data can be inspected on a CPU. A CUDA-capable NVIDIA GPU is needed to repeat the timing measurements, and Nsight Compute with access to GPU performance counters is needed to collect new instruction counts. The paper's measurements used a GeForce RTX 4070.
