# Deployment with complete post-processing

Appendix Table 11 in the paper. FastT2T and StruDiCO on TSP500/1000, Ts5/Tg5. Both initial decoding and each rewrite stage use native 2-opt until convergence, with a cap of 1000 iterations. This differs from the main TSP table, which disables 2-opt.

```bash
bash reproduce/deployment/run.sh --mode sample --run-id deployment-sample
bash reproduce/deployment/run.sh --mode full --run-id deployment-full
```

The sample selects FastT2T500 and StruDiCO1000. StruDiCO retains the deployment experiment's native decoder parameters: baseline `m1=m2=0.7` for TSP500 and `0.8` for TSP1000; PISG `m1=m2=0.6` for both sizes. CSVs report final cost/gap, paired final-gap improvements and wins; initial-stage metrics are also retained when provided. Timing is not a validation target. [Common setup and status rules](../README.md) apply.
