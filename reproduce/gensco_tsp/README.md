# GenSCO TSP robustness table

Appendix Table 4 in the paper. Use the [specified upstream and patch](../../solvers/gensco/README.md). Full coverage: TSP500/1000 × cycles10/20/40 × 2-opt0/20, with baseline and PISG.

```bash
bash reproduce/gensco_tsp/run.sh --mode sample --run-id gensco-tsp-sample
bash reproduce/gensco_tsp/run.sh --mode full --run-id gensco-tsp-full
```

The sample is TSP500/cycles10/2-opt0. All cells preserve sampling_steps4, search_runs8, batch16, and the scale-specific native budgets in `configs.json`. PISG uses hybrid scale0.5/rho0.5/layer3/linear, attention perturbation aggressive/rho0.5. CSVs contain cost, gap (%), paired gap improvements and wins. [Common setup and status rules](../README.md) apply.
