# Main TSP table

Table 1 in the paper. `configs.json` specifies all four solvers on TSP50/100/500/1000: DIFUSCO Ts100, T2T Ts50, FastT2T/StruDiCO Ts5; Tg0 for all and Tg5 for FastT2T/StruDiCO. This table disables 2-opt. Each configuration runs baseline and PISG on the same seeds.

```bash
bash reproduce/main_tsp/run.sh --mode sample --run-id tsp-sample
bash reproduce/main_tsp/run.sh --mode full --run-id tsp-full
```

The sample selects DIFUSCO100, T2T100, FastT2T500, StruDiCO1000, all Tg0. Outputs: `results.csv` with cost, gap (%), paired gap improvement and wins; `per_seed.csv` and native logs. [Common setup and status rules](../README.md) apply.

Keep StruDiCO's native merge decoder and its historical scale-specific `m1/m2` settings. The table's TSP500/1000 baseline and PISG settings differ as recorded in `configs.json`; they are not an ablation with every decoder setting held constant. Use the deployment table for its separate frozen post-processing protocol.
