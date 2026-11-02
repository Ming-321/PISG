# Main MIS table

Table 2 in the paper. Four solvers evaluate ER700–800; FastT2T and StruDiCO also evaluate RB200–300. Ts=100/50/5/5 respectively. Tg0 is included for all; Tg5 only for FastT2T and StruDiCO.

```bash
bash reproduce/main_mis/run.sh --mode sample --run-id mis-sample
bash reproduce/main_mis/run.sh --mode full --run-id mis-full
```

The sample covers all four ER configurations and FastT2T RB, Tg0. Outputs contain independent-set size (`cost`), gap (%), paired size improvement, and wins. FastT2T's native MIS gap sign is converted to the paper's lower-is-better convention; its original metric is retained in each run's `metrics.json`. Other native metrics are preserved. The gap here is each solver's native metric against its own reference values, whereas Table 2 reports gap against a common reference (41.35 on ER, 20.09 on RB), so gaps can differ slightly from the table; compare independent-set sizes and paired size improvements instead. [Common setup and status rules](../README.md) apply.
