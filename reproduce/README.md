# Reproducing the selected paper results

Table and figure numbers refer to the NeurIPS 2026 camera-ready version of the paper.

| Directory | Paper result | Full configuration coverage |
|---|---|---|
| [main_tsp](main_tsp/README.md) | Table 1 | Four solvers, TSP50/100/500/1000; reported Ts/Tg budgets |
| [main_mis](main_mis/README.md) | Table 2 | Four solvers on ER; FastT2T/StruDiCO on RB; reported Ts/Tg budgets |
| [fig_signal](fig_signal/README.md) | Figure 3 | TSP rank profile and MIS conflict separation |
| [gensco_tsp](gensco_tsp/README.md) | Appendix Table 4 | TSP500/1000, cycles 10/20/40, 2-opt 0/20 |
| [gensco_mis](gensco_mis/README.md) | Appendix Table 5 | ER/RB, cycles 500/1000/2000, frozen selected settings |
| [deployment](deployment/README.md) | Appendix Table 11 | FastT2T/StruDiCO, TSP500/1000, Ts=5/Tg=5, full 2-opt |

Figures 1, 4, 5 and Appendix Tables 10 and 14 are excluded. No additional appendix analyses, parameter searches, or runtime benchmarks are scheduled.

Install each solver via `solvers/<solver>/README.md` and prepare [resources](../docs/resources.md). By default the controller uses Conda environments named `difusco`, `t2t`, `fastt2t`, `strudico`, and `gensco`; resources are under `resources/<solver>/`. For other locations, copy `local.example.json` to `local.json` and edit it. Paths in that file are interpreted relative to the shell's current directory, so absolute resource/interpreter paths are recommended. `local.json` is Git-ignored.

```bash
bash reproduce/run.sh --mode sample --run-id sample
bash reproduce/main_tsp/run.sh --mode full --run-id tsp-full
# A subset; IDs are listed in each configs.json.
bash reproduce/main_tsp/run.sh --mode sample --only fastt2t_tsp500_ts5_tg0 --seeds 41 --run-id example
```

`sample` and `full` use the same inference configuration. The sample selects 13 paired configurations and seeds 41, 42, 43 (78 solver calls); full uses every selected table configuration and seeds 41–45. Every call retains the original full evaluation set, checkpoint, decoder, batch size, sampling count, and search budget. The sample does not launch Figure 3 GPU captures. Figure 3 has its own full and historical-CSV entrances.

| Sample group | Paired configurations |
|---|---|
| Main TSP | DIFUSCO100/Ts100, T2T100/Ts50, FastT2T500/Ts5, StruDiCO1000/Ts5; Tg0, no 2-opt |
| Main MIS | All four solvers on ER, FastT2T on RB; Tg0 |
| Deployment | FastT2T500 and StruDiCO1000; Ts5/Tg5, 2-opt to convergence, cap1000 |
| GenSCO TSP | TSP500, cycles10, sampling_steps4, search_runs8, 2-opt0 |
| GenSCO MIS | ER, cycles500, frozen `cfg-b7b7b8c7b5205b897ce2` |

Outputs are under `outputs/<run-id>/<table>/`. `results.csv` contains method means, actual/expected run counts, quality metrics, paired improvements and wins. `per_seed.csv` contains each run's status and metrics. TSP improvement is baseline gap minus PISG gap; MIS improvement is PISG size minus baseline size. Wins require a strictly positive difference. Standard deviations are sample standard deviations across seeds for tables. Figure-specific aggregation follows its original protocol.

Failed or unparsed metrics are empty and flagged, and do not enter means. A summary reports `incomplete` until all its expected runs finish. Historical results never fill failed cells. A nonzero controller exit status indicates missing or failed runs; other configurations still execute. The same run ID resumes completed work; changed configurations or seeds require a new ID. After a concrete fix, `--retry-failed` permits one retry of affected failed/invalid runs and retains the previous attempt. Do not use retries to pick favorable seeds or retune settings.

Use cost/gap and paired differences to compare results; modest numerical differences are acceptable. `reference/<table>/bands.csv` records the original five-seed min/max bands and paired-difference bands for the provided historical cells. These bands are engineering references, not confidence intervals. Historical CSVs are reference evidence; their presence does not prove a new reproduction. Runtime is intentionally omitted from the comparison target.

After the controller exits, `python3 reproduce/_common/compare.py --output outputs/<run-id>` writes `comparison.csv`, including matching-seed reference means, five-seed bands and source rounding allowance. Incomplete rows remain pending. This numerical screen does not substitute for checking actual runtime errors and native decoder behavior. If a parsing issue is fixed, `python3 reproduce/_common/summarize.py --output outputs/<run-id>` can reparse existing exit=0 logs without another GPU call; use it only after the controller has stopped.
