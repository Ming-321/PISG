# Figure 3 signal characterization

Figure 3 in the paper, both panels:

- Panel a: FastT2T TSP100, mean normal-minus-perturbed signal by edge rank (ranks 0/1/2/rest), steps 0 and 3.
- Panel b: FastT2T MIS-ER, mean endpoint signal separation on conflict/nonconflict edges at threshold0.3, steps 0 and 4.

Full reproduction captures the native tensors for seeds41–45 and automatically generates `results.csv` and `per_seed.csv`:

```bash
bash reproduce/run.sh --table fig_signal --mode full --run-id signal
# Equivalent direct entry:
bash reproduce/fig_signal/run.sh --output outputs/signal/fig_signal
# Check the supplied historical CSV aggregation only; no GPU experiment.
bash reproduce/fig_signal/run.sh --output outputs/signal-reference/fig_signal --from-reference
```

The root sample mode never launches these captures. The historical entry labels every row `source=historical_reference`; the full entry labels rows `source=new_run`. Captures and commands are saved per task/seed. Unfinished captures are retained and require a new output directory rather than being silently reused.

The historical analysis and this script measure the difference between **saved probability tensors**, as recorded in `signal_space`. The paper defines the diagnostic in probability space accordingly. Guidance itself is fused in logit space; the plotted diagnostic is a probability-space difference. The script preserves the computation that produced the current figure.

TSP computes node-level incident-edge ranks, excludes the diagonal, averages within seed, then reports the five-seed mean and population standard deviation (rounded to six decimals as in the original CSV). MIS first averages endpoint separation within each graph, excludes graphs with no conflict edges from both group means, and pools those graph-level values over seeds. Its standard deviation is across graph-level conflict means, not seed means. Ratios are conflict mean divided by nonconflict mean. Rank0 corresponds to the figure's 1st-ranked edge. Only the selected panel rows are emitted; no threshold sweep or additional mechanism experiments are scheduled.
