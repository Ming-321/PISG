# Historical reference evidence

These files contain the original per-seed results reported in the paper, for comparison. They are not results of the new reproduction controller and never fill missing cells in new output CSVs.

| Directory | Historical source | Provided coverage |
|---|---|---|
| main_tsp | Five-seed runs reported in Table 1 | Tg0 cost/gap for four solvers and four scales |
| main_mis | Five-seed runs reported in Table 2; DIFUSCO rows from the rerun with the training-time input encoding | Tg0 size/gap for the reported solver/distribution pairs |
| deployment | Five-seed runs reported in Appendix Table 11 | Final gap for four solver/scale pairs |
| gensco_tsp | Original GenSCO robustness five-seed run | All 12 scale/cycles/2-opt pairs, cost/gap |
| gensco_mis | Five-seed runs reported in Appendix Table 5; RB baseline rows from the unguided run at the RB PISG flip rate | All six distribution/cycles pairs, size |
| fig_signal | Original FastT2T signal analysis underlying Figure 3 | TSP per-seed rank means; MIS per-seed and pooled means at threshold0.3 |

Table directories provide `per_seed.csv` and `bands.csv`. Bands are the minimum/maximum and mean across the original five seeds for each method and for paired improvements. They are engineering references, not confidence intervals. Gaps retain the precision supplied by the source CSVs (the DIFUSCO TSP references round to four decimal places); new results may have more digits. Original runtime fields are omitted because timing is not a reproduction target.

Main-table Tg5 cells are covered by the full scripts but are not included in these per-seed reference files. Deployment references only report gap. GenSCO MIS references only report size. Empty metric fields reflect these evidence limits rather than inferred or fabricated numbers.

Figure historical aggregation is available with `reproduce/fig_signal/run.sh --from-reference --output <directory>`; its output explicitly records historical provenance. Figure values use the original probability-space computation, with TSP seed-level and MIS graph-level aggregation as described in its README.
