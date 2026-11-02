# GenSCO MIS scaling table

Appendix Table 5 in the paper. Full coverage: ER/RB × cycles500/1000/2000. Apply the [GenSCO patch](../../solvers/gensco/README.md) first. `_common/gensco_mis.py` invokes the native flow decoder with the frozen configuration; it does not search for new parameters.

```bash
bash reproduce/gensco_mis/run.sh --mode sample --run-id gensco-mis-sample
bash reproduce/gensco_mis/run.sh --mode full --run-id gensco-mis-full
```

The sample selects ER/cycles500. `er.json` preserves `cfg-b7b7b8c7b5205b897ce2`; `rb.json` preserves `cfg-f32fe47a2da5f08df986`. These settings were selected on a test subset, so retain the paper's **test-subset-selected** qualification. Their flip rates and guidance/attention settings are part of the frozen protocol, not a newly tuned reproduction.

The frozen PISG configurations also raise GenSCO's search flip rate from the default `[0.25,0.4]` (ER `[0.38777864,0.5399507]`, RB `[0.35,0.5]`). The unguided baseline was evaluated at both flip rates, and the table compares against the stronger one: `baseline.json` (default flip rate) for ER and `rb_baseline.json` (RB PISG flip rate) for RB, at all budgets. Both baselines set the guidance scale to 0.

CSV `cost` is independent-set size (higher is better). Paired improvement is PISG size minus reference size; wins count positive differences. Gap is left empty where native evidence does not report it. [Common setup and status rules](../README.md) apply.
