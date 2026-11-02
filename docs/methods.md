# PISG implementation and preserved settings

PISG evaluates a normal denoising branch and a branch with selected internal gates/attention perturbed, then extrapolates their difference. No model retraining or extra checkpoint is required. The native solver's sampling, decoding, and optional rewriting remain in place.

| Solver | Main implementation locations | Published default used here |
|---|---|---|
| DIFUSCO | `difusco/models/gnn_encoder.py`, `pl_tsp_model.py`, `pl_mis_model.py` | scale2, layer7; adaptive/linear for TSP, uniform/constant for MIS; logit fusion; training-time input encoding for baseline and PISG |
| T2T | `diffusion/models/gnn_encoder.py`, native TSP/MIS models | scale2, layer7; adaptive/linear for TSP, uniform/constant for MIS; logit fusion |
| FastT2T | `diffusion/models/gnn_encoder.py`, `diffusion/consistency/{tsp,mis}.py` | scale2, layer7; adaptive/linear for TSP, uniform/constant for ER, adaptive/constant for RB; logit fusion |
| StruDiCO | `diffusion/models/gnn_encoder.py`, `diffusion/consistency/{tsp,mis}.py` | TSP scale0.1/layer7/adaptive/linear; ER0.1/layer9/uniform/constant; RB0.2/layer5/adaptive/constant; logit fusion |
| GenSCO | patched model, transformer, attention, and flow-decoder modules | frozen hybrid TSP settings and selected MIS JSONs |

Layer numbers are used as given, without shifting. For the four GNN solvers, the paper's layer labels (e.g., L7) and the `pisg_layers` values in configs and CLIs are the same 0-based indices into the 12-layer encoder (paper L7 = `--pisg_layers 7`; valid range 0–11). GenSCO's `pisg_layers` are 1-based decoder-layer indices, matching the paper's GenSCO appendix.

The native CLIs document PISG flags. Selected scripts pass explicit inference budgets, checkpoint/data paths, seeds, and guidance settings. The original source supports additional historical research options, but this reproduction package schedules only the results listed in `reproduce/README.md`.

StruDiCO retains its threshold merge decoder and stitch seed. Its main-table TSP500/1000 baseline and PISG decoder parameters follow the source experiment, including their differences. Deployment likewise preserves the recorded decoder settings: baseline `m1=m2=0.7` for TSP500 and `0.8` for TSP1000; PISG `m1=m2=0.6` for both sizes. Both settings use 2-opt at initial decoding and rewriting. FastT2T deployment likewise uses native 2-opt throughout. No batch-size or search-budget reduction is made for shared-GPU operation.

GenSCO TSP uses hybrid scale0.5/rho0.5/layer3/linear with aggressive attention perturbation/rho0.5. MIS uses `gensco_mis/er.json` and `rb.json`, retaining their test-subset-selected identity, original flip rates, guidance schedules and layers. These configurations raise GenSCO's search flip rate; the unguided baseline was evaluated at both the default and the PISG flip rate, and the table uses the stronger one (default for ER, `rb_baseline.json` for RB).

The Figure 3 diagnostic reads probabilities saved by FastT2T, whereas guidance is fused in logits. Its CSV explicitly records `signal_space=probability`, matching the paper's probability-space definition of the diagnostic; see the figure README for aggregation.
