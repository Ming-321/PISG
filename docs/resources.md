# Data and checkpoints

Place resources under `resources/<solver>/`, or set absolute resource roots in `reproduce/local.json`. The controller uses absolute paths and does not require copying resources into solver source directories. Resources are excluded from Git; symlinks to existing datasets/checkpoints are acceptable.

Use the original authors' acquisition instructions:

| Resource | Original acquisition entry |
|---|---|
| DIFUSCO checkpoints | [author Drive folder](https://drive.google.com/drive/folders/1IjaWtkqTAs7lwtFZ24lTRspE0h1N6sBH), linked by [DIFUSCO](https://github.com/Edward-Sun/DIFUSCO) |
| T2T TSP checkpoints | Same DIFUSCO categorical checkpoints; [T2T instructions](https://github.com/Thinklab-SJTU/T2TCO) |
| T2T MIS-ER checkpoint | [author checkpoint](https://drive.google.com/file/d/1f3tNzpANSecJsn5ClgzRQ0lX6v1TPpXn/view), linked by T2T |
| FastT2T checkpoints | [author Drive folder](https://drive.google.com/drive/folders/107mbgDsRqp-0Vf16_xDRGVFeDzeRNr-P), linked by [FastT2T](https://github.com/Thinklab-SJTU/Fast-T2T) |
| StruDiCO checkpoints | `ckpts/` in the [original StruDiCO repository](https://github.com/yuuuuwang/StruDiCO) |
| GenSCO checkpoints | [author Drive folder](https://drive.google.com/drive/folders/1hJCAvDy6AkAjslcqCnAJ08ouuNVGXRoH), linked by [GenSCO](https://github.com/Thinklab-SJTU/GenSCO) |
| TSP50/100 | [learning-tsp](https://github.com/chaitjo/learning-tsp) dataset instructions |
| TSP500/1000 | [Spider-scnu/TSP](https://github.com/Spider-scnu/TSP) datasets |
| MIS ER700–800 | [DIMES](https://github.com/DIMESTeam/DIMES) evaluation data and native solver dataset instructions |
| MIS RB200–300 | StruDiCO's existing [500 evaluation graphs](https://github.com/yuuuuwang/StruDiCO/tree/e2270d6caf8c7c8cd6393590be6510c59eb42768/data/mis/rb/rb200_300_test) and [500 reference labels](https://github.com/yuuuuwang/StruDiCO/tree/e2270d6caf8c7c8cd6393590be6510c59eb42768/data/mis/rb/rb200_300_test_label) |

The links are documented upstream acquisition entries; downloaded bytes must be checked against `resources.csv`, rather than assuming a mutable download always supplies the same snapshot. `resources.csv` records relative paths, sizes and SHA256 for the datasets/checkpoints used in this validation. It does not redistribute these files. A new random RB generation is not the same fixed evaluation set merely because it uses the same distribution.

The RB evaluation set is the existing author-provided data in [StruDiCO](https://github.com/yuuuuwang/StruDiCO) at commit `e2270d6caf8c7c8cd6393590be6510c59eb42768`. All 500 `.gpickle` graphs and 500 matching `.result` label files used here are byte-identical to that commit. Obtain `data/mis/rb/rb200_300_test/` and `data/mis/rb/rb200_300_test_label/` from that repository and place them under `resources/fastt2t/data/mis/rb/`; reuse the same directories under `resources/strudico/data/mis/rb/`. No new RB generation or separately published PISG dataset archive is needed. Check the acquired files against `resources.csv` using the command below.

Required native layouts:

```text
resources/difusco/ or resources/t2t/
  ckpts/tsp{50,100,500,1000}_categorical.ckpt
  ckpts/mis_er_categorical.ckpt
  data/tsp/tsp50_concorde_5.68759.txt
  data/tsp/tsp100_concorde_7.75585.txt
  data/tsp/tsp500_concorde_16.54581.txt
  data/tsp/tsp1000_concorde_23.11812.txt
  data/mis/er/er_700_800_test/*.gpickle
resources/fastt2t/
  checkpoint/tsp{50,100,500,1000}.ckpt
  checkpoint/mis_{er,rb}.ckpt
  data/tsp/                         # same named TSP files
  data/mis/er/er_700_800_test/*.gpickle
  data/mis/rb/rb200_300_test/*.gpickle
  data/mis/rb/rb200_300_test_label/*.result
resources/strudico/
  ckpts/tsp{50,100,500,1000}.ckpt
  ckpts/mis_{er,rb}.ckpt
  data/tsp/                         # same named TSP files
  data/mis/er/er_700_800_test/*.gpickle
  data/mis/rb/rb200_300_test/*.gpickle
  data/mis/rb/rb200_300_test_label/*.result
resources/gensco/
  ckpts/tsp{500,1000}.ckpt
  ckpts/mis_{er,rb}.ckpt
  datasets/tsp500_concorde_16.54581.npz
  datasets/tsp1000_concorde_23.11812.npz
  datasets/er_700_800_test.npz
  datasets/rb_200_300_test.npz
```

For GenSCO TSP, use our supplied `solvers/gensco/tsp_convert_fmt.py`; the pinned upstream snapshot does not contain this converter. From the repository root, after preparing the native TSP files:

```bash
mkdir -p resources/gensco/datasets
conda run -n gensco python solvers/gensco/tsp_convert_fmt.py --src_path resources/fastt2t/data/tsp/tsp500_concorde_16.54581.txt --save_path resources/gensco/datasets/tsp500_concorde_16.54581.npz
conda run -n gensco python solvers/gensco/tsp_convert_fmt.py --src_path resources/fastt2t/data/tsp/tsp1000_concorde_23.11812.txt --save_path resources/gensco/datasets/tsp1000_concorde_23.11812.npz
```

For MIS, use GenSCO's upstream `data/mis_convert_fmt.py` with `--src_path` pointing to the graph directory and `--save_path` to the corresponding NPZ path above, preserving order and graph IDs. The validated NPZ snapshots contain 128 ER graphs and 500 RB graphs. TSP50/100 use 1280 instances each; TSP500/1000 use 128 each. The four ER paths contain 128 graphs. FastT2T and StruDiCO RB use the same 500 graphs and reference labels, matching the recorded StruDiCO experiment commands. Do not substitute the partial RB folder included in some native StruDiCO snapshots. StruDiCO RB is outside the GPU sample.

The supplied converter was checked on both full TSP test sets: coordinates and tours match the validated NPZ snapshots exactly. TSP500 reference costs differ by at most `3.8147e-6` due to floating-point computation; TSP1000 costs match exactly. Thus a locally converted NPZ can differ from the recorded file hash; compare array contents when checking this known conversion difference.

Check a prepared solver's resources with the system Python:

```bash
python3 reproduce/_common/check_resources.py --solver fastt2t
# A custom resource root:
python3 reproduce/_common/check_resources.py --solver gensco --root /path/to/resources
```

Missing or changed resources are reported. Different serialization can change file hashes even if numerical content is equivalent; investigate an actual mismatch briefly rather than changing inference settings. Do not regenerate labels or select subsets to obtain favorable results.
