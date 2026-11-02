# Source provenance and attribution

`solvers/gensco/tsp_convert_fmt.py` is our TSP resource converter, copied from our GenSCO research code (`data/tsp_convert_fmt.py`). It is provided separately under the root MIT license; it is not part of the pinned upstream source or the eight-file model/decoder patch.

The source manifest records the upstream commits, the SHA256 hashes of the copied files as used in our experiments, and the exact GenSCO patch content. The four MIT solvers retain their native organization and original license files. PISG infrastructure, documentation, and configuration additions use the root MIT license.

| Solver | Original project | Upstream baseline |
|---|---|---|
| DIFUSCO | [Edward-Sun/DIFUSCO](https://github.com/Edward-Sun/DIFUSCO) | `65eb3b7bde76097c0d3438ce743fa6769baf9ce5` |
| T2T | [Thinklab-SJTU/T2TCO](https://github.com/Thinklab-SJTU/T2TCO) | `ad41945178d64d9e44fc7a93114a2bcf1ae8630f` |
| FastT2T | [Thinklab-SJTU/Fast-T2T](https://github.com/Thinklab-SJTU/Fast-T2T) | `26f9bfb78561cb23c97d81e92c622e5e3beb2507` |
| StruDiCO | [yuuuuwang/StruDiCO](https://github.com/yuuuuwang/StruDiCO) | `e2270d6caf8c7c8cd6393590be6510c59eb42768` |
| GenSCO | [Thinklab-SJTU/GenSCO](https://github.com/Thinklab-SJTU/GenSCO) | `6edb48ce34f03dc5f2fa36c13f45e628a724d630` |

Our StruDiCO research code imported the original code into a separate Git history. All 25 imported native source files match the specified upstream commit byte-for-byte; this content comparison establishes attribution without a shared Git ancestry.

Two copied files postdate the listed baselines: FastT2T's `LICENSE` was added upstream in `12be8e35e67ad9337e16aa718f3c2988a01beb40`, and StruDiCO's `environment.yml` in `43639f50a92722746805c3cbde8c9687b518a660`. Their source-manifest entries record these commits as `upstream_commit`.

Copied content includes native Python/Cython source and solver license files. Historical experiments, internal notes, old logs, generated C, data, and weights are not redistributed. Environment YAMLs are replaced with the dependency versions used for validation; shell installers and table controllers are added. Native algorithm implementations remain those used for the paper's experiments. The Figure 3 aggregation copies the selected rank/CMA computations from the FastT2T research analysis (same MIT source), without the additional appendix diagnostics.

This repository adds one integration fix to StruDiCO's `train.py`: the checkpoint callback uses the project name when the optional W&B logger name is omitted. This prevents evaluation startup from passing `None` to `os.path.join`; it does not change model, decoder or evaluation parameters. The source manifest retains the original file hash; the file manifest records the delivered version.

DIFUSCO's `pl_mis_model.py` and `train.py` add one correction: an MIS input-encoding option (`--mis_categorical_input_encoding`, default `training`). The official inference code feeds the categorical denoiser raw `{0,1}` states, while training uses the jittered `2x-1` encoding; the research MIS baseline kept the raw input whereas its PISG branches used the training encoding. The baseline and both PISG branches now share the selected encoding, matching the existing TSP path. The source manifest keeps the original hashes; the file manifest records the delivered files.

FastT2T's `train.py` changes only CLI help text, so that its recommended settings match the paper (scale 2.0 for MIS; `adaptive` for MIS-RB). Argument defaults and behavior are unchanged. The source manifest retains the original file hash; the file manifest records the delivered version.

The four PyTorch environments pin pip24.0 because Lightning1.8 contains legacy dependency metadata (`torch>=1.9.*`). This pip version is used in the validated FastT2T environment; the other existing environments retain an already installed Lightning package while using newer pip. A clean installation must keep the supplied pip version to avoid rejection of the legacy metadata.

Please cite PISG using `CITATION.cff` and the original solver publications in [citations.bib](citations.bib). These five entries retain the paper bibliography's titles/authors/years and add each original project's official OpenReview URL. The source URLs above are the authors' original projects, not our development forks. GenSCO is delivered only as a patch on reader-fetched upstream source; its upstream and third-party permissions remain separate from this repository's MIT license.
