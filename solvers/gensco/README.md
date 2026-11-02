# GenSCO patch installation

This directory contains a PISG patch, not redistributed GenSCO source. Clone [GenSCO upstream](https://github.com/Thinklab-SJTU/GenSCO) at `6edb48ce34f03dc5f2fa36c13f45e628a724d630` and apply the patch:

```bash
# From this repository's root: create the separate environment, clone,
# apply the patch and build the native decoder library.
bash solvers/gensco/install.sh
```

This uses Python3.11/JAX0.5.0 with CUDA12, Flax0.10.4, and the native Triton dependencies. Torch2.6 CPU is used for upstream checkpoint support, independently of the four PyTorch GPU environments. A C++ compiler and make are required. `environment.yml` records the runtime dependency versions. A fresh isolated installation was checked through patch application, native library build, imports, TSP/MIS CLI help and `pip check`; a minimal JAX JIT operation on CUDA also passed. These installation checks do not constitute GPU quality reproduction; the selected quality runs use their recorded evaluation environment.

If the environment is already prepared, use:

```bash
bash solvers/gensco/apply.sh
conda run --no-capture-output -n gensco make -C solvers/gensco/upstream/lib
```

`apply.sh` creates `upstream/` and refuses to overwrite an existing directory. It checks the patch before applying it. An optional argument selects another empty clone destination; then set `source.gensco` in `reproduce/local.json` to its absolute path. Do not apply this patch on a different upstream revision.

Eight files are patched: the two model modules, attention and encoder modules, continuous sampling, decoder utilities, and the TSP/MIS decoder entries. The selected MIS configuration wrapper is maintained under `reproduce/_common/`. The patch preserves frozen test-subset-selected settings; see [MIS reproduction](../../reproduce/gensco_mis/README.md).

Prepare `datasets/*.npz` and `ckpts/*.ckpt` under `resources/gensco/` as described in [resources](../../docs/resources.md). Then run:

```bash
bash examples/gensco_tsp.sh
bash reproduce/gensco_mis/run.sh --mode sample --run-id gensco-mis
```

The upstream repository does not provide a repository-level license in the specified snapshot. Consult its authors for upstream permissions as needed. Upstream includes separate third-party notices (including array-view CC BY-NC-SA4.0 and jax-triton Apache2.0); cloning retains these. The root MIT license applies to our additions and does not grant rights to all upstream code. [Source manifest](../../docs/source-manifest.json) records the upstream commit and the patch hash.
