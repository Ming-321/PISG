# DIFUSCO

Run from the repository root:

```bash
bash solvers/difusco/install.sh
```

This creates the `difusco` Conda environment and compiles the native Cython merge extension. It requires a C/C++ compiler. If the environment already exists, review `environment.yml` and build the extension with `conda run -n difusco python setup.py build_ext --inplace` from `solvers/difusco/difusco/utils/cython_merge/`; the install script does not delete existing environments.

The supplied versions use Python 3.10/PyTorch 2.6/CUDA 12.4 and require compatible GPU drivers. The four PyTorch recipes differ only in environment name. Their common dependencies were installed in a fresh isolated environment; all four solvers' native extensions built and CLI help checks passed from a standalone source export, and `pip check` passed. Minimal CUDA operations using Torch, torch-scatter and torch-sparse also passed. These installation checks do not constitute model quality reproduction; the selected quality runs use their recorded evaluation environments.

Prepare data and checkpoints under `resources/difusco/` following [resource instructions](../../docs/resources.md). The table controller can use other locations via `reproduce/local.json`, and automatically invokes this environment.

```bash
bash examples/difusco_tsp.sh
bash reproduce/main_tsp/run.sh --mode sample --only difusco_tsp100_ts100_tg0 --run-id difusco-sample
```

Native source lives in `difusco/`. The original MIT [LICENSE](LICENSE) is retained. Upstream identity and PISG changes are recorded in [sources](../../docs/sources.md) and [methods](../../docs/methods.md).
