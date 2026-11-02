#!/usr/bin/env python3
"""Evaluate the frozen GenSCO MIS configuration on the full test set."""
import argparse
import json
import sys
from pathlib import Path

# GenSCO is cloned and patched by the reader; imports resolve from its cwd.
sys.path.insert(0, str(Path.cwd()))
import numpy as np
from flax import nnx
from decoding.mis import mis_flow_decode
from training import load_ckpt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", required=True)
parser.add_argument("--ckpt", required=True)
parser.add_argument("--config", required=True)
parser.add_argument("--setting", choices=("baseline", "pisg"), required=True)
parser.add_argument("--cycles", type=int, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--batch-size", type=int, required=True)
parser.add_argument("--threads", type=int, required=True)
args = parser.parse_args()
cfg = json.loads(Path(args.config).read_text())
params, _, _, model_config, _, _ = load_ckpt(args.ckpt)
model = model_config.construct_model()
model = nnx.merge(nnx.graphdef(model), params)
kwargs = dict(sampling_steps=3, flip_rate=tuple(cfg["flip_rate"]), cycles=args.cycles,
              runs=1, decode_intermediate=True, batch_size=args.batch_size,
              threads_over_batches=args.threads, seed=args.seed,
              pisg_scale=cfg["pisg_scale"], pisg_rho=cfg["pisg_rho"],
              pisg_schedule=cfg["pisg_schedule"],
              pisg_parallel=True, pisg_target=cfg.get("pisg_target", "bias"))
if cfg["pisg_scale"] > 0:
    kwargs["pisg_layers"] = cfg["pisg_layers"]
for key in ("pisg_attn_mode", "pisg_attn_rho"):
    if key in cfg:
        kwargs[key] = cfg[key]
mis_flow_decode(dict(np.load(args.data)), model, **kwargs)
