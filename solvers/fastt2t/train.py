"""The handler for training and evaluation."""

import os
import argparse
from argparse import ArgumentParser

import torch
import wandb

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.utilities import rank_zero_info

torch.cuda.amp.autocast(enabled=True)
torch.cuda.empty_cache()

import warnings

warnings.filterwarnings("ignore")


def arg_parser():
    parser = ArgumentParser(
        description="Train a Pytorch-Lightning diffusion model on a TSP dataset."
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--storage_path", type=str, required=True)
    parser.add_argument("--training_split", type=str, default=None)
    parser.add_argument(
        "--training_split_label_dir",
        type=str,
        default=None,
        help="Directory containing labels for training split (used for MIS).",
    )
    parser.add_argument("--validation_split", type=str, default=None)
    parser.add_argument(
        "--validation_split_label_dir",
        type=str,
        default=None,
        help="Directory containing labels for validation split (used for MIS).",
    )
    parser.add_argument("--test_split", type=str, default=None)
    parser.add_argument(
        "--test_split_label_dir",
        type=str,
        default=None,
        help="Directory containing labels for test split (used for MIS).",
    )
    parser.add_argument("--validation_examples", type=int, default=64)
    parser.add_argument(
        "--graph_type",
        type=str,
        default="undirected",
        choices=["undirected", "directed"],
    )

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr_scheduler", type=str, default="constant")

    parser.add_argument("--num_workers", type=int, default=64)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--use_activation_checkpoint", action="store_true")

    parser.add_argument("--diffusion_schedule", type=str, default="linear")
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--inference_diffusion_steps", type=int, default=5)
    parser.add_argument("--inference_schedule", type=str, default="cosine")
    parser.add_argument("--inference_trick", type=str, default="ddim")
    parser.add_argument("--sequential_sampling", type=int, default=1)
    parser.add_argument("--parallel_sampling", type=int, default=1)

    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--sparse_factor", type=int, default=-1)
    parser.add_argument("--aggregation", type=str, default="sum")
    parser.add_argument("--two_opt_iterations", type=int, default=0)
    parser.add_argument("--save_numpy_heatmap", action="store_true")

    parser.add_argument("--project_name", type=str, default="tsp_diffusion")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_logger_name", type=str, default=None)
    parser.add_argument(
        "--resume_id", type=str, default=None, help="Resume training on wandb."
    )
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--resume_weight_only", action="store_true")

    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_test", action="store_true")

    parser.add_argument("--rewrite_ratio", type=float, default=0.25)
    parser.add_argument("--norm", action="store_true", default=False)
    parser.add_argument("--rewrite", action="store_true")
    parser.add_argument("--rewrite_steps", type=int, default=1)
    parser.add_argument("--steps_inf", type=int, default=1)
    parser.add_argument("--guided", action="store_true")

    parser.add_argument(
        "--consistency",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Consistency mode (default: on). Use --no-consistency to disable.",
    )
    parser.add_argument("--boundary_func", default="truncate")
    parser.add_argument("--alpha", type=float)

    parser.add_argument(
        "--use_intermediate",
        action="store_true",
        help="set true when use intermediate x0 to decode tours",
    )
    parser.add_argument("--c1", type=float, default=50, help="coefficient of F1")
    parser.add_argument("--c2", type=float, default=50, help="coefficient of F2")

    parser.add_argument(
        "--offline", action="store_true", help="set true when use offline wandb"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for Python/NumPy/PyTorch/Lightning.",
    )
    parser.add_argument(
        "--pisg",
        action="store_true",
        help="Enable Perturbed Instance-Structure Guidance during the Ts "
             "(diffusion) phase only. Does NOT affect the Tg (rewrite) phase; "
             "use --pisg_tg to also enable PISG during rewrite. "
             "Only effective with --do_test --consistency. "
             "Recommended configs — TSP: s=2.0 l=7 adaptive linear; "
             "MIS: s=1.0 l=7 uniform constant.",
    )
    parser.add_argument(
        "--pisg_tg",
        action="store_true",
        help="Also apply PISG during the Tg (rewrite/guided) phase. "
             "Requires --pisg to be set. When off (default), the rewrite "
             "phase uses vanilla forward passes even if --pisg is on.",
    )
    parser.add_argument(
        "--pisg_scale",
        type=float,
        default=1.0,
        help="Base guidance strength for PISG extrapolation. "
             "Formula: x_guided = x_normal + scale * (x_normal - x_perturbed). "
             "0 = degrade to baseline (no guidance). "
             "Recommended: 2.0 (TSP), 1.0 (MIS).",
    )
    parser.add_argument(
        "--pisg_layers",
        type=str,
        nargs="+",
        default=["7"],
        help="GNN layers to apply PISG gate perturbation. "
             "Selectors: 'all' (every layer), 'first-half', 'second-half'. "
             "Or explicit layer indices: '0' '5' '11' (0-indexed, up to n_layers-1). "
             "Selectors cannot be mixed with indices. "
             "Recommended: 7 (both TSP and MIS with default 12-layer GNN).",
    )
    parser.add_argument(
        "--pisg_perturb",
        type=str,
        default="uniform",
        choices=["uniform", "zero", "adaptive", "random-gate", "shuffle-gate", "input-noise"],
        help="Gate perturbation strategy for constructing the degraded branch. "
             "uniform: replace gates with 0.5 (structure-agnostic). "
             "adaptive: replace with per-instance/per-graph mean gate value. "
             "zero: replace with 0 (fully suppress gating). "
             "random-gate: replace with random values in [0,1]. "
             "shuffle-gate: randomly permute gate values within each graph. "
             "input-noise: add Gaussian noise to xt instead of perturbing gates (non-structural baseline). "
             "Recommended: adaptive (TSP), uniform (MIS).",
    )
    parser.add_argument(
        "--pisg_fusion_space",
        type=str,
        default="logit",
        choices=["logit", "probability"],
        help="Space in which PISG extrapolation is performed. "
             "logit: extrapolate raw model outputs, then apply softmax. "
             "probability: apply softmax first, then extrapolate and re-normalize. "
             "logit is generally more stable and recommended.",
    )
    parser.add_argument(
        "--pisg_scale_schedule",
        type=str,
        default="constant",
        choices=["constant", "linear", "reverse-linear"],
        help="Schedule for PISG guidance strength across denoising steps. "
             "constant: use pisg_scale at every step. "
             "linear: ramp from 0 to pisg_scale over steps (weaker early, stronger late). "
             "reverse-linear: ramp from pisg_scale to 0 over steps (stronger early, weaker late). "
             "Recommended: linear (TSP), constant (MIS).",
    )
    parser.add_argument(
        "--pisg_active_steps",
        type=str,
        default="all",
        help="Which denoising steps to apply PISG guidance (others use scale=0). "
             "all: every step. early: first half. late: second half. "
             "first: step 0 only. last: final step only. "
             "Or comma-separated indices like '0,1,2'.",
    )
    parser.add_argument(
        "--logit_temperature",
        type=float,
        default=1.0,
        help="Temperature for logit scaling before softmax (non-PISG path only). "
             "T>1 smooths, T<1 sharpens. Default 1.0 (no change). "
             "Ignored when --pisg is enabled.",
    )
    parser.add_argument(
        "--save_intermediates",
        action="store_true",
        help="Save per-step x0_pred tensors to .pt files for offline analysis.",
    )
    parser.add_argument(
        "--intermediate_dir",
        type=str,
        default="intermediates",
        help="Directory for saving intermediate x0_pred tensors.",
    )

    args = parser.parse_args()
    return args


def main(args):
    print(args)
    epochs = args.num_epochs
    project_name = args.project_name
    os.makedirs(args.storage_path, exist_ok=True)
    run_name = args.wandb_logger_name or f"{args.task}_{'test' if args.do_test else 'train'}"

    if args.seed is not None:
        seed_everything(args.seed, workers=True)

    if args.pisg:
        if not args.do_test:
            warnings.warn(
                "--pisg is only effective during inference (--do_test). Ignoring PISG settings for training."
            )
            args.pisg = False
        elif not args.consistency:
            warnings.warn("--pisg requires --consistency mode. Ignoring PISG settings.")
            args.pisg = False

    if args.offline or (
        "WANDB_MODE" in os.environ and os.environ["WANDB_MODE"] in {"offline", "disabled"}
    ):
        requested_mode = os.environ.get("WANDB_MODE", "offline")
        if requested_mode not in {"offline", "disabled"}:
            requested_mode = "offline"
        os.environ["WANDB_MODE"] = requested_mode
        try:
            wandb.init(mode=requested_mode, dir=args.storage_path)
        except Exception as exc:
            warnings.warn(
                f"W&B {requested_mode} init failed ({exc}). Falling back to disabled mode."
            )
            os.environ["WANDB_MODE"] = "disabled"
            wandb.init(mode="disabled", dir=args.storage_path)
    else:
        wandb.login(key=os.environ["WANDB_API_KEY"])

    if args.task == "tsp":
        from diffusion.pl_tsp_model import TSPModel

        model_class = TSPModel
        saving_mode = "min"
    elif args.task == "mis":
        from diffusion.pl_mis_model import MISModel

        model_class = MISModel
        saving_mode = "max"
    else:
        raise NotImplementedError

    model = model_class(param_args=args)
    wandb_id = os.getenv("WANDB_RUN_ID") or wandb.util.generate_id()
    wandb_logger = WandbLogger(
        name=run_name,
        project=project_name,
        entity=args.wandb_entity,
        save_dir=os.path.join(args.storage_path),
        id=args.resume_id or wandb_id,
    )
    rank_zero_info(
        f"Logging to {wandb_logger.save_dir}/{wandb_logger.name}/{wandb_logger.version}"
    )
    rank_zero_info(args)

    checkpoint_callback = ModelCheckpoint(
        monitor="val/solved_cost",
        mode=saving_mode,
        save_top_k=1,
        save_last=True,
        dirpath=os.path.join(
            wandb_logger.save_dir,
            run_name,
            wandb_logger._id,
            "checkpoints",
        ),
    )
    lr_callback = LearningRateMonitor(logging_interval="step")
    using_cuda = torch.cuda.is_available() and args.device == "cuda"
    trainer_devices = torch.cuda.device_count() if using_cuda else 1
    trainer_accelerator = "gpu" if using_cuda else "cpu"
    trainer_kwargs = {}
    if trainer_devices and trainer_devices > 1:
        trainer_kwargs["strategy"] = DDPStrategy(static_graph=True)
    trainer = Trainer(
        accelerator=trainer_accelerator,
        devices=trainer_devices,
        max_epochs=epochs,
        callbacks=[TQDMProgressBar(refresh_rate=1), checkpoint_callback, lr_callback],
        logger=wandb_logger,
        check_val_every_n_epoch=1,
        precision=16 if args.fp16 else 32,
        inference_mode=False,
        **trainer_kwargs,
    )

    ckpt_path = args.ckpt_path
    if args.do_train:
        if args.do_test:
            trainer.test(model)
        if args.resume_weight_only:
            model = model_class.load_from_checkpoint(ckpt_path, param_args=args)
            trainer.fit(model)
        else:
            trainer.fit(model, ckpt_path=ckpt_path)
    elif args.do_test:
        trainer.test(model, ckpt_path=ckpt_path)
    trainer.logger.finalize("success")


if __name__ == "__main__":
    args = arg_parser()
    main(args)
