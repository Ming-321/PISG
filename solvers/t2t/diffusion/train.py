"""The handler for training and evaluation."""

import json
import os
from argparse import ArgumentParser

import torch
import wandb

# os.environ["WANDB_MODE"] = "offline"
# wandb.init()
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.utilities import rank_zero_info

from pl_tsp_model import TSPModel
from pl_mis_model import MISModel

torch.cuda.amp.autocast(enabled=True)
torch.cuda.empty_cache()


def arg_parser():
    parser = ArgumentParser(
        description="Train a Pytorch-Lightning diffusion model on a TSP dataset."
    )
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--storage_path", type=str, required=True)
    parser.add_argument(
        "--validation_split", type=str, default="data/tsp/tsp50_val_concorde.txt"
    )
    parser.add_argument(
        "--test_split", type=str, default="data/tsp/tsp50_test_concorde.txt"
    )
    parser.add_argument("--validation_examples", type=int, default=64)
    parser.add_argument("--test_examples", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--use_activation_checkpoint", action="store_true")

    parser.add_argument("--diffusion_schedule", type=str, default="linear")
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--inference_diffusion_steps", type=int, default=1000)
    parser.add_argument("--inference_schedule", type=str, default="linear")
    parser.add_argument("--inference_trick", type=str, default="ddim")
    parser.add_argument("--sequential_sampling", type=int, default=1)
    parser.add_argument("--parallel_sampling", type=int, default=1)

    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--sparse_factor", type=int, default=-1)
    parser.add_argument("--aggregation", type=str, default="sum")
    parser.add_argument("--two_opt_iterations", type=int, default=1000)
    parser.add_argument("--save_numpy_heatmap", action="store_true")

    parser.add_argument("--project_name", type=str, default="tsp_diffusion")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_logger_name", type=str, default=None)
    parser.add_argument(
        "--resume_id", type=str, default=None, help="Resume training on wandb."
    )
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--resume_weight_only", action="store_true")

    parser.add_argument("--do_valid_only", action="store_true")
    parser.add_argument("--do_test_only", action="store_true")

    parser.add_argument("--rewrite_ratio", type=float, default=0.25)
    parser.add_argument("--norm", action="store_true")
    parser.add_argument("--rewrite", action="store_true")
    parser.add_argument("--rewrite_steps", type=int, default=3)
    parser.add_argument("--inference_steps", type=int, default=10)
    parser.add_argument("--metrics_out", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--pisg", action="store_true")
    parser.add_argument("--pisg_scale", type=float, default=2.0)
    parser.add_argument("--pisg_layers", nargs="+", default=["7"])
    parser.add_argument("--pisg_perturb", type=str, default="adaptive")
    parser.add_argument("--pisg_scale_schedule", type=str, default="linear")
    parser.add_argument("--pisg_fusion_space", type=str, default="logit")
    parser.add_argument("--pisg_tg", action="store_true")

    args = parser.parse_args()
    return args


def maybe_load_weights(model, ckpt_path, resume_weight_only):
    if not ckpt_path or not resume_weight_only:
        return ckpt_path

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    rank_zero_info(
        f"Loaded weights from {ckpt_path} with "
        f"{len(missing_keys)} missing and {len(unexpected_keys)} unexpected keys."
    )
    if missing_keys:
        rank_zero_info(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        rank_zero_info(f"Unexpected keys: {unexpected_keys}")
    return None


def resolve_trainer_config():
    if not torch.cuda.is_available():
        return {"accelerator": "cpu", "devices": 1, "strategy": None}

    num_devices = torch.cuda.device_count()
    strategy = DDPStrategy(static_graph=True) if num_devices > 1 else None
    return {"accelerator": "gpu", "devices": num_devices, "strategy": strategy}


def dump_metrics(metrics_out, stage, metrics):
    if metrics_out is None:
        return
    if int(os.environ.get("RANK", "0")) != 0:
        return

    metrics_dir = os.path.dirname(metrics_out)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
    payload = {}
    if os.path.exists(metrics_out):
        try:
            with open(metrics_out, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except json.JSONDecodeError:
            payload = {}
    payload[stage] = metrics
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main(args):
    if args.seed is not None:
        seed_everything(args.seed, workers=True)

    if args.wandb_logger_name is None:
        args.wandb_logger_name = args.task
    args.wandb_logger_name += "_norm" if args.norm else ""
    args.wandb_logger_name += "_{}".format(args.parallel_sampling)
    args.wandb_logger_name += "_{}".format(args.inference_diffusion_steps)
    if args.rewrite:
        args.wandb_logger_name += "_{}".format(args.rewrite_ratio)
    else:
        args.wandb_logger_name += "_no_rewrite"
    if args.pisg:
        args.wandb_logger_name += "_pisg"

    project_name = args.project_name

    if args.task == "tsp":
        model_class = TSPModel
        saving_mode = "min"
    elif args.task == "mis":
        model_class = MISModel
        saving_mode = "max"
    else:
        raise NotImplementedError

    model = model_class(param_args=args)
    os.makedirs(os.path.join(args.storage_path, f"models"), exist_ok=True)
    wandb_id = os.getenv("WANDB_RUN_ID") or wandb.util.generate_id()
    wandb_logger = WandbLogger(
        name=args.wandb_logger_name,
        project=project_name,
        entity=args.wandb_entity,
        save_dir=os.path.join(args.storage_path, f"models"),
        id=args.resume_id or wandb_id,
    )
    rank_zero_info(
        f"Logging to {wandb_logger.save_dir}/{wandb_logger.name}/{wandb_logger.version}"
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val/solved_cost",
        mode=saving_mode,
        save_top_k=3,
        save_last=True,
        dirpath=os.path.join(
            wandb_logger.save_dir,
            args.wandb_logger_name,
            wandb_logger._id,
            "checkpoints",
        ),
    )
    lr_callback = LearningRateMonitor(logging_interval="step")

    trainer_config = resolve_trainer_config()
    trainer_kwargs = dict(
        accelerator=trainer_config["accelerator"],
        devices=trainer_config["devices"],
        callbacks=[TQDMProgressBar(refresh_rate=20), checkpoint_callback, lr_callback],
        logger=wandb_logger,
        check_val_every_n_epoch=1,
        precision=16 if args.fp16 else 32,
        inference_mode=False,
    )
    if trainer_config["strategy"] is not None:
        trainer_kwargs["strategy"] = trainer_config["strategy"]
    trainer = Trainer(**trainer_kwargs)

    rank_zero_info(f"{'-' * 100}\n" f"{str(model.model)}\n" f"{'-' * 100}\n")

    ckpt_path = maybe_load_weights(model, args.ckpt_path, args.resume_weight_only)

    if args.do_valid_only:
        valid_metrics = trainer.validate(model, ckpt_path=ckpt_path)
        dump_metrics(args.metrics_out, "validate", valid_metrics)
    elif args.do_test_only:
        test_metrics = trainer.test(model, ckpt_path=ckpt_path)
        dump_metrics(args.metrics_out, "test", test_metrics)
    else:
        valid_metrics = trainer.validate(model, ckpt_path=ckpt_path)
        dump_metrics(args.metrics_out, "validate", valid_metrics)
        test_metrics = trainer.test(model, ckpt_path=ckpt_path)
        dump_metrics(args.metrics_out, "test", test_metrics)
    trainer.logger.finalize("success")


if __name__ == "__main__":
    args = arg_parser()
    main(args)
