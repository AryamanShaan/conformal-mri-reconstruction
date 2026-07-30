from __future__ import annotations

import pathlib
from argparse import ArgumentParser

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

# from fastMRI.quantile_regression_batches.knee_slice_data import KneeSliceDataModule
from quantile.knee_slice_data import KneeSliceDataModule
# from fastMRI.quantile_regression_batches.quantile_regression_module import FrozenVarNetQuantileModule
from quantile.quantile_regression_module import FrozenVarNetQuantileModule


def _str_to_bool(value):
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False

    raise ValueError(f"Cannot interpret {value!r} as boolean.")


def cli_main(args) -> None:
    pl.seed_everything(args.seed)

    data_module = KneeSliceDataModule(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed_sampler=(args.num_gpus > 1),
    )

    model = FrozenVarNetQuantileModule(
        varnet_checkpoint=args.varnet_checkpoint,
        quantile=args.quantile,
        pinball_loss_type=args.pinball_loss_type,
        model_type=args.model_type,
        output_version=args.output_version,
        unet_chans=args.unet_chans,
        unet_pools=args.unet_pools,
        unet_drop_prob=args.unet_drop_prob,
        mask_type=args.mask_type,
        min_k_fraction=args.min_k_fraction,
        max_k_fraction=args.max_k_fraction,
        center_fractions=args.center_fractions,
        accelerations=args.accelerations,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
    )

    logger = CSVLogger(save_dir=str(args.default_root_dir), name="csv_logs")

    print("\n" + "###" * 20)
    print(f"model_type:       {args.model_type}")
    print(f"output_version:   {args.output_version}")
    print(f"quantile:         {args.quantile}")
    print(f"pinball_loss_type: {args.pinball_loss_type}")
    print(f"lr:               {args.lr}")
    print(f"lr_step_size:     {args.lr_step_size}")
    print(f"lr_gamma:         {args.lr_gamma}")
    print(f"weight_decay:     {args.weight_decay}")
    print(f"mask_type:        {args.mask_type}")
    print(f"batch_size:       {args.batch_size}")
    print(f"max_epochs:       {args.max_epochs}")
    print("###" * 20 + "\n")

    trainer = pl.Trainer(
        accelerator=("gpu" if args.num_gpus > 0 else "cpu"),
        devices=args.num_gpus,
        strategy=("ddp" if args.num_gpus > 1 else "auto"),
        default_root_dir=args.default_root_dir,
        max_epochs=args.max_epochs,
        deterministic=args.deterministic,
        callbacks=args.callbacks,
        logger=logger,
    )

    trainer.fit(model, datamodule=data_module, ckpt_path=args.resume_from_checkpoint)


def build_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--config",
        default=None,
        type=pathlib.Path,
        help="Optional YAML config file. CLI args override config values.",
    )
    parser.add_argument(
        "--num_gpus",
        default=1,
        type=int,
        help="Number of GPUs to use.",
    )
    parser.add_argument(
        "--default_root_dir",
        default=pathlib.Path("/scratch/fastmri/logs/quantile_regression"),
        type=pathlib.Path,
        help="Directory for logs and checkpoints.",
    )
    parser.add_argument(
        "--max_epochs",
        default=50,
        type=int,
        help="Max number of training epochs.",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed.",
    )
    parser.add_argument(
        "--deterministic",
        default=True,
        type=_str_to_bool,
        help="Whether to use deterministic training where possible.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        type=pathlib.Path,
        help="Optional checkpoint to resume from.",
    )

    parser = KneeSliceDataModule.add_data_specific_args(parser)
    parser.set_defaults(
        data_path=pathlib.Path("/scratch/fastmri/knee"),
        batch_size=1,
    )

    parser = FrozenVarNetQuantileModule.add_model_specific_args(parser)

    config_args, _ = parser.parse_known_args()
    if config_args.config is not None:
        with open(config_args.config, "r") as f:
            config = yaml.safe_load(f) or {}
        if not isinstance(config, dict):
            raise ValueError("Config file must contain a YAML mapping at top level.")
        parser.set_defaults(**config)

    args = parser.parse_args()
    args.data_path = pathlib.Path(args.data_path)
    args.default_root_dir = pathlib.Path(args.default_root_dir)
    if args.resume_from_checkpoint is not None:
        args.resume_from_checkpoint = pathlib.Path(args.resume_from_checkpoint)
    if args.varnet_checkpoint is None:
        raise ValueError("--varnet_checkpoint must be provided, either via CLI or config.")
    args.varnet_checkpoint = pathlib.Path(args.varnet_checkpoint)

    checkpoint_dir = args.default_root_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    args.callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            monitor="validation_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            every_n_epochs=1,
            verbose=True,
            filename="epoch{epoch:02d}-val_loss{validation_loss:.4f}",
        )
    ]

    if args.resume_from_checkpoint is None:
        last_ckpt = checkpoint_dir / "last.ckpt"
        if last_ckpt.exists():
            args.resume_from_checkpoint = last_ckpt

    return args


def run_cli() -> None:
    args = build_args()
    cli_main(args)


if __name__ == "__main__":
    run_cli()


# Example run from the repo root:
#
# export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
# python quantile_regression_batches/train_quantile_regression.py \
#   --config configs/quantile_regression_batches_upper_batch_1.yaml
