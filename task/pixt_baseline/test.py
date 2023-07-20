import sys

sys.path.append("C:\\pixt")

import os
import glob
import yaml
import clip
import torch
from omegaconf import DictConfig, OmegaConf

from datamodule import BaselineLitDataModule
from loss import BaseLoss
from module import BaselineLitModule
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import Trainer


def _set_gpu_environ(cfg: DictConfig) -> None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["trainer"]["devices"])


def main(cfg) -> None:
    _set_gpu_environ(cfg)

    lit_data_module = BaselineLitDataModule(
        img_dir=cfg["datamodule"]["image_dir"],
        annotation_dir=cfg["datamodule"]["annotation_dir"],
        num_workers=cfg["datamodule"]["num_workers"],
        batch_size=cfg["datamodule"]["batch_size"],
        test_batch_size=cfg["datamodule"]["test_batch_size"],
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("RN50", device=device)

    base_loss = BaseLoss(ce_loss_weight=cfg["loss"]["ce_loss_weight"])

    lit_module = BaselineLitModule(
        clip_model=model,
        classes_ko_dir=cfg["module"]["classes_ko_dir"],
        classes_en_dir=cfg["module"]["classes_en_dir"],
        max_length=cfg["module"]["max_length"],
        base_loss_func=base_loss,
        optim=torch.optim.Adam,
        lr=cfg["module"]["lr"],
    )

    save_dir = os.path.join(cfg["logger"]["save_root"], cfg["logger"]["log_dirname"])
    logger = TensorBoardLogger(save_dir=save_dir, name=cfg["logger"]["name"])

    checkpoint_callback = ModelCheckpoint(
        dirpath=logger.log_dir,
        monitor=cfg["callbacks"]["checkpoint"]["monitor"],
        save_top_k=cfg["callbacks"]["checkpoint"]["save_top_k"],
        mode=cfg["callbacks"]["checkpoint"]["mode"],
    )
    callbacks = [checkpoint_callback]

    trainer = Trainer(
        accelerator=cfg["trainer"]["accelerator"],
        devices=cfg["trainer"]["devices"],
        logger=logger,
        callbacks=callbacks,
        max_epochs=cfg["trainer"]["max_epochs"],
    )

    trainer.test(model=lit_module, datamodule=lit_data_module, ckpt_path=cfg["ckpt_path"])


if __name__ == "__main__":
    root_dir = "c:\pixt\outputs\pixt_baseline\lightning_logs\\version_5"
    config_path = root_dir + "\config.yaml"
    ckpt_path = glob.glob(root_dir + "\*.ckpt")[0]
    print(config_path)
    print(ckpt_path)
    cfg = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    cfg["ckpt_path"] = ckpt_path
    main(cfg)
