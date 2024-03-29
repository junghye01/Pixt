import sys

sys.path.append("/home/irteam/junghye-dcloud-dir/Pixt/code/Pixt")

import os
import yaml
import clip
import torch
from omegaconf import DictConfig, OmegaConf
from datamodule import BaselineLitDataModule
from network import ModifiedResNet
from loss import MultiLabelSoftMarginLoss, MSELoss
from metrics import Accuracy
from module import BaselineLitModule
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import Trainer


def _set_gpu_environ(cfg: DictConfig) -> None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["trainer"]["devices"])
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"



def main(cfg) -> None:
    _set_gpu_environ(cfg)

<<<<<<< HEAD
=======
    # wandb.init(project='clip',name='MSELoss_train')
>>>>>>> fc800e12d5250084fa102bbd35bf2791745d9071

    lit_data_module = BaselineLitDataModule(
        img_dir=cfg["datamodule"]["image_dir"],
        max_length=cfg["datamodule"]["max_length"],
        classes_ko_dir=cfg["datamodule"]["classes_ko_dir"],
        classes_en_dir=cfg["datamodule"]["classes_en_dir"],
        annotation_dir=cfg["datamodule"]["annotation_dir"],
        num_workers=cfg["datamodule"]["num_workers"],
        batch_size=cfg["datamodule"]["batch_size"],
        test_batch_size=cfg["datamodule"]["test_batch_size"],
       
    )

    # image encoder
    # image_width = cfg["module"]["encoder"]["image"]["width"]
    # image_heads = image_width * 32 // 64
    # image_encoder = ModifiedResNet(
    #     input_resolution=cfg["module"]["encoder"]["image"]["input_resolution"],
    #     layers=cfg["module"]["encoder"]["image"]["layers"],
    #     width=image_width,
    #     heads=image_heads,
    #     output_dim=cfg["module"]["encoder"]["image"]["output_dim"],
    # )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("RN50", device=device)

<<<<<<< HEAD
    base_loss = BaseLoss(base_loss_weight=cfg["loss"]["ce_loss_weight"],batch_size=cfg['datamodule']['batch_size'])
=======
    base_loss = MultiLabelSoftMarginLoss(base_loss_weight=cfg["loss"]["ce_loss_weight"])
>>>>>>> fc800e12d5250084fa102bbd35bf2791745d9071
    accuracy = Accuracy()

    lit_module = BaselineLitModule(
        clip_model=model,
        base_loss_func=base_loss,
        accuracy=accuracy,
        optim=torch.optim.AdamW,
        lr=cfg["module"]["lr"],
        save_dir=os.path.join(cfg["logger"]["save_root"], cfg["logger"]["log_dirname"]),
        classes_ko_dir=cfg["module"]["classes_ko_dir"],
        classes_en_dir=cfg["module"]["classes_en_dir"],
    )

    save_dir = os.path.join(cfg["logger"]["save_root"], cfg["logger"]["log_dirname"])
    logger = TensorBoardLogger(save_dir=save_dir, name=cfg["logger"]["name"])
    if not os.path.exists(logger.log_dir):
        os.makedirs(logger.log_dir)
    cfg["log_dir"] = logger.log_dir
    OmegaConf.save(cfg, f"{logger.log_dir}/config.yaml")

    checkpoint_callback = ModelCheckpoint(
        dirpath=logger.log_dir,
        monitor=cfg["callbacks"]["checkpoint"]["monitor"],
        save_top_k=cfg["callbacks"]["checkpoint"]["save_top_k"],
        mode=cfg["callbacks"]["checkpoint"]["mode"],
    )
    callbacks = [checkpoint_callback]

  

    trainer = Trainer(
        accelerator=cfg["trainer"]["accelerator"],
        devices=[0],
        logger=logger,
        callbacks=callbacks,
        max_epochs=cfg["trainer"]["max_epochs"],
        accumulate_grad_batches=cfg['trainer']['accumulate_grad_batches'], # 그래디언트 누적
        
    )
    trainer.fit(model=lit_module, datamodule=lit_data_module)


if __name__ == "__main__":
 
    
    config_path = "/home/irteam/junghye-dcloud-dir/Pixt/code/Pixt/task/pixt_baseline/config/RN50_baseline.yaml"
    cfg = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    main(cfg)

    cfg["datamodule"]["annotation_dir"][
        "train"
    ] = "c:\\pixt\data\\annotation\\annotation_remove_mgf\\train.csv"
    cfg["datamodule"]["annotation_dir"][
        "valid"
    ] = "c:\\pixt\data\\annotation\\annotation_remove_mgf\\valid.csv"
    main(cfg)
