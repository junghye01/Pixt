from typing import Optional
from beartype import beartype
import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader
from dataset import Pixt_Dataset, Pixt_Test_Dataset
from dataset.transform import Pixt_ImageTransform


class BaselineLitDataModule(pl.LightningDataModule):
    @beartype
    def __init__(
        self,
        img_dir: str,
        annotation_dir: dict,
        num_workers: int,
        batch_size: int,
        test_batch_size: int,
    ) -> None:
        super().__init__()
        self._img_dir = img_dir
        self._annotation_dir = annotation_dir
        self._num_workers = num_workers
        self._batch_size = batch_size
        self._test_batch_size = test_batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        image_transform = Pixt_ImageTransform()
        self._dataset_train = Pixt_Dataset(
            self._img_dir,
            self._annotation_dir["train"],
            transform=image_transform,
        )
        self._dataset_valid = Pixt_Dataset(
            self._img_dir,
            self._annotation_dir["valid"],
            transform=image_transform,
        )
        self._dataset_test = Pixt_Test_Dataset(
            self._img_dir,
            transform=image_transform,
        )

    def collate_fn(self, samples):
        input_data = {}
        input_data["image_tensor"] = torch.stack(
            [sample["image_tensor"] for sample in samples], dim=0
        )
        input_data["text_ko"] = [sample["text_ko"] for sample in samples]
        
        return input_data

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._dataset_train,
            shuffle=True,
            drop_last=True,
            num_workers=self._num_workers,
            batch_size=self._batch_size,
            persistent_workers=False,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._dataset_valid,
            shuffle=False,
            drop_last=False,
            num_workers=self._num_workers,
            batch_size=self._batch_size,
            persistent_workers=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._dataset_test,
            shuffle=False,
            drop_last=False,
            num_workers=self._num_workers,
            batch_size=self._test_batch_size,
            persistent_workers=False,
        )
