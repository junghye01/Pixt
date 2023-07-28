from typing import Optional
from beartype import beartype
import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader
from dataset import Pixt_Dataset, Pixt_Test_Dataset
from dataset.transform import Pixt_ImageTransform, Pixt_TextTransform, Pixt_TargetTransform


class BaselineLitDataModule(pl.LightningDataModule):
    @beartype
    def __init__(
        self,
        img_dir: str,
        max_length: int,
        classes_ko_dir: str,
        classes_en_dir: str,
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

        self._image_transform = Pixt_ImageTransform()
        self._text_transform = Pixt_TextTransform(
            max_length=max_length,
            classes_ko_dir=classes_ko_dir,
            classes_en_dir=classes_en_dir,
        )
        self._target_transform = Pixt_TargetTransform(max_length=max_length)

    def setup(self, stage: Optional[str] = None) -> None:
        self._dataset_train = Pixt_Dataset(
            self._img_dir,
            self._annotation_dir["train"],
            image_transform=self._image_transform,
        )
        self._dataset_valid = Pixt_Dataset(
            self._img_dir,
            self._annotation_dir["valid"],
            image_transform=self._image_transform,
        )
        self._dataset_test = Pixt_Test_Dataset(
            self._img_dir,
            image_transform=self._image_transform,
        )

    def collate_fn(self, samples):
        image_tensor = torch.stack([sample["image_tensor"] for sample in samples], dim=0)
        text_dict = self._text_transform([sample["text_ko"] for sample in samples])
        target_tensor = self._target_transform(text_dict["text_en"], text_dict["text_input"])

        input_data = text_dict
        text_dict["image_tensor"] = image_tensor
        text_dict["target_tensor"] = target_tensor
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
            shuffle=True,
            drop_last=True,
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
