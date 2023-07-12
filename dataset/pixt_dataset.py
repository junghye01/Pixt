import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class Pixt_Dataset(Dataset):
    def __init__(self, img_dir: str, annotation_dir: str, transform: nn.Module):
        super().__init__
        self._img_dir = img_dir
        self._annotation_df = self._set_annotation_df(annotation_dir)
        self._transform = transform

    def _set_annotation_df(self, annotation_dir: str) -> pd.DataFrame:
        return pd.read_csv(annotation_dir)

    def __len__(self):
        return self._annotation_df.shape[0]

    def _get_image(self, index: int) -> torch.Tensor:
        data = self._annotation_df.loc[index]
        file_path = self._img_dir + data["dataset"] + "/" + str(data["image_number"]) + ".webp"
        return self._transform(Image.open(file_path).convert("RGB")).float()

    def _get_tags_ko(self, index: int) -> list:
        return eval(self._annotation_df.loc[index]["tags_ko"])

    def __getitem__(self, index: int) -> dict:
        image_tensor = self._get_image(index)
        tags_ko = self._get_tags_ko(index)

        input_data = {"image_tensor": image_tensor, "text_ko": tags_ko}
        return input_data
