from typing import Any
import lightning.pytorch as pl
import clip
import torch
import torch.nn as nn
import json


class BaselineLitModule(pl.LightningModule):
    def __init__(
        self,
        clip_model: nn.Module,
        base_loss_func: nn.Module,
        accuracy: nn.Module,
        optim: torch.optim,
        lr: float,
    ):
        super().__init__()
        self._clip_model = clip_model
        self._base_loss_func = base_loss_func
        self._accuracy = accuracy
        self._optim = optim
        self._lr = lr
        self.automatic_optimization = False
        self.saved_path = (
            "/home/irteam/junghye-dcloud-dir/Pixt/code/Pixt/outputs/pixt_baseline/result.json"
        )
        self._test_log_dict = {}

    def _parse_batch(
        self, batch: list[dict[str, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_tensor = batch.get("image_tensor", None)
        text_ko = batch.get("text_ko", None)
        text_en = batch.get("text_en", None)
        text_input = batch.get("text_input", None)
        text_tensor = batch.get("text_tensor", None)
        target_tensor = batch.get("target_tensor", None)

        return image_tensor, text_tensor, target_tensor, text_en, text_input

    def configure_optimizers(self):
        return self._optim(
            self._clip_model.parameters(),
            lr=self._lr,
            betas=(0.9, 0.98),
            eps=1e-6,
            weight_decay=0.2,
        )

    def training_step(self, batch, batch_idx) -> None:
        optim = self.optimizers()
        image, text, target, text_en, text_input = self._parse_batch(batch)

        image_features = self._clip_model.encode_image(image)
        text_features = self._clip_model.encode_text(text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        similarity = image_features @ text_features.T

        loss = self._base_loss_func(similarity, target)
        acc = self._accuracy(similarity, text_en, text_input)

        optim.zero_grad()
        self.manual_backward(loss)
        optim.step()

        self.log("train/mlsm_loss", loss, on_step=True, on_epoch=True, batch_size=image.shape[0])
        self.log("train/accuracy", acc, on_step=True, on_epoch=True, batch_size=image.shape[0])

    def validation_step(self, batch, batch_idx) -> None:
        image, text, target, text_en, text_input = self._parse_batch(batch)

        image_features = self._clip_model.encode_image(image)
        text_features = self._clip_model.encode_text(text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        similarity = image_features @ text_features.T

        loss = self._base_loss_func(similarity, target)
        acc = self._accuracy(similarity, text_en, text_input)

        self.log("valid/mlsm_loss", loss, on_step=True, on_epoch=True, batch_size=image.shape[0])
        self.log("valid/accuracy", acc, on_step=True, on_epoch=True, batch_size=image.shape[0])

    def test_step(self, batch, batch_idx) -> None:
        image_tensor = batch.get("image_tensor", None)
        classes_list = list(set([tag_en.lower() for tag_en in self._tags_en_all_list]))
        text_tensor = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes_list])

        image_features = self._clip_model.encode_image(image_tensor)
        text_tensor = text_tensor.to(image_features.device)
        text_features = self._clip_model.encode_text(text_tensor)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(10)

        # batch에 있는 이미지 파일명

        batch_image_filenames = batch.get("image_filename", None)

        print(batch_image_filenames, batch_idx)
        image_filename = batch_image_filenames[0]

        image_prediction_dict = {}

        for value, index in zip(values, indices):
            image_prediction_dict[classes_list[index]] = value.item()

        self._test_log_dict[image_filename] = image_prediction_dict
        print("result", self._test_log_dict)

    def on_test_end(self) -> None:
        with open(self.saved_path, "w", encoding="utf-8") as f:
            json.dump(self._test_log_dict, f, indent="\t")
