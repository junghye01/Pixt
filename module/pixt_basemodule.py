from typing import Any, Optional
import lightning.pytorch as pl
import clip
import random
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineLitModule(pl.LightningModule):
    def __init__(
        self,
        clip_model: nn.Module,
        classes_ko_dir: str,
        classes_en_dir: str,
        max_length: int,
        base_loss_func: nn.Module,
        optim: torch.optim,
        lr: float,
    ):
        super().__init__()
        self._clip_model = clip_model
        self._classes_ko_dir = classes_ko_dir
        self._classes_en_dir = classes_en_dir
        self._tags_ko_all_list = self._get_tags_all_list(classes_ko_dir)
        self._tags_en_all_list = self._get_tags_all_list(classes_en_dir)
        self._max_length = max_length
        self._base_loss_func = base_loss_func
        self._optim = optim
        self._lr = lr
        self.automatic_optimization = False

    def _get_tags_all_list(self, classes_dir: str) -> list:
        return torch.load(classes_dir)

    def _get_text_and_target_tensor(self, text_ko: list[list]) -> tuple[torch.Tensor]:
        # true label인 tag 담기
        text_input_ko_list = []
        for tags_ko in text_ko:  # batch size 만큼 iteration
            for tag_ko in tags_ko:  # data sample 당 tag 개수만큼 iteration
                text_input_ko_list.append(tag_ko)
        text_input_ko_list = list(set(text_input_ko_list))

        # false label with random sampling 인 tag 담기
        while True:
            random_sample = random.sample(self._tags_ko_all_list, 1)[0]
            if random_sample not in text_input_ko_list:
                text_input_ko_list.append(random_sample)
            if len(text_input_ko_list) == self._max_length:
                break

        # 한국어(text_input_ko_list)에서 영어(text_input_en_list)로 번역하기
        text_input_en_list = [
            self._tags_en_all_list[self._tags_ko_all_list.index(tag_ko)]
            for tag_ko in text_input_ko_list
        ]
        # tokenize 수행 및 tensor 변환
        text_tensor = torch.cat([clip.tokenize(f"a photo of a {c}") for c in text_input_en_list])

        # 한국어(text_ko)에서 영어(text_en)로 번역하기
        text_en = []
        for tags_ko in text_ko:  # batch size 만큼 iteration
            tmp = []
            for tag_ko in tags_ko:  # data sample 당 tag 개수만큼 iteration
                tmp.append(self._tags_en_all_list[self._tags_ko_all_list.index(tag_ko)])
            text_en.append(tmp)

        # target tensor 생성
        target_tensor_list = []
        for tags_en in text_en:
            target_tensor = torch.zeros_like(torch.empty(self._max_length))
            for tag_en in tags_en:
                if tag_en in text_input_en_list:
                    target_tensor[text_input_en_list.index(tag_en)] = 1
            target_tensor_list.append(target_tensor)
        target_tesnor = torch.stack(target_tensor_list, dim=0)

        return text_tensor, target_tesnor

    def _parse_batch(
        self, batch: list[dict[str, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_tensor = batch.get("image_tensor", None)
        text_ko = batch.get("text_ko")
        text_tensor, target_tensor = self._get_text_and_target_tensor(text_ko)

        text_tensor = text_tensor.to("cuda")
        target_tensor = target_tensor.to("cuda")

        return image_tensor, text_tensor, target_tensor

    def configure_optimizers(self):
        return self._optim(
            self._clip_model.parameters(),
            lr=self._lr,
            betas=(0.9, 0.98),
            eps=1e-6,
            weight_decay=0.2,
        )

    def training_step(self, batch, batch_idx) -> dict[str, Any]:
        optim = self.optimizers()
        image, text, target = self._parse_batch(batch)

        logits_per_image, logits_per_text = self._clip_model(image, text)
        loss = self._base_loss_func(logits_per_image, logits_per_text, target)

        optim.zero_grad()
        self.manual_backward(loss)
        optim.step()

        self.log("train\ce_loss", loss, on_step=False, on_epoch=True, batch_size=image.shape[0])
        loss_dict = {"loss": loss}
        return loss_dict

    def validation_step(self, batch, batch_idx) -> dict[str, Any]:
        image, text, target = self._parse_batch(batch)
        logits_per_image, logits_per_text = self._clip_model(image, text)
        loss = self._base_loss_func(logits_per_image, logits_per_text, target)

        self.log("valid\ce_loss", loss, on_step=False, on_epoch=True, batch_size=image.shape[0])
        loss_dict = {"loss": loss}
        return loss_dict

    # def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT | None:
    #     return super().test_step(*args, **kwargs)
