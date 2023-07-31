from typing import Any
import lightning.pytorch as pl
import clip
import random
import torch
import torch.nn as nn
import json


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
        save_dir : None,
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
        
        self._test_log_dict = {}

    def _get_tags_all_list(self, classes_dir: str) -> list:
        return torch.load(classes_dir)

    def _get_text_and_target_tensor(self, text_ko: list[list]) -> tuple[torch.Tensor]:
        # true label인 tag 담기
        text_input_ko_list = []
        for tags_ko in text_ko:  # batch size 만큼 iteration
            for tag_ko in tags_ko:  # data sample 당 tag 개수만큼 iteration
                text_input_ko_list.append((tag_ko,1))
        
        # 중복된 라벨 제거
        unique_text_input_ko_list=list(set(text_input_ko_list))
        
        # false label with negative sampling 인 tag 담기
        num_false_labels=self._max_length-len(unique_text_input_ko_list)
        available_false_labels=list(set(self._tags_ko_all_list)-set([tag_ko for tag_ko,_ in unique_text_input_ko_list]))

        sampled_false_labels=random.sample(available_false_labels,num_false_labels)
        unique_text_input_ko_list.extend([(random_sample,0) for random_sample in sampled_false_labels])


        # 한국어(text_input_ko_list)에서 영어(text_input_en_list)로 번역하기
        text_input_en_list=[self._tags_en_all_list[self._tags_ko_all_list.index(tag_ko)] for tag_ko,_ in unique_text_input_ko_list]
        text_tensor=torch.cat([clip.tokenize(f"a photo of a {c}") for c in text_input_en_list])

        
        # target tensor
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
            target_tensor=torch.zeros(self._max_length,dtype=torch.float)
            indices=[text_input_en_list.index(tag_en) for tag_en in tags_en if tag_en in text_input_en_list]
            target_tensor[indices]=1
            target_tensor_list.append(target_tensor)
        target_tesnor = torch.stack(target_tensor_list, dim=0)
      

        return text_tensor, target_tesnor

    def _parse_batch(
        self, batch: list[dict[str, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_tensor = batch.get("image_tensor", None)
        text_ko = batch.get("text_ko", None)
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

    def test_step(self, batch, batch_idx) -> None:
        image_tensor = batch.get("image_tensor", None)
        classes_list = list(set([tag_en.lower() for tag_en in self._tags_en_all_list]))
        text_tensor = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes_list])

        # lower case와 original case 맵핑
        tag_mapping={tag_en.lower():tag_en for tag_en in self._tags_en_all_list}
        
        image_features = self._clip_model.encode_image(image_tensor)
        text_tensor = text_tensor.to(image_features.device)
        text_features = self._clip_model.encode_text(text_tensor)


        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(10)

        # batch에 있는 이미지 파일명
        batch_image_filenames=batch.get("image_filename",None)

        print(batch_image_filenames,batch_idx)
        image_filename=batch_image_filenames[0]

        image_prediction_dict = {}
        
        for value, index in zip(values, indices):
            eng_tag_lower=classes_list[index]
            eng_tag=tag_mapping[eng_tag_lower]
            ko_tag=self._tags_ko_all_list[self._tags_en_all_list.index(eng_tag)]

            #print('뭐지',eng_tag,ko_tag)
            formatted_tag=f"{eng_tag_lower}({ko_tag})"
            image_prediction_dict[formatted_tag] = round(value.item(),6)
      

        self._test_log_dict[image_filename] = image_prediction_dict
        print('result',self._test_log_dict)

    def on_test_end(self) -> None:
        with open(os.path.join(self.save_dir,'test_results.json'),"w",encoding='utf-8') as f:
            json.dump(self._test_log_dict,f,indent='\t')
            print(f'{self.save_dir}에 저장되었습니다')
