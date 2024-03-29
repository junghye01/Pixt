from typing import Any
import lightning.pytorch as pl
import clip
import torch
import torch.nn as nn
import json
import os
from pathlib import Path
from datetime import datetime
import sys

sys.path.append('/home/irteam/junghye-dcloud-dir/Pixt/code/Pixt/module')

from projection import ProjectionHead

class BaselineLitModule(pl.LightningModule):
    def __init__(
        self,
        clip_model: nn.Module,
        base_loss_func: nn.Module,
        accuracy: nn.Module,
        optim: torch.optim,
        lr: float,
        save_dir : None,
        classes_ko_dir:None,
        classes_en_dir:None,
    ):
        super().__init__()
        self._clip_model = clip_model
        self._base_loss_func = base_loss_func
        self._accuracy = accuracy
        self._optim = optim
        self._lr = lr
        self.automatic_optimization = False
        self.save_dir=save_dir
        self._test_log_dict = {}
      
    
        self.classes_ko_dir=classes_ko_dir
        self.classes_en_dir=classes_en_dir
        self.embedding_dim=1024
        self.projection=ProjectionHead(embedding_dim=self.embedding_dim)
      
        
        #load tags
        if classes_ko_dir is not None and classes_en_dir is not None:
            self._tags_ko_all_list = self._get_tags_all_list(classes_ko_dir)
            self._tags_en_all_list = self._get_tags_all_list(classes_en_dir)

    def _get_tags_all_list(self, classes_dir: str) -> list:
        return torch.load(classes_dir)

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
        optimizer=self._optim(
            self._clip_model.parameters(),
            lr=self._lr,
            betas=(0.9,0.98),
            eps=1e-6,
            weight_decay=1e-3,
        )

      
        lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=2,
            factor=0.5
           
        )

        return{
            "optimizer":optimizer,
            "lr_scheduler":lr_scheduler
        }


    
    def forward(self,image,text):
        image_features=self._clip_model.encode_image(image)
        text_features=self._clip_model.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_features = image_features.to('cuda').half()
        text_features = text_features.to('cuda').half()

        return image_features,text_features


    def training_step(self, batch, batch_idx) -> None:
        optim = self.optimizers()
   
        image, text, target, text_en, text_input= self._parse_batch(batch)

        image_features=self._clip_model.encode_image(image)
        text_features=self._clip_model.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_features = image_features.to('cuda').half()
        text_features = text_features.to('cuda').half()

        # getting image and text embedding (with same dimensions)
        image_embeddings=self.projection(image_features)
        text_embeddings=self.projection(text_features)

        target=target.to('cuda').half()

    
        logits = text_embeddings @ image_embeddings.T
        
        images_similarity=image_embeddings @ image_embeddings.T
        texts_similarity=text_embeddings @ text_embeddings.T


        loss = self._base_loss_func(images_similarity,texts_similarity,logits)
        
       # acc = self._accuracy(similarity, text_en, text_input)


        print(f'{batch_idx} ,{loss}')
        optim.zero_grad()
        self.manual_backward(loss)
        optim.step()

        self.log("train/ce_loss", loss, on_step=True, on_epoch=True, batch_size=image.shape[0])
        #self.log("train/accuracy", acc, on_step=True, on_epoch=True, batch_size=image.shape[0])
       



    def validation_step(self, batch, batch_idx) -> dict[str, Any]:
        image, text, target,text_en,text_input= self._parse_batch(batch)
        
        
        image_features = self._clip_model.encode_image(image)
        text_features = self._clip_model.encode_text(text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_features=image_features.to('cuda').half()
        text_features=text_features.to('cuda').half()
      
        # getting image and text embeddings (with same dimension)
        image_embeddings=self.projection(image_features)
        text_embeddings=self.projection(text_features)

        # calculating the loss
        logits=text_embeddings @ image_embeddings.T
        images_similarity=image_embeddings @ image_embeddings.T
        texts_similarity=text_embeddings @ text_embeddings.T
       
        target=target.to('cuda').half()

        loss = self._base_loss_func(images_similarity,texts_similarity,logits)
        
        #acc = self._accuracy(logits, text_en, val_text_input)

        self.log("valid/ce_loss", loss, on_step=True, on_epoch=True, batch_size=image.shape[0])
        #self.log("valid/accuracy", acc, on_step=True, on_epoch=True, batch_size=image.shape[0])
       

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

        image_features=image_features.to('cuda').half()
        text_features=text_features.to('cuda').half()


        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        #similarity=image_features @ text_features.T
        values, indices = similarity[0].topk(10)
        

        # batch에 있는 이미지 파일명

        batch_image_filenames = batch.get("image_filename", None)

        #print(batch_image_filenames, batch_idx)
        image_filename = batch_image_filenames[0]

        print('파일명',image_filename)
        print(values)
        image_prediction_dict = {}

        for value, index in zip(values, indices):
            eng_tag_lower=classes_list[index]
            eng_tag=tag_mapping[eng_tag_lower]
            ko_tag=self._tags_ko_all_list[self._tags_en_all_list.index(eng_tag)]

            formatted_tag=f"{eng_tag_lower}({ko_tag})"
            image_prediction_dict[formatted_tag] = round(value.item(),6)
      
        print('예측 결과',image_prediction_dict)
        self._test_log_dict[image_filename] = image_prediction_dict
        #print("result", self._test_log_dict)

    def on_test_end(self) -> None:
        with open(os.path.join(self.save_dir,'test_230820.json'),"w",encoding='utf-8') as f:
            json.dump(self._test_log_dict,f,indent='\t',ensure_ascii=False)
            print(f'{self.save_dir}에 test log가 저장되었습니다')


