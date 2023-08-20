import sys

sys.path.append("/home/irteam/junghye-dcloud-dir/Pixt/code/Pixt")


import os
import glob
import yaml
import clip
import torch
import torch.nn as nn


from loss import BaseLoss
from metrics import Accuracy
from module import BaselineLitModule

from lightning.pytorch.callbacks import ModelCheckpoint

from PIL import Image
import matplotlib.pyplot as plt
from typing import List
import pandas as pd


class ModelLoader(nn.Module):
    def __init__(self,root_dir:str):
        super(ModelLoader,self).__init__()
        self.root_dir=root_dir
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model,_=clip.load("RN50", device=self.device)
        self.classes_ko_dir=None
        self.classes_en_dir=None

    def forward(x):
        return x

    def load_model(self):
        config_path=os.path.join(self.root_dir,'config.yaml')
        ckpt_path=glob.glob(os.path.join(self.root_dir,'*.ckpt'))[0]
        cfg=yaml.load(open(config_path,'r'),Loader=yaml.FullLoader)

        self.classes_ko_dir=cfg['module']['classes_ko_dir']
        self.classes_en_dir=cfg['module']['classes_en_dir']

        self.base_loss=BaseLoss(base_loss_weight=cfg['loss']['ce_loss_weight'],batch_size=cfg['datamodule']['batch_size'])

        self.accuracy=Accuracy()

        lit_module = BaselineLitModule(
            clip_model=self.model,
            base_loss_func=self.base_loss,
            accuracy=self.accuracy,
            optim=torch.optim.Adam,
            lr=cfg["module"]["lr"],
            save_dir=os.path.join(cfg["logger"]["save_root"], cfg["logger"]["log_dirname"]),
            classes_ko_dir=self.classes_ko_dir,
            classes_en_dir=self.classes_en_dir,
        )
        lit_module.load_state_dict(torch.load(ckpt_path)["state_dict"])

        return lit_module

    def model_forward(self,image,text):
        return self.load_model().forward(image,text)


class ModelTester(nn.Module):
    def __init__(self,model,image_transform,image_paths:List[str],ann_dir:str):
        super(ModelTester,self).__init__()
        self.model=model
        self.image_transform=image_transform
        self.test_log_dict={}
        self._tags_en_all_list=self._get_tags_all_list(self.model.classes_en_dir)
        self._tags_ko_all_list=self._get_tags_all_list(self.model.classes_ko_dir)
        self.image_paths=image_paths
        self.ann_dir=ann_dir
   

    def forward(self,image,text):
        return self.model(image,text)

    def _get_tags_all_list(self, classes_dir: str) -> list:
        return torch.load(classes_dir)

    def show_image(self):
        num_images = len(self.image_paths)
        num_cols = 4  
        num_rows = (num_images + num_cols - 1) // num_cols

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
        axs = axs.flatten()

        for i, image_path in enumerate(self.image_paths):
            img = Image.open(image_path).convert("RGB")
            axs[i].imshow(img)
            axs[i].axis("off")

       
        for i in range(num_images, num_cols * num_rows):
            axs[i].axis("off")

        plt.tight_layout()
        plt.show()
    
    def test_step(self):

        for image_path in self.image_paths:
            # image -> image tensor
            image=self.image_transform(Image.open(image_path).convert('RGB')).float()
            image=image.unsqueeze(0).to('cuda')

            # text -> text token
            classes_list = list(set([tag_en for tag_en in self._tags_en_all_list]))
            text_tensor = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes_list])
            text_tensor=text_tensor.to('cuda')

            true_labels = self._get_true_labels([image_path], self.ann_dir)

        # prediction
            with torch.no_grad():
                image_features,text_features=self.model(image,text_tensor)

                similarity = (100.0 *image_features @ text_features.T).softmax(dim=-1)

                values,indices=similarity[0].topk(10)

             
                image_filename=os.path.basename(image_path)
                image_prediction_dict={}
                for value,index in zip(values,indices):
                    eng_tag=classes_list[index]
                    ko_tag=self._tags_ko_all_list[self._tags_en_all_list.index(eng_tag)]

                    formatted_tag = f"{eng_tag}({ko_tag})"
                    image_prediction_dict[formatted_tag] = round(value.item(), 6)

            self.test_log_dict[image_filename] = {
                    "prediction": image_prediction_dict,
                    "true_label": true_labels,
            }

        return self.test_log_dict

    def _get_true_labels(self,image_paths,ann_dir):
        ann=pd.read_csv(ann_dir)
        true_labels=[]
        for i, image_path in enumerate(image_paths):
            dataset,file_name=image_path.split('/')[-2],image_path.split('/')[-1]

            filtered_row=ann[(ann['dataset']==dataset) & (ann['image_number']==int(file_name[0]))]

            true_label=eval(filtered_row['tags_ko'].iloc[0])
            true_label_modified=[]
            for label in true_label:
                en_label=self._tags_en_all_list[self._tags_ko_all_list.index(label)]
                formatted_tag=f'{en_label}({label})'
                true_label_modified.append(formatted_tag)
            true_labels.append(true_label_modified)
        return true_labels

