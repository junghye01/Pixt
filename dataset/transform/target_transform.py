import torch
import torch.nn as nn


class Pixt_TargetTransform:
    def __init__(self, max_length) -> None:
        self._max_length = max_length

    def _get_target_tensor(self, text_en: list[list], text_input: list) -> torch.Tensor:
        # target tensor 생성
        target_tensor_list = []
        for tags_en in text_en:
            target_tensor = torch.zeros(self._max_length, dtype=torch.float)
            indices = [text_input.index(tag_en) for tag_en in tags_en if tag_en in text_input]
            target_tensor[indices] = 1
            target_tensor_list.append(target_tensor)
        target_tensor = torch.stack(target_tensor_list, dim=0)
        return target_tensor

    def __call__(self, text_en: list[list], text_input: list) -> torch.Tensor:
        return self._get_target_tensor(text_en, text_input)


class Pixt_TargetTransform2:
    def __init__(self,batch_size) -> None:
        self._batch_size=batch_size

    def _get_target_tensor(self, text_en: list[list], text_input: list) -> torch.Tensor:
        # target tensor 생성
        target_tensor_list=[]
        for tags_en in text_en:
            target_tensor=torch.zeros(self._batch_size,dtype=torch.float)
            indices=[text_input.index(tag_en) for tag_en in tags_en if tag_en in text_input]
            target_tensor[indices]=1
            target_tensor_list.append(target_tensor)
        target_tensor=torch.stack(target_tensor_list,dim=0)
        return target_tensor

    def __call__(self, text_en: list[list], text_input: list) -> torch.Tensor:
        return self._get_target_tensor(text_en, text_input)