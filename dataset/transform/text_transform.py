import torch
import torch.nn as nn
import random
import clip


class Pixt_TextTransform:
    def __init__(self, max_length: int, classes_ko_dir: str, classes_en_dir: str) -> None:
        self._max_length = max_length
        self._tags_ko_all_list = self._get_tags_all_list(classes_ko_dir)
        self._tags_en_all_list = self._get_tags_all_list(classes_en_dir)

    def _get_tags_all_list(self, classes_dir: str) -> list:
        return torch.load(classes_dir)

    def _translate_text_ko_to_en(self, text_ko: list[list]) -> list[list]:
        text_en = []
        for tags_ko in text_ko:  # batch size 만큼 iteration
            tmp = []
            for tag_ko in tags_ko:  # data sample 당 tag 개수만큼 iteration
                tmp.append(self._tags_en_all_list[self._tags_ko_all_list.index(tag_ko)])
            text_en.append(tmp)
        return text_en

    def _get_true_labels(self, text: list[list]) -> list:
        # true label인 tag 담기
        true_labels = []
        for tags in text:  # batch size 만큼 iteration
            for tag in tags:  # data sample 당 tag 개수만큼 iteration
                true_labels.append(tag)
        # 중복된 라벨 제거
        return list(set(true_labels))

    def _get_false_labels(self, true_labels: list) -> list:
        # false label with negative sampling 인 tag 담기
        available_false_labels = [tag for tag in self._tags_en_all_list if tag not in true_labels]
        num_false_labels = self._max_length - len(true_labels)
        sampled_false_labels = random.sample(available_false_labels, num_false_labels)
        return sampled_false_labels

    def _get_input_labels_with_negative_sampling(
        self, true_labels: list, false_labels: list
    ) -> list:
        return true_labels + false_labels

    def _get_text_tensor(self, text_input: list) -> torch.Tensor:
        text_tensor = torch.cat([clip.tokenize(f"a photo of a {c}") for c in text_input])
        return text_tensor

    def __call__(self, text_ko: list[list]) -> dict:
        text_en = self._translate_text_ko_to_en(text_ko)
        true_labels = self._get_true_labels(text_en)
        false_labels = self._get_false_labels(true_labels)
        text_input = self._get_input_labels_with_negative_sampling(true_labels, false_labels)
        text_tensor = self._get_text_tensor(text_input)

        return {
            "text_ko": text_ko,
            "text_en": text_en,
            "text_input": text_input,
            "text_tensor": text_tensor,
        }
