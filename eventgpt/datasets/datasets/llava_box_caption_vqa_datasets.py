"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
import torch

from PIL import Image

from eventgpt.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset

from collections import OrderedDict


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "question": ann["question"],
                "question_id": ann["question_id"],
                "answers": "; ".join(ann["answer"]),
                "image": sample["image"],
            }
        )


class LlavaBoxCaptionVQADataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def collater(self, samples):
        image_list, question_list, answer_list, weight_list = [], [], [], []

        num_answers = []

        image_path_list = []
        options_list = []
        bbox_list = []
        dense_caption_list = []

        for sample in samples:
            image_list.append(sample["image"])
            image_path_list.append(sample["image_path"])
            # question_list.append(sample["text_input"])
            # options_list.append(sample["options"])
            bbox_list.append(sample["bbox"])
            dense_caption_list.append(sample["dense_caption"])

            # weight_list.extend(sample["weights"])

            # answers = sample["answers"]

            # answer_list.extend(answers)
            # num_answers.append(len(answers))

        return {
            "image": torch.stack(image_list, dim=0),
            "image_path": image_path_list,
            # "text_input": question_list,
            # "text_output": answer_list,
            # "options": options_list,
            "bbox": bbox_list,
            # "weight": torch.Tensor(weight_list),
            # "n_answers": torch.LongTensor(num_answers),
            "dense_caption": dense_caption_list  # get from __getitem__
        }

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(e)
            image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        # question = self.text_processor(ann["question"])

        dense_caption = ann["dense_caption"]
        # options = ann["options"]
        bbox = ann["bbox"]

        # answer_weight = {}
        # for answer in ann["answer"]:
        #     if answer in answer_weight.keys():
        #         answer_weight[answer] += 1 / len(ann["answer"])
        #     else:
        #         answer_weight[answer] = 1 / len(ann["answer"])

        # answers = list(answer_weight.keys())
        # weights = list(answer_weight.values())

        return {
            "image": image,
            "image_path": image_path,
            # "text_input": f"Given the image, answer the question. Question: {question}", 
            # "text_input": f"Given the image, answer the question. Question: '{question} {options}'. Answer:", 
            # "options": options,  # list
            "bbox": bbox,
            # "answers": answers,
            # "weights": weights,
            "dense_caption": dense_caption
        }


class LlavaBoxCaptionVQAEvalDataset(VQAEvalDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """

        self.vis_root = vis_root

        self.annotation = json.load(open(ann_paths[0]))

        answer_list_path = ann_paths[1]
        if os.path.exists(answer_list_path):
            self.answer_list = json.load(open(answer_list_path))
        else:
            self.answer_list = None

        try:
            self.coco_fmt_qust_file = ann_paths[2]
            self.coco_fmt_anno_file = ann_paths[3]
        except IndexError:
            self.coco_fmt_qust_file = None
            self.coco_fmt_anno_file = None

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        dense_caption = ann["dense_caption"]
        options = ann["options"]
        bbox = ann["bbox"]

        return {
            "image": image,
            "image_path": image_path,
            # "text_input": f"Context: {dense_caption}. Question: {question}. Answer:", #question,
            "text_input": f"Given the image, answer the question. Question: '{question} {options}'. Answer:", 
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],
            "dense_caption": dense_caption,
            "options": options,
            "bbox": bbox,
        }
