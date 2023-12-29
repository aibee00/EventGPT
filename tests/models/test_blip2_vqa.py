"""
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#

Integration tests for BLIP2 models.
"""

import functools
from typing import List, Optional
import cv2
import torch
import sys
from pathlib import Path
import json
import random

sys.path.append(Path(__file__).parents[2].as_posix())

from eventgpt.common.utils import get_cache_path
from eventgpt.models import load_model, load_model_and_preprocess
from PIL import Image
import numpy as np
import torch.nn as nn

from eventgpt.models.blip2_models.sam_model.prompt_encoder import PromptEncoder
from transformers import BertTokenizer

from plot_box_to_img import plot_bbox, save, decode_norm, plot_text, parse_bbox

# setup device to use
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BERT_LOCAL_PATH = get_cache_path("bert-base-uncased")


def random_coords(caption):
    """ Random coords in caption

    caption: str: like:
        "neon green car : [ 1, 388, 910, 1012 ] ; man wearing black and white sneakers : [ 994, 285, 1173, 791 ] ; a black car with blue grill : [ 411, 761, 2317, 1428 ] ; woman wearing a gray dress : [ 169, 497, 431, 1070 ] ; a child walking in the room : [ 1775, 316, 1946, 635 ] ; man wearing blue shirt : [ 1154, 234, 1329, 752 ] ; woman wearing black shirt : [ 1311, 296, 1473, 744 ] ; a computer monitor : [ 2267, 488, 2493, 667 ] ; woman wearing black pants : [ 1446, 181, 1570, 574 ] ; people are standing around : [ 0, 11, 2560, 1412 ] ; words on top of screen : [ 5, 1, 1145, 74 ] ; the name of the photographer : [ 1969, 1347, 2333, 1434 ] ; a pile of sacks of potatoes : [ 1805, 3, 2550, 220 ] ;"
    """
    def get_random_coords(coords):
        """
        coords: str like:
            " [ 1, 388, 910, 1012 ] "
        """
        coords = coords.strip()
        coords = coords.replace("[", "")
        coords = coords.replace("]", "")
        coords = coords.split(",")
        x1 = int(coords[0])
        y1 = int(coords[1])
        x2 = int(coords[2])
        y2 = int(coords[3])
        x = torch.randint(x1, x2, (1,)).item() + 1
        y = torch.randint(y1, y2, (1,)).item() + 1
        return f" [ {x}, {y}, {x2}, {y2} ] "

    caption = caption.strip()
    caption = caption[:-1] if caption[-1] == ";" else caption
    caption = caption.split(";")
    caption_n = {}
    for i, c in enumerate(caption):
        annotation, coords = c.split(':')
        caption_n[annotation] = get_random_coords(coords)
    # convert caption_n to string format as caption
    caption = ""
    for k, v in caption_n.items():
        caption += f"{k}:{v};"
    return caption


def print_func_name(func):
    if hasattr(func, "tracing"):  # Only decorate once
        return func
    
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # print func name
        print(f"func name: {func.__name__}")
        with torch.no_grad():
            return func(self, *args, **kwargs)
    
    wrapper.tracing = True
    return wrapper


class DecorateTestMeta(type):
    def __new__(cls, name, bases, attrs):
        # 遍历类的属性和方法
        for k, v in attrs.items():
            # 如果是test_开头的方法（可调用对象）
            if k.startswith("test_") and callable(v):
                attrs[k] = print_func_name(v)
        
        return super().__new__(cls, name, bases, attrs)
    

def print_func_wrapper(kclass):
    # 遍历类的层次结�n
    for key in dir(kclass):
        value = getattr(kclass, key)
        # 如消k是否是合法的方法名
        if key.startswith("test_") and callable(value):
            # 如消k是否是合法的方法名
            setattr(kclass, key, print_func_name(value))
    return kclass


@print_func_wrapper
# class TestBlip2(metaclass=DecorateTestMeta):
class TestBlip2():
    def __init__(self) -> None:
        self.prompt_encoder=PromptEncoder(
                embed_dim=768,
                image_embedding_size=(16, 16),
                input_image_size=(224, 224),
                mask_in_chans=16,
            ).to(device)
        self.tokenizer = self.init_tokenizer()
        self.word_embeddings = nn.Embedding(
            30523, 768, padding_idx=self.tokenizer.pad_token_id
        ).to(device)

    @classmethod
    def init_tokenizer(cls, truncation_side="right"):
        # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side=truncation_side)
        tokenizer = BertTokenizer.from_pretrained(BERT_LOCAL_PATH, truncation_side=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer
    
    def get_options_embeds(self, options, batch_size, device, max_num_box=32):
        """ Combine question and options and the prompt suffix: '###Answer:' """
        if options:
            # Apply position_encoding to options
            if not isinstance(options, torch.Tensor):
                max_len = max([len(option) for option in options])
                labels = - torch.ones(batch_size, max_len).long().to(device)
                for i in range(batch_size):
                    labels[i, :len(options[i])].fill_(1)
                    try:
                        _dummy_box = [0,0,0,0]
                        options[i].extend([_dummy_box for i in range(max_len - len(options[i]))])  # padding
                    except:
                        print(f"options[i]:{options[i]}, len(options[i]):{len(options[i])}, max_len:{max_len}")
                        raise
                options = torch.tensor(options, dtype=torch.long).to(device)
            options = options.view(-1, 4)  # (B,N,4) -> (B*N,4)
            labels = labels.view(-1)  # (B,N,4) -> (B*N,4)
            options_embeds, _ = self.prompt_encoder(points=None, boxes=(options, labels), masks=None)  # (B*N, 2, prompt_embed_dim), 2 means 2 corner points
            # reshape to (B, N*2, prompt_embed_dim)
            options_embeds = options_embeds.view(batch_size, -1, self.prompt_encoder.embed_dim)
            num_options = options_embeds.size(1)
            if num_options < max_num_box:  # padding
                pad_ids = torch.ones((batch_size, max_num_box - num_options), dtype=torch.long).to(device).fill_(self.tokenizer.pad_token_id)
                pad_embeds = self.word_embeddings(pad_ids)  # (B, N_s, hidden_size)
                options_embeds = torch.cat((options_embeds, pad_embeds), dim=1)
            elif num_options > max_num_box:  # trucation
                options_embeds = options_embeds[:, :max_num_box]
        else:
            options_embeds = None

        return options_embeds

    @staticmethod
    def _get_new_img_path(image_path, new_image_dir):
        """ 从原始label中的img_path转换为新的img_name形式
        Input:
            image_path: 原文件路径 Path Like: "/training/wphu/Dataset/eventgpt/eventgpt/images/volvo_jinan_xfh_20210617/20210617/ch01019_20210617172500/1140.jpg"
            new_image_dir: 新文件目录，默认为None，则为保存到self.save_dir/images下。Path Like: '/training/wphu/Dataset/eventgpt/eventgpt/fewshot_data_eventgpt/images_expand/images'

        Return:
            img_path: Path Like: '/training/wphu/Dataset/eventgpt/eventgpt/fewshot_data_eventgpt/images_expand/images/volvo_jinan_xfh_20210617_ch01019_20210617172500_1140.jpg'
        
        Note:
            old img_path: 指 label 中的image_path形式
        """
        if not isinstance(image_path, Path):
            image_path = Path(image_path)
        if not isinstance(new_image_dir, Path):
            new_image_dir = Path(new_image_dir) if new_image_dir is not None else None
        image_dir = image_path.parent
        img_name = image_path.name
        sub_paths = image_dir.as_posix().split('/')
        new_img_path = new_image_dir / f'{sub_paths[-3]}__{sub_paths[-1]}__{img_name}'
        return new_img_path
    
    def test_blip2_llava_style_box_caption(self, model, vis_processors, image_index=60, verbose=0, save_img_path="./test_output/llava_styple_box_caption_stage1", img_with_bbox=False, image_path_with_bbox=None):
        """ test_blip2_llava_style_box_caption
        这个函数对应的实验是: "给定bbox预测caption"
        所以，在可视化的时候，可视化的不再是pred_bboxes,而是某个bbox的caption.
        """
        total_num_images = len(labels)
        num_iters = 20 if total_num_images > 30 else total_num_images  # 测试样本的个数
        chosen_set = set()
        for i in range(num_iters):
            if RANDOM_IMAGE:
                image_index = random.choice(range(total_num_images))
                while image_index in chosen_set:
                    image_index = random.choice(range(total_num_images))
                chosen_set.add(image_index)
            else:
                image_index = i
            
            print(f" {i}-th Image Index: {image_index} ".center(50, '-'))

            # Get label
            image_file = labels[image_index]["image"]
            image_path = Path(image_root, image_file).as_posix()

            if img_with_bbox:
                # 替换成带 bbox 的图片路径
                assert image_path_with_bbox is not None, "Please provide image_path_with_bbox!"
                image_path = self._get_new_img_path(image_file, image_path_with_bbox)

            if verbose >=1:
                print(f"Image_path: {image_path}")
            raw_image = Image.open(image_path).convert("RGB")

            ann = labels[image_index]
            dense_caption = ann["dense_caption"]

            # preprocess the image
            # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

            # generate caption
            caption = model.generate({"image": image, "dense_caption": dense_caption}, use_nucleus_sampling=True, max_length=1024, temperature=1)
            print(f"Caption: {caption}")
            print(f"\nAnnotation: {dense_caption}\n")

            # plot bbox on image
            if Plot:=True:
                W, H = raw_image.size
                
                desc, gtbbox = dense_caption.split("<bbox>")
                gt_box = decode_norm([eval(gtbbox)], H, W)

                txt_pos = [(int(i[0]), int(i[1])) for i in gt_box]
                txt_pos_gt = [(pos[0], pos[1] + 40) for pos in txt_pos]
                # convert raw_image to cv format RGB
                image_np = np.array(raw_image)
                image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                plot_text(image_cv2, [desc], txt_pos_gt, color=(0, 255, 0), fontScale=2)
                plot_text(image_cv2, caption, txt_pos, color=(0, 0, 255), fontScale=2)
                plot_bboxes(image_cv2, bbox=None, gt_box=gt_box)
                save_img(image_cv2, save_path=save_img_path, name=f"blip2_opt2p7b_llava_box_caption_{i}_{image_index}.jpg", verbose=verbose)


def plot_bboxes(raw_image, bbox: Optional[List[List[int]]], gt_box: Optional[List[List[int]]]):
    """
    bbox: List or np.ndarray
    """
    if isinstance(bbox, list) and bbox:
        if not isinstance(bbox[0]):
            bbox = np.array([bbox])

    if isinstance(gt_box, list) and gt_box:
        if not isinstance(gt_box[0]):
            gt_box = np.array([gt_box])
    
    if bbox is not None and bbox.size > 0:
        plot_bbox(raw_image, bbox, c='r')

    if gt_box is not None and gt_box.size > 0:
        plot_bbox(raw_image, gt_box, c='g')


def save_img(raw_image, save_path='./test_output/', image_index=None, name=None, verbose=1):
    save_path = Path(save_path)
    if not save_path.exists():
        save_path.mkdir(parents=True)
    if name is None:
        name=f"blip2_opt2p7b_onlycar_{image_index}.jpg"
    save(raw_image, save_path / name, verbose=verbose)


if __name__ == "__main__":
    tb = TestBlip2()
    image_index = 600

    if stage1:=False:
        image_root = get_cache_path("eventgpt_dataset/BoxCaptionVQA/images")  # testset on stage1
        label_path = get_cache_path("eventgpt_dataset/BoxCaptionVQA/annotations/vqa_llava_style_box_caption_test.json")  # testset on stage1
    else:
        image_root = get_cache_path("eventgpt_dataset/AibeeQA/images")
        label_path = get_cache_path("eventgpt_dataset/AibeeQA/annotations/label_result_530_llava-style-box_caption_en_test.json")  # AibeeQA, testset on stage2

    assert Path(image_root).exists(), f"Image root {image_root} not exists!"
    assert Path(label_path).exists(), f"Label path {label_path} not exists!"
    
    RANDOM_IMAGE = False
    Image_save_path = "./test_output/llava_style_box_caption_stage2"

    USE_NEW_IMAGE_WITH_BBOX = False  # 是否使用新的图片(带 bbox 信息)
    image_path_with_bbox = Path("/training/wphu/Dataset/eventgpt/eventgpt/fewshot_data_eventgpt/person_index/images/")
    
    # Load label
    with open(label_path) as json_file:
        labels = json.load(json_file)

    ## loads BLIP2-OPT-2.7b caption model, without finetuning on coco.
    model_name="blip2_opt_llava_box_caption_roi"
    model_type="pretrain_opt2.7b_llava_box_caption_roi"

    model, vis_processors, _ = load_model_and_preprocess(
        name=model_name, model_type=model_type, is_eval=True, device=device
    )

    tb.test_blip2_llava_style_box_caption(
        model, 
        vis_processors, 
        image_index, 
        verbose=0, 
        save_img_path=Image_save_path,
        img_with_bbox=USE_NEW_IMAGE_WITH_BBOX,
        image_path_with_bbox=image_path_with_bbox
    )

