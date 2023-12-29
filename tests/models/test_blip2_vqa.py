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
from eventgpt.common.utils import get_cache_path

sys.path.append(Path(__file__).parents[2].as_posix())

from eventgpt.models import load_model, load_model_and_preprocess
from PIL import Image
import numpy as np
import torch.nn as nn

from eventgpt.models.blip2_models.sam_model.prompt_encoder import PromptEncoder
from transformers import BertTokenizer

from plot_box_to_img import plot_bbox, save, decode_norm, plot_text, parse_bbox

# setup device to use
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BERT_LOCAL_PATH = "/training/wphu/Checkpoints/bert-base-uncased"

# load sample image
raw_image = Image.open("docs/_static/merlion.png").convert("RGB")
raw_image = Image.open("docs/_static/ch01001_20210728154845.jpg").convert("RGB")

# load sample , new add by me
# label_path = "/ssd/wphu/Dataset/eventgpt/eventgpt/annotations/label_test.json"
# label_path = "/ssd/wphu/Dataset/eventgpt/eventgpt/annotations/vqa_test.json"
label_path = "/training/wphu/Dataset/eventgpt/eventgpt/annotations/vqa_box_caption_test.json"
image_root = "/training/wphu/Dataset/eventgpt/eventgpt/images"
# Load label
# with open(label_path) as json_file:
    # labels = json.load(json_file)
# Get image
def get_index_by_name(name, task):
    for i, label in enumerate(labels):
        if label["image"] == name and task == label['task']:
            print(f"name: {name}, index: {i}")
            print("find it")
            return i
    print(f"Not found! name: {name}")
    return -1

# image_index = get_index_by_name('GACNE-guangzhou-xhthwk-20210717/ch01001_20210717161746.jpg')  # ch1
# image_index = get_index_by_name('GACNE-guangzhou-xhthwk-20210717/ch01002_20210717195940.jpg')  # ch2
# image_index = get_index_by_name('GACNE-guangzhou-xhthwk-20210717/ch01003_20210717163112.jpg')  # ch3
# image_index = get_index_by_name('GACNE-guangzhou-xhthwk-20210717/ch01004_20210717134417.jpg')  # ch4
# image_index = get_index_by_name('GACNE-guangzhou-xhthwk-20210717/ch01001_20210717200617.jpg', '[TASK: 3]')
# image_file = labels[image_index]["image"]

# image_file = "GACNE-guangzhou-xhthwk-20210717/ch01003_20210717194151.jpg"
# image_path = Path(image_root, image_file).as_posix()
# print(f"Init Image_path: {image_path}, task: {labels[image_index]['task']}")
# raw_image = Image.open(image_path).convert("RGB")

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
    
    def test_blip2_Qformer(self):
        # Get label
        image_index = 30
        annotation = labels[image_index]["caption"]
        ann = labels[image_index]
        # context = f"<context>{ann['bbox']}##{ann['dense_caption']}##{ann['caption_blip2']}</context>"
        context = f"<context>{ann['dense_caption']}</context>"
        prompt = labels[image_index]["prompt"]
        answer = labels[image_index]["answer"]

        question = labels[image_index]["question"]
        options = labels[image_index]["options"]
        options = [options]

        batch_size = 1
        options_embeds = self.get_options_embeds(options, batch_size, device).to(device)  # (B, N_s, hidden_size)

        # Convert text to embedding
        question_tokens = self.tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(device)
        question_embeds = self.word_embeddings(question_tokens.input_ids)  # (B, N_t, hidden_size)

        # Suffix of prompt '###Answer: '
        suffix = ["###Answer: " for i in range(batch_size)]
        suffix_tokens = self.tokenizer(
            suffix,
            padding="max_length",
            truncation=True,
            max_length=16,
            return_tensors="pt",
        ).to(device)
        suffix_embeds = self.word_embeddings(suffix_tokens.input_ids)  # (B, N_s, hidden_size)

        text_embeds = torch.cat((question_embeds, options_embeds, suffix_embeds), dim=1)  # torch.Size([3, 144, 768])

        # loads BLIP2-OPT-2.7b caption model, without finetuning on coco.
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_sam", model_type="pretrain_eval", is_eval=True, device=device
        )

        # preprocess the image
        # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        # generate caption
        # caption = model.generate({"image": image}, use_nucleus_sampling=True, max_length=512)
        caption = model.generate({"image": image, "prompt_embeds": text_embeds}, use_nucleus_sampling=True, max_length=512)
        
        # assert caption == ["the merlion fountain in singapore"]
        print(f"\n Caption: {caption}")
        print(f"\n Question: {question}")
        print(f"\n Options: {options}")
        print(f"\n Answer: {answer}")
    

    def test_blip2_Qformer_raw(self):
        # Get label
        annotation = labels[image_index]["caption"]
        ann = labels[image_index]
        # context = f"<context>{ann['bbox']}##{ann['dense_caption']}##{ann['caption_blip2']}</context>"
        context = f"<context>{ann['dense_caption']}</context>"
        prompt = labels[image_index]["prompt"]
        # question = labels[image_index]["question"]

        # loads BLIP2-OPT-2.7b caption model, without finetuning on coco.
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_context", model_type="pretrain_eval", is_eval=True, device=device
        )

        # preprocess the image
        # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        # generate caption
        # caption = model.generate({"image": image}, use_nucleus_sampling=True, max_length=512)
        caption = model.generate({"image": image, "prompt": context + prompt}, use_nucleus_sampling=True, max_length=512)

        # assert caption == ["the merlion fountain in singapore"]
        print(f"\n Caption: {caption}")
        print(f"\n Annotation: {annotation}")
        print(f"\n context: {context}")
        print(f"\n prompt: {prompt}")
        # print(f"\n question: {question}")

    
    def test_blip2_opt2p7b(self):
        # loads BLIP2-OPT-2.7b caption model, without finetuning on coco.
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_opt_instruct", model_type="pretrain_opt2.7b", is_eval=True, device=device
        )

        # preprocess the image
        # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        # context
        ann = labels[image_index]
        # context = f"<context>{ann['bbox']}##{ann['dense_caption']}##{ann['caption_blip2']}</context>"
        context = f"<context>{ann['dense_caption']}</context>"

        # prompt
        # prompt = labels[image_index]["prompt"]
        prompt = labels[image_index]["question"]

        # generate caption
        # caption = model.generate({"image": image, "prompt": context + prompt, "image_path": [image_path]}, use_nucleus_sampling=True, max_length=512, temperature=1)
        caption = model.generate({"image": image}, use_nucleus_sampling=True, max_length=512, temperature=1)

        print(f"\n Caption: {caption}")
        # Get label
        annotation = labels[image_index]["caption"]
        print(f"\n Annotation: {annotation}")

        import pdb; pdb.set_trace()

        # assert caption == ["the merlion fountain in singapore"]

        # # generate multiple captions
        # captions = model.generate({"image": image}, num_captions=3)

        # assert len(captions) == 3

    def test_blip2_opt2p7b_instruct(self, image_index, model, vis_processors):
        image_file = labels[image_index]["image"]
        image_path = Path(image_root, image_file).as_posix()
        print(f"Image_path: {image_path}, task: {labels[image_index]['task']}")
        raw_image = Image.open(image_path).convert("RGB")

        # loads BLIP2-OPT-2.7b caption model, without finetuning on coco.
        # model, vis_processors, _ = load_model_and_preprocess(
        #     name="blip2_opt_instruct", model_type="pretrain_opt2.7b", is_eval=True, device=device
        # )

        # preprocess the image
        # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        # context
        ann = labels[image_index]
        # context = f"<context>{ann['bbox']}##{ann['dense_caption']}##{ann['caption_blip2']}</context>"
        # context = f"<context>{ann['dense_caption']}</context>"

        # prompt
        answer = labels[image_index]["answer"]
        question = labels[image_index]["question"]
        options = labels[image_index]["options"]
        prompt = f"Given the image, answer the question. Question: '{question} {options}'. Answer:"

        # generate caption
        caption = model.generate({"image": image, "prompt": prompt, "image_path": [image_path]}, use_nucleus_sampling=True, max_length=512, temperature=1)
        # caption = model.generate({"image": image}, use_nucleus_sampling=True, max_length=512, temperature=1)

        # plot
        plot = False
        if plot:
            bbox = caption[0].split('[')[-1].split(']')[0].split(',')
            bbox = [int(x) for x in bbox]
            gt_box = answer[0].split('[')[-1].split(']')[0].split(',')
            gt_box = [int(x) for x in gt_box]

            # convert raw_image to cv format RGB
            image_np = np.array(raw_image)
            raw_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            plot_bbox(raw_image, [bbox], c='r')
            plot_bbox(raw_image, [gt_box], c='g')
            save_path = Path('./test_output/')
            if not save_path.exists():
                save_path.mkdir(parents=True)
            save(raw_image, save_path / f"test_blip2_opt2p7b_instruct_{image_index}.jpg")

        print(f"Caption: {caption}; answer: {answer}")
        print(f"prompt: {prompt}")


    def test_blip2_opt2p7b_oracle(self):
        # loads BLIP2-OPT-2.7b caption model, without finetuning on coco.
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_opt_oracle_context", model_type="pretrain_opt2.7b", is_eval=True, device=device
        )

        # preprocess the image
        # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        # prompt
        # context = labels[image_index]["dense_caption"]
        question = labels[image_index]["question"]
        options = labels[image_index]["options"]

        print(f"==question: {question}")
        print(f"==options: {options}")

        # generate caption
        caption = model.generate({"image": image, "text_input": question, "options": options}, use_nucleus_sampling=True, max_length=512, temperature=1)
        # caption = model.generate({"image": image, "context": context, "question": question}, use_nucleus_sampling=True, max_length=512, temperature=1)
        # caption = model.generate({"image": image}, use_nucleus_sampling=True, max_length=64, temperature=1)

        print(f"\n Caption: {caption}")
        # Get label
        # annotation = labels[image_index]["caption"]
        # print(f"\n Annotation: {annotation}")

        answer = labels[image_index]["answer"]
        print(f"\n Answer: {answer}")

        import pdb; pdb.set_trace()

    def test_blip2_box_caption(self, image_index = 600):
        # Get label
        image_file = labels[image_index]["image"]
        image_path = Path(image_root, image_file).as_posix()
        print(f"Image_path: {image_path}")
        raw_image = Image.open(image_path).convert("RGB")

        ann = labels[image_index]
        dense_caption = ann["dense_caption"]
        print(f"==dense_caption: {dense_caption}")
        annotation, box = dense_caption

        batch_size = 1
        box_embeds = self.get_options_embeds([[box]], batch_size, device).to(device)  # (B, N_s, hidden_size)

        # loads BLIP2-OPT-2.7b caption model, without finetuning on coco.
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_box_caption", model_type="pretrain_eval", is_eval=True, device=device
        )

        # preprocess the image
        # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        # generate caption
        # caption = model.generate({"image": image}, use_nucleus_sampling=True, max_length=512)
        for i in range(10):
            caption = model.generate({"image": image, "prompt_embeds": box_embeds}, use_nucleus_sampling=True, max_length=64)
            print(f"Caption: {caption}")
        print(f"\nAnnotation: {annotation}\n")
        import pdb; pdb.set_trace()

    def test_blip2_sam(self, image_index=6):
        # Get label
        image_file = labels[image_index]["image"]
        image_path = Path(image_root, image_file).as_posix()
        print(f"Image_path: {image_path}")
        raw_image = Image.open(image_path).convert("RGB")

        ann = labels[image_index]
        dense_caption = ann["dense_caption"]
        print(f"==dense_caption: {dense_caption}")
        annotation, box = dense_caption

        batch_size = 1
        box_embeds = self.get_options_embeds([[box]], batch_size, device).to(device)  # (B, N_s, hidden_size)

        # loads BLIP2-OPT-2.7b caption model, without finetuning on coco.
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_sam", model_type="pretrain_sam", is_eval=True, device=device
        )

        # preprocess the image
        # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        # generate caption
        # caption = model.generate({"image": image}, use_nucleus_sampling=True, max_length=512)
        for i in range(10):
            caption = model.generate({"image": image, "dense_caption": [dense_caption]}, use_nucleus_sampling=True, max_length=64)
            print(f"Caption: {caption}")
        print(f"\nAnnotation: {annotation}\n")
        import pdb; pdb.set_trace()

    def test_blip2_sam_roi(self, image_index=60):
        # Get label
        image_file = labels[image_index]["image"]
        image_path = Path(image_root, image_file).as_posix()
        print(f"Image_path: {image_path}")
        raw_image = Image.open(image_path).convert("RGB")

        ann = labels[image_index]
        dense_caption = ann["dense_caption"]
        annotation, box = dense_caption

        batch_size = 1
        box_embeds = self.get_options_embeds([[box]], batch_size, device).to(device)  # (B, N_s, hidden_size)

        # loads BLIP2-OPT-2.7b caption model, without finetuning on coco.
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_sam_roi", model_type="pretrain_sam_roi", is_eval=True, device=device
        )

        # preprocess the image
        # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        # generate caption
        # caption = model.generate({"image": image}, use_nucleus_sampling=True, max_length=512)
        for i in range(10):
            caption = model.generate({"image": image, "dense_caption": [dense_caption]}, use_nucleus_sampling=True, max_length=64)
            print(f"Caption: {caption}")
        print(f"\nAnnotation: {annotation}\n")
        print(f"dense_caption: {dense_caption}\n")

    def test_blip2_oracle_personcar(self, image_index=60):
        # Get label
        image_file = labels[image_index]["image"]
        image_path = Path(image_root, image_file).as_posix()
        print(f"Image_path: {image_path}")
        raw_image = Image.open(image_path).convert("RGB")

        ann = labels[image_index]
        bbox = ann["bbox"]

        # loads BLIP2-OPT-2.7b caption model, without finetuning on coco.
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_opt_oracle_personcar", model_type="pretrain_opt2.7b_oracle_personcar", is_eval=True, device=device
        )

        # preprocess the image
        # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        # generate caption
        for i in range(10):
            caption = model.generate({"image": image}, use_nucleus_sampling=True, max_length=256, temperature=1)
            print(f"Caption: {caption}")
        print(f"\nAnnotation: {bbox}\n")

    def test_blip2_oracle_onlycar(self, model, vis_processors, image_index=60):
        total_num_images = 8000
        num_iters = 5  # 测试样本的个数
        chosen_set = set()
        results = []
        for i in range(num_iters):
            image_index = random.choice(range(total_num_images))
            while image_index in chosen_set:
                image_index = random.choice(range(total_num_images))
            chosen_set.add(image_index)
            print(f" {i}-th Image Index: {image_index} ".center(50, '-'))

            # Get label
            image_file = labels[image_index]["image"]
            image_path = Path(image_root, image_file).as_posix()
            print(f"Image_path: {image_path}")
            raw_image = Image.open(image_path).convert("RGB")

            ann = labels[image_index]
            dense_caption = ann["dense_caption"]

            # preprocess the image
            # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

            # generate caption
            caption = model.generate({"image": image, "dense_caption": dense_caption}, use_nucleus_sampling=True, max_length=32, temperature=1)
            print(f"Caption: {caption}")
            print(f"\nAnnotation: {dense_caption}\n")

            # plot bbox on image
            W, H = raw_image.size
            try:
                bbox_vis = decode_norm([eval(caption[0])], H, W)
            except:
                try:
                    print(f"caption: {caption[0]}, trying again to split out the first bbox...")
                    bbox_vis = decode_norm([eval(caption[0].split(' [')[0])], H, W)
                except:
                    print(f"caption: {caption[0]}, Skip it since it is unregular")
                    continue
            
            desc, gtbbox = dense_caption.split(":")
            gt_box = decode_norm([eval(gtbbox)], H, W)

            print(f"bbox: {bbox_vis}, gt_box: {gt_box}")
            results.append({'bbox': bbox_vis.tolist(), 'gt_box': gt_box.tolist()})
            
            txt_pos = [(int(i[0]), int(i[1])) for i in gt_box]
            # convert raw_image to cv format RGB
            image_np = np.array(raw_image)
            image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            plot_text(image_cv2, [desc], txt_pos, fontScale=1)
            plot_bboxes(image_cv2, bbox_vis, gt_box)
            save_img(image_cv2, save_path='./test_output/onlycar', name=f"blip2_opt2p7b_onlycar_{i}_{image_index}.jpg")

        with open('results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    def test_blip2_oracle_onlycar_override(self, model, vis_processors, image_index=60):
        """ 重写test_blip2_oracle_onlycar
        这个函数对应的实验是: "给定bbox预测caption"
        所以，在可视化的时候，可视化的不再是pred_bboxes,而是某个bbox的caption.
        """
        total_num_images = 8000
        num_iters = 100  # 测试样本的个数
        chosen_set = set()
        for i in range(num_iters):
            image_index = random.choice(range(total_num_images))
            while image_index in chosen_set:
                image_index = random.choice(range(total_num_images))
            chosen_set.add(image_index)
            print(f" {i}-th Image Index: {image_index} ".center(50, '-'))

            # Get label
            image_file = labels[image_index]["image"]
            image_path = Path(image_root, image_file).as_posix()
            print(f"Image_path: {image_path}")
            raw_image = Image.open(image_path).convert("RGB")

            ann = labels[image_index]
            dense_caption = ann["dense_caption"]

            # preprocess the image
            # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

            # generate caption
            caption = model.generate({"image": image, "dense_caption": dense_caption}, use_nucleus_sampling=True, max_length=32, temperature=1)
            print(f"Caption: {caption}")
            print(f"\nAnnotation: {dense_caption}\n")

            # plot bbox on image
            if Plot:=True:
                W, H = raw_image.size
                
                desc, gtbbox = dense_caption.split(":")
                gt_box = decode_norm([eval(gtbbox)], H, W)

                txt_pos = [(int(i[0]), int(i[1])) for i in gt_box]
                txt_pos_gt = [(pos[0], pos[1] + 40) for pos in txt_pos]
                # convert raw_image to cv format RGB
                image_np = np.array(raw_image)
                image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                plot_text(image_cv2, [desc], txt_pos_gt, color=(0, 255, 0), fontScale=2)
                plot_text(image_cv2, caption, txt_pos, color=(0, 0, 255), fontScale=2)
                plot_bboxes(image_cv2, bbox=None, gt_box=gt_box)
                save_img(image_cv2, save_path='./test_output/onlycar_override', name=f"blip2_opt2p7b_onlycar_caption_{i}_{image_index}.jpg")

    def test_blip2_oracle_onlyperson(self, model, vis_processors, image_index=60):
        total_num_images = 8000
        num_iters = 100  # 测试样本的个数
        chosen_set = set()
        for i in range(num_iters):
            image_index = random.choice(range(total_num_images))
            while image_index in chosen_set:
                image_index = random.choice(range(total_num_images))
            chosen_set.add(image_index)
            print(f" {i}-th Image Index: {image_index} ".center(50, '-'))

            # Get label
            image_file = labels[image_index]["image"]
            image_path = Path(image_root, image_file).as_posix()
            print(f"Image_path: {image_path}")
            raw_image = Image.open(image_path).convert("RGB")

            ann = labels[image_index]
            bbox = ann["bbox"]

            # preprocess the image
            # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

            # generate caption
            caption = model.generate({"image": image}, use_nucleus_sampling=True, max_length=256, temperature=1)
            print(f"Caption: {caption}")
            print(f"\nAnnotation: {bbox}\n")

            # plot bbox on image
            if Plot:=True:
                W, H = raw_image.size
                gt_desc, gt_boxes = parse_bbox(bbox)
                gt_boxes = decode_norm(gt_boxes, H, W)

                try:
                    pred_desc, pred_bbox = caption[0].split(":")
                except:
                    print(f"caption 格式错误: {caption}")
                finally:
                    # 去掉末尾除']'的额外字符
                    while pred_bbox != "" and pred_bbox[-1] != "]":
                        pred_bbox = pred_bbox[:-1]
                    
                    if pred_bbox == "":
                        continue

                    if pred_bbox[0] != "[":
                        continue

                    if len(pred_bbox.split(',')) != 4:
                        continue

                    pred_bbox = decode_norm([eval(pred_bbox)], H, W)
                
                    # 可视化预测结果
                    # convert raw_image to cv format RGB
                    image_np = np.array(raw_image)
                    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                    plot_bboxes(image_cv2, bbox=pred_bbox, gt_box=gt_boxes)
                    save_img(image_cv2, save_path="./test_output/onlyperson", name=f"blip2_opt2p7b_onlyperson_{i}_{image_index}.jpg")
     
    def test_blip2_opt2p7b_coco(self):
        # loads BLIP2-OPT-2.7b caption model,
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_opt",
            model_type="caption_coco_opt2.7b",
            is_eval=True,
            device=device,
        )

        # preprocess the image
        # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        # generate caption
        caption = model.generate({"image": image})

        assert caption == ["a statue of a mermaid spraying water into the air"]

        # generate multiple captions
        captions = model.generate({"image": image}, num_captions=3)

        assert len(captions) == 3

    def test_blip2_opt6p7b(self):
        # loads BLIP2-OPT-2.7b caption model,
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_opt", model_type="pretrain_opt6.7b", is_eval=True, device=device
        )

        # preprocess the image
        # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        # generate caption
        caption = model.generate({"image": image})

        assert caption == ["a statue of a merlion in front of a water fountain"]

        # generate multiple captions
        captions = model.generate({"image": image}, num_captions=3)

        assert len(captions) == 3

    def test_blip2_opt6p7b_coco(self):
        # loads BLIP2-OPT-2.7b caption model,
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_opt",
            model_type="caption_coco_opt6.7b",
            is_eval=True,
            device=device,
        )

        # preprocess the image
        # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        # generate caption
        caption = model.generate({"image": image})

        assert caption == ["a large fountain spraying water into the air"]

        # generate multiple captions
        captions = model.generate({"image": image}, num_captions=3)

        assert len(captions) == 3

    def test_blip2_flant5xl(self):
        # loads BLIP2-FLAN-T5XL caption model,
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device
        )

        # preprocess the image
        # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        # generate caption
        caption = model.generate({"image": image})

        assert caption == ["marina bay sands, singapore"]

        # generate multiple captions
        captions = model.generate({"image": image}, num_captions=3)

        assert len(captions) == 3

    def test_blip2_flant5xxl(self):
        # loads BLIP2-FLAN-T5XXL caption model,
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_t5",
            model_type="pretrain_flant5xxl",
            is_eval=True,
            device=device,
        )

        # preprocess the image
        # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        # generate caption
        caption = model.generate({"image": image})

        assert caption == ["the merlion statue in singapore"]

        # generate multiple captions
        captions = model.generate({"image": image}, num_captions=3)

        assert len(captions) == 3


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

    if sys.argv[1] == "0":
        tb.test_blip2_Qformer()
    elif sys.argv[1] == "1":
        tb.test_blip2_opt2p7b()
    elif sys.argv[1] == "2":
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_opt_instruct", model_type="pretrain_opt2.7b", is_eval=True, device=device
        )
        num_images = int(sys.argv[2])
        for image_index in range(num_images):
            print(f" Sample {image_index} ".center(100, '-'))
            tb.test_blip2_opt2p7b_instruct(image_index, model, vis_processors)
    elif sys.argv[1] == "3":
        tb.test_blip2_opt2p7b_oracle()
    elif sys.argv[1] == "4":
        tb.test_blip2_box_caption(image_index)
    elif sys.argv[1] == "5":
        tb.test_blip2_sam(image_index)
    elif sys.argv[1] == "6":
        tb.test_blip2_sam_roi(image_index)
    elif sys.argv[1] == "7" or sys.argv[1] == "9":
        label_path = "/ssd/wphu/Dataset/eventgpt/eventgpt/annotations/vqa_oracle_test.json"
        # Load label
        with open(label_path) as json_file:
            labels = json.load(json_file)
        
        if sys.argv[1] == "7":
            tb.test_blip2_oracle_personcar(image_index)
        else:
            # loads BLIP2-OPT-2.7b caption model, without finetuning on coco.
            model, vis_processors, _ = load_model_and_preprocess(
                name="blip2_opt_oracle_onlyperson", model_type="pretrain_opt2.7b_oracle_onlyperson", is_eval=True, device=device
            )
            tb.test_blip2_oracle_onlyperson(model, vis_processors, image_index)
    
    elif sys.argv[1] == "8":
        label_path = "/ssd/wphu/Dataset/eventgpt/eventgpt/annotations/vqa_oracle_test.json"
        # Load label
        with open(label_path) as json_file:
            labels = json.load(json_file)

        # loads BLIP2-OPT-2.7b caption model, without finetuning on coco.
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_opt_oracle_onlycar", model_type="pretrain_opt2.7b_oracle_onlycar", is_eval=True, device=device
        )
        tb.test_blip2_oracle_onlycar = tb.test_blip2_oracle_onlycar_override
        tb.test_blip2_oracle_onlycar(model, vis_processors, image_index)

    elif sys.argv[1] == "10":
        # image_root = "eventgpt_dataset/BoxCaptionVQA/images"  # pretrain data(2万) image generated by grit, testset on stage1
        # label_path = "eventgpt_dataset/BoxCaptionVQA/annotations/vqa_llava_style_box_caption_test.json"  # pretrain data(2万) label generated by grit, testset on stage1

        image_root = get_cache_path("eventgpt_dataset/AibeeQA/images")
        label_path = "eventgpt_dataset/AibeeQA/annotations/label_result_530_llava-style-box_caption_en_test.json"  # AibeeQA, testset on stage2

        assert Path(image_root).exists(), f"Image root {image_root} not exists!"
        assert Path(label_path).exists(), f"Label path {label_path} not exists!"
        
        RANDOM_IMAGE = False
        Image_save_path = "./test_output/llava_style_box_caption_stage2"

        USE_NEW_IMAGE_WITH_BBOX = False  # 是否使用新的图片(带 bbox 信息)
        image_path_with_bbox = Path("/training/wphu/Dataset/eventgpt/eventgpt/fewshot_data_eventgpt/person_index/images/")
        
        # Load label
        with open(get_cache_path(label_path)) as json_file:
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

