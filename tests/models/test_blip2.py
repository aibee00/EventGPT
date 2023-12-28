"""
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#

Integration tests for BLIP2 models.
"""

import pytest
import torch
import sys
from pathlib import Path
import json
import os
from tqdm import tqdm
from multiprocessing import Pool

sys.path.append(Path(__file__).parents[2].as_posix())

from eventgpt.models import load_model, load_model_and_preprocess
from PIL import Image
import pickle
import numpy as np

# setup device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load sample image
raw_image = Image.open("docs/_static/merlion.png").convert("RGB")
raw_image = Image.open("docs/_static/ch01001_20210728154845.jpg").convert("RGB")

# load sample , new add by me
label_path = "/ssd/wphu/Dataset/eventgpt/eventgpt/annotations/label_train.json"
# label_path = "/ssd/wphu/Dataset/eventgpt/eventgpt/annotations/label_test.json"
# label_path = "/ssd/wphu/Dataset/eventgpt/eventgpt/annotations/vqa_test.json"
image_root = "/ssd/wphu/Dataset/eventgpt/eventgpt/images"
# Load label
with open(label_path) as json_file:
    labels = json.load(json_file)
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
image_index = get_index_by_name('GACNE-guangzhou-xhthwk-20210717/ch01001_20210717200617.jpg', '[TASK: 3]')
image_index = 30
image_file = labels[image_index]["image"]

# image_file = "GACNE-guangzhou-xhthwk-20210717/ch01003_20210717194151.jpg"
image_path = Path(image_root, image_file).as_posix()
print(f"Image_path: {image_path}, task: {labels[image_index]['task']}")
raw_image = Image.open(image_path).convert("RGB")

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


def forward_image(sample, vis_processors, model):
    """ Forward image

    sample: dict
    """
    image_file = sample['image']
    image_path = Path(image_root, image_file).as_posix()
    print(f"Image_path: {image_path}")
    raw_image = Image.open(image_path).convert("RGB")
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    model.update_image_embeds({"image": image, "image_path": image_file})


def get_processed_image(params):
    vis_processors, label = params
    image_file = label['image']
    image_path = Path(image_root, image_file).as_posix()
    print(f"Image_path: {image_path}")
    raw_image = Image.open(image_path).convert("RGB")
    image = vis_processors["eval"](raw_image).unsqueeze(0)
    return image


class TestBlip2:
    def test_blip2_Qformer(self):
        # Get label
        image_index = 30
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
        
        assert caption == ["the merlion fountain in singapore"]
        print(f"\n Caption: {caption}")
        print(f"\n Annotation: {annotation}")
        print(f"\n context: {context}")
        print(f"\n prompt: {prompt}")
        # print(f"\n question: {question}")

    
    def test_blip2_opt2p7b(self):
        # loads BLIP2-OPT-2.7b caption model, without finetuning on coco.
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device
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
        caption = model.generate({"image": image, "prompt": context + prompt}, use_nucleus_sampling=True, max_length=512, temperature=1)
        # caption = model.generate({"image": image}, use_nucleus_sampling=True, max_length=512, temperature=1)

        print(f"\n Caption: {caption}")
        # Get label
        annotation = labels[image_index]["caption"]
        print(f"\n Annotation: {annotation}")

        import pdb; pdb.set_trace()

        # assert caption == ["the merlion fountain in singapore"]

        # # generate multiple captions
        # captions = model.generate({"image": image}, num_captions=3)

        # assert len(captions) == 3

    def test_blip2_opt2p7b_oracle(self):
        # loads BLIP2-OPT-2.7b caption model, without finetuning on coco.
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_opt_oracle", model_type="pretrain_opt2.7b", is_eval=True, device=device
        )

        # preprocess the image
        # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        # prompt
        prompt = labels[image_index]["task"]

        # generate caption
        caption = model.generate({"image": image, "prompt": prompt}, use_nucleus_sampling=True, max_length=64, temperature=1)
        # caption = model.generate({"image": image}, use_nucleus_sampling=True, max_length=64, temperature=1)

        print(f"\n Caption: {caption}")
        # Get label
        annotation = labels[image_index]["caption"]
        print(f"\n Annotation: {annotation}")

        import pdb; pdb.set_trace()

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

    def save_image_to_embeds(self, labels):
        """ 测试了一下，一个image转为embeds保存需要1.4M(.npy: 1.4M, .pt: 14M)，而Image存储只要1M.
        """
        infer_vit = True
        images = []

        # loads BLIP2-OPT-2.7b caption model, without finetuning on coco.
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_context", model_type="pretrain_eval", is_eval=True, device=device
        )

        # save all image embeds
        root_path = os.path.dirname(image_root)
        save_dir = os.path.join(root_path, "image_embeds")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if (Path(save_dir) / "images.pkl").exists():
            with open(os.path.join(save_dir, "images.pkl"), "rb") as f:
                images = pickle.load(f)
        else:
            # ## Multi process
            # pool = Pool(32)
            # params = [[vis_processors, label] for label in labels]
            # images = pool.map(get_processed_image, params)
            # pool.close()
            # pool.join()
            # print("Processing images done!")

            # single process
            images = []
            for label in tqdm(labels):
                image_file = label['image']
                image_path = Path(image_root, image_file).as_posix()
                print(f"Image_path: {image_path}")
                raw_image = Image.open(image_path).convert("RGB")
                image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
                images.append(image)

            # save images to pkl
            with open(os.path.join(save_dir, "images.pkl"), "wb") as f:
                pickle.dump(images, f)

        if infer_vit:
            # Inference on vit
            bs = 10
            # for bs_image, bs_label in tqdm(zip(images[::bs], labels[::bs])):
            num_steps = len(labels) // bs if len(labels) % bs == 0 else len(labels) // bs + 1
            for step in tqdm(range(num_steps)):
                if not images[step*bs:(step+1)*bs]:
                    continue
                            
                bs_image = torch.concatenate(images[step*bs:(step+1)*bs], dim=0).to(device)
                bs_label = labels[step*bs:(step+1)*bs]
                bs_image_embeds = model.get_image_embeds(bs_image)

                # save image_embeds
                for i, (image_embeds, label) in enumerate(zip(bs_image_embeds, bs_label)):
                    image_file = label['image']
                    save_path = os.path.join(save_dir, image_file.replace(".jpg", ".npy"))
                    save_path = Path(save_path)
                    if not save_path.parent.exists():
                        os.makedirs(save_path.parent)
                    print(f"image_embeds saved: {save_path.as_posix()}")
                    # torch.save(image_embeds, save_path.as_posix())
                    if not save_path.exists():
                        np.save(save_path.as_posix(), image_embeds.numpy())
        

if __name__ == "__main__":
    tb = TestBlip2()

    if sys.argv[1] == "0":
        tb.test_blip2_Qformer()
    elif sys.argv[1] == "1":
        tb.test_blip2_opt2p7b()
    elif sys.argv[1] == "2":
        tb.test_blip2_opt2p7b_oracle()
    else:
        # 直接处理的话图片太多会OOM
        max_length = len(labels)
        max_items = 10000
        steps = max_length // max_items if (max_length % max_items) == 0 else max_length // max_items + 1
        for i in range(steps):
            tb.save_image_to_embeds(labels[i * max_items: (i + 1) * max_items])
