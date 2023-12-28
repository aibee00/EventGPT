"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import random
from packaging import version
import numpy as np

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torchvision.ops as ops

from eventgpt.common.registry import registry
from eventgpt.models.blip2_models.blip2 import Blip2Base, disabled_train
# from eventgpt.models.blip2_models.modeling_opt import OPTForCausalLM, OPTConfig
from transformers import AutoTokenizer, OPTForCausalLM, OPTConfig
import transformers


@registry.register_model("blip2_opt_llava_box_caption_roi")  # copy from onlycar model
class Blip2OPTLlavaBoxCaptionRoi(Blip2Base):
    """
    BLIP2 OPT model.
    Supported model types:
        - pretrained_opt2.7b: pretrained model with OPT2.7b
        - pretrained_opt6.7b: pretrained model with OPT6.7b
        - caption_coco_opt2.7b: fintuned image captioning model with OPT2.7b
        - caption_coco_opt6.7b: fintuned image captioning model with OPT6.7b
    Usage:
        >>> from eventgpt.models import load_model
        >>> model = load_model("blip2_opt", "caption_coco_opt2.7b")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_opt2.7b": "configs/models/blip2/blip2_pretrain_opt2.7b.yaml",
        "pretrain_opt6.7b": "configs/models/blip2/blip2_pretrain_opt6.7b.yaml",
        "caption_coco_opt2.7b": "configs/models/blip2/blip2_caption_opt2.7b.yaml",
        "caption_coco_opt6.7b": "configs/models/blip2/blip2_caption_opt6.7b.yaml",
        "pretrain_opt2.7b_llava_box_caption": "configs/models/blip2/blip2_opt2.7b_llava_box_caption.yaml",
        "pretrain_opt2.7b_llava_box_caption_roi": "configs/models/blip2/blip2_opt2.7b_llava_box_caption_roi.yaml",
        "pretrain_opt2.7b_llava_box_caption_vit_L364": "configs/models/blip2/blip2_opt2.7b_llava_box_caption_vit_L364.yaml",
    }

    # questions_list stage1 brieflly caption
    # questions = [
    #     "<image>\nA short image caption for the bounding box:{} in this image.",
    #     "<image>\nA short image description for the bounding box:{} in this image.",
    #     "<image>\nWrite a short description for the bounding box:{} in this image.",
    #     "<image>\nWrite a description for the bounding box:{} in this image.",
    #     "<image>\nProvide a description of what is presented in the the bounding box:{} in this image.",
    #     "<image>\nBriefly describe the content of the the bounding box:{} in this image.",
    #     "<image>\nCan you briefly explain what you see in the the bounding box:{} in this image?",
    #     "<image>\nCould you use a few words to describe what you perceive in the bounding box:{} in this image?",
    #     "<image>\nPlease provide a short depiction of the bounding box:{} in this picture.",
    #     "<image>\nUsing language, provide a short account of the bounding box:{} in this image.",
    #     "<image>\nUse a few words to illustrate what is happening in the bounding box:{} in this picture."
    # ]

    # Stage2 questions for detailed caption
    questions = [
        "<image>\nA detailed image caption for the bounding box:{} in this image.",
        "<image>\nA detailed image description for the bounding box:{} in this image.",
        "<image>\nWrite a detailed description for the bounding box:{} in this image.",
        "<image>\nWrite a description for the bounding box:{} in this image.",
        "<image>\nProvide a description of what is presented in the the bounding box:{} in this image.",
        "<image>\nDetailly describe the content of the the bounding box:{} in this image.",
        "<image>\nCan you detailly explain what you see in the the bounding box:{} in this image?",
        "<image>\nCould you use a few words to detailly describe what you perceive in the bounding box:{} in this image?",
        "<image>\nPlease provide a detailed depiction of the bounding box:{} in this picture.",
        "<image>\nUsing language, provide a detailed account of the bounding box:{} in this image.",
        "<image>\nUse a few words to detailly illustrate what is happening in the bounding box:{} in this picture."
    ]

    # questions_list stage1 caption
    questions = [
        "A detailed image caption for the bounding box:{} in this image. Caption:",
        "A detailed image description for the bounding box:{} in this image. Caption:",
        "Write a detailed description for the bounding box:{} in this image. Caption:",
        "Write a description for the bounding box:{} in this image. Caption:",
        "Provide a description of what is presented in the the bounding box:{} in this image. Caption:",
        "Detailly describe the content of the the bounding box:{} in this image. Caption:",
        "Can you detailly explain what you see in the the bounding box:{} in this image? Caption:",
        "Could you use a few words to detailly describe what you perceive in the bounding box:{} in this image? Caption:",
        "Please provide a detailed depiction of the bounding box:{} in this picture. Caption:",
        "Using language, provide a detailed account of the bounding box:{} in this image. Caption:",
        "Use a few words to detailly illustrate what is happening in the bounding box:{} in this picture. Caption:"
    ]

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        box_feat_size=8,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        opt_model="facebook/opt-2.7b",
        prompt="",
        max_txt_len=32,
        apply_lemmatizer=False,
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()
        self.image_size = img_size
        self.box_feat_size = box_feat_size
        transformers_version = version.parse(transformers.__version__)
        assert transformers_version >= version.parse("4.27"), "BLIP-2 OPT requires transformers>=4.27"

        self.vit_model = vit_model
        
        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False)
        self.opt_model = OPTForCausalLM.from_pretrained(
            opt_model, torch_dtype=torch.float16
        )
        for name, param in self.opt_model.named_parameters():
            param.requires_grad = False
        self.eos_token_id = self.opt_tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[0]

        # self.opt_proj = nn.Linear(
        #     self.Qformer.config.hidden_size, self.opt_model.config.hidden_size
        # )
        if vit_model == "swint":
            print(f"fpn.fpn_layer4.out_channels: {self.visual_encoder.fpn.fpn_layer4.out_channels}")
            self.opt_proj = nn.Linear(
                self.visual_encoder.fpn.fpn_layer4.out_channels, self.opt_model.config.hidden_size
            )
        else:
            self.opt_proj = nn.Linear(
                self.visual_encoder.num_features, self.opt_model.config.hidden_size
            )

        self.max_txt_len = max_txt_len
        self.prompt = prompt
        prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)
        
        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None  

        self.debug = False
        self.SEP_PROMPT_CAPTION = '\n' #"</s>"
        self.SEP_TEXT_BBOX = "<bbox>"
    
    @staticmethod
    def normalize_bbox(bbox, image_size):
        """
        Normalize bounding box coordinates based on image size.
        
        Args:
            bbox (torch.Tensor): Bounding box coordinates in the format [x_min, y_min, x_max, y_max].
            image_size (tuple): Tuple containing (image_width, image_height).
            
        Returns:
            torch.Tensor: Normalized bounding box coordinates.
        """
        width, height = image_size
        bbox_normalized = bbox.clone()
        bbox_normalized[0] /= width  # Normalize x_min
        bbox_normalized[1] /= height  # Normalize y_min
        bbox_normalized[2] /= width  # Normalize x_max
        bbox_normalized[3] /= height  # Normalize y_max
        return bbox_normalized 
    
    @staticmethod
    def denorm(bbox, image_size):
        """
        Denormalize bounding box coordinates based on image size.
        
        Args:
            bbox (torch.Tensor): Bounding box coordinates in the format [x_min, y_min, x_max, y_max].
            image_size (tuple): Tuple containing (image_width, image_height).
            
        Returns:
            torch.Tensor: Denormalized bounding box coordinates.
        """
        width, height = image_size
        bbox_denormalized = bbox.clone()
        bbox_denormalized[0] *= width  # Denormalize x_min
        bbox_denormalized[1] *= height  # Denormalize y_min
        bbox_denormalized[2] *= width  # Denormalize x_max
        bbox_denormalized[3] *= height  # Denormalize y_max
        bbox_denormalized = bbox_denormalized.round().int()  # Round to integer values
        return bbox_denormalized

    def preprocess_box_caption(self, dense_caption):
        """ Preprocess of prompt and caption label format
        """
        # Add prompt
        text = []
        prompt = []
        for dcap in dense_caption:
            desc, bbox = dcap.split(self.SEP_TEXT_BBOX)
            # cur_prompt = self.prompt.format(desc)
            # text.append(cur_prompt + bbox + '\n')
            cur_prompt = self.prompt.format(bbox)
            text.append(cur_prompt + desc + '\n')
            prompt.append(cur_prompt)
        return text, prompt
    

    def preprocess_box_caption_random_choice_quesitons(self, dense_caption, image_feature):
        """ Preprocess of prompt and caption label format
        v2 changes: 更改prompt的问题，改为随机从一个question_list选择一个
        """
        # Add prompt
        text = []
        prompt = []
        box_embeds = []  # [bz, 1+16*16, 768]
        for bz_index, dcap in enumerate(dense_caption):
            desc, bbox = dcap.split(self.SEP_TEXT_BBOX)

            question = random.choice(self.questions)
            cur_prompt = question.format(bbox)

            # roi
            patch_size = int(np.sqrt(image_feature.shape[-2] - 1))  # -1 is for cls_token
            bbox_denormalized = self.denorm(torch.tensor(eval(bbox)), (patch_size, patch_size))  # [4,]
            bbox_embed = self.get_roi_feature(image_feature[bz_index: bz_index+1], bbox_denormalized, output_size=(8,8))  # [1, 1+16*16, embed_dim]
            bbox_embed_proj = self.opt_proj(bbox_embed)  # [1, 1+16*16, 768]
            box_embeds.append(bbox_embed_proj)

            text.append(cur_prompt + self.SEP_PROMPT_CAPTION + desc + '\n')
            prompt.append(cur_prompt)
        
        box_embeds = torch.cat(box_embeds, dim=0)  # [bz, 1+16*16, 768]
        return text, prompt, box_embeds
    

    def get_prompt_for_predict_answers(self, text_input, image_feature):
        """ 注意这里的 text_input, 是question 和bbox的组合，不是annos 和bbox
        1. split out question and bbox from dense_caption
        2. get bbox embeds according to bbox coordinates
        3. return prompt and bbox_embeds
        Input:
          - text_input: List[string]
        """
        # Add prompt
        prompt = []
        box_embeds = []  # [bz, 1+16*16, 768]
        for bz_index, dcap in enumerate(text_input):
            print(f"dcap: {dcap}")
            question, bbox = dcap.split(self.SEP_TEXT_BBOX)

            # roi
            patch_size = int(np.sqrt(image_feature.shape[-2] - 1))  # -1 is for cls_token
            bbox_denormalized = self.denorm(torch.tensor(eval(bbox)), (patch_size, patch_size))  # [4,]
            bbox_embed = self.get_roi_feature(image_feature[bz_index: bz_index+1], bbox_denormalized, output_size=(8,8))  # [1, 1+16*16, embed_dim]
            bbox_embed_proj = self.opt_proj(bbox_embed)  # [1, 1+16*16, 768]
            box_embeds.append(bbox_embed_proj)

            prompt.append(question + self.SEP_PROMPT_CAPTION)
        
        box_embeds = torch.cat(box_embeds, dim=0)  # [bz, 1+16*16, 768]
        return prompt, box_embeds
    

    def get_roi_feature(self, feature_map, boxes, output_size=None):
        """
        Args:
            feature_map: [batch_size, 1+height*width, embed_dim] which need convert to [batch_size, num_features, height, width]
            boxes: Tensor([4,]) which need to convert to [batch_size, num_boxes, 4]
        """
        # permute and reshape
        cls_feature = feature_map[:, 0, :].unsqueeze(1)  # [1, 1, embed_dim]
        image_feature = feature_map[:, 1:, :]
        num_features = feature_map.shape[-1]
        height = width = int(np.sqrt(image_feature.shape[-2]))
        image_feature = image_feature.permute(0, 2, 1).reshape(feature_map.shape[0], num_features, height, width)

        # conver boxes to roi format (R, 5), and the 5 is (bz_idx, x1, y1, x2, y2), and the R is num_boxes
        boxes = boxes.reshape(1, -1, 4)  # [1,1,4]
        boxes = torch.cat([torch.zeros((1, boxes.shape[1], 1)), boxes], axis=-1)  # to [1,1,5]
        boxes = boxes.reshape(-1, 5).float().to(self.device)  # [1, 5]

        # boxes的坐标必须是基于feature_map的
        # 这里我们用ops.RoiPool来实现获取roi_feature的过程
        output_size = image_feature.shape[-2:] if output_size is None else output_size  # [h, w]
        roi_feature = ops.roi_pool(
            image_feature,
            boxes,
            output_size=output_size
        )  # [batch_size, num_features, output_size[0], output_size[1]], e.g. [1, embed_dim, h, w]

        # readd back cls_feature and convert roi_feature to [batch_size, 1+height*width, embed_dim]
        roi_feature = roi_feature.reshape(feature_map.shape[0], num_features, -1)
        roi_feature = roi_feature.permute(0, 2, 1)
        roi_feature = torch.cat([cls_feature, roi_feature], dim=1)  # [batch_size, 1+height*width, embed_dim]

        return roi_feature


    def forward(self, samples):
        image = samples["image"]
        with self.maybe_autocast():
            if self.vit_model == "swint":
                image_feats = self.visual_encoder(image)
                image_embeds = []
                for i, image_feat in enumerate(image_feats):
                    image_embed = self.ln_vision(image_feat)
                    image_embed = self.opt_proj(image_embed)
                    image_embeds.append(image_embed)
                inputs_opt = torch.concatenate(image_embeds, dim=1)
            else:
                image_embeds = self.visual_encoder(image)  # (bz, 1+16*16, embed_dim), 16=224/14, 14 is the first conv layer's kernal_size
                # print(f"image_embeds shape: {image_embeds.shape}")
                '''
                >>> print(f"image_embeds shape: {image_embeds.shape}")
                image_embeds shape: torch.Size([4, 257, 1408])
                image_embeds = concat(cls_tokens, x), cls_tokens: [batch_size, 1, embed_dim], x: [batch_size, seq_len, embed_dim]
                '''
                image_embeds = self.ln_vision(image_embeds)
                inputs_opt = self.opt_proj(image_embeds)

        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(image.device)

        self.opt_tokenizer.padding_side = "right"

        dense_caption = samples['dense_caption']

        # text, prompt = self.preprocess_box_caption(dense_caption)
        text, prompt, box_embeds = self.preprocess_box_caption_random_choice_quesitons(dense_caption, image_embeds)
        atts_box_embeds = torch.ones(box_embeds.size()[:-1], dtype=torch.long).to(image.device)

        # prompt token + target token
        opt_tokens = self.opt_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(image.device)

        targets = opt_tokens.input_ids.masked_fill(
            opt_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
        )
        
        # prompt, here the prompt's length maybe different,
        # so we need to padding to same length templly
        prompt_tokens = self.opt_tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_txt_len  # max_len: 64
        )
        self.prompt_length = prompt_tokens.attention_mask.sum(1)
        del prompt_tokens  # rm useless buffers

        # do not apply loss to the prompt
        for i,length in enumerate(self.prompt_length):
            offset = 1  # offset one for bos: </s> or '\n'
            targets[i, : length + offset] = -100  

        # do not apply loss to the query tokens or image tokens
        empty_targets = (
            torch.ones(atts_opt.size(), dtype=torch.long).to(image.device).fill_(-100)  # mask image token
        )

        # do not apply loss to the box embed tokens
        empty_targets_box_embeds = (
            torch.ones(atts_box_embeds.size(), dtype=torch.long).to(image.device).fill_(-100)  # mask box embed token
        )

        targets = torch.cat([empty_targets, empty_targets_box_embeds, targets], dim=1)

        if self.debug:
            text_vis = self.opt_tokenizer.convert_ids_to_tokens(opt_tokens.input_ids[0])
            print(f"{len(text_vis)}({self.max_txt_len}) text: {text_vis}")
            for i in range(len(text)):
                print(f"  ={i}=text: {text[i]}")
                print(f"  ={i}=targets: {self.opt_tokenizer.convert_ids_to_tokens(targets[i])}")
        
        input_token = opt_tokens
        inputs_embeds = self.opt_model.model.decoder.embed_tokens(input_token.input_ids)
        inputs_embeds = torch.cat([inputs_opt, box_embeds, inputs_embeds], dim=1)  # [img, text]
        attention_mask = torch.cat([atts_opt, atts_box_embeds, input_token.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.opt_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["image"]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))

            inputs_opt = self.opt_proj(image_embeds)
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(
                image.device
            )

            if "dense_caption" in samples and samples["dense_caption"] is not None:
                # Add desc to prompt
                dense_caption = samples["dense_caption"]
                text, prompt_list, box_embeds = self.preprocess_box_caption_random_choice_quesitons([dense_caption], image_embeds)
                atts_box_embeds = torch.ones(box_embeds.size()[:-1], dtype=torch.long).to(image.device)
                prompt = prompt_list[0]  # only one batch
                # text: prompt + SEP + desc
            else:
                raise ValueError("Please provide dense_caption in the samples!")

            print(f"==prompt: {prompt}")

            opt_tokens = self.opt_tokenizer(
                prompt,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(image.device)
            
            # new version for transformers>=4.27
            inputs_embeds = self.opt_model.get_input_embeddings()(opt_tokens.input_ids)
            # inputs_embeds = torch.cat([inputs_opt, context_embeds, inputs_embeds],dim=1)
            inputs_embeds = torch.cat([inputs_opt, box_embeds, inputs_embeds],dim=1)  # [img, context, prompt]
            attention_mask = torch.cat([atts_opt, atts_box_embeds, opt_tokens.attention_mask], dim=1)
            
            outputs = self.opt_model.generate(
                inputs_embeds=inputs_embeds, 
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.opt_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            
            output_text = [text.strip() for text in output_text]
            return output_text
        
        
    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=0,
        temperature=1,
        **kwargs
    ):
        image = samples["image"]
        with self.maybe_autocast():
            image_feats = self.visual_encoder(image)
            image_embeds = self.ln_vision(image_feats)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )

            inputs_opt = self.opt_proj(image_embeds)
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(
                image.device
            )

            if isinstance(samples["text_input"], str):
                samples["text_input"] = [samples["text_input"]]
            if prompt:
                text_input = [prompt.format(question) for question in samples["text_input"]]
            else:
                text_input = samples["text_input"]
            
            print(f"text_input 1: {text_input}")

            # split out question and bbox
            prompt_list, bbox_embeds = self.get_prompt_for_predict_answers(text_input, image_feats)
            text_input = prompt_list[0] # only one batch
            bbox_atts = torch.ones(bbox_embeds.size()[:-1], dtype=torch.long).to(image.device)

            print(f"text_input 2: {text_input}")

            self.opt_tokenizer.padding_side = "left"
            opt_tokens = self.opt_tokenizer(
                text_input,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(image.device)
        
            # require transformers>=4.27
            inputs_embeds = self.opt_model.get_input_embeddings()(opt_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_opt, bbox_embeds, inputs_embeds],dim=1)
            attention_mask = torch.cat([atts_opt, bbox_atts, opt_tokens.attention_mask], dim=1)
            
            outputs = self.opt_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=min_len,
                eos_token_id=self.eos_token_id,
                length_penalty=length_penalty,
                temperature=temperature,
            )
            output_text = self.opt_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            output_text = [text.strip() for text in output_text]
        if self._apply_lemmatizer or ("apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]):
            output_text = self._lemmatize(output_text)

        return output_text
    
    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer
        
    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        box_feat_size = cfg.get("box_feat_size", 8)  # default: 8
        num_query_token = cfg.get("num_query_token")
        opt_model = cfg.get("opt_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        
        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            box_feat_size=box_feat_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            opt_model=opt_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer,
        )
        model.load_checkpoint_from_config(cfg)

        return model
