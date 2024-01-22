EventGPT
-----------
This repository is used for predict the relationship or event between two objects in the images or videos. 

It is a multi-modal model equipped with ImageEncoder and LLM.

# 1. Config
## 1.1 Environments
Start a docker image to run environment.

Note: Before you run the docker as following, you may need to change the local mount path(instead with your own path) first in the `start_docker.sh`. If you can not access to `harbor.aibee.cn/auto_car/visualglm:lavis.v1.1`, you may need to download the docker from [BaiduWebDisk](https://pan.baidu.com/s/11oeqB3JV4X3cYJO73U7hjQ?pwd=nce3), (提取码: nce3).

Before you start docker, you need to load the docker image first.：
```bash
docker load -i visualglm_lavis_v1.1.tar
cd EventGPT  # 切换到工程目录
bash start_docker.sh
```

>Note: Concat part of .tar files: `cat <filename>.part-* > <filename>.tar`

## 1.2 Datasets

This model train by two stages. 

The stage1 trained on the first dataset.

The stage2 trained on AibeeQA dataset, which was labeled by us in handcraft way.

Assumption: The data is putted at path `cache`, which can be configed at `[eventgpt/configs/default.yaml](https://github.com/aibee00/EventGPT/blob/8e2008b2fbae8f1a7fd17943190a62f119782979/eventgpt/configs/default.yaml#L8)`.

You can download dataset from [BaiduWebDisk](https://pan.baidu.com/s/11oeqB3JV4X3cYJO73U7hjQ?pwd=nce3). (提取码: nce3). You should extract it under the path: `./cache`

## 1.3 Run/Model/Datasets/Processors and some others configs

This can be set in the following yaml files:

- `eventgpt/configs/models/blip2/blip2_opt2.7b_llava_box_caption_roi.yaml`
- `eventgpt/projects/blip2/train/pretrain_stage1_llava_box_caption_roi.yaml`
- `eventgpt/projects/blip2/train/pretrain_stage2_llava_box_caption_roi.yaml`

# 2. How to train

## **!!!Prepare**
Before you run, please make sure you have downloaded the following files from [BaiduWebDisk](https://pan.baidu.com/s/11oeqB3JV4X3cYJO73U7hjQ?pwd=nce3). (提取码: nce3):
- `EventGPT/cache/bert-base-uncased`
- `EventGPT/cache/eventgpt_dataset/AibeeQA`
- `EventGPT/cache/eventgpt_dataset/BoxCaptionVQA`

## 2.1 Stage1

```bash
bash run_scripts/blip2/train/pretrain_stage1_llava_box_caption_roi.sh 
```

## 2.2 Stage2
Note: Before you run stage2, please config path of checkpoint which was trained at stage1.
Config at [here](https://github.com/aibee00/EventGPT/blob/2d12aef2b4950419037fdfefac87a099d7c7c83f/eventgpt/projects/blip2/train/pretrain_stage2_llava_box_caption_roi.yaml#L15)

```bash
bash run_scripts/blip2/train/pretrain_stage2_llava_box_caption_roi.sh 
```

# 3. How to evaluate

Note: Before you run evaluation, you may need to config some parameters first at the file `tests/models/test_blip2_vqa.py`.

Run docker and run the script like this:

```bash
python tests/models/test_blip2_vqa.py |& tee test.log
```

# 4. Experiments

To increase the size of the input image and the box features, you need to modify the corresponding configurations in the following YAML files. The specific parameters to change are '**image_size**' and '**box_feat_size**'.:

- `eventgpt/configs/models/blip2/blip2_opt2.7b_llava_box_caption_roi.yaml`
- `eventgpt/projects/blip2/train/pretrain_stage1_llava_box_caption_roi.yaml`
- `eventgpt/projects/blip2/train/pretrain_stage2_llava_box_caption_roi.yaml`

