EventGPT
-----------
This repository is used for predict the relationship or event between two objects in the images or videos. 

It is a multi-modal model equipped with ImageEncoder and LLM.

# 1. Config
## 1.1 Environments
Start a docker image to run environment.

Note: Before you run the docker as following, you may need to change the local mount path(instead with your own path) first in the `start_docker.sh`. If you can not access to `harbor.aibee.cn/auto_car/visualglm:lavis.v1.1`, you may need to download the docker from [BaiduWebDisk](https://pan.baidu.com/s/1padY_ImFKWV7Eigs29RFZg?pwd=6zp4), (提取码: 6zp4).

Before you start docker, you need to load the docker image first.：
```bash
docker load -i visualglm_lavis_v1.1.tar
bash start_docker.sh
```

>Note: Concat part of .tar files: `cat <filename>.part-* > <filename>.tar`

## 1.2 Datasets

This model train by two stages. 

The stage1 trained on the first dataset.

The stage2 trained on AibeeQA dataset, which was labeled by us in handcraft way.

Assumption: The data is putted at path `cache`, which can be configed at `eventgpt/configs/default.yaml`.

You can download dataset from [BaiduWebDisk](https://pan.baidu.com/s/1NLldQg2QTjCrWmgynlfKyA?pwd=32kq). (提取码: 32kq). You should extract it under the path: `./cache`

## 1.3 Run/Model/Datasets/Processors and some others configs

This can be set in the following yaml files:

- `eventgpt/configs/models/blip2/blip2_opt2.7b_llava_box_caption_roi.yaml`
- `eventgpt/projects/blip2/train/pretrain_stage1_llava_box_caption_roi.yaml`
- `eventgpt/projects/blip2/train/pretrain_stage2_llava_box_caption_roi.yaml`

# 2. How to train

## 2.1 Stage1

```bash
bash run_scripts/blip2/train/pretrain_stage1_llava_box_caption_roi.sh 
```

## 2.2 Stage2

```bash
bash run_scripts/blip2/train/pretrain_stage2_llava_box_caption_roi.sh 
```

# 3. How to evaluate

Note: Before you run evaluation, you may need to config some parameters first at the file `tests/models/test_blip2_vqa.py`.

Run docker and run the script like this:

```bash
python tests/models/test_blip2_vqa.py |& tee test.log
```


