# export TORCH_DISTRIBUTED_DEBUG=DETAIL 
# export CUDA_LAUNCH_BLOCKING=1
# python -m torch.distributed.run --nproc_per_node=4 train.py --cfg-path eventgpt/projects/blip2/train/pretrain_stage1_box_caption.yaml
TRANSFORMERS_OFFLINE=1 \
python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path eventgpt/projects/blip2/train/pretrain_stage1_llava_box_caption_roi.yaml
