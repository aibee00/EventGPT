from collections import OrderedDict

import torch
from torch import nn

from . import registry
from .make_layers import conv_with_kaiming_uniform
from .layers import DropBlock2D, DyHead
from . import fpn as fpn_module
from . import swint
from . import swint_v2
from . import swint_vl
from . import swint_v2_vl
from eventgpt.common.dist_utils import download_cached_file


@registry.BACKBONES.register("SWINT-FPN-RETINANET")
def build_retinanet_swint_fpn_backbone(cfg):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    if cfg.MODEL.SWINT.VERSION == "v1":
        body = swint.build_swint_backbone(cfg)
    elif cfg.MODEL.SWINT.VERSION == "v2":
        body = swint_v2.build_swint_backbone(cfg)
    elif cfg.MODEL.SWINT.VERSION == "vl":
        body = swint_vl.build_swint_backbone(cfg)
    elif cfg.MODEL.SWINT.VERSION == "v2_vl":
        body = swint_v2_vl.build_swint_backbone(cfg)

    in_channels_stages = cfg.MODEL.SWINT.OUT_CHANNELS
    out_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
    in_channels_p6p7 = out_channels
    fpn = fpn_module.FPN(
        in_channels_list=[
            0,
            in_channels_stages[-3],
            in_channels_stages[-2],
            in_channels_stages[-1],
            ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7, out_channels),
        drop_block=DropBlock2D(cfg.MODEL.FPN.DROP_PROB, cfg.MODEL.FPN.DROP_SIZE) if cfg.MODEL.FPN.DROP_BLOCK else None,
        use_spp=cfg.MODEL.FPN.USE_SPP,
        use_pan=cfg.MODEL.FPN.USE_PAN,
        return_swint_feature_before_fusion=cfg.MODEL.FPN.RETURN_SWINT_FEATURE_BEFORE_FUSION,
        drop_first_feature=cfg.MODEL.FPN.DROP_FIRST_FEATURE
    )
    if cfg.MODEL.FPN.USE_DYHEAD:
        dyhead = DyHead(cfg, out_channels)
        model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn), ("dyhead", dyhead)]))
    else:
        model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    return model


@registry.BACKBONES.register("SWINT-FPN")
def build_swint_fpn_backbone(cfg):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    if cfg.MODEL.SWINT.VERSION == "v1":
        body = swint.build_swint_backbone(cfg)
    elif cfg.MODEL.SWINT.VERSION == "v2":
        body = swint_v2.build_swint_backbone(cfg)
    elif cfg.MODEL.SWINT.VERSION == "vl":
        body = swint_vl.build_swint_backbone(cfg)
    elif cfg.MODEL.SWINT.VERSION == "v2_vl":
        body = swint_v2_vl.build_swint_backbone(cfg)

    in_channels_stages = cfg.MODEL.SWINT.OUT_CHANNELS
    out_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stages[-4],
            in_channels_stages[-3],
            in_channels_stages[-2],
            in_channels_stages[-1],
            ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
        drop_block=DropBlock2D(cfg.MODEL.FPN.DROP_PROB, cfg.MODEL.FPN.DROP_SIZE) if cfg.MODEL.FPN.DROP_BLOCK else None,
        use_spp=cfg.MODEL.FPN.USE_SPP,
        use_pan=cfg.MODEL.FPN.USE_PAN
    )
    if cfg.MODEL.FPN.USE_DYHEAD:
        dyhead = DyHead(cfg, out_channels)
        model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn), ("dyhead", dyhead)]))
    else:
        model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    return model


def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES, \
        "cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry".format(
            cfg.MODEL.BACKBONE.CONV_BODY
        )
    return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)


##### new add begin #####
def build_model(config):
    model_type = getattr(config, "MODEL.TYPE", 'swint')
    if model_type == "swint":
        model = build_backbone(cfg=config)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")
    return model


def convert_weights_to_fp16(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

#         if isinstance(l, (nn.MultiheadAttention, Attention)):
#             for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
#                 tensor = getattr(l, attr)
#                 if tensor is not None:
#                     tensor.data = tensor.data.half()

    model.apply(_convert_weights_to_fp16)


def create_swint(img_size=224, drop_path_rate=0.4, use_checkpoint=False, precision="fp16", load_local=True):
    from eventgpt.models.swint_models.config import cfg

    model = build_model(cfg)

    if load_local:
        # Use local path instead
        cached_file = cfg.MODEL.BACKBONE.PRETRAINED_PATH
    else:
        url = "https://huggingface.co/GLIPModel/GLIP/blob/main/glip_tiny_model_o365_goldg_cc_sbu.pth"
        cached_file = download_cached_file(
            url, check_hash=False, progress=True
        )

    state_dict = torch.load(cached_file, map_location="cpu") 
    # interpolate_pos_embed(model,state_dict)

    incompatible_keys = model.load_state_dict(state_dict, strict=False)
    # print(incompatible_keys)
    
    if precision == "fp16":
        # model.to("cuda") 
        convert_weights_to_fp16(model)
    return model

