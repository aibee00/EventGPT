import sys
import copy
sys.path.append('.')
from eventgpt.models.swint_models import build_backbone
from eventgpt.models.swint_models import swint, swint_v2, swint_vl, swint_v2_vl


def build_model(config):
    model_type = getattr(config, "MODEL.TYPE", 'swint')
    if model_type == "swint":
        model = build_backbone(cfg=config)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")
    return model


def build_swint_body(cfg):
    if cfg.MODEL.SWINT.VERSION == "v1":
        body = swint.build_swint_backbone(cfg)
    elif cfg.MODEL.SWINT.VERSION == "v2":
        body = swint_v2.build_swint_backbone(cfg)
    elif cfg.MODEL.SWINT.VERSION == "vl":
        body = swint_vl.build_swint_backbone(cfg)
    elif cfg.MODEL.SWINT.VERSION == "v2_vl":
        body = swint_v2_vl.build_swint_backbone(cfg)
    return body


if __name__ == '__main__':
    import torch
    from eventgpt.models.swint_models.config import cfg

    # only load backbone
    # model = build_swint_body(cfg)

    # load model
    model = build_model(cfg)
    print(model)

    import pdb; pdb.set_trace()

    # pretrain weights
    pth = 'pretrain_weights/glip_tiny_model_o365_goldg_cc_sbu_lvisbest.pth'
    weights = torch.load(pth)

    # 仅仅加载weights中的weights['model']的module.body和module.fpn
    model_weight = weights['model']

    for k in model_weight.copy().keys():
        if k.startswith('module.backbone.'):
            model_weight[k[16:]] = model_weight.pop(k)

        elif k.startswith('module.fpn.'):
            model_weight[k[11:]] = model_weight.pop(k)

        else:
            model_weight.pop(k)

    # for k in model_weight.copy().keys():
    #     if k.startswith('module.backbone.body.'):
    #         model_weight[k[21:]] = model_weight.pop(k)
    #     else:
    #         model_weight.pop(k)

    incompatible_keys = model.load_state_dict(model_weight, strict=False)
    print(incompatible_keys)
    
    # save model structure into txt
    with open('swint_model_structure.txt', 'w') as f:
        f.write(str(model))
    
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    for i in y:
        print(i.shape)
    # print([i.shape for i in y])
    """
    swint->
    [torch.Size([2, 96, 56, 56]), 
    torch.Size([2, 192, 28, 28]), 
    torch.Size([2, 384, 14, 14]), -> 
    torch.Size([2, 768, 7, 7])]
    """
    """ output, flatten all last two dimensions and sum = 1049
    # torch.Size([1, 256, 28, 28])
    torch.Size([1, 256, 14, 14]) 265x256 -> prj -> llm
    torch.Size([1, 256, 7, 7])
    torch.Size([1, 256, 4, 4])
    torch.Size([1, 256, 2, 2])
    """

