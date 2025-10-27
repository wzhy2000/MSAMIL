import timm
import torch
import types
import torchvision
import torch.nn as nn

from functools import partial
from utils.constants import MODEL2CONSTANTS
from utils.transform_utils import get_eval_transforms, get_transforms_with_randaugment
from torchvision.models import resnet50


def get_encoder(model_name, target_img_size=224):
    print('loading model checkpoint')
    if model_name == 'resnet50':
        # model = resnet50_baseline(pretrained=True) # 1024-d from layer3 of ResNet50
        def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)

            return x

        model = resnet50(weights=None)
        del model.layer4, model.fc
        model._forward_impl = types.MethodType(_forward_impl, model)
        state_dict = torch.hub.load_state_dict_from_url(
            "https://download.pytorch.org/models/resnet50-19c8e357.pth"
        )
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("layer4.") and not k.startswith("fc.")}
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        
    elif model_name == 'resnet18':
        # model = torchvision.models.resnet18(pretrained=True)
        model = torchvision.models.resnet18(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])
        model.add_module("flatten", nn.Flatten())

    elif model_name == 'uni_v1':
        uni_ckpt_path = 'checkpoints/uni/pytorch_model.bin'
        model = timm.create_model("vit_large_patch16_224",
                            init_values=1e-5, 
                            num_classes=0, 
                            dynamic_img_size=True)
        model.load_state_dict(torch.load(uni_ckpt_path, map_location="cpu"), strict=True)
        
    elif model_name == 'conch_v1':
        conch_ckpt_path = 'checkpoints/conch/pytorch_model.bin'
        from conch.open_clip_custom import create_model_from_pretrained
        model, _ = create_model_from_pretrained("conch_ViT-B-16", conch_ckpt_path)
        model.forward = partial(model.encode_image, proj_contrast=False, normalize=False)
    else:
        raise NotImplementedError('model {} not implemented'.format(model_name))
    
    constants = MODEL2CONSTANTS[model_name]
    # img_transforms = get_eval_transforms(mean=constants['mean'],
    #                                      std=constants['std'],
    #                                      target_img_size = target_img_size)
    img_transforms = get_transforms_with_randaugment(mean=constants['mean'],
                                         std=constants['std'],
                                         target_img_size = target_img_size)
    print("img_transforms:", img_transforms)

    return model, img_transforms