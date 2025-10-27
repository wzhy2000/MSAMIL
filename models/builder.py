import timm
import torch
import types
import torchvision
import torch.nn as nn

from functools import partial
from utils.constants import MODEL2CONSTANTS
from utils.transform_utils import get_eval_transforms, get_transforms_with_randaugment
from torchvision.models import resnet50
from models.patch_encoder_models import encoder_factory


def get_encoder(model_name, target_img_size=224, pretrained=None):
    print('loading model checkpoint')
    if model_name == 'resnet50':
        if pretrained:
            model = Pretrained_extract_features(name='resnet50', dim=32, pretrained=pretrained)
        else:
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

    elif model_name == 'resnet18':
        if pretrained:
            model = Pretrained_extract_features(name='resnet18', dim=32, pretrained=pretrained)
        else:
            model = torchvision.models.resnet18(pretrained=True)
            model = nn.Sequential(*list(model.children())[:-1])
            model.add_module("flatten", nn.Flatten())

    elif model_name == 'uni_v1':
        ckpt_path = 'checkpoints/uni_v1/pytorch_model.bin'
        model = timm.create_model("vit_large_patch16_224",
                            init_values=1e-5, 
                            num_classes=0, 
                            dynamic_img_size=True)
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)

    elif model_name == 'uni_v2':
        model = encoder_factory(model_name)

    elif model_name == 'virchow2':
        model = encoder_factory(model_name)

    elif model_name == 'gigapath':
        # Older versions of timm have compatibility issues. Please ensure that you use a newer version by running the following command: pip install timm>=1.0.3.
        # model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        model = encoder_factory(model_name)

    elif model_name == 'conch_v1':
        conch_ckpt_path = 'checkpoints/conch/pytorch_model.bin'
        from conch.open_clip_custom import create_model_from_pretrained
        model, preprocess = create_model_from_pretrained("conch_ViT-B-16", conch_ckpt_path)
        model.forward = partial(model.encode_image, proj_contrast=False, normalize=False)

    else:
        raise NotImplementedError('model {} not implemented'.format(model_name))
    
    
    # img_transforms = get_eval_transforms(mean=constants['mean'],
    #                                      std=constants['std'],
    #                                      target_img_size = target_img_size)
    # img_transforms = get_transforms_with_randaugment(mean=constants['mean'],
    #                                  std=constants['std'],
    #                                  target_img_size = target_img_size)
    if model_name == 'uni_v2' or model_name == 'virchow2':   # or model_name == 'gigapath'
        img_transforms = model.eval_transforms
    elif model_name == 'conch_v1':
        img_transforms = preprocess
    else:
        constants = MODEL2CONSTANTS
        img_transforms = get_transforms_with_randaugment(mean=constants['mean'],
                                         std=constants['std'],
                                         target_img_size = target_img_size)
    print("img_transforms:", img_transforms)

    return model, img_transforms