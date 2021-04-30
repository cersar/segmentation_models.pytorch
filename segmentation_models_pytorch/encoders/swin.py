""" Each encoder should have following attributes and methods and be inherited from `_base.EncoderMixin`

Attributes:

    _out_channels (list of int): specify number of channels for each encoder feature tensor
    _depth (int): specify number of stages in decoder (in other words number of downsampling operations)
    _in_channels (int): default number of input channels in first Conv2d layer for encoder (usually 3)

Methods:

    forward(self, x: torch.Tensor)
        produce list of features of different spatial resolutions, each feature is a 4D torch.tensor of
        shape NCHW (features should be sorted in descending order according to spatial resolution, starting
        with resolution same as input `x` tensor).

        Input: `x` with shape (1, 3, 64, 64)
        Output: [f0, f1, f2, f3, f4, f5] - features with corresponding shapes
                [(1, 3, 64, 64), (1, 64, 32, 32), (1, 128, 16, 16), (1, 256, 8, 8),
                (1, 512, 4, 4), (1, 1024, 2, 2)] (C - dim may differ)

        also should support number of features according to specified depth, e.g. if depth = 5,
        number of feature tensors = 6 (one with same resolution as input and 5 downsampled),
        depth = 3 -> number of feature tensors = 4 (one with same resolution as input and 3 downsampled).
"""
from copy import deepcopy

import torch.nn as nn
import torch
import numpy as np

from swin_transformer.models.swin_transformer import SwinTransformer

from ._base import EncoderMixin


class SwinEncoder(SwinTransformer, EncoderMixin):
    def __init__(self, out_channels, depth=4, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3
        self.scale_factor = [4, 2, 2, 2]

        del self.head
        del self.avgpool

    def forward(self, x):

        features = []
        features.append(x)

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for i in range(self._depth):
            x = self.layers[i](x)
            b,s,c = x.shape
            features.append(x.permute(0,2,1).reshape((b,c,int(np.sqrt(s)),-1)))

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop('head.weight')
        state_dict.pop('head.bias')
        super().load_state_dict(state_dict, **kwargs)


new_settings = {
    "Swin-T": {
        "imagenet": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth"
    },
    "Swin-S": {
        "imagenet": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth"
    },
    "Swin-B": {
        "imagenet": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth"
    }
}

pretrained_settings = {}
for model_name, sources in new_settings.items():
    if model_name not in pretrained_settings:
        pretrained_settings[model_name] = {}

    for source_name, source_url in sources.items():
        pretrained_settings[model_name][source_name] = {
            "url": source_url,
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }


swin_encoders = {
    "Swin-T": {
        "encoder": SwinEncoder,
        "pretrained_settings": pretrained_settings["Swin-T"],
        "params": {
            "embed_dim": 96,
            "out_channels": (3, 96, 192, 384, 768),
            "depths": [2, 2, 6, 2]
        },
    },
    "Swin-S": {
        "encoder": SwinEncoder,
        "pretrained_settings": pretrained_settings["Swin-S"],
        "params": {
            "embed_dim": 96,
            "out_channels": (3, 96, 192, 384, 768),
            "depths": [2, 2, 18, 2]
        },
    },
    "Swin-B": {
        "encoder": SwinEncoder,
        "pretrained_settings": pretrained_settings["Swin-B"],
        "params": {
            "embed_dim": 128,
            "out_channels": (3, 128, 256, 512, 1024),
            "depths": [2, 2, 18, 2]
        },
    },
}
