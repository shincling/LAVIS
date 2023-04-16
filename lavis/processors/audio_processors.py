"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import re

from lavis.common.registry import registry
from lavis.processors.base_processor import BaseProcessor
from lavis.processors.randaugment import RandomAugment
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchaudio import transforms as audio_transforms
import torch


class BlipImageBaseProcessor(BaseProcessor):
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms.Normalize(mean, std)

@registry.register_processor("blip2_audio_train")
# class Blip2AudioTrainProcessor(BlipImageBaseProcessor):
class Blip2AudioTrainProcessor(BaseProcessor):
    def __init__(
        self, n_mel, max_frames=160000, mean=None, std=None, min_scale=0.5, max_scale=1.0
    ):
        # super().__init__(mean=mean, std=std)


        # normalize the audio signal to [-1, 1] and padding to 160000 frames
        self.audio_transform = transforms.Compose(
            [
                audio_transforms.MelSpectrogram(sample_rate=16000, n_mels=n_mel),
                # audio_transforms.AmplitudeToDB(),
            ]
        )
        

    def __call__(self, item):
        item_tensor = torch.FloatTensor(item)
        # print("Item tensor shape: ", item_tensor.shape, item_tensor.dtype)
        feats = self.audio_transform(item_tensor).transpose(0, 1)
        # print("Feat shape: ", feats.shape)
        return feats

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        audio_frames= cfg.get("max_frames",160000)
        n_mel = cfg.get("n_mel", None)
        print("N_mel now is: ", n_mel, "!!!")

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        min_scale = cfg.get("min_scale", 0.5)
        max_scale = cfg.get("max_scale", 1.0)

        return cls(
            # image_size=image_size,
            n_mel = n_mel,
            max_frames=audio_frames,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
        )
