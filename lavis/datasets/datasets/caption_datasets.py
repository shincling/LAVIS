"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict

from lavis.datasets.datasets.base_dataset import BaseDataset
from PIL import Image
import soundfile as sf
import numpy as np

from transformers import AutoProcessor
<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes

class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "caption": ann["caption"],
                "image": sample["image"],
            }
        )


class CaptionDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        caption = self.text_processor(ann["caption"])

        return {
            "image": image,
            "text_input": caption,
            "image_id": self.img_ids[ann["image_id"]],
        }


class CaptionEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        return {
            "image": image,
            "image_id": ann["image_id"],
            "instance_id": ann["instance_id"],
        }

class MedCaptionDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        caption = self.text_processor(ann["caption"])

        return {
            "image": image,
            "text_input": caption,
            "image_id": self.img_ids[ann["image_id"]],
        }

class AudioCaptionDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        audio_processor = vis_processor
        super().__init__(audio_processor, text_processor, vis_root, ann_paths)
        # 这里是在原有的vis的基础上继承audio_processor, 所以底下做一些名字上的映射吧
        self.audio_processor = self.vis_processor
        self.audio_root = self.vis_root

        # self.audio_wav2vec_processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
        self.audio_wav2vec_processor = AutoProcessor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
        self.audio_ids = {}
        n = 0
        for ann in self.annotation:
            audio_id = ann["audio_id"]
            if audio_id not in self.audio_ids.keys():
                self.audio_ids[audio_id] = n
                n += 1

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        audio_path = os.path.join(self.audio_root, ann["audio"])
        audio_npy, sr = sf.read(audio_path)
        assert sr == 16000 and audio_npy.ndim == 1, "audio should be 1-channel and sr=16000, but got {} and {}".format(sr, audio_npy.shape[0])

        # normalize 
        if np.max (np.abs(audio_npy)) == 0:
            print("Warning: audio {} is all zeros".format(audio_path))
        audio_npy = audio_npy / np.max(np.abs(audio_npy))

        # padding or truncating to 10s 
        if audio_npy.shape[0] < 160000:
            audio_npy = np.pad(audio_npy, (0, 160000-audio_npy.shape[0]), 'constant')
        else:
            audio_npy = audio_npy[:160000]

<<<<<<< Updated upstream
        # audio = self.audio_processor(audio_npy)
        audio = self.audio_wav2vec_processor(audio_npy, sampling_rate=16000, return_tensors="pt",)
=======
        # audio = self.audio_processor(audio_npy) 

        audio = self.audio_wav2vec_processor(audio_npy, sampling_rate=16000, return_tensors="pt",)
        # audio = self.wav2vec_model(**audio).last_hidden_state
        # print('Warning: we use wav2vec2_processor to process audio!!!!!!')
        # print('audio with w2v feats shape:', audio["input_values"].shape)
>>>>>>> Stashed changes

        caption = self.text_processor(ann["caption"])

        return {
            "audio": audio,
            "text_input": caption,
            "audio_id": self.audio_ids[ann["audio_id"]],
        }