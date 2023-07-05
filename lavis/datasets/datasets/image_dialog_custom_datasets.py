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

class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "dialog": ann["dialog"],
                "image": sample["image"],
                "caption": ann["caption"],
            }
        )


class IMAGEDialogDataset(BaseDataset, __DisplMixin):
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

        r"""
        {
            "image_id": 378461,
            "image": "coco/images/train2014_123287/COCO_train2014_000000378461.jpg",
            "caption": "厨房，厨房有浅色的木橱柜，黑色洗碗机和前面的水槽或窗户",
            "dialog": "[{'question': '水槽是什么颜色的？\\n', 'answer': '白色\\n'}]"
        },
        {
            "image_id": 332243,
            "image": "coco/images/train2014_123287/COCO_train2014_000000332243.jpg",
            "caption": "一只长颈鹿从高高的树上的喂食箱里拿食物，旁边是一只在草地上吃草的斑马",
            "dialog": "[{'question': '比较《哈利·波特与魔法石》和《哈利·波特与密室》的情节。', 'answer': '在《哈利·波特与魔法石》中，主要情节围绕着哈利阻止伏地魔拿回魔法石展开。他需要证明自己的魔法能力并完成各种任务才能成功。在《哈利·波特与密室》中，主要情节围绕着哈利和他的朋友试图揭开有关魔法密室及其致命生物的谜团。故事还聚焦于哈利为证明自己对被指责释放黑暗魔法的冤屈而努力奋斗的故事。两本书都聚焦于哈利的冒险经历，力图拯救魔法世界免于邪恶的威胁。'}, {'question': '在下面的文章中，找到语法错误的句子并将其纠正！It’s been two years since I first play the piano.', 'answer': '自从我第一次弹钢琴以来已经过去两年了。'}, {'question': '图中所示有多少斑马？\\n', 'answer': '1\\n'}]"
        },
        """

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]
        if isinstance(ann['dialog'] , str):
            ann['dialog'] = eval(ann['dialog'])
        ann["dialog"] = [(round['question'], round['answer']) for round in ann["dialog"]]
        # print(ann["dialog"])

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        context = ""
        # print("ann[dialog]", ann["dialog"], len(ann["dialog"]), type(ann["dialog"]))
        for i, (old_query, response) in enumerate(ann["dialog"]):
            if i == len(ann["dialog"]) - 1:
                context += "[Round {}]\n问：{}\n答：".format(i, old_query)
            else:
                context += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
        
        # split the context and ans with the last
        final_response = response

        context = self.text_processor(context)
        final_response = self.text_processor(final_response)
        # print("context", context)
        # print("final_response", final_response)

        return {
            "image": image,
            "text_input": context,
            "text_output": final_response,
            "image_id": self.img_ids[ann["image_id"]],
        }
