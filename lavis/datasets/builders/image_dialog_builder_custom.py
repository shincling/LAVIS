"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.coco_caption_datasets import (
    COCOCapDataset,
    COCOCapEvalDataset,
)

from lavis.datasets.datasets.image_dialog_custom_datasets import  IMAGEDialogDataset
from lavis.common.registry import registry


@registry.register_builder("image_dialog_ft")
class IMAGEDialogBuilder(BaseDatasetBuilder):
    # train_dataset_cls = COCOCapDataset
    train_dataset_cls = IMAGEDialogDataset
    eval_dataset_cls = COCOCapEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_image_dialog.yaml"

    }