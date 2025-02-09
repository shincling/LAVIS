"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import os
import time
import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

import lavis.common.dist_utils as dist_utils
from lavis.common.dist_utils import download_cached_file
from lavis.common.utils import is_url
from lavis.common.logger import MetricLogger
from lavis.models.base_model import BaseModel
from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel
from lavis.models.eva_vit import create_eva_vit_g
from transformers import BertTokenizer
from lavis.models.blip2_models.PointTransformer import PointTransformerV2


class Blip2Base(BaseModel):
    @classmethod
    def init_tokenizer(cls):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 2
        encoder_config.query_length = num_query_token

        # Note(Jing): close the dropout to see whether it's the reason for the gap between .train() and .eval()
        encoder_config.attention_probs_dropout_prob = 0.0
        encoder_config.hidden_dropout_prob = 0.0
        Qformer = BertLMHeadModel.from_pretrained(
            "bert-base-uncased", config=encoder_config
        )                 
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    @classmethod
    def init_vision_encoder(
        cls, img_size, drop_path_rate, use_grad_checkpoint, precision
    ):
        visual_encoder = create_eva_vit_g(
            img_size, drop_path_rate, use_grad_checkpoint, precision
        )
        ln_vision = LayerNorm(visual_encoder.num_features)
        return visual_encoder, ln_vision

    @classmethod
    def init_audio_encoder(
        cls, wav2vec_path = "facebook/wav2vec2-base-960h"
    ):
        from transformers import AutoProcessor, Wav2Vec2Model
        # audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        audio_encoder = Wav2Vec2Model.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
        audio_encoder.freeze_feature_extractor()
        return audio_encoder, None

    @classmethod
    # def init_cloud_encoder(cls, model_name, max_cloud_size, drop_path_rate, use_grad_checkpoint, pretrained_model_path = None,):
    def init_cloud_encoder(cls, model_name, pretrained_model_path = None,):
        if (model_name == "point_transformer"):
             cloud_encoder = PointTransformerV2(in_channels=6,
                                            #  num_classes=20,
                                             patch_embed_depth=1,
                                             patch_embed_channels=48,
                                             patch_embed_groups=6,
                                             patch_embed_neighbours=8,
                                             enc_depths=(2, 2, 6, 2),
                                             enc_channels=(96, 192, 384, 512),
                                             enc_groups=(12, 24, 48, 64),
                                             enc_neighbours=(16, 16, 16, 16),
                                            #  dec_depths=(1, 1, 1, 1),
                                            #  dec_channels=(48, 96, 192, 384),
                                            #  dec_groups=(6, 12, 24, 48),
                                            #  dec_neighbours=(16, 16, 16, 16),
                                             grid_sizes=(0.06, 0.15, 0.375, 0.9375),
                                             attn_qkv_bias=True,
                                             pe_multiplier=False,
                                             pe_bias=True,
                                             attn_drop_rate=0.,
                                             drop_path_rate=0.3,
                                             enable_checkpoint=False,
                                             unpool_backend="map",
                                             num_features=256,
                                             checkpoint_path=pretrained_model_path,)
        else:
            raise KeyError("cloud encoder must be point_transformer")
        # TODO: 这个要根据point transformer的特征维度相应修改
        ln_cloud = LayerNorm(cloud_encoder.enc_channels[-1])
        return cloud_encoder, ln_cloud

    def load_from_pretrained(self, url_or_filename):
        # import pdb; pdb.set_trace()
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            print(url_or_filename)
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]
        # tmp_dict = torch.load("/data2/shij/llama/only_opt.pth", map_location="cpu")['model']
        # lm_head_state_dict = {'opt_model.lm_head.weight' : state_dict["opt_model.lm_head.weight"][:130528]}
        # state_dict.pop("opt_model.lm_head.weight")

        # Note(Jing): 下面三行应该用，避免chatglm的参数被覆盖出现问题
        for k in list(state_dict.keys()):
            if k.startswith("opt_model."):
                state_dict.pop(k)

        # new_state_dict = {}
        # for k,v in state_dict.items():
        #     # if "chatglm" in k:
        #         # new_state_dict[k.replace('chatglm_', "opt_")] = v
        #     if "proj" in k:
        #         new_state_dict[k.replace('chatglm_', "opt_")] = v
        #     else:
        #         new_state_dict[k] = v
        # state_dict = new_state_dict

        # state_dict.update(lm_head_state_dict)
        logging.warn("Remove the LLM realted parameters to use the GLM orignal weights.")
        
        msg = self.load_state_dict(state_dict, strict=False)

        logging.info("Missing keys {}".format(msg.missing_keys))
        print("Missing keys {}".format(msg.missing_keys), len(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


def compute_sim_matrix(model, data_loader, **kwargs):
    k_test = kwargs.pop("k_test")

    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation:"

    logging.info("Computing features for evaluation...")
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i : min(num_text, i + text_bs)]
        text_input = model.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=35, # TODO(jing): this could increase to ~300
            return_tensors="pt",
        ).to(model.device)
        text_feat = model.forward_text(text_input)
        text_embed = F.normalize(model.text_proj(text_feat))
        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds, dim=0)
    text_ids = torch.cat(text_ids, dim=0)
    text_atts = torch.cat(text_atts, dim=0)

    vit_feats = []
    image_embeds = []
    for samples in data_loader:
        image = samples["image"]

        image = image.to(model.device)
        image_feat, vit_feat = model.forward_image(image)
        image_embed = model.vision_proj(image_feat)
        image_embed = F.normalize(image_embed, dim=-1)

        vit_feats.append(vit_feat.cpu())
        image_embeds.append(image_embed)

    vit_feats = torch.cat(vit_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)

    sims_matrix = []
    for image_embed in image_embeds:
        sim_q2t = image_embed @ text_embeds.t()
        sim_i2t, _ = sim_q2t.max(0)
        sims_matrix.append(sim_i2t)
    sims_matrix = torch.stack(sims_matrix, dim=0)

    score_matrix_i2t = torch.full(
        (len(data_loader.dataset.image), len(texts)), -100.0
    ).to(model.device)

    num_tasks = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        image_inputs = vit_feats[start + i].repeat(k_test, 1, 1).to(model.device)
        score = model.compute_itm(
            image_inputs=image_inputs,
            text_ids=text_ids[topk_idx],
            text_atts=text_atts[topk_idx],
        ).float()
        score_matrix_i2t[start + i, topk_idx] = score + topk_sim

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full(
        (len(texts), len(data_loader.dataset.image)), -100.0
    ).to(model.device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        image_inputs = vit_feats[topk_idx.cpu()].to(model.device)
        score = model.compute_itm(
            image_inputs=image_inputs,
            text_ids=text_ids[start + i].repeat(k_test, 1),
            text_atts=text_atts[start + i].repeat(k_test, 1),
        ).float()
        score_matrix_t2i[start + i, topk_idx] = score + topk_sim

    if dist_utils.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
        )
        torch.distributed.all_reduce(
            score_matrix_t2i, op=torch.distributed.ReduceOp.SUM
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("Evaluation time {}".format(total_time_str))

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()
