"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import random
import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.modeling_opt import OPTForCausalLM, OPTConfig
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel, BertConfig
from lavis.models.blip2_models.modeling_chatglm import ChatGLMForConditionalGeneration, ChatGLMConfig
from lavis.models.blip2_models.tokenization_chatglm import ChatGLMTokenizer
from lavis.models.blip2_models.blip2_llama import prompts_list_image_cn, prompts_list_music_cn, prompts_list_signal_cn
import re


@registry.register_model("blip2_chatglm")
class Blip2GLM(Blip2Base):
    """
    BLIP2 OPT model.
    Supported model types:
        - pretrained_opt2.7b: pretrained model with OPT2.7b
        - pretrained_opt6.7b: pretrained model with OPT6.7b
        - caption_coco_opt2.7b: fintuned image captioning model with OPT2.7b
        - caption_coco_opt6.7b: fintuned image captioning model with OPT6.7b
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_opt", "caption_coco_opt2.7b")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_chatglm6b": "configs/models/blip2/blip2_pretrain_chatglm6b.yaml",
        "pretrain_chatglm6b-audio": "configs/models/blip2/blip2_pretrain_chatglm6b-audio.yaml",
        "pretrain_chatglm6b-speech": "configs/models/blip2/blip2_pretrain_chatglm6b-speech.yaml",
        "pretrain_chatglm6b-music": "configs/models/blip2/blip2_pretrain_chatglm6b-music.yaml",
        "pretrain_chatglm6b-signal": "configs/models/blip2/blip2_pretrain_chatglm6b-signal.yaml",
        "pretrain_chatglm6b-cloud": "configs/models/blip2/blip2_pretrain_chatglm6b-cloud.yaml",
        "pretrain_chatglm6b_image_dialog_ft": "configs/models/blip2/blip2_pretrain_chatglm6b-image-dialog-ft.yaml",
    }

    def __init__(
        self,
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        opt_model="THUDM/chatglm-6b",
        prompt="",
        max_txt_len=32,
        modality="image",
        freeze_qformer = False,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        # self.visual_encoder, self.ln_vision = self.init_vision_encoder(
        #     img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        # )
        # if freeze_vit:
        #     for name, param in self.visual_encoder.named_parameters():
        #         param.requires_grad = False               
        #     self.visual_encoder = self.visual_encoder.eval()
        #     self.visual_encoder.train = disabled_train
        #     logging.info("freeze vision encoder")

        self.modality = modality
        logging.info(f"modality: {self.modality}")
        logging.info(f"freeze_qformer: {freeze_qformer}")
        if modality == "image":
            self.visual_encoder, self.ln_vision = self.init_vision_encoder(
                img_size, drop_path_rate, use_grad_checkpoint, vit_precision
            )
            if freeze_vit:
                for name, param in self.visual_encoder.named_parameters():
                    param.requires_grad = False               
                self.visual_encoder = self.visual_encoder.eval()
                self.visual_encoder.train = disabled_train
                logging.info("freeze vision encoder")

            self.Qformer, self.query_tokens = self.init_Qformer(
                num_query_token, self.visual_encoder.num_features
            )
            if freeze_qformer:
                for name, param in self.Qformer.named_parameters():
                    param.requires_grad = False               
                self.query_tokens.requires_grad = False
                self.Qformer = self.Qformer.eval()
                self.Qformer.train = disabled_train
                logging.info("freeze Qformer")

        elif modality in ["audio", "music"]:
            self.audio_encoder, _ = self.init_audio_encoder()
            if freeze_vit:
                for name, param in self.audio_encoder.named_parameters():
                    param.requires_grad = False               
                self.audio_encoder = self.audio_encoder.eval()
                logging.info("freeze audio encoder")
            self.Qformer, self.query_tokens = self.init_Qformer(
                num_query_token, 1024, #768 # 512, #self.visual_encoder.num_features
            )
        elif modality == "signal":
            self.signal_encoder, _ = self.init_audio_encoder()
            self.audio_encoder = self.signal_encoder
            if freeze_vit:
                for name, param in self.signal_encoder.named_parameters():
                    param.requires_grad = False               
                self.signal_encoder = self.signal_encoder.eval()
                logging.info("freeze audio encoder")
            self.Qformer, self.query_tokens = self.init_Qformer(
                num_query_token, 1024, #768 # 512, #self.visual_encoder.num_features
            )
        elif modality == "cloud":
            self.cloud_encoder, self.ln_cloud = self.init_cloud_encoder(
                model_name="point_transformer", pretrained_model_path="/data2/shij/data/cloud_cap/model/point_transformer/resave_best.pth"
            )
            if freeze_vit:
                for name, param in self.cloud_encoder.named_parameters():
                    param.requires_grad = False
                self.cloud_encoder = self.cloud_encoder.eval()
                self.cloud_encoder.train = disabled_train
                logging.info("freeze point cloud encoder")

            self.num_query_token = num_query_token
            self.Qformer, self.query_tokens = self.init_Qformer(
                self.num_query_token, self.cloud_encoder.enc_channels[-1]
            )
            self.eos_token_id = self.tokenizer.eos_token_id
        elif modality == "speech":
            self.speech_adaptor_type = "bert"
            if self.speech_adaptor_type == "bert":
                self.speech_bert_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
                self.speech_bert_adaptor = BertModel.from_pretrained("bert-base-chinese")
                self.bert_hidden_size = self.speech_bert_adaptor.config.hidden_size
                self.pre_bert_proj = nn.Linear(512, self.bert_hidden_size)
                self.post_bert_proj = nn.Linear(
                    # self.bert_hidden_size, self.opt_model.config.hidden_size
                    self.bert_hidden_size, 4096 # NOTE(JING): hard code for now 
                )
        else:
            raise NotImplementedError("modality {} not implemented".format(modality))

        if hasattr(self, "Qformer"): # speech modal has not Qformer
            self.Qformer.cls = None
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None

        self.opt_tokenizer = ChatGLMTokenizer.from_pretrained(opt_model, trust_remote_code=True)
        self.opt_model = ChatGLMForConditionalGeneration.from_pretrained(
            opt_model, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).half()

        # self.opt_tokenizer = AutoTokenizer.from_pretrained("/data2/shij/data/cloud_cap/model/chatglm-6b", trust_remote_code=True)
        # self.opt_model = AutoModel.from_pretrained("/data2/shij/data/cloud_cap/model/chatglm-6b", trust_remote_code=True, torch_dtype=torch.bfloat16).half()

        # self.opt_model.gradient_checkpointing_enable()
        for name, param in self.opt_model.named_parameters():
            param.requires_grad = False
        # self.eos_token_id = self.opt_tokenizer(
        #     "\n", add_special_tokens=False
        # ).input_ids[0]
        self.ignore_pad_token_for_loss = True

        if hasattr(self, "Qformer"): # speech modal has not Qformer
            self.opt_proj = nn.Linear(
                self.Qformer.config.hidden_size, self.opt_model.config.hidden_size
            )

        self.max_txt_len = max_txt_len
        self.prompt = prompt
        prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.input_ids.size(1)

    def prompt_wrap(self, img_embeds, atts_img, prompt, use_speech=True, video=False, music=False):
        if not self.modality:
            if use_speech:
                special_token = '<SpeechHere>'
            else:
                special_token = '<ImageHere>'
            if video:
                special_token = '<VideoHere>'
            if music:
                special_token = '<MusicHere>'
                # prompt = prompt.replace('<Image>', '<Music>').replace('</Image>', '</Music>')
                prompt = "<Music>" + special_token+ "</Music>"
        else:
            modal_token = self.modality.capitalize()
            special_token = f"<{modal_token}Here>"
            prompt = f"<{modal_token}><{modal_token}Here></{modal_token}>"
        
        if prompt:
            batch_size = img_embeds.shape[0]
            p_before, p_after = prompt.split(special_token)
            p_before_tokens = self.opt_tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_after_tokens = self.opt_tokenizer(
                p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_before_embeds = self.opt_model.transformer.word_embeddings(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.opt_model.transformer.word_embeddings(p_after_tokens.input_ids).expand(batch_size, -1, -1)
            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img

    def forward(self, samples):
        if self.modality == "image":
            image = samples["image"]
            image_embeds = self.ln_vision(self.visual_encoder(image))
        elif self.modality in ["audio", "music"]:
        # image_embeds = image
            audio = samples["audio"]
            audio_embeds = self.audio_encoder(input_values=audio['input_values'].squeeze().to(self.audio_encoder.device), return_dict=True)
            audio_embeds = audio_embeds.last_hidden_state
            image, image_embeds = audio_embeds, audio_embeds
        elif self.modality == "signal":
            signal = samples["audio"]
            signal_embeds = self.signal_encoder(input_values=signal['input_values'].squeeze().to(self.signal_encoder.device), return_dict=True)
            signal_embeds = signal_embeds.last_hidden_state
            image, image_embeds = signal_embeds, signal_embeds
        else:
            raise NotImplementedError("modality {} not implemented".format(self.modality))

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        image_embeds = self.opt_proj(query_output.last_hidden_state)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.bool).to(image.device)

        image_prompt = '<Image><ImageHere></Image>'
        image_embeds, image_atts = self.prompt_wrap(image_embeds, image_atts, image_prompt, use_speech=False, music=True)


        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            # opt_tokens = self.preprocess_function_train(samples, image_atts.device, src_txt_len=32, max_tgt_len=32)
            # opt_tokens = self.preprocess_function_train(samples, image_atts.device, src_txt_len=50, max_tgt_len=150)
            opt_tokens = self.preprocess_function_train(samples, image_atts.device, src_txt_len=64, max_tgt_len=128)
            empty_targets = (
                torch.ones(image_atts.size(),
                           dtype=torch.long).to(image_atts.device).fill_(0)
            )

            opt_tokens['input_ids'] = torch.cat(
                    [empty_targets, opt_tokens['input_ids']], dim=1
                ).to(image_embeds.device)
            opt_tokens['labels'] = torch.cat(
                [empty_targets.fill_(-100), opt_tokens['labels']], dim=1
            ).to(image_embeds.device)


            outputs = self.opt_model(
                **opt_tokens,
                input_image=image_embeds,
                return_dict=True,
            )
            loss = outputs.loss
            # logits = outputs.logits
            # _, pred = logits[0].max(1)
            # # outputs = pred.tolist()
            # outputs = pred.tolist()[context_length:]
            # response = self.opt_tokenizer.decode(outputs)

     
            # file_ = open('/raid/cfl/cn_pretraining_multi_dialog/LAVIS/lavis/output/BLIP2/samples1.txt', 'a', encoding='utf-8')
            # file_.write("pred: %s \n" % response)
            # file_.write("true: %s \n" % samples["text_output"][0])
            # file_.write("\n")
            if random.random() < 0.1:
                print("*"*20)
                print("source:", samples["text_input"][0], len(samples["text_input"][0]))
                # print("tgt:", samples["text_output"][0], len(samples["text_input"][0]))
                print("Pred:", self.opt_tokenizer.decode(outputs.logits[0].argmax(1)))
                # print("tgt:", text[0], len(text[0]))
                print("Pred:", self.opt_tokenizer.decode(outputs.logits[0][32:].argmax(1)))
      

        return {"loss": loss}

    

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=1,
        max_length=2048,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        do_sample=True,
        num_captions=1,
        temperature=0.95,
        temp_prompt="skip",
        **kwargs
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        # max_length=2048
        # num_beams=1
        # do_sample=True
        # top_p=0.7
        # temperature=0.95
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, **kwargs}

        image = samples["image"]
        with torch.cuda.amp.autocast(
            enabled=(self.device != torch.device("cpu"))
        ):          
            image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_embeds = self.opt_proj(query_output.last_hidden_state)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.bool).to(image.device)


            if "prompt" in samples.keys():
                prompt = samples["prompt"]
            else:
                prompt = samples["text_input"]
                # prompt = self.prompt

            image_prompt = '<Image><ImageHere></Image>'
                   
            image_embeds, image_atts = self.prompt_wrap(image_embeds, image_atts, image_prompt, use_speech=False)


            device_type = "cuda" if "cuda" in str(self.device) else "cpu"
            opt_tokens = self.opt_tokenizer([prompt], return_tensors="pt", padding=True).to(self.device)
            opt_tokens = opt_tokens.to(self.device)
            context_length = opt_tokens.input_ids.size(1)   


            empty_targets = (
                torch.ones(image_atts.size(), dtype=torch.long).to(image.device).fill_(0)
            )
            opt_tokens['input_ids'] = torch.cat([empty_targets, opt_tokens.input_ids], dim=1)
            # opt_tokens['attention_mask'] = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)
            opt_tokens = opt_tokens.to(image.device)

            del opt_tokens['attention_mask']
            del opt_tokens['position_ids']
            context_length = opt_tokens.input_ids.size(1)

            with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):

                outputs = self.opt_model.generate(
                    **opt_tokens, **gen_kwargs, input_image=image_embeds,
                )

                outputs = outputs.tolist()[0][context_length -  2:]
                response = self.opt_tokenizer.decode(outputs)
                response = response.strip()
                response = response.replace("[[训练时间]]", "2023年")
                return [response]
            

    @torch.no_grad()
    def generate_demo(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
        **kwargs
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        max_length=2048
        num_beams=1
        do_sample=True
        top_p=0.7
        temperature=0.95
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, **kwargs}

        use_image = samples["use_image"]
        image = samples["image"]
        with torch.cuda.amp.autocast(
            enabled=(self.device != torch.device("cpu"))
        ):          
            if use_image:
                image = samples["image"].to(self.device)
                image_embeds = self.ln_vision(self.visual_encoder(image))
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                    image.device
                )

                query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

                image_embeds = self.opt_proj(query_output.last_hidden_state)
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.bool).to(image.device)

            if "prompt" in samples.keys():
                prompt = samples["prompt"]
            else:
                prompt = samples["text_input"]
                # prompt = self.prompt

            # if isinstance(prompt, str):
            #     prompt = [prompt] * image.size(0)
            # else:
            #     assert len(prompt) == image.size(
            #         0
            #     ), "The number of prompts must be equal to the batch size."


            device_type = "cuda" if "cuda" in str(self.device) else "cpu"
            opt_tokens = self.opt_tokenizer([prompt], return_tensors="pt", padding=True).to(image.device)
            opt_tokens = opt_tokens.to(image.device)
            context_length = opt_tokens.input_ids.size(1)    
            
            with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
                if use_image:
                    image_prompt = '<Image><ImageHere></Image>'
                   
                    image_embeds, image_atts = self.prompt_wrap(image_embeds, image_atts, image_prompt, use_speech=False)

                    empty_targets = (
                        torch.ones(image_atts.size(), dtype=torch.long).to(image.device).fill_(0)
                    )
                    opt_tokens['input_ids'] = torch.cat([empty_targets, opt_tokens.input_ids], dim=1)
                    # opt_tokens['attention_mask'] = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)
                    opt_tokens = opt_tokens.to(image.device)

                    del opt_tokens['attention_mask']
                    del opt_tokens['position_ids']
                    context_length = opt_tokens.input_ids.size(1)
                    print('context_length: ', context_length)
                    outputs = self.opt_model.generate(
                            **opt_tokens, **gen_kwargs, input_image=image_embeds,
                        )
                else:
                    outputs = self.opt_model.generate(
                        **opt_tokens, **gen_kwargs, 
                    )
                print('output length: ', len(outputs.tolist()[0]))
                # print(outputs.tolist()[0])
                # response = self.opt_tokenizer.decode(outputs.tolist()[0])
                # print(response)
                outputs = outputs.tolist()[0][context_length -  2:]
                response = self.opt_tokenizer.decode(outputs)
                response = response.strip()
                response = response.replace("[[训练时间]]", "2023年")
                response = self.process_response(response)
                return [response]
            
    @torch.no_grad()
    def generate_audio_or_music(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
        **kwargs
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        # self.Qformer.bert.encoder.train()
        max_length=2048
        num_beams=5
        do_sample=True
        # do_sample=False
        top_p=0.7
        temperature=0.95
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, **kwargs}

        audio = samples["audio"]
        with torch.cuda.amp.autocast(
            enabled=(self.device != torch.device("cpu"))
        ):          
            audio = samples["audio"]
            audio_embeds = self.audio_encoder(input_values=audio['input_values'].to(self.audio_encoder.device), return_dict=True)
            audio_embeds = audio_embeds.last_hidden_state
            image, image_embeds = audio_embeds, audio_embeds
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_embeds = self.opt_proj(query_output.last_hidden_state)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.bool).to(image.device)


            if "prompt" in samples.keys():
                prompt = samples["prompt"]
            else:
                prompt = samples["text_input"]
                # prompt = self.prompt

            image_prompt = '<Image><ImageHere></Image>'
                   
            image_embeds, image_atts = self.prompt_wrap(image_embeds, image_atts, image_prompt, use_speech=False)


            device_type = "cuda" if "cuda" in str(self.device) else "cpu"
            opt_tokens = self.opt_tokenizer([prompt], return_tensors="pt", padding=True).to(self.device)
            opt_tokens = opt_tokens.to(self.device)
            context_length = opt_tokens.input_ids.size(1)   


            empty_targets = (
                torch.ones(image_atts.size(), dtype=torch.long).to(image.device).fill_(0)
            )
            opt_tokens['input_ids'] = torch.cat([empty_targets, opt_tokens.input_ids], dim=1)
            # opt_tokens['attention_mask'] = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)
            opt_tokens = opt_tokens.to(image.device)

            del opt_tokens['attention_mask']
            del opt_tokens['position_ids']
            context_length = opt_tokens.input_ids.size(1)

            with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):

                outputs = self.opt_model.generate(
                    **opt_tokens, **gen_kwargs, input_image=image_embeds,
                )

                outputs = outputs.tolist()[0][context_length -  2:]
                response = self.opt_tokenizer.decode(outputs)
                response = response.strip()
                response = response.replace("[[训练时间]]", "2023年")
                return [response]

    def process_response(self, response):
        response = response.strip()
        response = response.replace("[[训练时间]]", "2023年")
        response = response.replace("ChatGLM-6B", "百灵（Lark）")
        response = response.replace("ChatGLM", "百灵（Lark）")
        response = response.replace("清华大学 KEG 实验室和智谱 AI 公司", "中科院自动化所认知计算小组")
        response = response.replace("清华大学 KEG 实验室", "中科院自动化所")
        response = response.replace("智谱 AI 公司", "认知计算小组")
        response = response.replace("中科院自动化所和智谱AI", "中科院自动化所")
        response = response.replace("智谱AI", "")
        
        
        punkts = [
            [",", "，"],
            ["!", "！"],
            [":", "："],
            [";", "；"],
            ["\?", "？"],
        ]
        for item in punkts:
            response = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], response)
            response = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], response)
        return response
            
    @torch.no_grad()
    def _generate(
            self,
            **kwargs,
    ):
        MASK, gMASK = 150000, 150001
        bos, eos = 150004, 150005

        if "eos_token_id" not in kwargs:
            kwargs["eos_token_id"] = eos

        stop = False

        return_seqs = []

        while True:
            output_ids = super().generate(**kwargs)
            kwargs['inputs_opt'] = None
            return_seqs = []
            max_length = 0

            for i in range(output_ids.shape[0]):
                output_seq = output_ids[i].tolist()
                mask_token = MASK if MASK in output_seq else gMASK
                mask_position = output_seq.index(mask_token)
                bos_position = output_seq.index(bos)
                if eos in output_seq:
                    eos_position = output_seq.index(eos)
                else:
                    eos_position = len(output_seq)

                return_seq = output_seq[:mask_position] + output_seq[bos_position + 1:eos_position] + output_seq[
                                                                                                      mask_position + 1:bos_position]
                max_length = max(max_length, len(return_seq))
                return_seqs.append(return_seq)

            for i in range(output_ids.shape[0]):
                return_seqs[i] = [0] * (max_length - len(return_seqs[i])) + return_seqs[i]  # padding
                if mask_token not in return_seqs[i]:
                    stop = True

            if stop:
                break

            for return_seq in return_seqs:
                return_seq += [bos]

            kwargs['input_ids'] = torch.tensor(return_seqs, dtype=torch.long, device=kwargs['input_ids'].device)

        return torch.tensor(return_seqs, dtype=torch.long, device=kwargs['input_ids'].device)

    
    def preprocess_function_train(self, examples, device, src_txt_len=None, max_tgt_len=None):
        if src_txt_len is None:
            src_txt_len = self.max_txt_len
        if src_txt_len is None:
            max_tgt_len = self.max_txt_len

        max_seq_length = src_txt_len + max_tgt_len

        model_inputs = {
            "input_ids": [],
            "labels": [],
        }

        
        if examples.get("text_input") and examples.get("text_output"):
            prompt_text = examples["text_input"]
            answer_text = examples["text_output"]
        
        else:
            # if self.modality == "audio":
            if self.modality == "music":
                prompts_list = prompts_list_music_cn
            elif self.modality == "image":
                prompts_list = prompts_list_image_cn
            elif self.modality == "signal":
                prompts_list = prompts_list_signal_cn
            else:
                raise NotImplementedError

            prompt_text = []
            answer_text = []
            for sample_text in examples["text_input"]:
                # 如果是QA的text input，就用QA的prompt
                # given_text.append("描述这段音乐。" + sample_text)
                # prompt_text.append("描述这段音乐。")
                # continue

                # 特殊符号，一般用来分割问题和答案
                if "<|||>" in sample_text:
                    ques_text = sample_text.split("<|||>")[0]
                    # if ques_text[-1] not in ["?", "？", "。", "!", "！", "…", "."]:
                        # ques_text += "?"
                    temp_prompt = ques_text
                    # temp_prompt = prompt_input.format(ques_text)
                    prompt_text.append(temp_prompt)
                    answer_text.append(sample_text.split("<|||>")[1])

                # 正常caption描述类文本
                else: 
                    # temp_prompt = random.choice(prompts_list)
                    temp_prompt = random.choice(prompts_list)
                    # TODO(jing): 验证通用的空prompt是否有效
                    # temp_prompt = "[gMASK]"
                    # temp_prompt = prompt_input.format(random.choice(prompts_list_new))
                    prompt_text.append(temp_prompt)
                    # given_text.append(temp_prompt + " " + sample_text)
                    answer_text.append(sample_text)

        # for question, answer in zip(examples["text_input"], examples["text_output"]):
        for question, answer in zip(prompt_text, answer_text):
            prompt = question
            a_ids = self.opt_tokenizer.encode(text=prompt, add_special_tokens=False)
            b_ids = self.opt_tokenizer.encode(text=answer, add_special_tokens=False)

            if len(a_ids) > src_txt_len - 1:
                a_ids = a_ids[: src_txt_len - 1]

            if len(b_ids) > max_tgt_len- 2:
                b_ids = b_ids[: max_tgt_len - 2]

            input_ids = self.opt_tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)

            context_length = input_ids.index(self.opt_tokenizer.bos_token_id)
            mask_position = context_length - 1
            labels = [-100] * context_length + input_ids[mask_position+1:]
            
            pad_len = max_seq_length - len(input_ids)
            input_ids = input_ids + [self.opt_tokenizer.pad_token_id] * pad_len
            labels = labels + [self.opt_tokenizer.pad_token_id] * pad_len
            if self.ignore_pad_token_for_loss:
                labels = [(l if l != self.opt_tokenizer.pad_token_id else -100) for l in labels]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["labels"].append(labels)
        model_inputs["input_ids"] = torch.LongTensor(model_inputs["input_ids"]).to(device)
        model_inputs["labels"] = torch.LongTensor(model_inputs["labels"]).to(device)
        return model_inputs

    @torch.no_grad()
    def generate_pure_text(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
        temp_prompt="skip",
        **kwargs
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        max_length=2048
        num_beams=1
        do_sample=True
        top_p=0.7
        temperature=0.95
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, **kwargs}

        with torch.cuda.amp.autocast(
            enabled=(self.device != torch.device("cpu"))
        ):          
            if "prompt" in samples.keys():
                prompt = samples["prompt"]
            else:
                prompt = samples["text_input"]
                # prompt = self.prompt
            device_type = "cuda" if "cuda" in str(self.device) else "cpu"

            opt_tokens = self.opt_tokenizer([prompt], return_tensors="pt", padding=True).to(self.device)
            opt_tokens = opt_tokens.to(self.device)
            context_length = opt_tokens.input_ids.size(1)   

            with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):

                outputs = self.opt_model.generate(
                    # **opt_tokens, **gen_kwargs, input_image=image_embeds,
                    **opt_tokens, **gen_kwargs, input_image=None,
                )

                outputs = outputs.tolist()[0][context_length -  2:]
                response = self.opt_tokenizer.decode(outputs)
                response = response.strip()
                response = response.replace("[[训练时间]]", "2023年")
                return [response]


    @torch.no_grad()
    def generate_cloud(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=1,
        max_length=30,
        min_length=1,
        top_p=0.95,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=0.7,
    ):
        image = samples["cloud"]
        device = image["coord"].device
        self.chatglm_proj = self.opt_proj
        self.chatglm_tokenizer = self.opt_tokenizer
        self.chatglm_model = self.opt_model
        # with self.maybe_autocast():
        with torch.cuda.amp.autocast(
            enabled=(self.device != torch.device("cpu"))
        ):          
            # fake_cloud_encoder_result = torch.rand(image["coord"].shape[0], 256, 384).to(device)        # [batch_size, 256, 384]
            # image_embeds = self.ln_cloud(fake_cloud_encoder_result)

            image_embeds = self.ln_cloud(self.cloud_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
    
            # Q-Former 的 learnable query 映射到 llama 的特征空间后得到的特征以及 attention mask
            inputs_query = self.chatglm_proj(query_output.last_hidden_state)

            if "text_input" in samples.keys():
                prompt = samples["text_input"]
            else:
                prompt = self.prompt

            # 把 prompt 变成 list
            prompt = [prompt] if isinstance(prompt, str) else prompt
            assert len(prompt) == image_embeds.size(0), "number of prompt != batch size"
            input_ids_list = []
            for p in prompt:
                prompt_ids = self.chatglm_tokenizer.encode(p, add_special_tokens=True)
                input_ids_list.append(torch.tensor(prompt_ids, dtype=torch.long).to(device))
                
            input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.chatglm_tokenizer.pad_token_id)
            # 把 learnable query 对应的占位用 input_ids 添加进去
            pad_input_ids = torch.ones(image_embeds.shape[0], self.num_query_token, dtype=torch.long).to(device).fill_(self.chatglm_tokenizer.pad_token_id)
            input_ids = torch.cat([pad_input_ids, input_ids], dim=1)


            outputs = self.chatglm_model.generate(
                input_ids=input_ids,
                # query_embeds=inputs_query,
                input_image=inputs_query,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )

            # 去掉前面的 query 以及prompt部分
            outputs = outputs[:, input_ids.shape[1]:]
            output_text = self.chatglm_tokenizer.batch_decode(
                outputs, skip_special_tokens=True, spaces_between_special_tokens=False
            )
            output_text = [text.strip() for text in output_text]
            
            # output_text = self.postprocess_text(output_text, device = device)
            return output_text

    @torch.no_grad()
    def generate_speech(
            self,
            samples,
            num_beams=5,
            max_length=30,
            min_length=1,
            **kwargs,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        max_length = 80
        num_beams = 10
        do_sample = False # True
        top_p = 1.0 # 0.7
        temperature = 1 # 0.95
        repetition_penalty = 1.0

        # do_sample = True
        # top_p = 0.7
        # temperature = 0.95

        # max_length=80
        # num_beams=1
        # do_sample=True # True
        # top_p=0.7
        # temperature=0.95 # 0.95
     
        # gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
        #               "temperature": temperature, **kwargs}
        gen_kwargs = {
            "max_length": max_length,
            "num_beams": num_beams,
            "do_sample": do_sample,
            "top_p": top_p,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            **kwargs
        }

        # get speech inputs
        speech_embeds = samples["speech_input"]  # B x T x C
        speech_embeds_attn_mask = (speech_embeds.sum(-1) != 0.0).long()  # B x T
        # speech_embeds_attn_mask = torch.ones(speech_embeds.size()[:-1], dtype=torch.bool).to(speech_embeds.device)
        speech_embeds = speech_embeds.to(self.device)
        speech_embeds_attn_mask = speech_embeds_attn_mask.to(self.device)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            
            if self.speech_adaptor_type == "fc":
                speech_embeds = self.speech_feat_proj(speech_embeds)  # B x T x C
                speech_embeds = nn.ReLU()(speech_embeds)
            else:
                speech_embeds = nn.ReLU()(self.pre_bert_proj(speech_embeds))  # B x T x C
                sbert_outputs = self.speech_bert_adaptor(
                    inputs_embeds=speech_embeds,
                    attention_mask=speech_embeds_attn_mask,
                )
                speech_embeds = nn.ReLU()(self.post_bert_proj(sbert_outputs[0]))  # B x T x C

            if "prompt" in samples.keys():
                prompt = samples["prompt"]
            else:
                prompt = samples["text_input"]

            speech_prompt = '<Speech><SpeechHere></Speech>'
            speech_embeds, speech_embeds_attn_mask = self.prompt_wrap(speech_embeds, speech_embeds_attn_mask, speech_prompt)

            opt_tokens = self.opt_tokenizer(prompt, return_tensors="pt", padding=True).to(speech_embeds.device)

            opt_tokens = opt_tokens.to(speech_embeds.device)
            context_length = opt_tokens.input_ids.size(1)

            empty_targets = (
                torch.ones(speech_embeds_attn_mask.size(), dtype=torch.long).to(speech_embeds.device).fill_(0)
            )

            opt_tokens['input_ids'] = torch.cat([empty_targets, opt_tokens.input_ids], dim=1)
            # opt_tokens['attention_mask'] = torch.cat([speech_embeds_attn_mask, opt_tokens.attention_mask], dim=1)
            opt_tokens = opt_tokens.to(speech_embeds.device)
            context_length = opt_tokens.input_ids.size(1)
            del opt_tokens['attention_mask']
            del opt_tokens['position_ids']

            outputs = self.opt_model.generate(
                **opt_tokens, **gen_kwargs,
                input_speech=speech_embeds
            )
            # outputs_list = outputs.tolist()
            # response = [self.opt_tokenizer.decode(_[context_length-2:]).strip().replace("[[训练时间]]", "2023年") for _ in outputs_list]
            try:
                # outputs = outputs.tolist()[0][context_length - 4:]
                outputs = outputs.tolist()[0][context_length - 2:]
                response = self.opt_tokenizer.decode(outputs)
                response = response.strip()
                response = response.replace("[[训练时间]]", "2023年")
                response = [response]
            except:
                response = ['Error']

        return response

    @classmethod
    def from_config(cls, cfg):

        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        opt_model = cfg.get("opt_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        modality = cfg.get("modality", "image")

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        print("max_txt_len", max_txt_len)
        freeze_qformer = cfg.get("freeze_qformer", False)

        model = cls(
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            opt_model=opt_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            modality=modality,
            freeze_qformer=freeze_qformer
        )
        model.load_checkpoint_from_config(cfg)

        return model