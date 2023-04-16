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

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.modeling_opt import OPTForCausalLM, OPTConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

prompt_input =  (
    "Above is an medical image illustration of a patient, write a response that appropriately completes the following request."
    "\n\n Input:\n{}\n\n Response:"
)
    

prompts_list_new = [
    "Describe the image.",
    "Describe the preceding.",
    "Summarize the image.",
    "What is this picture about?",
    "What is the image about?",
    "What does this condition show?",
    "What could we see see from the image?",
    "Tell me the description of the image.",
    "How is the situation in the image?",
    "How about my condition?",
    "What is the condition?",
    "What is the patient's condition?",
    "Summarize my condition.",
]
prompts_list = [
    "Describe the image. Response:",
    "Describe the preceding. Answer:",
    "Summarize the image. Response:",
    # "A picture of",
    "What is the image about? Answer:",
    # "The image shows",
    # "We could see from the image that",
    # "The description of the image is",
    "How is the situation in the image? Answer:",
]


prompts_list_question = [
    "{}Answer:",
    "{}Response:",
    # "{}\nAnswer:",
    # "Question: {}\nAnswer:",
    # "Question: {}Answer:",
    # "{}",
    # "{}\n",
]

# prompts_list = [
#     "Describe the image. Response:",
# ]
# prompts_list_question = [
#     "{}Answer:",
#     "Question: {}Answer:",
# ]

@registry.register_model("blip2_llama")
class Blip2LLaMA(Blip2Base):
    """
    BLIP2 llama model.
    Supported model types:
        - pretrained_llama2.7b: pretrained model with llama2.7b
        - pretrained_llama6.7b: pretrained model with llama6.7b
        - caption_coco_llama2.7b: fintuned image captioning model with llama2.7b
        - caption_coco_llama6.7b: fintuned image captioning model with llama6.7b
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_llama", "caption_coco_llama2.7b")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        # "pretrain_llama2.7b": "configs/models/blip2/blip2_pretrain_llama2.7b.yaml",
        "pretrain_llama7b": "configs/models/blip2/blip2_pretrain_llama7b.yaml",
        "pretrain_alpaca7b": "configs/models/blip2/blip2_pretrain_alpaca7b.yaml",
        "pretrain_vicuna13b": "configs/models/blip2/blip2_pretrain_vicuna13b.yaml",
        "pretrain_zidong13b": "configs/models/blip2/blip2_pretrain_zidong13b.yaml",
    }

    def __init__(
        self,
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        # llama_model="facebook/llama-2.7b",
        llama_model="decapoda-research/llama-7b-hf", # Notice: the series of llama-xb-hf gets bugs (https://huggingface.co/decapoda-research/llama-7b-hf/discussions/9), waiting for fix 202303018
        prompt="",
        max_txt_len=320, # Notice(Jing): the max_txt_len is 320 for llama-7b-hf
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        """
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False               
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
        """

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, 512, #self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        # Notice(Jing): 呼应L47的注释， 暂时用本地的LLAMA模型，修改了bug了，之后可以考虑替换成Huggingface上的
        # llama_model = "/data2/shij/llama/llama-7b-hf"
        logging.info("load LLAMA model:{}".format(llama_model))
        self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_model)
        self.llama_model = AutoModelForCausalLM.from_pretrained(llama_model)
        self.llama_tokenizer.pad_token_id = 0 # 这里不太规范，tokenizer的设置里没有，先加到这里吧

        # self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_model, use_fast=False)
        # self.llama_model = llamaForCausalLM.from_pretrained(
        #     llama_model, torch_dtype=torch.float16
        # )
        for name, param in self.llama_model.named_parameters():
            # 暂时先不用 bf16？llama没用
            param.data = param.data.bfloat16()
            param.requires_grad = False
            continue 
            # if 'layers.31' in name : #or 'decoder.block.22.layer' in name: #or 'decoder.block.21.layer' in name:
                # logging.info("open LLAMA model:{}".format(name))
                # param.requires_grad = True
            # else:
                # param.requires_grad = False
                # param.data = param.data.bfloat16()

        self.eos_token_id = self.llama_tokenizer(
            # "\n", add_special_tokens=False
            "</s>", add_special_tokens=False
        ).input_ids[0]

        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )

        # self.max_txt_len = max_txt_len
        self.max_txt_len = 240
        self.prompt = prompt
        prompt_tokens = self.llama_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)

    def forward(self, samples):
        # image = samples["image"]
        # image_embeds = self.ln_vision(self.visual_encoder(image))
        image = samples["audio"]
        image_embeds = image

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

        inputs_llama = self.llama_proj(query_output.last_hidden_state) # bs, len_query, dim
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)

        self.llama_tokenizer.padding_side = "right"

        if prompts_list is not None:
            given_text = []
            prompt_text = []
            for sample_text in samples["text_input"]:
                # 如果是QA的text input，就用QA的prompt
                if "<|||>" in sample_text:
                    ques_text = sample_text.split("<|||>")[0]
                    # if ques_text[-1] not in ["?", "？", "。", "!", "！", "…", "."]:
                        # ques_text += "?"
                    temp_prompt = random.choice(prompts_list_question).format(ques_text)
                    # temp_prompt = prompt_input.format(ques_text)
                    prompt_text.append(temp_prompt)
                    given_text.append(temp_prompt + " " + sample_text.split("<|||>")[1])
                else: 
                    temp_prompt = random.choice(prompts_list)
                    # temp_prompt = prompt_input.format(random.choice(prompts_list_new))
                    prompt_text.append(temp_prompt)
                    given_text.append(temp_prompt + " " + sample_text)
            text = [t+'</s>' for t in given_text]

        llama_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(image.device)

        targets = llama_tokens.input_ids.masked_fill(
            llama_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        if self.prompt:
            targets[:, : self.prompt_length] = -100  # do not apply loss to the prompt
        if prompt_text is not None:
            # 这里专门来处理我自己定义的prompt
            prompt_lens = self.llama_tokenizer(prompt_text, padding="longest", return_tensors="pt").attention_mask.sum(1).tolist() # 
            for idx, length in enumerate(prompt_lens):
                targets[idx, : length] = -100  # do not apply loss to the prompt

        empty_targets = (
            torch.ones(atts_llama.size(), dtype=torch.long).to(image.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        # inputs_embeds = self.llama_model.model.decoder.embed_tokens(llama_tokens.input_ids) # 这里可能会出问题，调试的时候看一下
        inputs_embeds = self.llama_model.model.embed_tokens(llama_tokens.input_ids) # 看了源码，直接是embed_tokens就可以
        inputs_embeds = torch.cat([inputs_llama, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_llama, llama_tokens.attention_mask], dim=1)

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        print("*"*20)
        print("tgt:", samples["text_input"][0], len(samples["text_input"][0]))
        print("Pred:", self.llama_tokenizer.decode(outputs.logits[0][self.prompt_length[0]:].argmax(1)))
        print("tgt:", text[0], len(text[0]))
        print("Pred:", self.llama_tokenizer.decode(outputs.logits[0][32:].argmax(1)))
        loss = outputs.loss

        return {"loss": loss}

    @torch.no_grad()
    def generate(
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
        early_stopping=False,
        temp_prompt=None,
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

            inputs_llama = self.llama_proj(query_output.last_hidden_state)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)

            if "prompt" in samples.keys():
                prompt = samples["prompt"]
            else:
                prompt = self.prompt

            if temp_prompt == "skip": # 给部署的模型直接跳过去
                pass
            else:
                if temp_prompt is None:
                    prompt = "Question: What does this show? Answer:"
                    # prompt = "Let's start a new dialogue. Tell me a joke. Answer: \n"
                else:
                    prompt = temp_prompt
            # """
            
            print("Prompt:", prompt)

            prompt = [prompt] * image.size(0)

            llama_tokens = self.llama_tokenizer(prompt, return_tensors="pt").to(image.device)
            input_ids = llama_tokens.input_ids
            attention_mask = torch.cat([atts_llama, llama_tokens.attention_mask], dim=1)

            if use_nucleus_sampling:
                query_embeds = inputs_llama.repeat_interleave(num_captions, dim=0)
                num_beams = 1
            else:
                query_embeds = inputs_llama.repeat_interleave(num_beams, dim=0)

            device_type = "cuda" if "cuda" in str(self.device) else "cpu"
            with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
                inputs_embeds = self.llama_model.model.embed_tokens(llama_tokens.input_ids)
                inputs_embeds = torch.cat([inputs_llama, inputs_embeds], dim=1)

            outputs = self.llama_model.generate(
                # input_ids=input_ids,
                # query_embeds=query_embeds,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
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
                early_stopping=early_stopping,
            )

            # import pdb; pdb.set_trace()
            # Todo(jing): 是否加入并不一定的，只给Inputs_embeds的时候，输出不返回prompt的
            # prompt_length = llama_tokens.input_ids.shape[1]
            # output_text = self.llama_tokenizer.batch_decode(outputs[:, prompt_length:], skip_special_tokens=True)
            output_text = self.llama_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            output_text = [text.strip() for text in output_text]
            return output_text

    @classmethod
    def from_config(cls, cfg):

        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
        )
        model.load_checkpoint_from_config(cfg)

        return model
