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
from transformers.generation.utils import GenerationConfig

prompt_input =  (
    "Above is an medical image illustration of a patient, write a response that appropriately completes the following request."
    "\n\n Input:\n{}\n\n Response:"
)


USE_PROMPT = True
PROMPT_PREFIX = (
    "以下是一个描述任务的指令，并配有一个提供详细上下文信息的输入。"
    "请写一个完成该指令的适当回复。\n\n"
    "### 指令:\n{}\n\n### 输入: 图像-[ ]\n\n### 回复:" # 这里insert的地方，一定要注意保证[ ]不要被tokenizer识别为一个token，这样长度才不会出问题，也才好插入query_emb
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

prompts_list_image_cn = [
    "生成给定图片的概况。",
    "生成给定图片的标题。",
    "这幅图像描述了什么？",
    "为这幅图像生成概述。",
    "为给定的图像生成标题。",
    "描述给定图像的内容是什么？",
    "根据给定的图像生成简要摘要。",
    "通过分析给定的图像来生成标题。",
    "请生成一份关于给定图像的概述。",
    "使用自然语言生成为给定图像的简要描述。",
    "请提供有关给定图像的简要信息。",
    "为给定图像生成简洁的标题和概述。",
    "请生成一份关于这幅图像的描述.",
]

prompts_list_music_cn = [
    "生成这段音乐的概述。",
    "描述这段音乐。",
    "请描述这段音乐。",
    "请描述之前给定的音乐材料。",
    "请描述之前给定的材料。",
    "概括这段音乐的内容。",
    "这段音乐是关于什么的？",
    "这段音乐的情况是什么？",
    "请以自己的话描述这段音乐。",
    "试着描述这段音乐，表达对它的感受和印象。",
    "请用你自己的话概括之前给定的材料，强调其中的重点和关键信息。",
    "请描述之前给定的材料，让读者能够清晰地理解它所涉及的主题和背景。",
    "请以简明扼要的语言概括这段音乐的内容和主旨，让读者能够快速了解它的特点和风格。",
    "这段音乐片段的信息有哪些？请用你自己的话进行描述，并解释它们对你的意义。",
    "请描述这段音乐所呈现的场景和情境。",
]

prompts_list_signal_cn = [
    "生成给定信号的概况。",
    "请描述给定信号。",
    "描述下这段信号的情况。",
    "告诉我这段信号的相关信息。"
    "这段信号是什么？"
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

# 适合已经有“回答：”这种的
prompts_list_question = [
    "{}\n\n ###",
    "{} \n\n",
    "{}",
    "{}\n",
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
        "pretrain_llama7b-linba": "configs/models/blip2/blip2_pretrain_llama7b-linba.yaml",
        "pretrain_alpaca7b": "configs/models/blip2/blip2_pretrain_alpaca7b.yaml",
        "pretrain_vicuna13b": "configs/models/blip2/blip2_pretrain_vicuna13b.yaml",
        "pretrain_zidong13b": "configs/models/blip2/blip2_pretrain_zidong13b.yaml",
        "pretrain_zidong13b-audio": "configs/models/blip2/blip2_pretrain_zidong13b-audio.yaml",
        "pretrain_baichuan7b": "configs/models/blip2/blip2_pretrain_baichuan7b.yaml",
        "pretrain_baichuan13b": "configs/models/blip2/blip2_pretrain_baichuan13b.yaml",
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
        modality="image",
        freeze_qformer=False
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()


        self.llama_model = llama_model
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
        elif modality == "audio":
            self.audio_encoder, _ = self.init_audio_encoder()
            if freeze_vit:
                for name, param in self.audio_encoder.named_parameters():
                    param.requires_grad = False               
                self.audio_encoder = self.audio_encoder.eval()
                logging.info("freeze audio encoder")
            self.Qformer, self.query_tokens = self.init_Qformer(
                num_query_token, 1024, #768 # 512, #self.visual_encoder.num_features
            )
        else:
            raise NotImplementedError("modality {} not implemented".format(modality))

        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        # Notice(Jing): 呼应L47的注释， 暂时用本地的LLAMA模型，修改了bug了，之后可以考虑替换成Huggingface上的
        # llama_model = "/data/shij/llama/zidongv2/step1000_v2"
        logging.info("load LLAMA model:{}".format(llama_model))
        # self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_model)
        # self.llama_model = AutoModelForCausalLM.from_pretrained(llama_model)

        self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_model, trust_remote_code=True)
        self.llama_model = AutoModelForCausalLM.from_pretrained(llama_model, trust_remote_code=True)#.half()
        # self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_model)
        # self.llama_model = AutoModelForCausalLM.from_pretrained(llama_model)#.half()
        if "Baichuan-13B" in llama_model: 
            self.llama_model.generation_config = GenerationConfig.from_pretrained(llama_model)
            self.fixed_gen_config = True
        else:
            self.fixed_gen_config = False

        self.llama_tokenizer.pad_token_id = 0 # 这里不太规范，tokenizer的设置里没有，先加到这里吧

        # self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_model, use_fast=False)
        # self.llama_model = llamaForCausalLM.from_pretrained(
        #     llama_model, torch_dtype=torch.float16
        # )
        for name, param in self.llama_model.named_parameters():
            # 暂时先不用 bf16？llama没用
            param.data = param.data.bfloat16()
            param.requires_grad = False
            # if 'layers.31' in name : #or 'decoder.block.22.layer' in name: #or 'decoder.block.21.layer' in name:
                # logging.info("open LLAMA model:{}".format(name))
                # param.requires_grad = True
            # else:
                # param.requires_grad = False
                # param.data = param.data.bfloat16()
        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False               
            self.query_tokens.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            logging.info("freeze Qformer") 

        self.eos_token_id = self.llama_tokenizer(
            "</s>", add_special_tokens=False
        ).input_ids[0]

        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        # self.max_txt_len = 240
        self.prompt = prompt
        prompt_tokens = self.llama_tokenizer(self.prompt, return_tensors="pt", add_bos_token=False, add_eos_token=False, max_length=self.max_txt_len)
        self.prompt_length = prompt_tokens.attention_mask.sum(1)

        self.query_insert_mode = "before" # blip2默认的放到前面
        # self.query_insert_mode = "middle" # 改成可以支持放到中间的

    def prompt_wrap(self, img_embeds, atts_img):
        """
        wrap the image embedding with prompt
        image embedding: [batch_size, query_len, hidden_size]
        atts_img: [batch_size, query_len]
        """
        if not self.modality:
            raise NotImplementedError("modality not specified")
            return img_embeds, atts_img
        else:
            modal_token = self.modality.capitalize()
            special_token = f"<{modal_token}Here>"
            prompt = f"<{modal_token}><{modal_token}Here></{modal_token}>"

            batch_size = img_embeds.shape[0]
            p_before, p_after = prompt.split(special_token)
            p_before_tokens = self.llama_tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_after_tokens = self.llama_tokenizer(
                p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
            return wrapped_img_embeds, wrapped_atts_img
        

    def forward(self, samples):
        self.Qformer.bert.encoder.eval()

        if self.modality == "image":
            image = samples["image"]
            image_embeds = self.ln_vision(self.visual_encoder(image))
        elif self.modality == "audio":
        # image_embeds = image
            audio = samples["audio"]
            audio_embeds = self.audio_encoder(input_values=audio['input_values'].squeeze().to(self.audio_encoder.device), return_dict=True)
            audio_embeds = audio_embeds.last_hidden_state
            image, image_embeds = audio_embeds, audio_embeds
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

        inputs_llama = self.llama_proj(query_output.last_hidden_state) # bs, len_query, dim
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        inputs_llama, atts_llama = self.prompt_wrap(inputs_llama, atts_llama)

        self.llama_tokenizer.padding_side = "right"

        if USE_PROMPT:
            given_text = []
            prompt_text = []
            # decide which prompt to use
            if self.modality == "image":
                prompts_list = prompts_list_image_cn
            elif self.modality == "audio":
                prompts_list = prompts_list_music_cn

            for sample_text in samples["text_input"]:
                # 如果是QA的text input，就用QA的prompt
                # given_text.append("描述这段音乐。" + sample_text)
                # prompt_text.append("描述这段音乐。")
                # continue

                # 特殊符号，一般用来分割问题和答案
                if "<|||>" in sample_text:
                    ques_text = sample_text.split("<|||>")[0]
                    # if ques_text[-1] not in ["?", "？", "。", "!", "！", "…", "."]:
                        # ques_text += "?"
                    temp_prompt = random.choice(prompts_list_question).format(ques_text)
                    # temp_prompt = prompt_input.format(ques_text)
                    prompt_text.append(temp_prompt)
                    given_text.append(temp_prompt + " " + sample_text.split("<|||>")[1])

                # 正常caption描述类文本
                else: 
                    if 0:
                        temp_prompt = random.choice(prompts_list) + "<reserved_103>"
                        # temp_prompt = PROMPT_PREFIX.format(random.choice(prompts_list))
                        # temp_prompt = PROMPT_PREFIX.format(prompts_list[0])
                        # temp_prompt = prompt_input.format(random.choice(prompts_list_new))
                        prompt_text.append(temp_prompt)
                        # given_text.append(temp_prompt + " " + sample_text)
                        given_text.append(temp_prompt + sample_text)
                    else:
                        # temp_prompt = "<reserved_103>"
                        temp_prompt = "<s>"
                        prompt_text.append(temp_prompt) # 变成 <Image>IMAGE_EMBEDDING<\Image> 请描述xxx<s>答案</s>的形式，参考了lora fine-tune的一些设置。
                        given_text.append(temp_prompt + sample_text) # 变成 <Image>IMAGE_EMBEDDING<\Image> <reserved_103>答案</s>的形式，跟baichuan-13B-chat的先对齐一下看看。
                        # given_text.append(temp_prompt + "<s>" +  sample_text) 
                        # given_text.append(temp_prompt + "<s>" +  sample_text) # 变成 <Image>IMAGE_EMBEDDING<\Image> 请描述xxx<s>答案</s>的形式，参考了lora fine-tune的一些设置。
            text = [t+'</s>' for t in given_text]

        llama_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_bos_token=False,
        ).to(image.device)

        # if 1: # 强行去除开头的<s>
        #     llama_tokens = {k:v[:,1:] for k,v in llama_tokens.items()}

        targets = llama_tokens.input_ids.masked_fill(
            llama_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        if self.prompt:
            targets[:, : self.prompt_length] = -100  # do not apply loss to the prompt
        if prompt_text is not None:
            # 这里专门来处理我自己定义的prompt
            prompts_tokens = self.llama_tokenizer(prompt_text, padding="longest", return_tensors="pt", add_bos_token=True)
            prompt_lens = prompts_tokens.attention_mask.sum(1) # 
            for idx, length in enumerate(prompt_lens.tolist()):
                targets[idx, : length] = -100  # do not apply loss to the prompt

            if self.query_insert_mode == "middle":
                insert_idx = prompt_lens - 10 # [bs] 这个值是跟prompt的长度有关的，从后往前是固定的，所以-一个固定的值
                assert prompts_tokens.input_ids[-1][insert_idx[-1]: insert_idx[-1]+2].tolist() == [29961, 4514] # 每一个batch验证一下insert_idx是否是 [ ]
                split_idx = (insert_idx + 1, llama_tokens.input_ids.size(1) - insert_idx -1)  # 用来给prompt分割成两部分,以在中间插入query_emb
                split_idx = torch.stack(split_idx).t() # [bs, 2]

        empty_targets = (
            torch.ones(atts_llama.size(), dtype=torch.long).to(image.device).fill_(-100)
        ) # (bs, query_length) 为query_emb添加空target
        targets = torch.cat([empty_targets, targets], dim=1) # 不论是before还是middle，总的添加的长度是不变的，所以直接cat
        attention_mask = torch.cat([atts_llama, llama_tokens.attention_mask], dim=1) # ditto
        inputs_embeds = self.llama_model.model.embed_tokens(llama_tokens.input_ids) # 看了源码，直接是embed_tokens就可以

        if self.query_insert_mode == "before":
            inputs_embeds = torch.cat([inputs_llama, inputs_embeds], dim=1)

        elif self.query_insert_mode == "middle":
            embeds_list = []
            for idx, in_index in enumerate(insert_idx):
                part1, part2 = inputs_embeds[idx].split((split_idx[idx].tolist()), dim=0) # divide into two parts
                new_tensor = torch.cat([part1, inputs_llama[idx], part2], dim=0) # insert query_emb
                embeds_list.append(new_tensor)
            inputs_embeds = torch.stack(embeds_list, dim=0)

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )

        if 1 and random.random() < 0.05:
            print("*"*20)
            print("tgt:", samples["text_input"][0], len(samples["text_input"][0]))
            print("Pred:", self.llama_tokenizer.decode(outputs.logits[0].argmax(1)))
            print("tgt:", text[0], len(text[0]))
            print("target(-100) from 32-50:", targets[0][32:50], targets[0].shape)
            if self.query_insert_mode == "before":
                print("Pred:", self.llama_tokenizer.decode(outputs.logits[0][32:].argmax(1)))
            elif self.query_insert_mode == "middle":
                print("Pred:", self.llama_tokenizer.decode(outputs.logits[0][split_idx[0][0]+32+8:].argmax(1)))

        loss = outputs.loss
        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        samples,
        do_sample=False,
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
        self.Qformer.bert.encoder.train()
        image = samples["image"]
        with torch.cuda.amp.autocast(
            enabled=(self.device != torch.device("cpu"))
        ):          
            image_embeds = self.ln_vision(self.visual_encoder(image)) # 1,257, 1408
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1) # [1,32,768]
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_llama = self.llama_proj(query_output.last_hidden_state) # [1,32,4096]
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
            inputs_llama, atts_llama = self.prompt_wrap(inputs_llama, atts_llama)

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
            raw_prompt = prompt

            prompt = [prompt] * image.size(0)

            llama_tokens = self.llama_tokenizer(prompt, return_tensors="pt").to(image.device)
            input_ids = llama_tokens.input_ids # [1,5]
            attention_mask = torch.cat([atts_llama, llama_tokens.attention_mask], dim=1) # [1,32+5]

            if use_nucleus_sampling:
                query_embeds = inputs_llama.repeat_interleave(num_captions, dim=0)
                num_beams = 1
            else:
                query_embeds = inputs_llama.repeat_interleave(num_beams, dim=0)

            device_type = "cuda" if "cuda" in str(self.device) else "cpu"
            with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
                if self.query_insert_mode == "before":
                    inputs_embeds = self.llama_model.model.embed_tokens(llama_tokens.input_ids)
                    inputs_embeds = torch.cat([inputs_llama, inputs_embeds], dim=1)

                elif self.query_insert_mode == "middle":
                    prompt = PROMPT_PREFIX.format(raw_prompt)
                    prompt = [prompt] * image.size(0)
                    prompts_tokens = self.llama_tokenizer(prompt, padding="longest", return_tensors="pt", add_bos_token=True).to(image.device)
                    prompt_lens = prompts_tokens.attention_mask.sum(1) # 

                    insert_idx = prompt_lens - 10 # [bs] 这个值是跟prompt的长度有关的，从后往前是固定的，所以-一个固定的值
                    assert prompts_tokens.input_ids[-1][insert_idx[-1]: insert_idx[-1]+2].tolist() == [29961, 4514] # 每一个batch验证一下insert_idx是否是 [ ]
                    split_idx = (insert_idx + 1, prompts_tokens.input_ids.size(1) - insert_idx -1)  # 用来给prompt分割成两部分,以在中间插入query_emb
                    split_idx = torch.stack(split_idx).t() # [bs, 2]

                    inputs_embeds = self.llama_model.model.embed_tokens(prompts_tokens.input_ids)
                    embeds_list = []
                    for idx, in_index in enumerate(insert_idx):
                        part1, part2 = inputs_embeds[idx].split((split_idx[idx].tolist()), dim=0) # divide into two parts
                        new_tensor = torch.cat([part1, inputs_llama[idx], part2], dim=0) # insert query_emb
                        embeds_list.append(new_tensor)
                    inputs_embeds = torch.stack(embeds_list, dim=0)
                    attention_mask = torch.cat([atts_llama, prompts_tokens.attention_mask], dim=1) # [1,32+5]

            if not self.fixed_gen_config:
                outputs = self.llama_model.generate(
                    # input_ids=input_ids,
                    # query_embeds=query_embeds,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    do_sample=do_sample,
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
            else:
                outputs = self.llama_model.generate(
                    # input_ids=input_ids,
                    # query_embeds=query_embeds,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    generation_config=self.llama_model.generation_config
                )

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
        freeze_qformer = cfg.get("freeze_qformer", False)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)

        modality = cfg.get("modality", "image")

        model = cls(
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            freeze_qformer=freeze_qformer,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            modality=modality,
        )
        model.load_checkpoint_from_config(cfg)

        return model

    @torch.no_grad()
    def generate_audio(
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
        # The mel branch
        # image = samples["audio"]

        # The wave2vec branch
        audio = samples["audio"]
        print('audio: ', audio)
        print('audio shape:',audio['input_values'].shape)
        
        audio_embeds = self.audio_encoder(input_values=audio['input_values'].to(self.audio_encoder.device), return_dict=True)
        audio_embeds = audio_embeds.last_hidden_state
        image, image_embeds = audio_embeds, audio_embeds

        with torch.cuda.amp.autocast(
            enabled=(self.device != torch.device("cpu"))
        ):          
            # image_embeds = self.ln_vision(self.visual_encoder(image))
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

            llama_tokens = self.llama_tokenizer(prompt, return_tensors="pt", add_bos_token=False).to(image.device)
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
                inputs_embeds=inputs_embeds, # [1,42,5120]
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

            # Todo(jing): 是否加入并不一定的，只给Inputs_embeds的时候，输出不返回prompt的
            # prompt_length = llama_tokens.input_ids.shape[1]
            # output_text = self.llama_tokenizer.batch_decode(outputs[:, prompt_length:], skip_special_tokens=True)
            output_text = self.llama_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            output_text = [text.strip() for text in output_text]
            return output_text

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
        # self.Qformer.bert.encoder.train()
        with torch.cuda.amp.autocast(
            enabled=(self.device != torch.device("cpu"))
        ):          
            if "prompt" in samples.keys():
                prompt = samples["prompt"]
            else:
                prompt = self.prompt

            print("Prompt:", prompt)
            prompt = [prompt] 

            llama_tokens = self.llama_tokenizer(prompt, return_tensors="pt").to(self.llama_model.device)
            attention_mask = llama_tokens.attention_mask

            device_type = "cuda" if "cuda" in str(self.device) else "cpu"
            print("device_type:", device_type)
            # with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
                # inputs_embeds = self.llama_model.model.embed_tokens(llama_tokens.input_ids)
            inputs_embeds = self.llama_model.model.embed_tokens(llama_tokens.input_ids)

            outputs = self.llama_model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, generation_config=self.llama_model.generation_config)

            # Todo(jing): 是否加入并不一定的，只给Inputs_embeds的时候，输出不返回prompt的
            # prompt_length = llama_tokens.input_ids.shape[1]
            # output_text = self.llama_tokenizer.batch_decode(outputs[:, prompt_length:], skip_special_tokens=True)
            output_text = self.llama_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            output_text = [text.strip() for text in output_text]
            return output_text