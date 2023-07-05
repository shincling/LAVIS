import torch
from PIL import Image
import os
import base64
from io import BytesIO
import json
from flask import request
from flask_api import FlaskAPI
import requests
import torch
import numpy as np
from lavis.models import load_model_and_preprocess, load_preprocess_only
import soundfile as sf
import time
import random
from transformers import AutoProcessor
import sys

app = FlaskAPI(__name__)

modal_type = sys.argv[1]
assert modal_type in ['text', 'music', 'image', 'signal', 'all'], "modal_type must be one of ['text', 'music', 'image', 'signal']"

translation_api_cn2en_genel = 'http://172.18.30.134:6217/aliyuntranslate_gen/'
translation_api_en2cn_genel = 'http://172.18.30.134:6220/aliyuntranslate_en_gen/'
translation_api_cn2en_pro = 'http://172.18.30.134:5117/aliyuntranslate/'
translation_api_en2cn_pro = 'http://172.18.30.134:5118/aliyuntranslate_en/'
# print(requests.post(translation_api_en2an_genel, data={'text': tgt_text}).json()['Data']['Translated'])
# print(requests.post(translation_api_en2an_genel, data={'text': pred_text}).json()['Data']['Translated'])

audio_wav2vec_processor = AutoProcessor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")

def base64_to_wavfile(base64_str):
    wav_file = open("temp.wav", "wb")
    decode_string = base64.b64decode(base64_str)
    wav_file.write(decode_string)

# encode_string = base64.b64encode(open("audio.wav", "rb").read())

def base64_pil(base64_str):
    image = base64.b64decode(base64_str)
    image = BytesIO(image)
    image = Image.open(image)
    return image

def bytes2wav(bytes_base64):
    wav_bytes = base64.b64decode(bytes_base64)
    wav = np.frombuffer(buffer=wav_bytes, dtype=np.int16)
    wav=wav/32768
    return wav

def write_wave(path, data, n_chan=1):
    # remove the old file
    if os.path.exists(path):
        os.remove(path)
        print("remove the old file: %s" % path)
    sf.write(path, data, 16000)

def audio_file_reader(audio_path):
    # Notice(Jing): 这个跟caption_dataset.py的Audio的读取时候的处理保持一致
    audio_npy, sr = sf.read(audio_path)
    assert sr == 16000 and audio_npy.ndim == 1, "audio should be 1-channel and sr=16000, but got {} and {}".format(sr, audio_npy.shape[0])

    # normalize 
    audio_npy = audio_npy / np.max(np.abs(audio_npy))

    # padding or truncating to 10s 
    if audio_npy.shape[0] < 160000:
        audio_npy = np.pad(audio_npy, (0, 160000-audio_npy.shape[0]), 'constant')
    else:
        audio_npy = audio_npy[:160000]

    
    # audio_wav2vec_processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
    audio_npy = audio_wav2vec_processor(audio_npy, sampling_rate=16000, return_tensors="pt",)
    
    return audio_npy

def load_point_cloud(path:str):
    """
    从文件中读取点云
    path: 点云路径,绝对路径
    return: 点云, 字典类型, 包含 "coord", "color", "semantic_gt" 三个key
    """
    file_type = path.split(".")[-1]
    if file_type == "pth":
        cloud = torch.load(path)
        if(isinstance(cloud, tuple)):
            cloud = {"coord": cloud[0], "color": cloud[1], "semantic_gt": cloud[2]}
            cloud["color"] = ((cloud["color"] + 1) * 127.5).astype(np.uint8)
            cloud["color"] = cloud["color"].astype(np.float64)
            cloud["coord"] = cloud["coord"].astype(np.float64)
            # 把 coord 中的值归一化到 [-5, 5] 之间
            max_value = np.max(cloud["coord"])
            min_value = np.min(cloud["coord"])
            final_value = max(abs(max_value), abs(min_value))
            cloud["coord"] = cloud["coord"] / final_value  * 5.0

        # "coord" "color" "semantic_gt"
        if "semantic_gt" in cloud.keys():
            cloud["semantic_gt"] = cloud["semantic_gt"].reshape([-1])
            cloud["semantic_gt"] = cloud["semantic_gt"].astype(np.int64)
    elif file_type == "ply":
        cloud = {}
        plydata = plyfile.PlyData().read(path)
        points = np.array([list(x) for x in plydata.elements[0]])
        coords = np.ascontiguousarray(points[:, :3]).astype(np.float64)
        colors = np.ascontiguousarray(points[:, 3:6]).astype(np.float64)
        semantic_gt = np.zeros((coords.shape[0]), dtype=np.int64)
        cloud["coord"] = coords
        cloud["color"] = colors
        cloud["semantic_gt"] = semantic_gt
    else:
        raise ValueError("file type {} not supported".format(file_type))
    
    return cloud


# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
# load sample image
# raw_image = Image.open(
#     "/data2/shij/data/Med2/raw_data/1.3.12.2.1107.5.6.1.1586.30000017010600213448400000001.jpg"
# ).convert("RGB")

# default_image = "/data2/shij/data/Med2/raw_data/1.3.12.2.1107.5.6.1.1586.30000017010600213448400000001.jpg"


# loads BLIP-2 pre-trained model

# """
# caption_model, temp, _ = load_model_and_preprocess(
    # name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=torch.device("cuda:1"), pre_model_path= "/data2/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_t5/20230306094/checkpoint_199.pth"
# )
if modal_type == 'music':
    model, vis_processors, _ = load_model_and_preprocess(
        # name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device, pre_model_path= "/data2/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_t5/20230301150/checkpoint_99.pth"
        #name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device, pre_model_path="/data2/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_t5/20230303091/checkpoint_99.pth"
        # name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device, pre_model_path="/data2/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_t5_xxl/20230309020/checkpoint_99.pth"
        # name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device, pre_model_path="/data2/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_t5_xxl/20230314041/checkpoint_19.pth"
        # name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device, pre_model_path="/data2/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_t5_xxl/20230315070/checkpoint_99.pth"
        # name="blip2_llama", model_type="pretrain_llama7b", is_eval=True, device=device, pre_model_path="/data2/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_llama/20230320100/checkpoint_9.pth"
        # name="blip2_llama", model_type="pretrain_llama7b", is_eval=True, device=device, pre_model_path="/data2/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_llama/20230323061/checkpoint_9.pth"
        # name="blip2_llama", model_type="pretrain_llama7b", is_eval=True, device=device, pre_model_path="/data2/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_llama/20230324022/checkpoint_5.pth"
        # name="blip2_llama", model_type="pretrain_llama7b", is_eval=True, device=device, pre_model_path="/data2/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_llama/20230324080/checkpoint_9.pth"
        # name="blip2_llama", model_type="pretrain_llama7b", is_eval=True, device=device, pre_model_path="/data2/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_llama/20230327032/checkpoint_9.pth"
        # name="blip2_llama", model_type="pretrain_llama7b", is_eval=True, device=device, pre_model_path="/data2/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_llama/20230327061/checkpoint_8.pth"
        # name="blip2_llama", model_type="pretrain_llama7b", is_eval=True, device=device, pre_model_path="/data2/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_llama/20230327092/checkpoint_9.pth"
        # name="blip2_llama", model_type="pretrain_llama7b", is_eval=True, device=device, pre_model_path="/data2/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_llama/20230329084/checkpoint_19.pth"
        # name="blip2_llama", model_type="pretrain_llama7b", is_eval=True, device=device, pre_model_path="/data2/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_llama/20230330135/checkpoint_19.pth"
        # name="blip2_llama", model_type="pretrain_llama7b", is_eval=True, device=device, pre_model_path="/data2/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_llama/20230331073/checkpoint_9.pth"

        # name="blip2_llama", model_type="pretrain_llama7b-linba", is_eval=True, device=device, pre_model_path="/data/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_llama/20230412203/checkpoint_19.pth"
        # name="blip2_llama", model_type="pretrain_llama7b-linba", is_eval=True, device=device, pre_model_path="/data/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_llama/20230412203/checkpoint_19.pth"

        # name="blip2_llama", model_type="pretrain_vicuna13b", is_eval=True, device=device, pre_model_path="/data/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_vicuna13b/20230414151/checkpoint_29.pth"
        # name="blip2_llama", model_type="pretrain_zidong13b-audio", is_eval=True, device=device, pre_model_path="/data/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_music/20230419023/checkpoint_9.pth"
        # name="blip2_llama", model_type="pretrain_zidong13b-audio", is_eval=True, device=device, pre_model_path="/data/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_music/20230418024/checkpoint_49.pth"
        # name="blip2_llama", model_type="pretrain_zidong13b-audio", is_eval=True, device=device, pre_model_path="/data/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_music/20230419023/checkpoint_44.pth"
        # name="blip2_llama", model_type="pretrain_zidong13b-audio", is_eval=True, device=device, pre_model_path="/data/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_music/20230419181/checkpoint_9.pth"
        # name="blip2_llama", model_type="pretrain_zidong13b-audio", is_eval=True, device=device, pre_model_path="/data/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_music/20230420184/checkpoint_84.pth"
        # name="blip2_llama", model_type="pretrain_zidong13b-audio", is_eval=True, device=device, pre_model_path="/data/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_music/20230420184/checkpoint_129.pth"
        # name="blip2_llama", model_type="pretrain_zidong13b-audio", is_eval=True, device=device, pre_model_path="/data/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_music/20230423225/checkpoint_84.pth"
        # name="blip2_llama", model_type="pretrain_zidong13b-audio", is_eval=True, device=device, pre_model_path="/data/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_music/20230425103/checkpoint_14.pth"

        # name="blip2_chatglm", model_type="pretrain_chatglm6b-music", is_eval=True, device=device, pre_model_path="/data2/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_music_glm6b/20230514120/checkpoint_29.pth"
        name="blip2_chatglm", model_type="pretrain_chatglm6b-music", is_eval=True, device=device, pre_model_path="/data2/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_music_glm6b/20230516083/checkpoint_29.pth"

        # name="blip2_chatglm", model_type="pretrain_chatglm6b", is_eval=True, device=device, pre_model_path="/data2/shij/lavis_glm/lavis/checkpoints/checkpoint_4.pth"
)

if modal_type =="image":
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_chatglm", model_type="pretrain_chatglm6b",\
                                                          is_eval=True, device=device, pre_model_path="/data2/shij/lavis_glm/lavis/checkpoints/checkpoint_4.pth")
if modal_type =="signal":
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_chatglm", model_type="pretrain_chatglm6b-signal",\
                                                          is_eval=True, device=device, pre_model_path="/data2/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_signal_glm6b/20230518075/checkpoint_49.pth")
                                                        #   is_eval=True, device=device, pre_model_path="/data2/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_signal_glm6b/20230518075/checkpoint_49.pth")
                                                        #   is_eval=True, device=device, pre_model_path="/data2/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_signal_glm6b/20230519092/checkpoint_149.pth")
if modal_type =="cloud":
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_chatglm", model_type="pretrain_chatglm6b-cloud", \
            pre_model_path="/data2/shij/data/cloud_cap/model/stage2/cap_and_structure_v3_chatglm.pth",is_eval=True, device=device
    )

if modal_type =="all":
    all_model = "/data2/shij/codes/BLIP2/LAVIS/lavis/saved_checkpoints/merge_image_music_sig_cloud.pth"
    # all_model = "/data2/shij/codes/BLIP2/LAVIS/lavis/saved_checkpoints/merge_image_ft_music_sig_cloud.pth"
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_chatglm_all", model_type="pretrain_chatglm6b_dynamic",\
                                                        #   is_eval=True, device=device, pre_model_path="/data2/shij/codes/BLIP2/LAVIS/lavis/saved_checkpoints/merge_image_music_sig.pth")
                                                          is_eval=True, device=device, pre_model_path=all_model)
    print("load all-modal done.")
    if "cloud" in all_model:
        cloud_processor, _ = load_preprocess_only(name="blip2_chatglm", model_type="pretrain_chatglm6b-cloud", \
            pre_model_path="/data2/shij/data/cloud_cap/model/stage2/cap_and_structure_v3_chatglm.pth",is_eval=True, device=device 
        )
        print("load cloud-modal processor done.")

del _
# prepare the image
# image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

# res = model.generate(
    # {"image": image, "prompt": "Question: which city is this? Answer:"}
# )
# 'singapore'

# print("Default caption w/o prompt: \n", model.generate({"image": image, "prompt": "A picture of"}))
# """

# while True:
#     prompt = input("prompt: ")
#     res = model.generate({"image": image, "prompt": prompt})
#     print(res)

@app.route('/example/')
def example():
    return {'hello': 'world'}

@app.route("/", methods=['GET', 'POST'])
def notes_list():
    """
    List or create notes.
    """
    if request.method == 'POST':
        data = request.data
        text = data.get('text', '')

        given_image = data.get('image', '')
        given_image = base64_pil(given_image).convert("RGB")
        given_image = vis_processors["eval"](given_image).unsqueeze(0).to(device)

        input_text_en = requests.post(translation_api_cn2en_genel, data={'text': text}).json()['Data']['Translated']

        input_text_en = input_text_en.replace("Problem:", "Question:").strip()
        prompt = input_text_en
        # response = model.generate({"image": given_image, "prompt": prompt}, max_length=300, repetition_penalty=2.0, temp_prompt="skip", early_stopping=False)
        response = model.generate({"image": given_image, "prompt": prompt}, max_length=320, repetition_penalty=2.5, temp_prompt="skip", early_stopping=True)
        response = response[0]

        print("Response: ", response)
        if "</s>" in response:
            response = response.replace("</s>","")
            # response = response.split("</s>")[0]
        if response.endswith("</"):
            response = response[:-2]
        print("Response: ", response)
        # response = model.generate({"image": given_image, "prompt": prompt}, max_length=300, temp_prompt="skip")
        response_trans = requests.post(translation_api_en2cn_genel, data={'text': response}).json()['Data']['Translated']

        result_dict = {
            "input_text": text,
            "prompt": text,
            "response": response,
            "response_trans": response_trans, 
            "input_text_en": input_text_en,
        }
        return json.dumps(result_dict, ensure_ascii=False, indent=4)

@app.route("/image", methods=['GET', 'POST'])
def image_infer():
    """
    List or create notes.
    """
    if request.method == 'POST':
        data = request.data
        text = data.get('text', '')
        if isinstance(text, list):
            text = text[-1]
            assert isinstance(text, str), "text should be a string"

        given_image = data.get('image', '')
        given_image = base64_pil(given_image).convert("RGB")
        given_image = vis_processors["eval"](given_image).unsqueeze(0).to(device)

        prompt = text
        # response = model.generate({"image": given_image, "prompt": prompt}, max_length=300, repetition_penalty=2.0, temp_prompt="skip", early_stopping=False)
        response = model.generate_image({"image": given_image, "prompt": prompt}, max_length=320, repetition_penalty=2.5, temp_prompt="skip", early_stopping=True)
        response = response[0]

        print("Response: ", response)
        if "</s>" in response:
            response = response.replace("</s>","")
            # response = response.split("</s>")[0]
        if response.endswith("</"):
            response = response[:-2]
        print("Response: ", response)
        # response = model.generate({"image": given_image, "prompt": prompt}, max_length=300, temp_prompt="skip")
        # response_trans = requests.post(translation_api_en2cn_genel, data={'text': response}).json()['Data']['Translated']

        result_dict = {
            "input_text": text,
            "prompt": text,
            "response": response,
            "response_trans":response, 
            "input_text_en": 'Null',
        }
        return json.dumps(result_dict, ensure_ascii=False, indent=4)
        return json.dumps(result_dict, ensure_ascii=False, indent=4)

@app.route("/text", methods=['GET', 'POST'])
def text_infer():
    """
    List or create notes.
    """
    if request.method == 'POST':
        data = request.data
        text = data.get('text', '')

        prompt = text
        # response = model.generate({"image": given_image, "prompt": prompt}, max_length=300, repetition_penalty=2.0, temp_prompt="skip", early_stopping=False)
        response = model.generate_pure_text({"prompt": prompt}, max_length=320, repetition_penalty=2.5, temp_prompt="skip", early_stopping=True)
        response = response[0]

        print("Response: ", response)

        result_dict = {
            "input_text": text,
            "prompt": text,
            "response": response,
            "response_trans":response, 
            "input_text_en": 'Null',
            "output_with_embeds": response
        }
        return json.dumps(result_dict, ensure_ascii=False, indent=4)

@app.route("/audio", methods=['GET', 'POST'])
def audio_infer(use_prefix_prompt=False):
    """
    List or create notes.
    """
    infer_prompt = "指令： 给定上面的一段音乐，以下是一个描述任务的指令，请根据音乐相关的信息生成一个完成该指令的适当回复。\n\n 问题：{} \n\n 回答："
    if request.method == 'POST':
        data = request.data
        text = data.get('text', '')

        given_audio = data.get('audio', '')

        formatted_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + str(random.randint(0, 1000000))
        audio_file_name = "/data/shij/codes/BLIP2/temp_stuff/test_" + formatted_time + ".wav"
        write_wave(audio_file_name, bytes2wav(given_audio))
        audio_file_npy = audio_file_reader(audio_file_name)

        # mel branch 
        # given_audio = vis_processors["eval"](audio_file_npy).unsqueeze(0).to(device)

        given_audio= audio_file_npy

        prompt = text if not use_prefix_prompt else infer_prompt.format(text)
        # response = model.generate({"image": given_image, "prompt": prompt}, max_length=300, repetition_penalty=2.0, temp_prompt="skip", early_stopping=False)
        response = model.generate_audio({"audio": given_audio, "prompt": prompt, "modality": "audio"}, max_length=320, repetition_penalty=2.5, temp_prompt="skip", early_stopping=True)
        response = response[0]

        print("Response: ", response)
        if "</s>" in response:
            response = response.replace("</s>","")
            # response = response.split("</s>")[0]
        if response.endswith("</"):
            response = response[:-2]
        # print("Response: ", response)
        # response = model.generate({"image": given_image, "prompt": prompt}, max_length=300, temp_prompt="skip")
        # response_trans = requests.post(translation_api_en2cn_genel, data={'text': response}).json()['Data']['Translated']
        response_trans = response.replace(" ", "")

        result_dict = {
            "input_text": text,
            "prompt": text,
            "response": response,
            "response_trans": response_trans, 
            "output_with_embeds": response_trans
        }
        return json.dumps(result_dict, ensure_ascii=False, indent=4)

@app.route("/music", methods=['GET', 'POST'])
def music_infer(use_prefix_prompt=False):
    """
    List or create notes.
    """
    infer_prompt = "指令： 给定上面的一段音乐，以下是一个描述任务的指令，请根据音乐相关的信息生成一个完成该指令的适当回复。\n\n 问题：{} \n\n 回答："
    if request.method == 'POST':
        data = request.data
        text = data.get('text', '')

        given_audio = data.get('audio', '')

        formatted_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + str(random.randint(0, 1000000))
        audio_file_name = "/data2/shij/codes/BLIP2/temp_stuff/test_" + formatted_time + ".wav"
        write_wave(audio_file_name, bytes2wav(given_audio))
        audio_file_npy = audio_file_reader(audio_file_name)

        # mel branch 
        # given_audio = vis_processors["eval"](audio_file_npy).unsqueeze(0).to(device)

        given_audio= audio_file_npy

        prompt = text if not use_prefix_prompt else infer_prompt.format(text)
        # response = model.generate({"image": given_image, "prompt": prompt}, max_length=300, repetition_penalty=2.0, temp_prompt="skip", early_stopping=False)
        print("Final prompt:", prompt)
        response = model.generate_audio_or_music({"audio": given_audio, "prompt": prompt, "modality": "music"}, max_length=320, repetition_penalty=2.5, early_stopping=True)
        response = response[0]

        print("Response: ", response)
        if "</s>" in response:
            response = response.replace("</s>","")
            # response = response.split("</s>")[0]
        if response.endswith("</"):
            response = response[:-2]
        print("Response: ", response)
        # response = model.generate({"image": given_image, "prompt": prompt}, max_length=300, temp_prompt="skip")
        # response_trans = requests.post(translation_api_en2cn_genel, data={'text': response}).json()['Data']['Translated']
        response_trans = response.replace(" ", "")

        result_dict = {
            "input_text": text,
            "prompt": text,
            "response": response,
            "response_trans": response_trans, 
            "output_with_embeds": response_trans
        }
        return json.dumps(result_dict, ensure_ascii=False, indent=4)

@app.route("/signal", methods=['GET', 'POST'])
def signal_infer(use_prefix_prompt=False):
    """
    List or create notes.
    """
    if request.method == 'POST':
        data = request.data
        text = data.get('text', '')

        given_audio = data.get('audio', '')

        formatted_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + str(random.randint(0, 1000000))
        audio_file_name = "/data2/shij/codes/BLIP2/temp_stuff/test_" + formatted_time + ".wav"
        write_wave(audio_file_name, bytes2wav(given_audio))
        audio_file_npy = audio_file_reader(audio_file_name)

        # mel branch 
        # given_audio = vis_processors["eval"](audio_file_npy).unsqueeze(0).to(device)

        given_audio= audio_file_npy

        prompt = text 
        # response = model.generate({"image": given_image, "prompt": prompt}, max_length=300, repetition_penalty=2.0, temp_prompt="skip", early_stopping=False)
        print("Final prompt:", prompt)
        response = model.generate_signal({"signal": given_audio, "prompt": prompt, "modality": "signal"}, max_length=320, repetition_penalty=2.5, early_stopping=True)
        response = response[0]

        print("Response: ", response)
        if "</s>" in response:
            response = response.replace("</s>","")
            # response = response.split("</s>")[0]
        if response.endswith("</"):
            response = response[:-2]
        print("Response: ", response)
        # response = model.generate({"image": given_image, "prompt": prompt}, max_length=300, temp_prompt="skip")
        # response_trans = requests.post(translation_api_en2cn_genel, data={'text': response}).json()['Data']['Translated']
        response_trans = response.replace(" ", "")

        result_dict = {
            "input_text": text,
            "prompt": text,
            "response": response,
            "response_trans": response_trans, 
            "output_with_embeds": response_trans
        }
        return json.dumps(result_dict, ensure_ascii=False, indent=4)

@app.route("/cloud", methods=['GET', 'POST'])
def cloud_infer(use_prefix_prompt=False):
    """
    List or create notes.
    """
    if request.method == 'POST':
        data = request.data
        text = data.get('text', '')

        given_cloud = data.get('cloud', '')
        if given_cloud.endswith(".pth"):
            cloud_path = "/data2/shij/data/cloud_cap/example_data/" + given_cloud
            assert os.path.exists(cloud_path), "Cloud file not found!"
        else:
            formatted_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + str(random.randint(0, 1000000))
            cloud_file_name = "/data2/shij/codes/BLIP2/temp_stuff/test_" + formatted_time + ".pth"
            clund_data = base64.b64decode(given_cloud)
            with open(cloud_file_name, 'wb') as f:
                f.write(clund_data)
            cloud_path = cloud_file_name

        prompt = text 
        # response = model.generate({"image": given_image, "prompt": prompt}, max_length=300, repetition_penalty=2.0, temp_prompt="skip", early_stopping=False)
        print("Final prompt:", prompt)

        # cloud = torch.load(cloud_path)
        cloud = load_point_cloud(cloud_path)
        cloud = cloud_processor["eval"](cloud)
        for k in cloud.keys():
            if(isinstance(cloud[k], torch.Tensor)):
                cloud[k] = cloud[k].to(device)
                cloud[k] = cloud[k].unsqueeze(0)

        cloud_copy = cloud.copy()
        cloud = cloud_copy.copy()
        # result = model.generate_with_hidden_prompt({"cloud":cloud, "text_input": "请描述一下这个三维场景。"}, max_length=100, num_beams=1)
        response = model.generate_cloud({"cloud":cloud, "text_input": prompt}, max_length=100, num_beams=1)
        print("cloud response:", response)
        if isinstance(response, list):
            response = response[0]
        response_trans = response.replace(" ", "")
        result_dict = {
            "input_text": text,
            "prompt": text,
            "response": response,
            "response_trans": response_trans, 
            "output_with_embeds": response_trans
        }
        return json.dumps(result_dict, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # app.config[‘JSON_AS_ASCII’] = False 这个跟编码有关系，好像可以直接弄
    # app.run(debug=False, host='0.0.0.0', port=7124)
    if modal_type == "image":
        app.run(debug=False, host='0.0.0.0', port=7125)
    if modal_type == "music":
        app.run(debug=False, host='0.0.0.0', port=7126)
    if modal_type == "signal":
        app.run(debug=False, host='0.0.0.0', port=7127)
    if modal_type == "all":
        # app.run(debug=False, host='0.0.0.0', port=7128)
        app.run(debug=False, host='0.0.0.0', port=7129)

"""
model.generate({
    "image": image,
    "prompt": "Question: which city is this? Answer: singapore. Question: why?"})
# 'it has a statue of a merlion'    


# prepare context prompt
context = [
    ("which city is this?", "singapore"),
    ("why?", "it has a statue of a merlion"),
]
question = "where is the name merlion coming from?"
template = "Question: {} Answer: {}."
prompt = " ".join([template.format(context[i][0], context[i][1]) for i in range(len(context))]) + " Question: " + question + " Answer:"
print(prompt)
# generate model's response
model.generate({"image": image,"prompt": prompt})
# 'merlion is a portmanteau of mermaid and lion'
"""
