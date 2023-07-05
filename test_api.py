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
from lavis.models import load_model_and_preprocess
import soundfile as sf
import time
import random
from transformers import AutoProcessor
import sys
import torchaudio

os.environ['CURL_CA_BUNDLE'] = ''

app = FlaskAPI(__name__)

modal_type = sys.argv[1]
assert modal_type in ['text', 'music', 'image', 'signal', 'speech', "image_med", "image_air", 'all'], "modal_type must be one of ['text', 'music', 'image', 'signal']"

translation_api_cn2en_genel = 'http://172.18.30.134:6217/aliyuntranslate_gen/'
translation_api_en2cn_genel = 'http://172.18.30.134:6220/aliyuntranslate_en_gen/'
translation_api_cn2en_pro = 'http://172.18.30.134:5117/aliyuntranslate/'
translation_api_en2cn_pro = 'http://172.18.30.134:5118/aliyuntranslate_en/'
# print(requests.post(translation_api_en2an_genel, data={'text': tgt_text}).json()['Data']['Translated'])
# print(requests.post(translation_api_en2an_genel, data={'text': pred_text}).json()['Data']['Translated'])

audio_wav2vec_processor = AutoProcessor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
# audio_wav2vec_processor = AutoProcessor.from_pretrained("/data/shij/data/audio_cap/wav2vec2-large-xlsr-53-english")
audio_wav2vec_processor = AutoProcessor.from_pretrained("/data/shij/data/audio_cap/wav2vec2-large-xlsr-53-english")
# audio_wav2vec_processor = AutoProcessor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english", cache_dir="/data/shij/.cache/huggingface/hub/models--jonatasgrosman--wav2vec2-large-xlsr-53-english/")
# audio_wav2vec_processor = AutoProcessor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
# import pdb; pdb.set_trace()

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
    # audio_wav2vec_processor = AutoProcessor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
    audio_npy = audio_wav2vec_processor(audio_npy, sampling_rate=16000, return_tensors="pt",)
    
    return audio_npy

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
        )

if modal_type =="image":
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_chatglm", model_type="pretrain_chatglm6b",\
                                                          is_eval=True, device=device, pre_model_path="/data2/shij/lavis_glm/lavis/checkpoints/checkpoint_4.pth")
    # model, vis_processors, _ = load_model_and_preprocess(
    # # 通用Image與所裏LLM
    # # name="blip2_llama", model_type="pretrain_zidong13b", is_eval=True, device=device, pre_model_path="/data/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_wukong/20230427144/checkpoint_8.pth"
    # # name="blip2_llama", model_type="pretrain_zidong13b", is_eval=True, device=device, pre_model_path="/data/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_wukong/20230505181/checkpoint_29.pth"

    # # Baichuan-7B image
    # # name="blip2_llama", model_type="pretrain_baichuan7b", is_eval=True, device=device, pre_model_path="/data/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_image_baichuan7b/20230629100/checkpoint_0.pth"
    # # name="blip2_llama", model_type="pretrain_baichuan7b", is_eval=True, device=device, pre_model_path="/data/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_image_baichuan7b/20230629181/checkpoint_0.pth"
    # # name="blip2_llama", model_type="pretrain_baichuan7b", is_eval=True, device=device, pre_model_path="/data/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_image_baichuan7b/20230630222/checkpoint_0.pth"
    # name="blip2_llama", model_type="pretrain_baichuan7b", is_eval=True, device=device, pre_model_path="/data/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_image_baichuan7b/20230703164/checkpoint_0.pth"

if modal_type =="image_med":
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_chatglm", model_type="pretrain_chatglm6b_image_dialog_ft",\
                                                        #   is_eval=True, device=device, pre_model_path="/data2/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_med_image_cap_ft_glm6b/20230613090/checkpoint_94.pth")
                                                        #   is_eval=True, device=device, pre_model_path="/data2/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_med_image_cap_ft_glm6b/20230613162/checkpoint_19.pth")
                                                        #   is_eval=True, device=device, pre_model_path="/data2/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_med_image_cap_ft_glm6b/20230614025/checkpoint_19.pth")
                                                          is_eval=True, device=device, pre_model_path="/data2/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_med_image_cap_ft_glm6b/20230614064/checkpoint_59.pth")
if modal_type =="image_air":
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_chatglm", model_type="pretrain_chatglm6b_image_dialog_ft",\
                                                        #   is_eval=True, device=device, pre_model_path="/data2/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_med_image_cap_ft_glm6b/20230614064/checkpoint_59.pth")
                                                          is_eval=True, device=device, pre_model_path="/data2/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_aircraft_image_cap_ft_glm6b/20230615072/checkpoint_19.pth")
if modal_type =="signal":
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_chatglm", model_type="pretrain_chatglm6b-signal",\
                                                        #   is_eval=True, device=device, pre_model_path="/data2/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_signal_glm6b/20230518075/checkpoint_49.pth")
                                                        #   is_eval=True, device=device, pre_model_path="/data2/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_signal_glm6b/20230518075/checkpoint_49.pth")
                                                          is_eval=True, device=device, pre_model_path="/data2/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_signal_glm6b/20230519092/checkpoint_149.pth")
if modal_type =="speech":
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_chatglm", model_type="pretrain_chatglm6b-speech",\
                                                          is_eval=True, device=device, pre_model_path="/data2/shij/codes/BLIP2/LAVIS/lavis/saved_checkpoints/speech_js_ft_0607.pth")
                                                        #   is_eval=True, device=device, pre_model_path="/data2/shij/codes/BLIP2/LAVIS/lavis/saved_checkpoints/speech_1000h_v0.pth")
if modal_type =="all":
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_chatglm_all", model_type="pretrain_chatglm6b_dynamic",\
                                                          is_eval=True, device=device, pre_model_path="/data2/shij/codes/BLIP2/LAVIS/lavis/saved_checkpoints/merge_image_music_sig.pth")
# del temp
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
def notes_list(is_trans=False):
    """
    List or create notes.
    """
    prefix_str = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nUSER:{} ASSISTANT:"
    # prefix_str = ""
    if request.method == 'POST':
        data = request.data
        text = data.get('text', '')

        given_image = data.get('image', '')
        given_image = base64_pil(given_image).convert("RGB")
        given_image = vis_processors["eval"](given_image).unsqueeze(0).to(device)


        input_text_en = requests.post(translation_api_cn2en_genel, data={'text': text}).json()['Data']['Translated']
        input_text_en = input_text_en.replace("Problem:", "Question:").strip()
        prompt = input_text_en if is_trans else text
        if prefix_str:
            prompt = prefix_str.format(prompt)
        # response = model.generate({"image": given_image, "prompt": prompt}, max_length=300, repetition_penalty=2.0, temp_prompt="skip", early_stopping=False)
        model.train()
        # response = model.generate({"image": given_image, "prompt": prompt}, max_length=320, repetition_penalty=2.5, temp_prompt="skip", early_stopping=True)
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
        if prompt == input_text_en and is_trans:
            response_trans = requests.post(translation_api_en2cn_genel, data={'text': response}).json()['Data']['Translated']
        else:
            response_trans = response

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
        print("Revived text in model server-node: ", text)

        given_image = data.get('image', '')
        given_image = base64_pil(given_image).convert("RGB")
        given_image = vis_processors["eval"](given_image).unsqueeze(0).to(device)

        prompt = text
        response = model.generate({"image": given_image, "prompt": prompt}, max_length=2048, repetition_penalty=2.0, temp_prompt="skip", early_stopping=False)
        # response = model.generate({"image": given_image, "prompt": prompt}, max_length=320, repetition_penalty=2.5, temp_prompt="skip", early_stopping=True)
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
        }
        return json.dumps(result_dict, ensure_ascii=False, indent=4)

@app.route("/audio", methods=['GET', 'POST'])
def audio_infer(use_prefix_prompt=True):
    """
    List or create notes.
    """
    infer_prompt = "指令： 给定上面的一段音乐，以下是一个描述任务的指令，请根据音乐相关的信息生成一个完成该指令的适当回复。\n\n 问题：{} \n\n 回答："
    if request.method == 'POST':
        data = request.data
        text = data.get('text', '')

        given_audio = data.get('audio', '')

        if given_audio.endswith(".wav") and given_audio in os.listdir("/data/shij/codes/BLIP2/LAVIS/music_samples/"):
            print("Use the saved audio file: ", given_audio)
            audio_file_npy = audio_file_reader("/data/shij/codes/BLIP2/LAVIS/music_samples/" + given_audio)
        else:
            formatted_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + str(random.randint(0, 1000000))
            audio_file_name = "/data/shij/codes/BLIP2/temp_stuff/test_" + formatted_time + ".wav"
            write_wave(audio_file_name, bytes2wav(given_audio))
            audio_file_npy = audio_file_reader(audio_file_name)

        # mel branch 
        # given_audio = vis_processors["eval"](audio_file_npy).unsqueeze(0).to(device)

        given_audio= audio_file_npy

        prompt = text if not use_prefix_prompt else infer_prompt.format(text)
        # response = model.generate({"image": given_image, "prompt": prompt}, max_length=300, repetition_penalty=2.0, temp_prompt="skip", early_stopping=False)
        response = model.generate_audio({"audio": given_audio, "prompt": prompt}, max_length=320, repetition_penalty=2.5, temp_prompt="skip", early_stopping=True)
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

@app.route("/music", methods=['GET', 'POST'])
@app.route("/signal", methods=['GET', 'POST'])
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
        response = model.generate_audio_or_music({"audio": given_audio, "prompt": prompt}, max_length=320, repetition_penalty=2.5, early_stopping=True)
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

def get_ASRFeature(speech):
    waveform, samplereate = torchaudio.load(speech)
    print("waveform_size: ", waveform.size())
    waveform_bytes = waveform.numpy().tobytes()
    audio_base64 = base64.b64encode(waveform_bytes).decode('utf-8')

    url = 'http://172.18.30.121:8888/genfeats'
    asr_feature = requests.post(url, json={'waveform':audio_base64}).json()

    asr_feature = base64.b64decode(asr_feature)
    asr_feature = np.frombuffer(asr_feature, dtype=np.float32)
    asr_feature = torch.tensor(asr_feature, dtype=torch.float32).view(-1, 512)
    asr_feature = asr_feature.unsqueeze(0)
    print(asr_feature.size())

    return asr_feature

@app.route("/speech", methods=['GET', 'POST'])
def speech_infer(use_prefix_prompt=True):
    """
    List or create notes.
    """
    # infer_prompt = "指令： 给定上面的一段音乐，以下是一个描述任务的指令，请根据音乐相关的信息生成一个完成该指令的适当回复。\n\n 问题：{} \n\n 回答："
    # infer_prompt = "请忠实地识别该语音[gMASK]"
    infer_prompt = "以上是一段给定的语音，请识别给定的语音，直接生成合理的回复。"
    if request.method == 'POST':
        data = request.data
        text = data.get('text', '')

        given_audio = data.get('speech', '')

        formatted_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + str(random.randint(0, 1000000))
        audio_file_name = "/data2/shij/codes/BLIP2/temp_stuff/test_" + formatted_time + ".wav"
        write_wave(audio_file_name, bytes2wav(given_audio))
        audio_cif_feats = get_ASRFeature(audio_file_name)

        given_audio = audio_cif_feats

        prompt = text if not use_prefix_prompt else infer_prompt.format(text)
        # response = model.generate({"image": given_image, "prompt": prompt}, max_length=300, repetition_penalty=2.0, temp_prompt="skip", early_stopping=False)
        print("Final prompt:", prompt)
        response = model.generate_speech({"speech_input": given_audio, "prompt": prompt})
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

if __name__ == "__main__":
    # app.config[‘JSON_AS_ASCII’] = False 这个跟编码有关系，好像可以直接弄
    # app.run(debug=False, host='0.0.0.0', port=7124)
    if modal_type == "image_med":
        app.run(debug=False, host='0.0.0.0', port=7122)
    if modal_type == "image_air":
        app.run(debug=False, host='0.0.0.0', port=7121)
    if modal_type == "image":
        app.run(debug=False, host='0.0.0.0', port=7125)
    if modal_type == "music":
        app.run(debug=False, host='0.0.0.0', port=7126)
    if modal_type == "speech":
        app.run(debug=False, host='0.0.0.0', port=7126)
    if modal_type == "signal":
        app.run(debug=False, host='0.0.0.0', port=7127)
    if modal_type == "all":
        app.run(debug=False, host='0.0.0.0', port=7128)

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
