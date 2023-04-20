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

app = FlaskAPI(__name__)

translation_api_cn2en_genel = 'http://172.18.30.134:6217/aliyuntranslate_gen/'
translation_api_en2cn_genel = 'http://172.18.30.134:6220/aliyuntranslate_en_gen/'
translation_api_cn2en_pro = 'http://172.18.30.134:5117/aliyuntranslate/'
translation_api_en2cn_pro = 'http://172.18.30.134:5118/aliyuntranslate_en/'
# print(requests.post(translation_api_en2an_genel, data={'text': tgt_text}).json()['Data']['Translated'])
# print(requests.post(translation_api_en2an_genel, data={'text': pred_text}).json()['Data']['Translated'])


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
    # name="blip2_llama", model_type="pretrain_zidong13b-audio", is_eval=True, device=device, pre_model_path="/data/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_music/20230418124/checkpoint_49.pth"
    # name="blip2_llama", model_type="pretrain_zidong13b-audio", is_eval=True, device=device, pre_model_path="/data/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_music/20230419023/checkpoint_44.pth"
    name="blip2_llama", model_type="pretrain_zidong13b-audio", is_eval=True, device=device, pre_model_path="/data/shij/codes/BLIP2/LAVIS/lavis/output/BLIP2/Pretrain_stage2_music/20230419181/checkpoint_94.pth"

)
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

@app.route("/audio", methods=['GET', 'POST'])
def audio_infer():
    """
    List or create notes.
    """
    if request.method == 'POST':
        data = request.data
        text = data.get('text', '')

        given_audio = data.get('audio', '')

        formatted_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + str(random.randint(0, 1000000))
        audio_file_name = "/data/shij/codes/BLIP2/temp_stuff/test_" + formatted_time + ".wav"
        write_wave(audio_file_name, bytes2wav(given_audio))
        audio_file_npy = audio_file_reader(audio_file_name)

        given_audio = vis_processors["eval"](audio_file_npy).unsqueeze(0).to(device)

        prompt = text
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

if __name__ == "__main__":
    # app.config[‘JSON_AS_ASCII’] = False 这个跟编码有关系，好像可以直接弄
    # app.run(debug=False, host='0.0.0.0', port=7124)
    app.run(debug=False, host='0.0.0.0', port=7125)

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
