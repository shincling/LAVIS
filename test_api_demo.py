import torch
from PIL import Image
import os
import json
from flask import request
from flask_api import FlaskAPI
import requests
import io

import torch
from lavis.models import load_model_and_preprocess

app = FlaskAPI(__name__)

translation_api_cn2en_genel = 'http://172.18.30.134:6217/aliyuntranslate_gen/'
translation_api_en2cn_genel = 'http://172.18.30.134:6220/aliyuntranslate_en_gen/'
translation_api_cn2en_pro = 'http://172.18.30.134:5117/aliyuntranslate/'
translation_api_en2cn_pro = 'http://172.18.30.134:5118/aliyuntranslate_en/'
# print(requests.post(translation_api_en2an_genel, data={'text': tgt_text}).json()['Data']['Translated'])
# print(requests.post(translation_api_en2an_genel, data={'text': pred_text}).json()['Data']['Translated'])

# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
if 1:
    # load sample image
    raw_image = Image.open(
        "/data2/shij/data/Med2/raw_data/1.3.12.2.1107.5.6.1.1586.30000017010600213448400000001.jpg"
    ).convert("RGB")

    default_image = "/data2/shij/data/Med2/raw_data/1.3.12.2.1107.5.6.1.1586.30000017010600213448400000001.jpg"


    # loads BLIP-2 pre-trained model
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device
    )
    # prepare the image
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

    res = model.generate(
        {"image": image, "prompt": "Question: which city is this? Answer:"}
    )
    # 'singapore'

    print("Default caption w/o prompt: \n", model.generate({"image": image, "prompt": ""}))

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
        given_image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        input_text_en = requests.post(translation_api_cn2en_genel, data={'text': text}).json()['Data']['Translated']
        prompt = input_text_en
        response = model.generate({"image": image, "prompt": prompt})
        response_trans = requests.post(translation_api_en2cn_genel, data={'text': response}).json()['Data']['Translated']

        result_dict = {
            "input_text": text,
            "prompt": text,
            "response": response,
            "response_trans": response_trans, 
            "input_text_en": input_text_en,
        }
        return json.dumps(result_dict, ensure_ascii=False, indent=4)


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

from pydantic import BaseModel, Field
from opyrator.components.types import FileContent

class TextGenerationInput(BaseModel):
    # image_file: FileContent = Field(..., mime_type="image/png")
    image_file_jpg: FileContent = Field(..., mime_type="image/jpeg") 
    text: str = Field(
        ...,
        title="Text Input",
        description="The input text to use as basis to generate text.",
        max_length=1000,
    )
    temperature: float = Field(
        1.0,
        gt=0.0,
        multiple_of=0.001,
        description="The value used to module the next token probabilities.",
    )
    max_length: int = Field(
        30,
        ge=5,
        le=100,
        description="The maximum length of the sequence to be generated.",
    )
    repetition_penalty: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="The parameter for repetition penalty. 1.0 means no penalty.",
    )
    top_k: int = Field(
        50,
        ge=0,
        description="The number of highest probability vocabulary tokens to keep for top-k-filtering.",
    )
    do_sample: bool = Field(
        False,
        description="Whether or not to use sampling ; use greedy decoding otherwise.",
    )


class TextGenerationOutput(BaseModel):
    res_text: str = Field(...)
    all_results: str = Field(...)


def generate_text(input: TextGenerationInput) -> TextGenerationOutput:
    """Generate text based on a given prompt."""

    print("input", input)

    # if input.image_file:
        # input_image = Image.open(io.BytesIO(input.image_file.as_bytes())).convert("RGB")
    # else:
    input_image = Image.open(io.BytesIO(input.image_file_jpg.as_bytes())).convert("RGB")
    given_image = vis_processors["eval"](input_image).unsqueeze(0).to(device)
    
    input_text_en = requests.post(translation_api_cn2en_genel, data={'text': input.text}).json()['Data']['Translated']
    prompt = input_text_en
    response = model.generate({"image": given_image, "prompt": prompt})[0]
    response_trans = requests.post(translation_api_en2cn_genel, data={'text': response}).json()['Data']['Translated']

    result_dict = {
        "input_text":input.text,
        "prompt": prompt,
        "response": response,
        "response_trans": response_trans, 
        "input_text_en": input_text_en,
    }

    # return TextGenerationOutput(generated_text=res[0]["generated_text"])
    return TextGenerationOutput(all_results=str(result_dict), res_text=response_trans)


if __name__ == "__main__":
    # app.config[‘JSON_AS_ASCII’] = False 这个跟编码有关系，好像可以直接弄
    # app.run(debug=False, host='0.0.0.0', port=7124)
    pass