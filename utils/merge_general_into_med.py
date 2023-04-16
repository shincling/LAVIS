import json
import argparse
import random

# fuction to init and read arguments
def init_args():
    parser = argparse.ArgumentParser(description='Merge general data into med data')
    parser.add_argument('--med_cap_json_path', type=str, default='/export/home/.cache/lavis/med2/annotations/med_train_with_id.json', help='path to med json file')
    parser.add_argument('--med_qa_json_path', type=str, default='/export/home/.cache/lavis/med2/annotations/med2_qa1900_with_id.json', help='path to med json file')
    parser.add_argument('--general_json_path', type=str, default='/data2/shij/codes/BLIP2/LAVIS/utils/alpaca_data.json', help='path to general json file')
    parser.add_argument('--output_json_path', type=str, default='/data2/shij/codes/BLIP2/LAVIS/utils/merged_med_3xqa1900_general.json', help='path to output json file')
    parser.add_argument('--merge_qa', type=int, default=1, help='whether to merge qa data')
    args = parser.parse_args()
    return args


args = init_args()

general_json = json.load(open(args.general_json_path, 'r'))
general_json_bak = general_json.copy()
random.shuffle(general_json_bak)
med_cap_json = json.load(open(args.med_cap_json_path, 'r'))
med_qa_json = json.load(open(args.med_qa_json_path, 'r'))


def merge_general_info_into_one_string(general_info):
    instruction = general_info['instruction']
    # if instruction[-1] not in ['.', '?', '!', ';']:
        # instruction += '?'
    input = general_info['input']

    response = general_info['output']
    if input == "":
        general_info_str = "instruction: {}\n\n<|||>{}".format(instruction, response)
    else:
        general_info_str = "instruction: {}\n\ninput: {}\n\n<|||>{}\n".format(instruction, input, response)
    return general_info_str

# Create the general data with med formmat
med_general_json =[]
for i in range(len(med_cap_json)):
# for i in range(10):
    temp_dict = {}
    temp_dict['image'] = med_cap_json[i]['image']
    temp_dict['image_id'] = med_cap_json[i]['image_id']
    temp_dict['caption'] = merge_general_info_into_one_string(general_json[i])
    med_general_json.append(temp_dict)

if args.merge_qa > 0:
    final_mix_json = med_cap_json + args.merge_qa * med_qa_json + med_general_json 
else:
    final_mix_json = med_cap_json + med_general_json 

with open(args.output_json_path, 'w') as f:
    json.dump(final_mix_json, f, indent=4, ensure_ascii=False)
