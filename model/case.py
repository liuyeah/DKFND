import json
import os
import argparse

raw_data_path = "../dataset_full"
judge_path = "../dataset_full/judge_result"
determine_path = "../dataset_full/determine_result"

dataset_list = ['politifact', 'gossipcop', 'news_t', 'runtime_gos_test', 'alp']
shot_list = ['100', '64', '32', '16', '8']
parser = argparse.ArgumentParser()
parser.add_argument('-i', type=int)
args = parser.parse_args()
index = args.i  # 20

dataset = dataset_list[0]
shot = shot_list[0]

raw_data_files = [file for file in os.listdir(raw_data_path) if dataset in file and 'test' in file]
judge_files = [file for file in os.listdir(judge_path) if dataset in file]
determine_files = [file for file in os.listdir(determine_path) if dataset in file and shot in file]

with open(os.path.join(raw_data_path, raw_data_files[0]), 'r') as f:
    raw_data_list = json.load(f)
for j in judge_files:
    if 'dem_result' in j and shot in j:
        with open(os.path.join(judge_path, j), 'r') as f:
            inside_judge_list = json.load(f)
    elif 'google' in j:
        with open(os.path.join(judge_path, j), 'r') as f:
            outside_judge_list = json.load(f)
for d in determine_files:
    if 'dev_right_google_right' in d:
        with open(os.path.join(determine_path, d), 'r') as f:
            drgr = json.load(f)
    elif 'dev_right_google_wrong' in d:
        with open(os.path.join(determine_path, d), 'r') as f:
            drgw = json.load(f)
    elif 'dev_wrong_google_right' in d:
        with open(os.path.join(determine_path, d), 'r') as f:
            dwgr = json.load(f)
    elif 'wrong_determine' in d:
        with open(os.path.join(determine_path, d), 'r') as f:
            wd = json.load(f)

print("raw_data:")
print(raw_data_list[index]['text'])
print(raw_data_list[index]['keywords'])
print()
print("inside_judge_data:")
print(inside_judge_list['reason'][index])
print(inside_judge_list['decision'][index])
print()
print("outside_judge_data:")
print(outside_judge_list['reason'][index])
print(outside_judge_list['decision'][index])
print()
print("determine_data:")
if index in drgr['index']:
    print('drgr')
    print(drgr['final_decision'][drgr['index'].index(index)])
    print(drgr['final_reason'][drgr['index'].index(index)])
elif index in drgw['index']:
    print('drgw')
    print(drgw['final_decision'][drgw['index'].index(index)])
    print(drgw['final_reason'][drgw['index'].index(index)])
elif index in dwgr['index']:
    print('dwgr')
    print(dwgr['final_decision'][dwgr['index'].index(index)])
    print(dwgr['final_reason'][dwgr['index'].index(index)])
elif index in wd['index']:
    print('wd')
    print(wd['final_decision'][wd['index'].index(index)])
    print(wd['final_reason'][wd['index'].index(index)])
