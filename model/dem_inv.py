import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import time
import json
from LLM_detected_and_save import LLM_save
import torch
from transformers import pipeline, AutoTokenizer
from prompt_news_select import prompt_news_select
import numpy as np
import argparse
import random
from tqdm import tqdm
import re
from LLM_calls import load_llm, llm_call

prompt_keywords_rank = ("Given this news and the keywords in this news:\n "
                        "given news: {}\n keywords: {} \n"
                        "please find the news with the keywords closest to this given news among the following news based on the keywords. The following news and keywords are:\n {}\n"
                        "Please return the index of following news in a list")


def train_data_shot(shot, train_data):
    fake_num = shot
    real_num = shot
    train_news_index_list = []
    for index, news in enumerate(train_data):
        if news['label'] == 'fake' and fake_num > 0:
            fake_num -= 1
            train_news_index_list.append(index)
        elif news['label'] == 'real' and real_num > 0:
            real_num -= 1
            train_news_index_list.append(index)
    return train_news_index_list

def prompt_fomular_kg_local_check_0(news:dict, tokenizer):
    if news['title'] != ' ' or news['text'] != ' ':
        news_context = ' '.join([news['title'], news['text']])
    else:
        news_context = news['tweet']
    news_tokens_ids = tokenizer(news_context)

    if len(news_tokens_ids['input_ids']) < 5:
        news_context = ' '.join([news_context, news['tweet']])
    if len(news_tokens_ids['input_ids']) > 513:
        news_context = tokenizer.decode(news_tokens_ids['input_ids'][1:513])
    news_context = news_context.strip()
    content = 'I need your help determining the reliability of a passage, and I’ve provided descriptions of the relevant entities mentioned in the text. Here are some hints:\n'
    content += '1. **Entity Accuracy**: Check if the entities (e.g., names, attributes, roles) match the passage I’ve provided. Note that we do not require all relevant entities to appear in the article, you only need to verify that the entities present in the text do not conflict with the entities provided. You also do not need to check the relationships between the entities. However, if none of the entities appear, then the passage should be unreliable.\n'
    # content += '2. **Relation Validity**: Review the relationships between the entities. Are these relationships correctly represented according to the descriptions, or do they seem inconsistent?\n'
    # content += '3. **Consistency with Known Information**: Does the passage align with what is commonly known or accepted about the entities and their interactions?\n'
    # content += '4. **Context and Reliability**: Based on the above checks, please assess whether the passage is reliable overall. If there are any doubts or questionable points, provide your reasoning.\n\n'
    content += 'Please output your reason and reliability score between 0 and 1 in the following JSON format:\n'
    content += '{"reason": [your explanation for the decision],"reliability": <score>}\n\n'
    content += 'Here is the Passages:\n{}\n\n'.format(news_context)
    content += 'Here are the relevant entities:\n'
    for i, ent in enumerate(news['keywords'][:5]):
        content += '{}.{}\n'.format(i + 1, ent[0])
    content += '\nOutput:\n'

    return content


def prompt_fomular_kg_local_check(news_list, keywords, tokenizer):
    news = ''
    for i, n in enumerate(news_list):
        if n['title'] != ' ' or n['text'] != ' ':
            news_context = ' '.join([n['title'], n['text']])
        else:
            news_context = n['tweet']
        news_tokens_ids = tokenizer(news_context)

        if len(news_tokens_ids['input_ids']) < 5:
            news_context = ' '.join([news_context, n['tweet']])
        if len(news_tokens_ids['input_ids']) > 513:
            news_context = tokenizer.decode(news_tokens_ids['input_ids'][1:513])
        news_context = news_context.strip()
        news += 'Passage {}: \n{}\n'.format(i, news_context)
    content = 'I need your help determining the reliability of some passages, and I’ve provided some relevant entities mentioned in the text. Here are some hints:\n'
    content += '1. **Entity Accuracy**: Check if the entities (e.g., names, attributes, roles) match the passages I’ve provided. Note that we do not require all relevant entities to appear in the passage, you only need to verify that the entities present in the passage do not conflict with the entities provided. You also do not need to check the relationships between the entities. However, if none of the entities appear, then the passage should be unreliable.\n'
    # content += '2. **Relation Validity**: Review the relationships between the entities. Are these relationships correctly represented according to the descriptions, or do they seem inconsistent?\n'
    # content += '3. **Consistency with Known Information**: Does the passage align with what is commonly known or accepted about the entities and their interactions?\n'
    # content += '4. **Context and Reliability**: Based on the above checks, please assess whether the passage is reliable overall. If there are any doubts or questionable points, provide your reasoning.\n\n'
    content += 'Please output your reason and index of the two most relevant passage in the following JSON format:\n'
    content += '{"reason": <your explanation for the decision>,"index": [<index of passage which is the most relevant>, <index of passage which is the second relevant>]}\n\n'
    content += 'Here are the Passages:\n{}'.format(news)
    content += '\nHere are the relevant entities:\n'
    #for i, ent in enumerate(line['question_entity'].values()):
    #    content += '{}.{}: {}\n'.format(i + 1, ent['entity'], ent['description'])
    content += keywords
    content += '\nOutput:\n{"reason": #Explanation#,"index": [#Index 1#, #Index 2#]}'
    content == 'Please replace #Explanation# with your explanation for the decision and replace #Index 1#, #Index 2# with index of passage which is the first relevant and second relevant.'
    return content

def LLM_rank(pipe, tokenizer, dev_news_list, test_news_list):
    
    ranked_list = []
    for dev_news, test_news in tqdm(zip(dev_news_list, test_news_list), total=len(test_news_list)):
        case = 1
        try:
            if case == 1:
                test_keywords = ' '.join([i[0] for i in test_news['keywords'][:5]])
                prompt = prompt_fomular_kg_local_check(dev_news, test_keywords, tokenizer)
                messages = [{"role": "user", "content": prompt}]
                response = llm_call(messages, 'Llama', pipeline=pipe)
                print(response + '\n')
                json_pattern = r'(\{.*?\})'
                match = re.findall(json_pattern, response, re.DOTALL)
                json_str = match[0]
                json_str = json_str.replace('\n', '')
                print(json_str)
                index = json.loads(json_str)["index"]
                if -1 in index:
                    index[index.index(-1)] = 1
                if None in index:
                    index[index.index(None)] = 1
                set_index = set(index)
                index = list(set_index)
                if len(index) == 1:
                    index.append(index[0] + 1)
                print(index)
                print()
                news_list = []
                for i in index[:2]:
                    news_list.append(dev_news[i])
                ranked_list.append(news_list)
            else:
                for news in dev_news:
                    prompt = prompt_fomular_kg_local_check_0(news, tokenizer)
                    messages = [{"role": "user", "content": prompt}]
                    response = llm_call(messages, 'Llama', pipeline=pipe)
                    print(response)
        except:
            ranked_list.append(dev_news[:2])

    return ranked_list

def extract_train_news_and_ask_LLM(shot, dem_ger, a, b, test_news_dataset, source_folder,
                                   encode_folder, target_folder, output_folder, model_name, model_path):
    news_jsonl_files = [file for file in os.listdir(source_folder) if test_news_dataset in file]
    for news_jsonl in news_jsonl_files:
        if 'test' in news_jsonl and 'runtime_gos' not in news_jsonl:
            with open(os.path.join(source_folder, news_jsonl), 'r') as f:
                test_news_list = json.load(f)
        elif 'train' in news_jsonl:
            with open(os.path.join(source_folder, news_jsonl), 'r') as f:
                train_data_list = json.load(f)
        elif 'news_t' in news_jsonl:
            with open(os.path.join(source_folder, news_jsonl), 'r') as f:
                test_news_list = json.load(f)
            with open(os.path.join(source_folder, "selected_train_politifact.jsonl"), 'r') as f:
                train_data_list = json.load(f)
        elif 'runtime_gos_test' in news_jsonl:
            with open(os.path.join(source_folder, news_jsonl), 'r') as f:
                test_news_list = json.load(f)
            with open(os.path.join(source_folder, "selected_train_gossipcop.jsonl"), 'r') as f:
                train_data_list = json.load(f)
        elif 'alp' in news_jsonl:
            with open(os.path.join(source_folder, news_jsonl), 'r') as f:
                test_news_list = json.load(f)
            with open(os.path.join(source_folder, "selected_train_politifact.jsonl"), 'r') as f:
                train_data_list = json.load(f)

    train_index_list = train_data_shot(shot, train_data_list)
    train_news_list = [train_data_list[i] for i in train_index_list]

    train_encode_list = []
    test_news_encode_list = []
    for encoded_file in os.listdir(encode_folder):
        if test_news_dataset in encoded_file:
            if 'train' in encoded_file:
                train_encode_list = np.load(os.path.join(encode_folder, encoded_file))
            elif 'test' in encoded_file and 'runtime_gos' not in encoded_file:
                test_news_encode_list = np.load(os.path.join(encode_folder, encoded_file))
            elif 'news_t' in encoded_file:
                test_news_encode_list = np.load(os.path.join(encode_folder, encoded_file))
                train_encode_list = np.load(os.path.join(encode_folder, "encoded_selected_train_politifact.jsonl.npy"))
            elif 'runtime_gos_test' in encoded_file:
                test_news_encode_list = np.load(os.path.join(encode_folder, encoded_file))
                train_encode_list = np.load(os.path.join(encode_folder, "encoded_selected_train_gossipcop.jsonl.npy"))
            elif 'alp' in encoded_file:
                test_news_encode_list = np.load(os.path.join(encode_folder, encoded_file))
                train_encode_list = np.load(os.path.join(encode_folder, "encoded_selected_train_politifact.jsonl.npy"))

    train_news_encode_list = np.asarray([train_encode_list[i] for i in train_index_list])
    assert len(train_news_encode_list) == len(train_news_list)
    assert len(test_news_encode_list) == len(test_news_list)

    fake_num = 5
    real_num = 5

    debug = 0
    
    pipeline = load_llm(model_name, model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if dem_ger == True:
        test_news_list = test_news_list[a:b]
        test_news_encode_list = test_news_encode_list[a:b]
        start1 = time.time()
        if debug == 0:
            dem_news_index_list = prompt_news_select(train_news_list, train_news_encode_list,
                                                    test_news_encode_list, fake_num, real_num)
        else:
            n_dem = len(test_news_list)
            sampled_list = list(range(len(train_news_list)))
            dem_news_index_list = [random.sample(sampled_list, 4) for _ in range(n_dem)]

        dem_news_list_0 = [[train_news_list[i] for i in index] for index in dem_news_index_list]
        r_dem_news_list_0 = []
        f_dem_news_list_0 = []
        for news_list in dem_news_list_0:
            r_list = []
            f_list = []
            for news in news_list:
                if news['label'] == 'real':
                    r_list.append(news)
                else:
                    f_list.append(news)
            r_dem_news_list_0.append(r_list)
            f_dem_news_list_0.append(f_list)
        #pipe = pipeline("text-generation", model="../zephyr-7b-beta", torch_dtype=torch.bfloat16,
        #            device_map="cuda")
        start2 = time.time()
        with open('time2.1_{}.txt'.format(test_news_dataset), 'w') as t:
            t.writelines('Inside Detect: {} seconds'.format(start2 - start1))
        assert len(dem_news_list_0) == len(test_news_list)
        r_dem_news_list = LLM_rank(pipeline, tokenizer, r_dem_news_list_0, test_news_list)
        f_dem_news_list = LLM_rank(pipeline, tokenizer, f_dem_news_list_0, test_news_list)
        dem_news_list = []
        for r, f in zip(r_dem_news_list, f_dem_news_list):
            dem_news_list.append(r + f)
            
        end2 = time.time()
        with open('time3.1_{}.txt'.format(test_news_dataset), 'w') as t:
            t.writelines('Inside Judge: {} seconds'.format(end2 - start2))
            
        with open(os.path.join(target_folder, 'dem_news_{}shot_{}_{}_'.format(shot, a, b) + test_news_dataset), 'w') as f:
            json.dump(dem_news_list, f)
    else:
        with open(os.path.join(target_folder, 'dem_news_{}shot_'.format(shot) + test_news_dataset), 'r') as f:
            dem_news_list = json.load(f)
    assert len(dem_news_list) == len(test_news_list)

    start4 = time.time()
    LLM_save(shot, model_path, pipeline, test_news_dataset, target_folder, test_news_list, lc_use=True,
             dem_list=dem_news_list, output_folder=output_folder)
    end4 = time.time()
    with open('time4.1_{}.txt'.format(test_news_dataset), 'w') as t:
        t.writelines('Inside Determine: {} seconds'.format(end4 - start4))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--shot', type=int, default=100, choices=[100, 64, 32, 16, 8])
    parser.add_argument('--dataset', default='politifact', choices=['politifact', 'gossipcop'])
    parser.add_argument('--dem_ger', type=bool, default=False)
    parser.add_argument('--a', type=int, default=0)
    parser.add_argument('--b', type=int, default=4500)
    args = parser.parse_args()

    source_folder = '../dataset_full'
    encode_folder = '../dataset_full/encode'
    target_folder = '../dataset_full/judge_result'
    output_folder = './output_all/'
    model_name = "Zephyr"
    model_path = "../zephyr-7b-beta"
    print(f"{args.dataset}[{args.a}: {args.b}]")
    test_news_dataset = args.dataset
    dem_ger = args.dem_ger
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)
    extract_train_news_and_ask_LLM(args.shot, dem_ger, args.a, args.b,  test_news_dataset, source_folder, encode_folder,
                                   target_folder, output_folder, model_name, model_path)
