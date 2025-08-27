import argparse
import os
import time

# os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import json
import pickle
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch

model_name = '../deberta-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
module = AutoModel.from_pretrained(model_name, output_hidden_states=True)
# device = 'cuda:0'
module.eval()
# module.to(device)
module.cuda()


def news_encoder(des):
    des_token = tokenizer.encode(des, add_special_tokens=True)[:256]
    input_ids = torch.tensor(des_token).unsqueeze(dim=0).cuda()
    with torch.no_grad():
        outputs = module(input_ids)
        # svec0 = outputs[-1]
        # svec1 = svec0[-1]
        svec = outputs['last_hidden_state']
        svec_np = svec.detach().cpu().numpy()
        svec_np = svec_np[0]
    return svec_np[0]


def train_news_encoder(source_folder, target_folder, dataset_list):
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)
    # dataset_list = os.listdir(source_folder)
    # dataset_list = [dataset for dataset in dataset_list if 'jsonl' in dataset]
    for dataset in dataset_list:
        with open(os.path.join(source_folder, dataset), 'r', encoding='utf-8') as f:
            # train_news_list = f.readlines()[0]
            train_news_list = json.load(f)
        svecs_list = list()
        for index, train_news in enumerate(tqdm(train_news_list, total=len(train_news_list))):
            '''if train_news['title'] != ' ' or train_news['text'] != ' ':
                des = ' '.join([train_news['title'], train_news['text']])
            else:
                des = train_news['tweet']'''
            kewords_list = [i[0] for i in train_news['keywords'][:5]]
            des = ' '.join(kewords_list)
            des = des.strip()
            svec = news_encoder(des)
            svecs_list.append(svec)
        svecs_list = np.asarray(svecs_list)
        np.save(os.path.join(target_folder, 'encoded_' + dataset), svecs_list)
        print(os.path.join(target_folder, 'encoded_' + dataset) + ' done')
    print('finish')


if __name__ == '__main__':
    start2 = time.time()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--datafile',
                        default='politifact_test.jsonl',
                        choices=['selected_valid_politifact.jsonl', 'selected_train_politifact.jsonl',
                                 'politifact_test.jsonl', 'selected_valid_gossipcop.jsonl',
                                 'selected_train_gossipcop.jsonl', 'gossipcop_test.jsonl'])
    args = parser.parse_args()

    source_folder = '../dataset_full'
    target_folder = '../dataset_full/encode'

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', default='politifact', choices=['politifact', 'gossipcop'])
    # args = parser.parse_args()
    dataset_list = ['selected_valid_gossipcop.jsonl', 'selected_train_gossipcop.jsonl', 'gossipcop_test.jsonl']
    news_t_list = [args.datafile]  # time test
    train_news_encoder(source_folder, target_folder, news_t_list)
    end2 = time.time()
    with open('time_2.1_encode_{}.txt'.format(args.datafile.split('.')[0]), 'w') as t:
        t.writelines('time_encode: {} seconds'.format(end2 - start2))
