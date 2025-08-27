import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '5,6'
import json
import re
import argparse
from tqdm import tqdm
import time
import spacy  # version 3.5
import torch
from transformers import pipeline, AutoTokenizer
from wiki_query import entity_linking_with_spacy
from LLM_calls import load_llm, llm_call




prompt_keywords = ( "There is a news article and a list of concepts related to it. Your task is to determine how relevant these keywords are to the given news. Assign each keyword a relevancy score between 0 and 1, where 1 indicates the keyword is highly relevant, and 0 indicates it is not relevant at all.\n"
                    "################\n"
                    "Instructions:\n"
                    "1. Analyze the content of the news article to understand its main topics and themes.\n"
                    "2. Compare each concept to the news content and assign a relevancy score between 0 and 1.\n"
                    "3. Output the concepts and their scores in reverse order (from the most relevant to the least relevant).\n"
                    "4. Don't return your explanation about the concepts and their scores."
                    "################\n"
                    "Output Format:\n"
                    "[[\"concept 1\", <score>], [\"concept 2\", <score>],..., [\"concept N\", <score>]]\n"
                    "################\n"
                    "Example Input:\n"
                    "News Article:\n"
                    "Scientists have discovered a new method to reduce carbon emissions using advanced nanotechnology. This breakthrough could significantly impact efforts to combat climate change globally.\n"
                    "concepts:\n"
                    "Global warming, Carbon emissions, Cryptocurrency, Nanotechnology, Climate change\n"
                    "################\n"
                    "Example Output:\n"
                    "[[\"Global warming\", 0.8], [\"Carbon emissions\", 1.0], [\"Cryptocurrency\", 0.0], [\"Nanotechnology\", 1.0], [\"Climate change\", 0.9]]\n"
                    "################\n"
                    "Input: \n")

prompt_format = "Please replace #score# in Your Output Format with the relevancy score of corresponding word: "

# initialize language model
nlp = spacy.load("en_core_web_md")
# add pipeline (declared through entry_points in setup.py)
nlp.add_pipe("entityLinker", last=True)



def keywords_detect(pipe, news_dict, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if news_dict['title'] != ' ' or news_dict['text'] != ' ':
        news_context = ' '.join([news_dict['title'], news_dict['text']])
    else:
        news_context = news_dict['tweet']
    news_tokens_ids = tokenizer(news_context)

    if len(news_tokens_ids['input_ids']) < 5:
        news_context = ' '.join([news_context, news_dict['tweet']])
    if len(news_tokens_ids['input_ids']) > 513:
        news_context = tokenizer.decode(news_tokens_ids['input_ids'][1:513])
    news_context = news_context.strip()

    ent_dict = entity_linking_with_spacy(news_context, add_description=True)

    prompt_content = prompt_keywords + f"News article: \n{news_context}\n" + f"concepts: \n{', '.join([ent_dict[i]['entity'] for i in ent_dict.keys()])}\n" + "################\n" + "Your Output Format:\n" + "[{}]\n".format(', '.join([f"[\"{ent_dict[i]['entity']}\", #score#]" for i in ent_dict.keys()])) + prompt_format
    print(prompt_content + '\n')
    print()
    messages = [
        {"role": "user", "content": prompt_content},
    ]

    text = llm_call(messages, 'Zephyr', pipeline=pipe)
    print(text)
    
    text = text.replace('(', '[')
    text = text.replace(')', ']')
    text = text.replace('{', '[')
    text = text.replace('}', ']')
    n1 = text.count('[')
    n2 = text.count(']')
    if n1 - n2 > 0:
        for _ in range(n1 - n2):
            text += ']'
    matches = re.findall(r'\[.*\]', text, re.DOTALL)[0]
    try:
        matches = matches.replace('\n', '')
        keywords_list = json.loads(matches)
        print(len(keywords_list))
    except:
        print("re error")
        keywords_list = [[ent_dict[i]['entity'], 0.5] for i in ent_dict.keys()]
    
    print()
    keywords_list.sort(key=lambda element: element[1], reverse=True)
    print(keywords_list)
    return keywords_list

def keywords_save(batch, selected_folder, keywords_folder, datafile, model_name, model_path):
    
    if not os.path.exists(keywords_folder):
        os.mkdir(keywords_folder)
    dataset_list = [dataset for dataset in os.listdir(selected_folder) if datafile in dataset]  #time test
    #pipe = pipeline("text-generation", model="../Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16,
    #                device_map="cuda")
    pipe = load_llm(model_name, model_path)
    for dataset in dataset_list:
        #context = mp.get_context('spawn')
        #pool = context.Pool(2)
        
        print(dataset)
        with open(os.path.join(selected_folder, dataset), 'r') as f:
            #news_list0 = f.readlines()[0]
            news_list0 = json.load(f)
        #batch_size = int(len(news_list0))
        news_list = news_list0    #[batch*batch_size:(batch+1)*batch_size]
        print("len_news_list: {}".format(len(news_list)))
        # print(batch_size)
        keywords_list = []
        index_list = []
        for index, news in tqdm(enumerate(news_list), total=len(news_list)):
            print("No." + str(index))
            try:
                keywords_list.append(keywords_detect(pipe, news, model_path))
            except:
                print("error")
                index_list.append(index)

        if len(index_list) > 0:
            for index in index_list:
                print("No." + str(index))
                try:
                    keywords = keywords_detect(pipe, news_list[index], model_path)
                    keywords_list.insert(index, keywords)
                except:
                    print("error")
                    text = news_list[index]['title'] + news_list[index]['text'] + news_list[index]['tweet']
                    bad_list = text.strip().split(' ')[:6]
                    bad_list = [[word, 0.5] for word in bad_list]
                    keywords_list.insert(index, bad_list)

        for news_dict, keywords in tqdm(zip(news_list, keywords_list), total=len(news_list)):
            news_dict['keywords'] = keywords

        with open(os.path.join(keywords_folder, dataset), 'w') as fout:
            json.dump(news_list, fout)


if __name__ == "__main__":
    start1 = time.time()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--batch', type=int, default=0)
    parser.add_argument('--datafile',
                        default='alp.json',
                        choices=['selected_valid_politifact.jsonl', 'selected_train_politifact.jsonl',
                         'politifact_test.jsonl', 'selected_valid_gossipcop.jsonl', 'selected_train_gossipcop.jsonl',
                          'gossipcop_test.jsonl', 'news_t.jsonl', 'runtime_gos_test.jsonl', 'alp.json'])
    args = parser.parse_args()

    selected_folder = '../dataset_full'
    dataset_folder = '../extracted_dataset/'
    #keywords_folder = '../dataset_with_keywords_2'
    model_name = "Zephyr"
    model_path = "../zephyr-7b-beta"
    datafile = args.datafile
    keywords_save(args.batch, selected_folder, selected_folder, datafile, model_name, model_path)
    end1 = time.time()
    with open('time1_{}.txt'.format(args.datafile.split('.')[0]), 'w') as t:
        t.writelines('detect module: {} seconds'.format(end1 - start1))