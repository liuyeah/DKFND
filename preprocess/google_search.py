import os
import json
# os.environ['CUDA_VISIBLE_DEVICES'] = '5,6'
import torch
from serpapi import GoogleSearch
import time
from IPython.utils import io
from tqdm import tqdm
import argparse
import re
from triple_generate import prompt_fomular_triple_extraction
from LLM_calls import load_llm, llm_call
from wiki_query import entity_linking_with_spacy, entity_mapping_for_line, relation_mapping_for_line, triple_mapping
from kge.model import KgeModel
from kge.util.io import load_checkpoint


def google_search(news_keywords, b=4):
    params = {
        "engine": "google",
        "q": news_keywords,
        "api_key": "308d6efcf3e01926b5c449119c2cf421ecf0fffcdd4f7c5118b3d2b1f91ce302"
        # "3f8973b61b205b3e82f65b8e14873e6d012b441f8174ec9a332f907891115262",
    }
    with io.capture_output() as captured:  # disables prints from GoogleSearch
        search = GoogleSearch(params)
        res = search.get_dict()
    answer = None
    snippet = None
    title = None
    cop = []
    toret = []
    '''if "answer_box" in res.keys(): 
        print("answer_box")
        print(type(res["answer_box"]))
    if "organic_results" in res:
        print(type(res["organic_results"]))'''

    if "answer_box" in res.keys() and "answer" in res["answer_box"].keys():
        answer = res["answer_box"]["answer"]
    if "answer_box" in res.keys() and "snippet" in res["answer_box"].keys():
        snippet = res["answer_box"]["snippet"]
        title = res["answer_box"]["title"]
    elif 'answer_box' in res.keys() and 'snippet_highlighted_words' in res['answer_box'].keys():
        toret = res['answer_box']["snippet_highlighted_words"][0]
    elif (
            "answer_box" in res.keys()
            and "contents" in res["answer_box"].keys()
            and "table" in res["answer_box"]["contents"].keys()
    ):
        snippet = res["answer_box"]["contents"]["table"]
        title = res["answer_box"]["title"]
    elif "answer_box" in res.keys() and "list" in res["answer_box"].keys():
        snippet = res["answer_box"]["list"]
        title = res["answer_box"]["title"]
    elif "organic_results" in res:
        for i in range(len(res["organic_results"])):
            if "snippet" in res["organic_results"][i].keys() and "title" in res["organic_results"][i].keys() and len(
                    res["organic_results"][i]["snippet"]):
                snippet = res["organic_results"][i]["snippet"]
                title = res["organic_results"][i]["title"]
                cop.append((title, snippet))
                if i == b - 1:
                    break
    elif (
            "organic_results" in res
            and "rich_snippet" in res["organic_results"][0].keys()
    ):
        for i in range(len(res["organic_results"])):
            snippet = res["organic_results"][i]["rich_snippet"]
            title = res["organic_results"][i]["title"]
            cop.append((title, snippet))
            if i == b - 1:
                break
    else:
        snippet = None
    if snippet is not None:
        # title = title.replace("- Wikipedia", "").strip()
        if len(cop) == 0:
            toret0 = f"{title}: {snippet}"
            toret0 = f"{toret0} So the answer is {answer}." if answer is not None else toret0
            toret.append(toret0)
        else:
            for title, snippet in cop:
                toret.append(f"{title}: {snippet}")
    else:
        toret = []
    print(f"There are {len(toret)} messages.")
    return toret, res


def fix():
    path = '../dataset_full/google_search/google_selected_valid_politifact.jsonl'
    with open(path, 'r') as f:
        a = json.load(f)
    l = a.keys()
    d = {}
    for i in l:
        n = i[:-4]
        s = i[-4:]
        d[n + '_' + s] = a[i]
    with open(path, 'w') as f:
        json.dump(d, f)


def google_save(datafile, a, b, alp_token):
    # keywords_folder = '../dataset_with_keywords_2'
    selected_folder = '../dataset_full'
    full_folder = '../dataset_full'
    google_folder = '../dataset_full/google_search'
    if not os.path.exists(full_folder):
        os.mkdir(full_folder)
    if not os.path.exists(google_folder):
        os.mkdir(google_folder)
    # folder = selected_folder if 'selected' in datafile else keywords_folder
    folder = selected_folder
    with open(os.path.join(folder, datafile), 'r') as f:
        # news_list = f.readlines()[0]
        if alp_token == True:
            news_list = json.load(f)[a:b]
        else:
            news_list = json.load(f)
    # news_list = news_list[a: b]
    keywords_list = []

    for news in news_list:

        keywords_str = []
        for keyword in news['keywords'][:5]:
            keywords_str.append(keyword[0])
        keywords_list.append(', '.join(keywords_str))
    search_res_list = []
    search_toret_list = []
    search_result = {}
    for keywords in tqdm(keywords_list, total=len(news_list)):
        toret, res = google_search(keywords, 8)
        search_res_list.append(res)
        search_toret_list.append(toret)
    #
    # search_result_list = list(tqdm(map(google_search, keywords_list), total=len(news_list)))
    assert len(news_list) == len(search_res_list) == len(search_toret_list)
    for news_dict, toret, res in tqdm(zip(news_list, search_toret_list, search_res_list), total=len(news_list)):
        news_dict['raw_google'] = toret
        search_result[news_dict['news_id'] + '_' + news_dict['label']] = res
    output_name = datafile
    if alp_token == True:
        with open(os.path.join(folder, datafile), 'r') as f:
            raw_news_list = json.load(f)
        raw_news_list[a:b] = news_list
        with open(os.path.join(full_folder, datafile), 'w') as fout:
            json.dump(raw_news_list, fout)
    else:
        with open(os.path.join(full_folder, datafile), 'w') as fout:
            json.dump(news_list, fout)
    with open(os.path.join(google_folder, 'google_' + output_name), 'w') as fout:
        json.dump(search_result, fout)


forbidden_list = set(['--', 'I', 'you', 'he', 'she', 'it', 'we', 'they', 'this', 'that', 'these', 'those',
                      'anyone', 'everyone', 'someone', 'no one', 'nobody', 'somebody', 'everybody', 'anything',
                      'something', 'everything', 'nothing',
                      'the', 'a', 'an', 'one', 'the two', 'the other', 'other', 'another',
                      'book', 'song', 'country', 'school', 'friend', 'pet', 'job', 'event', 'restaurant', 'app',
                      'company', 'film', 'people', 'person',
                      'language', 'city', 'family member', 'hobby', 'sport', 'project', 'skill', 'neighborhood',
                      'website', 'community', 'judge', 'court'])

e_file_path = '../../kge/data/wikidata5m/entity_ids.json'
r_file_path = '../../kge/data/wikidata5m/relation_ids.json'


def read_KGC_id_dict(entity_file_name, relation_file_name):
    with open(entity_file_name) as e_f, \
            open(relation_file_name) as r_f:
        e_kgc_id_dict = {}
        r_kgc_id_dict = {}

        for line in tqdm(e_f):
            line = json.loads(line)
            e_kgc_id_dict[line['wiki_id']] = line['map_id']

        for line in tqdm(r_f):
            line = json.loads(line)
            r_kgc_id_dict[line['wiki_id']] = line['map_id']

    return e_kgc_id_dict, r_kgc_id_dict


e_kgc_id_dict, r_kgc_id_dict = read_KGC_id_dict(e_file_path, r_file_path)


def triple_extraction_decode(ans: str, entity):
    json_pattern = r'\[.*\]'
    match = re.match(json_pattern, ans, re.DOTALL)  # re.DOTALL 允许 . 匹配换行符
    phrase_flag = False
    e_flag = False
    if match:
        json_str = match.group()
        for e in entity:
            if e in json_str:
                e_flag = True
        if '{' in json_str and '}' in json_str and e_flag == True:
            json_str = json_str.replace('\n', '')
            # print(json.dumps(json_str))
            try:
                json_str = json.loads(json_str)
                phrase_flag = True
                return json_str, phrase_flag
            except:
                pass
    json_pattern = r'\{.*?\}'
    match = re.findall(json_pattern, ans)
    triple_list = []
    for m in match:
        for e in entity:
            if e in m:
                e_flag = True
        try:
            m = json.loads(m)
            triple_list.append(m)
        except:
            continue

    if e_flag == False:
        return ans, False

    if len(triple_list) > 0:
        return triple_list, True
    else:
        return ans, False


def triple_verication(list_raw: list):
    '''纯数字我们不处理，他仍然是合法的'''
    list_new = []

    for t in list_raw:
        head_list, tail_list = [], []
        if 'subject' not in t.keys():
            continue
        if 'object' not in t.keys():
            continue
        if 'predicate' not in t.keys():
            continue
        if isinstance(t['subject'], str):
            head_list = [t['subject']]
        elif isinstance(t['subject'], list):
            head_list = t['subject']

        if isinstance(t['object'], str):
            tail_list = [t['object']]
        elif isinstance(t['object'], list):
            tail_list = t['object']

        for s in head_list:
            if s.lower() in forbidden_list:
                continue
            for o in tail_list:
                if o.lower() in forbidden_list:
                    continue
                list_new.append({'subject': s, 'predicate': t['predicate'], 'object': o})

    return list_new


def score(triples, model, relation=True):
    # line = json.loads(line.strip())
    # triples = line[src_key]
    s_id_list = []
    p_id_list = []
    o_id_list = []

    for t in triples:
        s, p, o = t
        s_id = e_kgc_id_dict.get(s, -1)
        if s_id == -1:
            continue
        o_id = e_kgc_id_dict.get(o, -1)
        if o_id == -1:
            continue
        p_id = r_kgc_id_dict[p]
        # we use the relation in kgc model, so we do not check
        s_id_list.append(int(s_id))
        p_id_list.append(int(p_id))
        o_id_list.append(int(o_id))

    if len(s_id_list) == 0:
        # none link
        return 0
    else:
        if relation:
            s_tensor = torch.Tensor(s_id_list).long()
            p_tensor = torch.Tensor(p_id_list).long()
            o_tensor = torch.Tensor(o_id_list).long()
            scores = model.score_spo(s_tensor, p_tensor, o_tensor)
            score_list = scores.tolist()
            return sum(score_list) / len(score_list)
        else:
            s_tensor = torch.Tensor(s_id_list).long()
            o_tensor = torch.Tensor(o_id_list).long()
            scores = model.score_so(s_tensor, o_tensor)
            scores, _ = torch.max(scores, dim=1)
            score_list = scores.tolist()
            return sum(score_list) / len(score_list)

    # output_f.write(json.dumps(line, ensure_ascii=False) + '\n')


def google_select(datafile, model_name, model_path, a, b, alp_token):
    selected_folder = '../dataset_full'
    full_folder = '../dataset_full'
    google_folder = '../dataset_full/google_search'
    if not os.path.exists(full_folder):
        os.mkdir(full_folder)
    if not os.path.exists(google_folder):
        os.mkdir(google_folder)

    folder = selected_folder
    with open(os.path.join(folder, datafile), 'r') as f:
        if alp_token == True:
            news_list = json.load(f)[a:b]
        else:
            news_list = json.load(f)
    pipeline = load_llm(model_name, model_path)
    checkpoint = load_checkpoint('../../xkliu/kge/checkpoints/wikidata5m-complex.pt')
    model = KgeModel.create_from(checkpoint)
    for news in tqdm(news_list, total=len(news_list)):
        if len(news['raw_google']) == 0:
            news['google'] = [["", 100]]
            continue
        google_list = news['raw_google'][1:8]
        score_list = []
        entity_list = []
        for text in tqdm(google_list, total=len(google_list)):
            # EL
            ent_dict = entity_linking_with_spacy(text, add_description=True)
            entity_list = ['\"' + ent_dict[i]['entity'] + '\"' for i in ent_dict.keys()]
            entity_str = ', '.join(entity_list)

            # triple_generate
            prompt = prompt_fomular_triple_extraction(text, entity_str, provide_entity=True)
            print(prompt)
            print('*****************************************')
            messages = [{"role": "user", "content": prompt}]
            response = llm_call(messages, model_name, pipeline=pipeline)
            print(response)
            # triple_extract
            res, json_flag = triple_extraction_decode(response, entity_list)
            if json_flag:
                triple = triple_verication(res)
            else:
                triple = []
                # error_num += 1
            # mapping
            if len(triple) == 0:
                score_list.append(0)
                continue
            entity_map_dict = entity_mapping_for_line(triple, ent_dict)
            relation_map_dict = relation_mapping_for_line(triple)
            if relation_map_dict == None:
                score_list.append(0)
                continue
            triple_id_list = triple_mapping(triple, entity_map_dict, relation_map_dict)
            # score
            score_list.append(score(triple_id_list, model))

        if len(google_list) > 0:
            google_with_score = list(zip(google_list, score_list))
            google_with_score.sort(key=lambda element: element[1], reverse=True)
            news['google'] = [[news['raw_google'][0], 100], google_with_score[0]]
        else:
            news['google'] = [[i, 100] for i in news['raw_google'][:2]]
    if alp_token == True:
        with open(os.path.join(folder, datafile), 'r') as f:
            raw_news_list = json.load(f)
        raw_news_list[a:b] = news_list
        with open(os.path.join(full_folder, datafile), 'w') as fout:
            json.dump(raw_news_list, fout)
    else:
        with open(os.path.join(full_folder, datafile), 'w') as fout:
            json.dump(news_list, fout)


if __name__ == '__main__':
    start3 = time.time()

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--datafile',
                        default='politifact_test.jsonl',
                        choices=['selected_valid_politifact.jsonl', 'selected_train_politifact.jsonl',
                                 'politifact_test.jsonl', 'selected_valid_gossipcop.jsonl',
                                 'selected_train_gossipcop.jsonl', 'gossipcop_test.jsonl'])
    # parser.add_argument('--model_name', '-m', type=str, default='Zephyr', required=True, help='Model Name')
    # parser.add_argument('--model_path','-p',type=str, default="../zephyr-7b-beta", required=True, help="Path to model")
    parser.add_argument('--a', type=int, default=0)
    parser.add_argument('--b', type=int, default=4500)
    args = parser.parse_args()
    print(f"{args.datafile}[{args.a}: {args.b}]")
    '''model_name = "Llama"
    model_path = "../Meta-Llama-3-8B-Instruct"'''
    model_name = "Zephyr"
    model_path = "../zephyr-7b-beta"
    alp_token = True
    datafile = args.datafile
    google_save(datafile, args.a, args.b, alp_token)
    end3_0 = time.time()
    google_select(datafile, model_name, model_path, args.a, args.b, alp_token)
    end3_1 = time.time()
    with open('time2.2_3.2_{}.txt'.format(datafile.split('.')[0]), 'w') as t:
        t.writelines(
            'time_Outside_Investigation: Outside Investigate：{} seconds, Outside judge：{} seconds'.format(end3_0 - start3, end3_1 - end3_0))
