import os
import json
import time
import re
from transformers import pipeline
import sys
from prompt import prompt_generate
from tqdm import tqdm
from LLM_calls import load_llm, llm_call


def LLM_detect(pipe, prompt_content, prompt_infer, google_use=False):
    messages = [
        {
            "role": "system",
            "content": "You are an expert of news authenticity evaluation. As an expert of "
                       "news authenticity evaluation, you should analyze and evaluate"
                       "the authenticity of news",
        },
        {"role": "user", "content": prompt_content},
    ]
    '''prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    print(outputs[0]["generated_text"])
    a_text = outputs[0]["generated_text"][outputs[0]["generated_text"].find('<|assistant|>'):]'''
    a_text = llm_call(messages, 'Llama', pipeline=pipe)
    print(a_text)
    if "This is fake news" in a_text[:40]:
        prediction = 'fake'
    elif "This is real news" in a_text[:40]:
        prediction = 'real'
    elif a_text.count("This is fake news") > a_text.count("This is real news"):
        prediction = 'fake'
    elif a_text.count("This is fake news") < a_text.count("This is real news"):
        prediction = 'real'
    elif a_text.count("is fake") > a_text.count("is real"):
        prediction = 'fake'
    elif a_text.count("is fake") < a_text.count("is real"):
        prediction = 'real'
    elif a_text.count("is not real") > a_text.count("is not fake"):
        prediction = 'fake'
    elif a_text.count("is not real") < a_text.count("is not fake"):
        prediction = 'real'
    elif a_text.count("fake") > a_text.count("real"):
        prediction = 'fake'
    elif a_text.count("fake") < a_text.count("real"):
        prediction = 'real'
    else:
        prediction = 'no_idea'
    '''prompt_i = prompt_infer + "The inference is: {}\nThe reliability of the inference is: ".format(a_text)
    messages_r = [
        {"role": "user", "content": prompt_i},
    ]
    r_text = llm_call(messages_r, 'Llama', pipeline=pipe)
    print(r_text)
    reliablity = 5
    numbers = re.findall(r'\[(.*?)(\d+)(.*?)\]', r_text)
    extracted_numbers = [match[1] for match in numbers]
    if extracted_numbers:
        reliablity = int(extracted_numbers[0])
        print("Matched number:", reliablity)
    else:
        print("No number found")'''

    return prediction, 5, a_text


def f1score(tp, fp, fn):
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = 2 * p * r / (p + r)
    return f1

def LLM_save(shot, model_path, pipe, test_news_dataset, target_folder, news_list, dem_list=None, lc_use=False, output_folder='./output'):
    correct_detect = 0
    correct_fake = 0
    correct_real = 0
    wrong_fake = 0
    wrong_real = 0
    news_num = 0
    correct_score = 0
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    print("total num: {}".format(len(news_list)))
    index_list = []
    decision_list = []
    reason_list = []
    '''pipe = pipeline("text-generation", model="../zephyr-7b-beta", torch_dtype=torch.bfloat16,
                    device_map="cuda")'''
    for index, (news, dem) in tqdm(enumerate(zip(news_list, dem_list)), total=len(news_list)):
        print(f'No.{index}')
        prompt_news, prompt_infer = prompt_generate(news, model_path, dem)
        try:
            decision, reliablity, reason = LLM_detect(pipe, prompt_news, prompt_infer)
            decision_list.append([decision, reliablity])
            reason_list.append(reason)
            if decision == news['label']:
                correct_detect += 1
                if reliablity > 5:
                    correct_score += 1
                if news['label'] == 'fake':
                    correct_fake += 1
                else:
                    correct_real += 1
            else:
                if reliablity < 5:
                    correct_score += 1
                if decision == 'fake':
                    wrong_fake += 1
                elif decision == 'real':
                    wrong_real += 1
                elif decision == 'no_idea':
                    if news['label'] == 'fake':
                        wrong_real += 1
                    else:
                        wrong_fake += 1
            news_num += 1
        except:
            print("can't load this news!!! ")
            print("Unexpected error:", sys.exc_info()[0])
            index_list.append(index)
            time.sleep(30)

    if len(index_list) > 0:
        for index in tqdm(index_list, total=len(index_list)):
            '''pipe = pipeline("text-generation", model="../zephyr-7b-beta", torch_dtype=torch.bfloat16,
                            device_map="auto")'''
            print(f'No.{index}')
            news = news_list[index]
            dem = dem_list[index]
            prompt_news, prompt_infer = prompt_generate(news, model_path, dem)
            try:
                decision, reliablity, reason = LLM_detect(pipe, prompt_news, prompt_infer)
                decision_list.insert(index, [decision, reliablity])
                reason_list.insert(index, reason)
                if decision == news['label']:
                    correct_detect += 1
                    if reliablity > 5:
                        correct_score += 1
                    if news['label'] == 'fake':
                        correct_fake += 1
                    else:
                        correct_real += 1
                else:
                    if reliablity < 5:
                        correct_score += 1
                    if decision == 'fake':
                        wrong_fake += 1
                    elif decision == 'real':
                        wrong_real += 1
                    elif decision == 'no_idea':
                        if news['label'] == 'fake':
                            wrong_real += 1
                        else:
                            wrong_fake += 1
                news_num += 1
            except:
                print("can't load this news!!!")
                print("Unexpected error:", sys.exc_info()[0])
                decision_list.insert(index, ['no_idea', 5])
                reason_list.insert(index, 'no_idea')

    result = {'decision': decision_list, 'reason': reason_list}

    if lc_use:
        output_file = 'dem_{}_{}_{}shot_{}.txt'.format(test_news_dataset, 'use_lc', shot,
                                                       time.strftime("%Y%m%d%H%M%S", time.localtime()))
        result_file = 'dem_result_use_lc_{}_{}shot'.format(test_news_dataset, shot)
    else:
        output_file = 'dem_{}_{}_{}shot_{}.txt'.format(test_news_dataset, 'no_lc', shot,
                                                       time.strftime("%Y%m%d%H%M%S", time.localtime()))
        result_file = 'dem_result_no_lc_{}_{}shot'.format(test_news_dataset, shot)
    with open(os.path.join(target_folder, result_file), 'w') as f:
        json.dump(result, f)

    acc = correct_detect / news_num
    f1 = f1score(correct_fake, wrong_fake, wrong_real)
    print("****************************  RESULT  ********************************")
    print("")
    print("{0} Accuracy: {1:.4f} ({2}/{3})".format(news_list[0]['news_source'], acc, correct_detect, news_num))
    print(
        "correct_fake: {} correct_real: {} wrong_fake: {} wrong_real: {}".format(correct_fake, correct_real, wrong_fake,
                                                                                 wrong_real))
    print("F1 score: {0:.4f}  correct_score: {1}".format(f1, correct_score))
    print("**********************************************************************")
    with open(os.path.join(output_folder, output_file), 'w') as f:
        result = ["{0} Accuracy: {1:.4f} ({2}/{3})\n".format(test_news_dataset, acc,
                                                             correct_detect, news_num),
                  "correct_fake: {} correct_real: {} wrong_fake: {} wrong_real: {}\n".format(
                      correct_fake, correct_real, wrong_fake, wrong_real),
                  "F1 score: {0:.4f}  correct_score: {1}".format(f1, correct_score)
                  ]
        f.writelines(result)
