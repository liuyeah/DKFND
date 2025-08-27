import argparse
import os
import sys
import time
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
import json
import re
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm
from LLM_calls import load_llm, llm_call

def f1score(tp, fp, fn):
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = 2 * p * r / (p + r)
    return f1


def LLM_detected(pipe, prompt_content, prompt_infer):

    messages = [
        {
            "role": "system",
            "content": "You are an expert of news authenticity evaluation.",
        },
        {"role": "user", "content": prompt_content},
    ]
    '''prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    print(outputs[0]["generated_text"])
    a_text = outputs[0]["generated_text"][outputs[0]["generated_text"].find('<|assistant|>'):]'''
    a_text = llm_call(messages, 'Llama', pipeline=pipe)
    print(a_text)
    print()
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
    reliablity = 5
    numbers = re.findall(r'\[(.*?)(\d+)(.*?)\]', r_text)
    extracted_numbers = [match[1] for match in numbers]
    if extracted_numbers:
        reliablity = int(extracted_numbers[0])
        print("Matched number:", reliablity)
    else:
        print("No number found")'''
    
    return prediction, 5, a_text


prompt_google_search = ("I need your assistance in evaluating the authenticity of a news article."
                        "I will provide you the news article and additional information about this news. "
                        "Please analyze the following news and give your decision and reason. "
                        "The first sentence of your [Decision and Reason] must be [This is fake news] or [This is real news], and then give reason. "
                        "The news article is: \n{}\n"
                        "The additional information is: \n{}\n"
                        "[Decision and Reason]:"
                        )
("I need your assistance in evaluating the authenticity of a news article. "
                        "I will provide you the news article and additional information about this news. "
                        "Please analyze the following news and give your decision. "
                        "The first sentence of your [Decision] must be [This is fake news] or [This is real news]. "
                        "The news article is: \n{}\n"
                        "The additional information is: \n{}\n"
                        "[Decision]:"
                        )
prompt_google_search_v2 = ("I need your assistance in evaluating the authenticity of a news article "
                        "and please assess the reliablity of your own answers. "
                        "The reliablity indicates credibility of your answer, which is a score between 1 to 10. "
                        "The higher the score, the more reliable your answer. "
                        "Score between 1 to 5 means unreliable, 5 to 10 means reliable, and 5 means uncertainty. "
                        "I will provide you the news article and additional information about this news. "
                        "Your answer should include your decision, reliablity and reason for your decision. "
                        "The first sentence of your answer is your decision, which must be [This is fake news] or [This is real news]. "
                        "The second sentence of your answer is about reliablity like [The confidence level for my answer is <score between 1 to 10>]. "
                        "Then, give your reason for your decision.\n"
                        "The news article is: \n{}\n"
                        "The additional information is: \n{}\n"
                        "Your Answer:"
                        )
reliablity_ppt = ("Here's a news article and an inference about the authenticity of the news. Please assess the reliability of the inference. "
                  "Reliability is a score ranging from 1 to 10. A score of 1 is very unreliable and a score of 10 is very reliable. "
                  "The higher the score, the more reliable the inference. "
                  "If the inference is found to be inconsistent with the facts, give a score of 1 to 5, and if the inference is consistent with the facts, give a score of 5 to 10.\n "
                  "Please return the reliablity in the format of [<score>] in the first sentence of your answer. "
                  "Here are some example: \n"
                  "#\n"
                  "The news article is: Recent Studies Show Dark Chocolate May Improve Heart Health."
                  "The inference is: [This is real news]   Research confirms dark chocolate's flavonoids support cardiovascular health, backed by clinical studies demonstrating reduced blood pressure and improved circulation."
                  "The reliability of the inference is: [8]"
                  "#\n"
                  'The news article is: Scientists successfully developed a vaccine that reduces the severity of COVID-19.'
                  "The inference is: [This is fake news]  The vaccine claims lack peer-reviewed evidence and are not supported by reputable health organizations, undermining their credibility and scientific validity."
                  "The reliability of the inference is: [1]"
                  '#\n'
                  "The news article is: {}\n"
                  )

def news_cut(news, model_path, n, m):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    thr_n = n+1
    thr_m = m+1

    news_text_ids = tokenizer(news['text'])
    if len(news_text_ids['input_ids']) > thr_n:
        news_text = tokenizer.decode(news_text_ids['input_ids'][1:thr_n])
    else:
        news_text = news['text']
    news_text = news_text.strip()

    news_tweet_ids = tokenizer(news['tweet'])
    if len(news_tweet_ids['input_ids']) > thr_m:
        news_tweet = tokenizer.decode(news_tweet_ids['input_ids'][1:thr_m])
    else:
        news_tweet = news['tweet']
    news_tweet = news_tweet.strip()
    return news_text, news_tweet


def google_inv(test_news_dataset, source_folder, target_folder, output_folder):
    correct_detect = 0
    correct_fake = 0;correct_real = 0;wrong_fake = 0;wrong_real = 0
    correct_score = 0
    news_num = 0

    news_jsonl_files = [file for file in os.listdir(source_folder) if test_news_dataset in file]
    for news_jsonl in news_jsonl_files:
        if 'test' in news_jsonl and 'runtime_gos' not in news_jsonl:
            with open(os.path.join(source_folder, news_jsonl), 'r') as f:
                test_news_list = json.load(f)
        elif 'train' in news_jsonl:
            with open(os.path.join(source_folder, news_jsonl), 'r') as f:
                train_news_list = json.load(f)
        elif 'news_t' in news_jsonl:
            with open(os.path.join(source_folder, news_jsonl), 'r') as f:
                test_news_list = json.load(f)
            with open(os.path.join(source_folder, "selected_train_politifact.jsonl"), 'r') as f:
                train_news_list = json.load(f)
        elif 'runtime_gos_test' in news_jsonl:
            with open(os.path.join(source_folder, news_jsonl), 'r') as f:
                test_news_list = json.load(f)
            with open(os.path.join(source_folder, "selected_train_gossipcop.jsonl"), 'r') as f:
                train_news_list = json.load(f)
        elif 'alp' in news_jsonl:
            with open(os.path.join(source_folder, news_jsonl), 'r') as f:
                test_news_list = json.load(f)
            with open(os.path.join(source_folder, "selected_train_politifact.jsonl"), 'r') as f:
                train_news_list = json.load(f)

    index_list = []
    decision_list = []
    reason_list = []

    #pipe = pipeline("text-generation", model="../zephyr-7b-beta", torch_dtype=torch.bfloat16,
    #                device_map="cuda")
    model_name = "Zephyr"
    model_path = "../zephyr-7b-beta"
    pipe = load_llm(model_name, model_path)
    for index, news in tqdm(enumerate(test_news_list), total=len(test_news_list)):
        print(f'No.{index}')
        news_text, news_tweet = news_cut(news, model_path, n=256, m=50)
        news_article = ("news title: {}, news text: {}, news tweet: {}".format
                        (news['title'], news_text, news_tweet))
        googles = ''
        for i, g in enumerate(news['google']):
            googles += "{}. {}  \n".format(i, g[0])
        prompt_content = prompt_google_search.format(news_article, googles)
        prompt_infer = reliablity_ppt.format(news_article)
        try:
            decision, reliablity, reason = LLM_detected(pipe, prompt_content, prompt_infer)
            decision_list.append([decision, reliablity])
            reason_list.append(reason)
            if decision == news['label']:
                print("correct predict")
                correct_detect += 1
                if reliablity > 5:
                    correct_score += 1
                if news['label'] == 'fake':
                    correct_fake += 1
                else:
                    correct_real += 1
            else:
                print("wrong predict")
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
            print("can't load this news!!! len:{}".format(len(prompt_content)))
            print("Unexpected error:", sys.exc_info()[0])
            index_list.append(index)
            time.sleep(30)

    if len(index_list) > 0:
        for index in tqdm(index_list, total=len(index_list)):
            print(f'No.{index}')
            test_news = test_news_list[index]
            news_article = ("news title: {}, news text: {}, news tweet: {}".format
                            (test_news['title'], test_news['text'], test_news['tweet']))
            prompt_content = prompt_google_search.format(news_article, test_news['google'])
            prompt_infer = reliablity_ppt.format(news_article)
            try:
                decision, reliablity, reason = LLM_detected(pipe, prompt_content, prompt_infer)
                decision_list.insert(index, [decision, reliablity])
                reason_list.insert(index, reason)
                if decision == test_news['label']:
                    correct_detect += 1
                    if reliablity > 5:
                        correct_score += 1
                    if test_news['label'] == 'fake':
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
                        if test_news['label'] == 'fake':
                            wrong_real += 1
                        else:
                            wrong_fake += 1
                news_num += 1
            except:
                print("can't load this news!!! len:{}".format(len(prompt_content)))
                print("Unexpected error:", sys.exc_info()[0])
                decision_list.insert(index, 'no_idea')
                reason_list.insert(index, 'no_idea')

    result = {'decision': decision_list, 'reason': reason_list}
    with open(os.path.join(target_folder, 'google_result_' + test_news_dataset), 'w') as f:
        json.dump(result, f)
    acc = correct_detect / news_num
    f1 = f1score(correct_fake, wrong_fake, wrong_real)
    output_file = 'google_{}_{}.txt'.format(test_news_dataset, time.strftime("%Y%m%d%H%M%S", time.localtime()))
    print("****************************  RESULT  ********************************")
    print("")
    print("{0} Accuracy: {1:.4f} ({2}/{3})".format(test_news_dataset, acc, correct_detect, news_num))
    print(
        "correct_fake: {} correct_real: {} wrong_fake: {} wrong_real: {}".format(correct_fake, correct_real, wrong_fake,
                                                                                 wrong_real))
    print("F1 score: {0:.4f}  correct_score: {1}".format(f1, correct_score))
    print("**********************************************************************")
    with open(os.path.join(output_folder, output_file), 'w') as f:
        result = ["{0} Accuracy: {1:.4f} ({2}/{3})\n".format(test_news_dataset, acc, correct_detect, news_num),
                  "correct_fake: {} correct_real: {} wrong_fake: {} wrong_real: {}\n".format(correct_fake, correct_real,
                                                                                             wrong_fake, wrong_real),
                  "F1 score: {0:.4f}  correct_score: {1}".format(f1, correct_score)]
        print(result)
        f.writelines(result)


if __name__ == "__main__":
    start5 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='politifact', choices=['alp', 'runtime_gos_test', 'news_t', 'politifact', 'gossipcop'])
    args = parser.parse_args()
    test_news_dataset = args.dataset
    source_folder = '../dataset_full'
    target_folder = '../dataset_full/judge_result'
    output_folder = './output_all/'
    print(test_news_dataset)
    google_inv(test_news_dataset, source_folder, target_folder, output_folder)
    end5 = time.time()
    with open('time4.2_{}.txt'.format(test_news_dataset), 'w') as t:
        t.writelines('Outside Determine: {} seconds'.format(end5 - start5))