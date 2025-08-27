from transformers import pipeline, AutoTokenizer


instruction = ("I need your assistance in evaluating the authenticity of a news article. Please focus on whether the content of the article is true, not on the genre and rhetoric of the article. "
               "I will provide you the news article and additional information about this news. "
               "You have to answer that [This is fake news] or [This is real news]"
               " in the first sentence of your output and give your explanation about [target news]. ")
instruction_v2 = ("I need your assistance in evaluating the authenticity of a news article " 
                  "and please assess the reliablity of your own answers. "
                  "The reliablity indicates credibility of your answer, which is a score between 1 to 10. "
                  "The higher the score, the more reliable your answer. "
                  "Score between 1 to 5 means unreliable, 5 to 10 means reliable, and 5 means uncertainty."
                  "I will provide you the news article and some example related to this news. ")
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

def prompt_generate(news, model_path, dem=None, google_use=False):
    news_text, news_tweet = news_cut(news, model_path, n=256, m=50)
    input_news = "news title: {}, news text: {}, news tweet: {}".format(
        news['title'], news_text, news_tweet)

    result_ppt = ("Your answer should include your decision, reliablity and reason for your decision. "
               "The first sentence of your answer is your decision, which must be [This is fake news] or [This is real news]. "
               "The second sentence of your answer is about reliablity like [The confidence level for my answer is <score between 1 to 10>]. "
               "Then, give your reason for your decision.\n")
    infer_reliablity = reliablity_ppt.format(input_news)

    content = ("    [target news]:\n"
               "        [input news]: [{}]\n"
               "        [output]: ") .format(input_news)

    demostration = ('I will give you some examples of news. '
                    'Your answer after [output] should be consistent with the following examples:\n')
    demostration_v2 = ('There are some examples of news that might be real or fake:\n ')
    if dem is not None:
        for i, dem_news in enumerate(dem):
            dem_news_text, dem_news_tweet = news_cut(dem_news, model_path, n=128, m=50)
            input_example = "news title: {}, news text: {}, news tweet: {}".format(
                dem_news['title'], dem_news_text, dem_news_tweet,)
            demostration += ("    [example {}]:\n"
                             "        [input news]: [{}]\n"
                             "        [output]: [This is {} news]\n").format(i + 1,
                                                                             input_example,
                                                                             dem_news['label'])
    if google_use:
        google_result = ("I will give you additional information about this news: {}".format(news['google']))
        return instruction + google_result + demostration + content

    return instruction + demostration + content, infer_reliablity



