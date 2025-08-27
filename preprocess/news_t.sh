export CUDA_VISIBLE_DEVICES=0,3,4
python keywords_detect.py --datafile news_t.jsonl
python google_search.py --datafile news_t.jsonl
python news_encode.py --datafile news_t.jsonl