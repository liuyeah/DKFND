export CUDA_VISIBLE_DEVICES=1,3,4
#python keywords_detect.py --datafile runtime_gos_test.jsonl
python google_search.py --datafile runtime_gos_test.jsonl
python news_encode.py --datafile runtime_gos_test.jsonl