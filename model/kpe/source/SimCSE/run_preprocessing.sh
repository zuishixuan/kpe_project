export DATA_DIR='../data/data.xlsx'
#export DATA_DIR_CSV='../data/cnki/data.csv'
#export MODEL_DIR='../../model/chinese-bert-wwm-ext'
#export MODEL_OUT_DIR="../model/SimCSE/cnki/out"
export TOKENIZER_NAME='xlnet-base-cased'
export MAX_LENGTH=5

export DATA_DIR_CSV='../../data/cnki_2/data.csv'
export MODEL_DIR='../../model/chinese_roberta_wwm_ext_pytorch'
export MODEL_OUT_DIR="../../model/SimCSE/cnki_2/out"

# Creates jsonl files for train and dev

# python train_unsup.py ./data/news_title.txt ./path/to/huggingface_pretrained_model
# python train_unsup.py ../data/data3.csv --pretrained ./model/chinese-bert-wwm-ext --model_out ./model/out
# python train_unsup.py --train_file ../../data/cnki_2/data.csv --pretrained ../../model/chinese_roberta_wwm_ext_pytorch --model_out ../../model/SimCSE/cnki_2/out
python train_unsup.py --train_file $DATA_DIR_CSV --pretrained $MODEL_DIR --model_out $MODEL_OUT_DIR