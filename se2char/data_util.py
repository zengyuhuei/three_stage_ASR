
import torch
import tqdm
import logging
import os
from transformers import BertTokenizer
import re
import random
loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')
MAX_LENGTH = 512
# 全形英文變成半形 手動

def to_dict_by_id(path):
    # 不把它合併 使用原本斷詞過後的text 
    # ㄅㄆㄇㄈ -> [UNK]    
    # ㄅ ㄆ ㄇ ㄈ -> ㄅ ㄆ ㄇ [UNK]    
    original_by_id = {}
    with open(path, 'r') as f:
        for line in f:
            tmp = line.strip().split(' ',1)
            original_by_id[tmp[0]] = tmp[1]
    return original_by_id

def produce_data_dict( _id, context, tokenizer):
    #context += '。'
    data = {}
    encode_dict = tokenizer.encode_plus(
            context,
            add_special_tokens = True,
            max_length = MAX_LENGTH,
            padding = 'max_length',
            truncation=True
    )
    data['id'] = _id
    data['context'] = context
    data['token_type_ids'] = torch.LongTensor(encode_dict['token_type_ids'])
    data['attention_mask'] = torch.FloatTensor(encode_dict['attention_mask'])
    data['input_ids'] = torch.LongTensor(encode_dict['input_ids'])
    data['label'] = torch.LongTensor(encode_dict['input_ids'])

    
    return data

def preprocess(path, tokenizer):
    
    dataset = []
    original_by_id = to_dict_by_id(path)
    logging.info(f"Preprocessing ： {path}")
    for _id, v in tqdm.tqdm(original_by_id.items()):
        dataset.append(produce_data_dict( _id, original_by_id[_id], tokenizer))
    return dataset

