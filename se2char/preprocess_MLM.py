
from transformers import BertTokenizer
import re
import random
import torch
from tqdm import tqdm
import pickle
import os
import logging
loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                    level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')

MAX_LENGTH = 512


def is_only_english_letter(text):
    reg = re.compile(r"^[A-Za-z]+$")
    if reg.match(text):
        return True
    else:
        return False


def load_pickle(path: str, show_message: bool = False):
    if show_message:
        print(f"Reading pickle file from : {path}...")
    f = open(path, 'rb')
    pickle_object = pickle.load(f)
    return pickle_object


def to_dict_by_id(path):
    original_by_id = {}
    with open(path, 'r') as f:
        for line in f:
            tmp = line.strip().split(' ', 1)
            original_by_id[tmp[0]] = tmp[1]
    return original_by_id


def get_english_idx(text):
    res = re.search(r'[A-Za-z]+', text)
    if res:
        return res.span()
    else:
        return None


def split_word(word: str):
    t = []
    if is_only_english_letter(word):
        return [word]
    elif not get_english_idx(word):
        return [w for w in word]
    else:
        if re.findall(r'[\u4e00-\u9fff]+[a-zA-Z]+', word):
            # 處理類似 '卡拉OK' 的 token
            chinese_word = re.findall(r'[\u4e00-\u9fff]+', word)[0]
            t.extend([w for w in chinese_word])
            eng_word = re.findall(r'[a-zA-Z]+', word)
            t.extend(eng_word)
        elif re.findall(r'[a-zA-Z]+[\u4e00-\u9fff]+', word):
            # 處理類似 'X光' 的 token
            eng_word = re.findall(r'[a-zA-Z]+', word)
            t.extend(eng_word)
            chinese_word = re.findall(r'[\u4e00-\u9fff]+', word)[0]
            t.extend([w for w in chinese_word])
        return t


def split_lexicon(lexicon):
    splited_lexicons = []
    # chinese lexicon will end with digits
    start = 0
    for i in re.finditer('[0-9]', lexicon):
        splited_lexicons.append(lexicon[start:i.end()])
        start = i.end() + 1
    return splited_lexicons


def produce_data_dict_for_MLM(type:str, _id:str, text:str, tokenizer, random_words_list: list, lexicon_word_pair: dict, word_lexicon_pair: dict):

    preprocess_text = []
    text_len = 0
    label = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        padding = 'max_length',
        truncation=True
    )
    if type == 'test':
        return {
            'id': _id,
            'context' :text,
            'token_type_ids': label['token_type_ids'],
            'attention_mask': label['attention_mask'],
            'input_ids': label['input_ids'],
            'label': label['input_ids']  
        }
    else:
        label = label['input_ids']
        for i in text.split():
            only_chinese_i = re.sub('[A-Z]+', '', i)
            if only_chinese_i:
                if only_chinese_i in word_lexicon_pair:
                    exist_in_lexicon = True
                    lexicons = split_lexicon(word_lexicon_pair[only_chinese_i])
                else:
                    exist_in_lexicon = False
            words = split_word(i)
            chinese_idx = 0
            for word in words:
                text_len += 1
                if text_len > 510:
                    break
                if is_only_english_letter(word) or not exist_in_lexicon:
                    preprocess_text.append(word)
                else:
                    if torch.rand(1) < 0.15:
                        # 10% of the time, masking
                        if torch.rand(1) < 0.5:
                            # 50% of the time, replace with random word
                            preprocess_text.append(random.sample(random_words_list, 1)[0])  
                        else:
                            # 50% of the time, replace with homophone word which is in BERT vacab
                            homophone_words = lexicon_word_pair[lexicons[chinese_idx]]
                            homophone_words_in_bert_vocabs = list(set(homophone_words) & set(random_words_list))
                            if homophone_words_in_bert_vocabs:
                                preprocess_text.append(random.sample(homophone_words_in_bert_vocabs, 1)[0])
                            else:
                                preprocess_text.append(word)
                    else:
                        preprocess_text.append(word)
                    chinese_idx += 1
            if text_len > 510:
                    break
        
        encode_dict = tokenizer.encode_plus(
            ' '.join(preprocess_text),
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            padding = 'max_length'
        )
        assert len(encode_dict['input_ids']) == MAX_LENGTH
        assert len(label) == MAX_LENGTH
        
        return {
            'id': _id,
            'context' :text,
            'label': label,
            'token_type_ids': encode_dict['token_type_ids'],
            'attention_mask': encode_dict['attention_mask'],
            'input_ids': encode_dict['input_ids']
        }
        
    

def preprocess(file_path: str, tokenizer):

    dataset = []
    logging.info(f"Loading text from {file_path}...")
    original_by_id = to_dict_by_id(file_path)
    logging.info(f"Loading lexicon ...")
    lexicon_word_pair = load_pickle(f'data_for_MLM/lexicon_word_pair.pkl')
    word_lexicon_pair = load_pickle(f'data_for_MLM/word_lexicon_pair.pkl')

    logging.info(f"Creating vocab list from BERT's vocab...")
    random_words_list = []
    for k, v in tokenizer.get_vocab().items():
        if v in [i for i in range(670, 7992)]:
            random_words_list.append(k)
    logging.info(f"Preprocess...")
    for _id, v in tqdm(original_by_id.items()):
        dataset.append(produce_data_dict_for_MLM(
            type,_id, original_by_id[_id], tokenizer, random_words_list, lexicon_word_pair, word_lexicon_pair))
    return dataset

