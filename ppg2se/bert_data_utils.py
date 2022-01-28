#from kaldiio import ReadHelper
from tqdm import tqdm
import json
import pickle
import re
import os
from copy import deepcopy
import torch
import logging
from glob import glob
from transformers import BertTokenizer
import itertools
from random import sample
from scipy.stats import entropy
import numpy as np
PUNTUATIONS = ["，", "。", "？","、","！"]
SPECIAL_TOKEN = ["<UNK>"]
TOKEN_LIMIT = 510
FRAME_LIMIT = 1022
INPUT_MAX_LENGTH = 512
import pandas as pd

loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')

def load_ark(path: str, ppg_dim:int=233, show_loading_progess: bool=False):
    post = {}
    with open(path, 'r') as phone_post:
        if show_loading_progess==True:
            print(f"Reading ark file from : {path}...")
            phone_post = tqdm(phone_post.readlines())
        for line in phone_post:
            if line.startswith(" "):
                posterior = line.strip().split(' ')
                if posterior[-1] == ']':
                    posterior.pop(-1)
                posterior = [float(prob) for prob in posterior]
            
                assert len(posterior) == ppg_dim
                post[_id].append(posterior)
            else:
                _id = line.strip().split(' ')[0]
                if _id in post.keys() and show_loading_progess:
                    print(f"Key:{_id} is duplicated.")
                else:
                    post[_id] = []
    return post

def load_json(path: str, show_message: bool=False):
    if show_message:
        print(f"Reading json file from : {path}...")
    with open(path, 'r') as f:
        json_object = json.load(f)
        return json_object

def load_pickle(path: str, show_message: bool=False):
    if show_message:
        print(f"Reading pickle file from : {path}...")
    f = open(path, 'rb')
    pickle_object = pickle.load(f)
    return pickle_object

def load_dict_from_pickle(key_path: str, value_path: str):
    keys = load_pickle(f"{key_path}")
    values = load_pickle(f"{value_path}")
    dict_object = {k:v for k,v in zip(keys, values)}
    return dict_object

def load_text(path: str, use_single_chinese_word_for_json: bool=True):
    print(f"Reading txt file from : {path}...")
    text = {}
    with open(path, 'r') as f:#f.readlines()
        for l in f.readlines():
            l = l.split()
            _id = l[0]
            t = []
            if use_single_chinese_word_for_json:
                # split chinese charaters, keep english chcaraters/special token grouped.
                for _t in l[1:]:
                    if is_only_english_letter(_t) or _t in SPECIAL_TOKEN:
                        t.append(_t)
                    # if it has no english charater
                    elif not get_english_idx(_t):
                        for chinese_char in _t:
                            t.append(chinese_char)
                    else:
                        # split english and non-enlish char and append them to list in order
                        english_idx = get_english_idx(_t)
                        if english_idx[0] == 0:
                            # start with english
                            t.append(_t[:english_idx[1]])
                            for chinese_char in _t[english_idx[1]:]:
                                t.append(chinese_char)
                        else:
                            # start with non-english
                            for chinese_char in _t[:english_idx[0]]:
                                t.append(chinese_char)
                            t.append(_t[english_idx[0]:])
            else:
                # keep the Hyphenation in text file
                t = [_t for _t in l[1:]]
            text[_id] = t
    return text

def split_sentences(original_text, word_with_frame_idx):
    puntuation_locations = []
    for p in PUNTUATIONS:
        puntuation_locations.extend([i for i, v in enumerate(original_text) if v == p])
    puntuation_locations.sort()

    try:
        assert(len(original_text)-len(puntuation_locations) == len(word_with_frame_idx))
    except AssertionError:
        '''
        print(len(original_text), len(puntuation_locations), len(word_with_frame_idx))
        print(original_text)
        print(puntuation_locations)
        print(''.join([w['word'] for w in word_with_frame_idx]))
        '''
        print(original_text, word_with_frame_idx)
        raise AssertionError
    
    # add if the paragraph is not end with puntuation
    if len(original_text) - 1 not in puntuation_locations:
        original_text.append('。')
        puntuation_locations.append(len(original_text) - 1)
    sentences = []
    start_idx = 0
    
    for idx, puntuation_location in enumerate(puntuation_locations):
        sentences.append(word_with_frame_idx[start_idx:puntuation_location-idx])
        start_idx = puntuation_location-idx
    return sentences


def is_only_english_letter(text):
    reg = re.compile(r"^[A-Za-z]+$")
    if reg.match(text):
        return True
    else:
        return False

def get_english_idx(text):
    res = re.search(r'[A-Za-z]+', text)
    if res:
        return res.span()
    else:
        return None


def compose_sentences(sentences:list):
    composed_sentences = []
    for sentence_start_idx in range(len(sentences)):
        composed_sentence = []
        for sentence_end_idx in range(sentence_start_idx, len(sentences)):
            composed_sentence.extend(sentences[sentence_end_idx])
            # terminate when composed sentences is longer than max token length(512 - 2)
            if len(composed_sentence) > TOKEN_LIMIT:
                break
            else:
                composed_sentences.append(deepcopy(composed_sentence))
    return composed_sentences

def get_dict_id_sentence(txt:list, word_with_frame_idx:list):
    # return dict {'_id':id_start_end,'sent':character with space}
    data = {}
    ids = list(txt.keys())
    sentences_with_idx = {}
    composed_sentences = {}
    original_sentences_count = 0
    accepted_sentences_count = 0
    for _id in tqdm(ids):
        sentences_with_idx[_id] = split_sentences(txt[_id], word_with_frame_idx[_id])
        composed_sentences[_id] = compose_sentences(sentences_with_idx[_id])
        original_sentences_count += len(composed_sentences[_id])
        for s in composed_sentences[_id]:
            if s[-1]["end"] - s[0]["start"] <= FRAME_LIMIT:
                # make sure that the number of frame is less than 1022(1024 - 2)
                accepted_sentences_count += 1
                data[f'{_id}.{s[0]["start"]}.{s[-1]["end"]}'] = " ".join([c['word'] for c in s])
    print(f"sentences count after composition: {accepted_sentences_count}/{original_sentences_count}")
    return data

def produce_input_dict(_id:str, text:str, tokenizer):
   
    encode_dict = tokenizer.encode_plus(
            text,
            add_special_tokens = True,
            max_length = INPUT_MAX_LENGTH,
            padding = 'max_length',
            truncation=False
    )
   
    #assert (len([x for x in encode_dict['input_ids'] if x != 0]) == len(text.split(' '))+2)
    return {
                    'id' : _id,
                    'context' : text,
                    'token_type_ids' : torch.LongTensor(encode_dict['token_type_ids']),
                    'attention_mask' : torch.FloatTensor(encode_dict['attention_mask']),
                    'input_ids' : torch.LongTensor(encode_dict['input_ids'])
            }

def preprocess(text_path:str, word_with_frame_num_path:str, tokenizer, use_single_chinese_word_for_json: bool=True):
    logging.info("Preprocessing dataset...")
    logging.info("Loading Txt...")

    txt = load_text(text_path, use_single_chinese_word_for_json)
    logging.info("Text Loaded, Loading Word with Frame...")
    word_with_frame_idx = load_json(word_with_frame_num_path)
    assert(len(txt.keys()) == len(word_with_frame_idx.keys()))
    logging.info("Word with Frame Loaded, get_dict_id_sentence ...")
    data = get_dict_id_sentence(txt, word_with_frame_idx)
    logging.info("get_dict_id_sentence Finished...")
    
    logging.info("Produce_input_dict...")
    dataset = []
    for _id in tqdm(data.keys()):
        dataset.append(produce_input_dict(_id, data[_id], tokenizer))
    return dataset


def split_phone_posts(path: str, frame_limit: int=FRAME_LIMIT, position_dependent: bool=False, show_message: bool=False) -> list:
    if position_dependent:
        feature_dim = 939
        slience_feature_num = 15
    else:
        feature_dim = 237
        slience_feature_num = 3
    pkl_paths = glob(f"{path}/*")
    splited_phone_posts = {}
    if show_message:
        print(f'Frame Limit : {frame_limit}')
        print(f'Feature Dim : {feature_dim}')
        print(f'Silence Feature Num : {slience_feature_num}')
        pkl_paths = tqdm(pkl_paths)
    for json_path in pkl_paths:
        phone_post = load_pickle(json_path)
        for p in phone_post:
            # normalize entropy, and the natural logarithm is logarithm in base e
            #print(p[:3], max(p[3:]),float(entropy(p[3:])), float(entropy(p[3:]) / np.log(len(p[3:]))))
            value = float(entropy(p[slience_feature_num:]) / np.log(len(p[slience_feature_num:])))
            if pd.isna(value):
                p.insert(len(p), 0)
            else:
                p.insert(len(p), value)
            assert len(p) == feature_dim - 3 
        phone_post_id = json_path.split("/")[-1].replace(".pkl","")
        splited_phone_post_lens = []
        idx = 0
        # append splited phone posts
        for idx in range(0, int(len(phone_post)/frame_limit)+1):
            try:
                splited_phone_posts[f"{phone_post_id}_{idx}"] = phone_post[idx * frame_limit:(idx+1) * frame_limit]
            except IndexError:
                # handle last phone post with length < frame_limit
                splited_phone_posts[f"{phone_post_id}_{idx}"] = phone_post[idx * frame_limit:]
            splited_phone_post_lens.append(len(splited_phone_posts[f"{phone_post_id}_{idx}"]))
        assert sum(splited_phone_post_lens) == len(phone_post)
    for phone_post_id in splited_phone_posts:
        assert len(splited_phone_posts[phone_post_id]) <= frame_limit
    return splited_phone_posts
    

def generate_eps_data(path:str, frame_limit:int, sample_num:int=0):
    text = load_json(path,False)
    frame = []
    for key, values in text.items():
        for v in values:
            for i in range(v['start'],v['end'],frame_limit):
                if v['end'] - v['start'] < 5:
                    break
                if i+frame_limit > v['end']:
                    frame.append(f"{key}.{i}.{v['end']}")
                else:
                    frame.append(f"{key}.{i}.{i+frame_limit}")
    if sample_num:
        frame = sample(frame, sample_num)
    return frame


    