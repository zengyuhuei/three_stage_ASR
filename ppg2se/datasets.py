import torch
import numpy as np
import json
from torch.utils.data import Dataset
import random
import os
import pickle
from transformers import BertTokenizer, BertModel
from scipy.stats import entropy
import pandas as pd
np.seterr(divide='ignore',invalid='ignore')
EMBEDDING_SEQ_PED_TO = 513 # start token + (cls + tokens + sep)
EMBEDDING_DIM = 768
PAD = 0
def load_json(path: str, show_message: bool=False):
    if show_message:
        print(f"Reading json file from : {path}...")
    with open(path, 'r') as f:
        json_object = json.load(f)
        return json_object


def load_pickle(path: str, show_message: bool=False):
    if show_message:
        print(f"Reading pickle file from : {path}...")
    with open(path, 'rb') as f:
        pickle_object = pickle.load(f)
        return pickle_object
    

class PhoneDataset(Dataset):
    
    def __init__(self, mode: str, eps: bool=False):
        
        
        assert(mode in ['train', 'dev'])
        self.data_folder = f"../data/{mode}"
        self.eps = eps
        self.mode = mode
        self.FRAME_SEQ_PED_TO =  1024
        
        self.phone_posts_file = 'phone_post'
        #self.FRAME_DIM = 938
        
        self.NUM_PHONE_WITH_ENTROPY = 936
        self.slience_feature_num = 15
        self.FRAME_DIM = self.NUM_PHONE_WITH_ENTROPY + 3 # 935 + 3 + entropy
        
        # embedding 的所有id
        embedding_name = 'embedding_ids.pkl'
        
        print(f'frame feature dim: {self.FRAME_DIM}')
        print(f'max number of frames: {self.FRAME_SEQ_PED_TO}')
        if self.eps:
            print(f'Creating EPS embedding and input ids...')
            tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
            model = BertModel.from_pretrained('bert-base-chinese')      
            inputs = tokenizer("", return_tensors="pt")
            self.eps_embedding = model(**inputs).last_hidden_state.squeeze(0).tolist()
            self.eps_input_ids = inputs['input_ids'].squeeze(0).tolist()
            assert len(self.eps_embedding) == 2 
            self.eps_ids = load_json(self.data_folder+'/eps_ids.json')
       
        self.input_ids = load_json(self.data_folder+'/token_id.json')
        print(f'Loading {mode} data...')
        self.embedding_ids = load_pickle(f'{self.data_folder}/{embedding_name}')
        print(f'Total {self.mode} : {len(self.embedding_ids)}')
        
    def __len__(self):
        return len(self.embedding_ids)
    

    def __getitem__(self, idx):
        return self.embedding_ids[idx]
    
    def pad_feature(self, feature, pad_size):
        if len(feature) < pad_size:
            feature = feature + [PAD] * (pad_size - len(feature))
        return feature

   

    def pad_sequence(self, sequence, pad_size, pad_seq):
        if len(sequence) < pad_size:
            sequence = sequence + pad_seq * (pad_size - len(sequence))
        return sequence

    def add_speical_token_dim(self, sequence):
        # Add 3 dim for CLS, SEP, PAD to every sequence
        return [[0, 0, 0] + feature for feature in sequence]

    def add_cls_sep_to_dim(self, sequence, dim: int):
        assert(dim > 0)
        _cls = [1, 0, 0] + [0] * (dim - 3)
        _sep = [0, 1, 0] + [0] * (dim - 3)
        return [_cls] + sequence + [_sep] 

    def add_start_to_dim(self, sequence, dim: int):
        assert(dim > 0)
        start = [1] + [0] * (dim - 1)
        return [start] + sequence


    def generate_key_padding_mask(self, mask_size: int, total_size: int):
        # ignore padding ,padding -> 1, not padding -> 0
        return [0] * mask_size + [1] * (total_size - mask_size)

    def collate_fn(self, batch):
        phone_posts = []
        embeddings = []
        phone_post_key_padding_masks = []
        embedding_key_padding_masks = []
        embedding_ids = []
        input_ids = []
        for embedding_id in batch:
            if ".pkl" in embedding_id:
                embedding_id = embedding_id.replace(".pkl", "")
            # embedding_id: PTSNE20021114-00679_145-00696_068.108.166
            s = embedding_id.split(".")
            phone_post_id, start_idx, end_idx = s[0], int(s[1]), int(s[2])
            phone_post_path = f"{self.data_folder}/{self.phone_posts_file}/{phone_post_id}.pkl"
            embedding_path = f"{self.data_folder}/embedding_pkl/{embedding_id}.pkl"
            phone_post = load_pickle(phone_post_path)[start_idx:end_idx]
            if os.path.exists(embedding_path):
                embedding = load_pickle(embedding_path)
                input_id = self.pad_sequence(self.input_ids[embedding_id], pad_size=EMBEDDING_SEQ_PED_TO-1, pad_seq=[PAD] )
            elif embedding_id in self.eps_ids:
                embedding = self.eps_embedding
                input_id = self.pad_sequence(self.eps_input_ids, pad_size=EMBEDDING_SEQ_PED_TO-1, pad_seq=[PAD] )
            else:
                raise FileNotFoundError(f'{embedding_path} not found')
            
            
            for p in phone_post:
                # normalize entropy, and the natural logarithm is logarithm in base e
                #print(p[:3], max(p[3:]),float(entropy(p[3:])), float(entropy(p[3:]) / np.log(len(p[3:]))))
                value = float(entropy(p[self.slience_feature_num:]) / np.log(len(p[self.slience_feature_num:])))
                if pd.isna(value):
                    p.insert(len(p), 0)
                else:
                    p.insert(len(p), value)
                assert len(p) == self.NUM_PHONE_WITH_ENTROPY
               
            
            # create mask
            # phone_post need to add 2 for start and end token
            # embedding need to add 1 for start token, view [SEP] as end token
            phone_post_mask = self.generate_key_padding_mask(len(phone_post) + 2, self.FRAME_SEQ_PED_TO)
            embedding_mask = self.generate_key_padding_mask(len(embedding) + 1, EMBEDDING_SEQ_PED_TO)
            
        
            # expend 3 dim for special token to every phone post
            phone_post = self.add_speical_token_dim(phone_post)  

            # add cls and sep
            phone_post = self.add_cls_sep_to_dim(phone_post, dim=self.FRAME_DIM)
            embedding = self.add_start_to_dim(embedding, dim=EMBEDDING_DIM)

            # pad frame squence to 1024
            phone_post = self.pad_sequence(phone_post, pad_size=self.FRAME_SEQ_PED_TO, pad_seq=[[0, 0, 1] + [0] * (self.FRAME_DIM - 3) ])
            # pad embedding squence to 513
            embedding = self.pad_sequence(embedding, pad_size=EMBEDDING_SEQ_PED_TO, pad_seq=[embedding[-1]])
                
            assert len(embedding) == EMBEDDING_SEQ_PED_TO
            assert len(phone_post) == self.FRAME_SEQ_PED_TO
            # append to batch data
            phone_posts.append(phone_post)
            embeddings.append(embedding)
            phone_post_key_padding_masks.append(phone_post_mask)
            embedding_key_padding_masks.append(embedding_mask)
            embedding_ids.append(embedding_id)
            input_ids.append(input_id)
        return (torch.tensor(phone_posts), torch.tensor(embeddings), torch.tensor(phone_post_key_padding_masks).bool(), torch.tensor(embedding_key_padding_masks).bool(), torch.LongTensor(input_ids), embedding_ids)
        

