
from re import S
from tqdm import tqdm
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset
import data_util as data_util
import preprocess_MLM as MLM
from torch.utils.data import DataLoader
import pickle
class MATBNDataset(Dataset):
    def __init__(self, file_path, tokenizer, LMmask):
        if LMmask:
            self.data = MLM.preprocess(file_path, tokenizer)
        else:
            self.data = data_util.preprocess(file_path, tokenizer)
        
    def __getitem__(self, idx):
            return (self.data[idx]['id'], self.data[idx]['context'], self.data[idx]['input_ids'], \
                self.data[idx]['token_type_ids'], self.data[idx]['attention_mask'], self.data[idx]['label'])
    
    def __len__(self):
        return len(self.data)

    def collate_fn(self, samples):
        _ids = [s[0] for s in samples]
        contexts =  [s[1] for s in samples]
        input_ids =  torch.stack([torch.LongTensor(s[2]) for s in samples])
        token_type_ids =  torch.stack([torch.LongTensor(s[3]) for s in samples])
        attention_mask =  torch.stack([torch.FloatTensor(s[4]) for s in samples])
        label = torch.stack([torch.LongTensor(s[5]) for s in samples])
        return _ids, contexts, input_ids, token_type_ids, attention_mask, label

def load(file_path:str, LMmask:bool=False):
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese", do_lower_case=False)
    dataset = MATBNDataset(file_path, tokenizer=tokenizer, LMmask=LMmask)
    return dataset




            