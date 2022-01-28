from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset
import bert_data_utils as data_util
from torch.utils.data import DataLoader


class MATBNDataset(Dataset):
    def __init__(self, tokenizer,use_single_chinese_word_for_json,text, word_with_frame_num:str=None):
        self.tokenizer = tokenizer
       
        self.data = data_util.preprocess(
            text, word_with_frame_num, self.tokenizer, use_single_chinese_word_for_json)

        # print(self.data)
    def __getitem__(self, idx):
        return (self.data[idx]['id'], self.data[idx]['context'], self.data[idx]['input_ids'],
                self.data[idx]['token_type_ids'], self.data[idx]['attention_mask'])

    def __len__(self):
        return len(self.data)

    def collate_fn(self, samples):

        _ids = [s[0] for s in samples]
        context = [s[1] for s in samples]
        input_ids = torch.stack([s[2] for s in samples])
        token_type_ids = torch.stack([s[3] for s in samples])
        attention_mask = torch.stack([s[4] for s in samples])
        return _ids, context, input_ids, token_type_ids, attention_mask


def load(text, word_with_frame_num:str=None, use_single_chinese_word_for_json:bool=False):
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-chinese', do_lower_case=False)
    dataset = MATBNDataset(text=text, word_with_frame_num=word_with_frame_num, tokenizer=tokenizer,
                           use_single_chinese_word_for_json=use_single_chinese_word_for_json)
    return dataset
