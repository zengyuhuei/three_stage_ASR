import torch
from torch import nn
from torch.utils.data import DataLoader
from bert_dataset import load
from transformers import BertModel, BertTokenizer
import os
import logging
import argparse
from tqdm import tqdm
import pickle
import json
BATCH_SZIE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def process_command():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_path', dest="text_path")
    parser.add_argument('--type', dest="type")
    parser.add_argument('--word_with_frame_num_path', dest="word_with_frame_num_path")
    parser.add_argument('--use_single_chinese_word_for_json', dest="use_single_chinese_word_for_json", action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')
    args = process_command()
    logging.info('Loading pre-training BERT model')
    model = BertModel.from_pretrained("bert-base-chinese").to(DEVICE)
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=False)
    if DEVICE != 'cpu':
        model = nn.DataParallel(model, device_ids=[0, 1])
    model.to(DEVICE)
    
    model.eval()
   
    test = load(args.text_path, args.word_with_frame_num_path, use_single_chinese_word_for_json=False)
    
    test_loader = DataLoader(test, batch_size=BATCH_SZIE, shuffle = False,
                            collate_fn=test.collate_fn, num_workers=16)
    
    # create folder to store preprocessed files
    if not os.path.isdir(f"../data/{args.type}/embedding_pkl/"):
        os.mkdir(f"../data/{args.type}/embedding_pkl/")
        
    bert_input_id = {}
    with torch.no_grad():
        for _ , d in enumerate(tqdm(test_loader)):
            _ids = d[0]
            contexts = d[1]
            input_ids, token_type_ids, attention_masks = [t.to(DEVICE) for t in d[2:]]
            
            predict = model(input_ids=input_ids,
                            attention_mask=attention_masks,
                            token_type_ids=token_type_ids)[0]
            
            for x in zip(_ids, predict, attention_masks, contexts, input_ids):
                _id, embedding, attention_mask, context, input_id = x[0], x[1].tolist(), x[2].tolist(), x[3], x[4].tolist()
                pad_index = attention_mask.count(1)
                bert_input_id[_id] = input_id[:input_id.index(0)]
                with open(f'../data/{args.type}/embedding_pkl/{_id}.pkl', 'wb') as f:
                    pickle.dump(embedding[:pad_index], f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'../data/{args.type}/token_id.json', 'w') as w:
                json.dump(bert_input_id, w, ensure_ascii=False)
    
    