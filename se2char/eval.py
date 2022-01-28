
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from BertDecoder import BertDecoder
import logging
import os
import argparse
from sklearn import metrics
from data import load
from transformers import BertModel, BertTokenizer
import codecs
BATCH_SIZE = 16
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VOCAB_SIZE = 21128


def eval(args):
    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-chinese", do_lower_case=False)

    logging.info('Loading pre-training BERT model')
    bert = BertModel.from_pretrained("bert-base-chinese").to(DEVICE)
    logging.info('Loading model state from '+args.model_path)
    linear = BertDecoder(hidden_size=bert.config.hidden_size,
                         vocab_size=bert.config.vocab_size,
                         biLSTM=False).to(DEVICE)
    linear.load_state_dict(torch.load(
        args.model_path, map_location=torch.device(DEVICE))['model_state_dict'])
    if DEVICE != 'cpu':
        linear = nn.DataParallel(linear, device_ids=[0])
        bert = nn.DataParallel(bert, device_ids=[0])
    bert.eval()
    linear.eval()

    m = nn.Softmax(dim=2)
    logging.info('Loading testing data from ' + args.text_path)
    test = load(args.text_path)
    test_loader = DataLoader(test, batch_size=BATCH_SIZE,
                             shuffle=False, collate_fn=test.collate_fn)

    prediction = []
    reference = []
    pred = []
    ref = []
    with torch.no_grad():
        for _, d in enumerate(tqdm(test_loader)):

            input_ids, token_type_ids, attention_mask,label = [
                t.to(DEVICE) for t in d[2:]]
            
            embedding = bert(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)[0]

            predict = linear(embedding,biLSTM=False)
            
            output = torch.argmax(m(predict), dim=2)
            
            output = output.view(-1).tolist()
            target = input_ids.contiguous().view(-1).tolist()
            assert len(target) == len(output)
            prediction.extend(output)
            reference.extend(target)
            
    assert len(prediction) == len(reference)
    ref = [_a for _a, _b in zip(reference, prediction) if _a != 0]
    pred = [_b for _a, _b in zip(reference, prediction) if _a != 0]
    print(metrics.accuracy_score(ref, pred))
   

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path')
    parser.add_argument('--text_path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')
    eval(args)
