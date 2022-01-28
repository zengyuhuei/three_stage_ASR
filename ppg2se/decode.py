import codecs
from transformers import BertTokenizer, BertModel
from test_datasets import PhoneTestDataset
from PhoneToEnbed import generate_square_subsequent_mask, PhoneToEnbed
import os
import logging
import torch.nn as nn
from torch.utils.data import DataLoader
import math
import torch
from tqdm import tqdm
from se2char.SE2Char import SE2Char
import argparse

BATCH_SIZE = 32
MAX_LENGTH = 513
EMBEDDING = 768
CLS = torch.tensor([1] + [0] * (EMBEDDING - 1))


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#DEVICE = 'cpu'
SE2CHAE_PATH = './model_state/fine_tune_stage3_fixed_stage2_v8_ppg_output_layer/ckpt.5.pt'
PPG2SE_PATH = './model_state/v8_ppg_output_layer/ckpt.40.pt'
mse = nn.MSELoss()
cos = nn.CosineSimilarity(dim=0, eps=1e-6)


def eval(args):
    logging.info('Loading bert_decoder state from '+SE2CHAE_PATH)
    bert = BertModel.from_pretrained("bert-base-chinese")
    bert_decoder = SE2Char(hidden_size=bert.config.hidden_size, vocab_size=bert.config.vocab_size,biLSTM=True)
    bert_decoder.load_state_dict(torch.load(
        SE2CHAE_PATH, map_location=torch.device(DEVICE))['model_state_dict'])

    logging.info('Loading phone2embedding state from '+PPG2SE_PATH)
    phone2embedding = PhoneToEnbed(num_encoder_layers=4,num_decoder_layers=3,dropout=0,activation='gelu', DEVICE = DEVICE, max_len=1024, phone_size=939)
    phone2embedding.load_state_dict(torch.load(
        PPG2SE_PATH, map_location=torch.device(DEVICE))['model_state_dict'])
     
    phone2embedding.to(DEVICE)
    bert_decoder.to(DEVICE)
    logging.info('Loading testing data')
    test = PhoneTestDataset('test', test_folder=args.test_folder)
    test_loader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=test.collate_fn, num_workers=16)

    m = nn.Softmax(dim=2)
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-chinese', do_lower_case=False)
    
    sep_result = codecs.open(f'{args.result_path}/test_out', 'w+', 'utf8')
    
    output_sentences = {}
    with torch.no_grad():
        logging.info('Testing')
        phone2embedding.eval()
        bert_decoder.eval()
        for i, test_batch in enumerate(tqdm(test_loader)):
            phone_post_ids = test_batch[0]
            phone_posts, src_key_padding_masks = \
                [t.to(DEVICE) for t in test_batch[1:]]
            bsz = len(phone_post_ids)
            tgt = torch.zeros(MAX_LENGTH, bsz, EMBEDDING).to(DEVICE)
            
            tgt[0, :, :] = CLS
            memory = phone2embedding.encoder(phone2embedding.pos_encoder(phone2embedding.fc(phone_posts.transpose(
                0, 1)) * math.sqrt(phone2embedding.d_model)), src_key_padding_mask=src_key_padding_masks)
            
        
            for t in range(1, MAX_LENGTH):
                tgt_mask = generate_square_subsequent_mask(t).to(DEVICE)
                # decode = [t, bsz, embedding(768)]
                decode = phone2embedding.decoder(phone2embedding.pos_encoder(tgt[:t] * math.sqrt(phone2embedding.d_model)),
                                                 memory,
                                                 memory_key_padding_mask=src_key_padding_masks,
                                                 tgt_mask=tgt_mask)
            
                tgt[t] = decode[t-1]
            
            # tgt = [bsz, 513, 768]
            tgt = tgt.permute(1, 0, 2)
            
            vocab_prob = bert_decoder(tgt[:,1:],biLSTM=True)
            output = torch.argmax(m(vocab_prob), dim=2)
            pred_sentences = [''.join(tokenizer.convert_ids_to_tokens(
                pred, skip_special_tokens=False)) for pred in output]
            
            for i in range(len(pred_sentences)):
                output_sentences[phone_post_ids[i]] = pred_sentences[i].split('[SEP]')[
                    0].replace('[CLS]', '')
            
        
        phone_post_id_mapping = {}
        for phone_post_id in sorted(list(output_sentences.keys()), key=lambda x: (x.split('_')[:-1], int(x.split('_')[-1]))):
            
            parent_phone_post_id = "_".join(phone_post_id.split("_")[:-1])
            if parent_phone_post_id not in phone_post_id_mapping:
                phone_post_id_mapping[parent_phone_post_id] = [phone_post_id]
            else:
                phone_post_id_mapping[parent_phone_post_id].append(
                    phone_post_id)
        
        for parent_phone_post_id in phone_post_id_mapping:
            phone_post_ids = sorted(
                phone_post_id_mapping[parent_phone_post_id], key=lambda x: (x.split('_')[:-1], int(x.split('_')[-1])))
            sep_result.write(parent_phone_post_id + ' ' +
                            "".join([output_sentences[_id] for _id in phone_post_ids]) + '\n')
    sep_result.close()  
    
def process_command():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', dest='result_path')
    parser.add_argument('--test_folder', dest='test_folder')
    return parser.parse_args()
if __name__ == '__main__':
    args = process_command()
    loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')
    eval(args)
