import numpy as np
from SE2Char import SE2Char
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import logging
import pickle
import argparse
from data import load
import random
from transformers import BertModel

BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VOCAB_SIZE = 21128
EPOCHS = 30
loss_function = nn.CrossEntropyLoss(ignore_index=0)
# save model


def save(model, optimizer, path):
    torch.save(
        {
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        },
        path
    )

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(args):
    print(args.mask)
    print(args.bilstm)
    train = load(args.train_path, LMmask=args.mask)
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=train.collate_fn, num_workers=16)

    valid = load(args.dev_path, LMmask=args.mask)
    valid_loader = DataLoader(valid, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=valid.collate_fn, num_workers=16)

    logging.info('Loading pre-training BERT model')
    bert = BertModel.from_pretrained("bert-base-chinese").to(DEVICE)

    bert.eval()
    logging.info('Initial model')
    linear = SE2Char(hidden_size=bert.config.hidden_size,
                         vocab_size=bert.config.vocab_size,
                         biLSTM=args.bilstm).to(DEVICE)

    optimizer = torch.optim.Adam(linear.parameters(), lr=1e-5)  
    if DEVICE != 'cpu':
        linear = nn.DataParallel(linear, device_ids=[0, 1])
        bert = nn.DataParallel(bert, device_ids=[0, 1])
    
    valid_loss_min = np.Inf

    TRAIN_LOSS = []
    VALID_LOSS = []
    

    for epoch in range(1, EPOCHS+1):
        linear.train()
        train_loss = []
        logging.info('Training Epoch : '+str(epoch))

        for _, train_batch in enumerate(tqdm(train_loader)):
            linear.zero_grad()
            input_ids, token_type_ids, attention_mask, label = [
                t.to(DEVICE) for t in train_batch[2:]]
        
            train_embedding = bert(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
            
            train_predict = linear(train_embedding, biLSTM=args.bilstm)
            loss = loss_function(train_predict.view(-1,VOCAB_SIZE),label.view(-1).long())
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            
        with torch.no_grad():
            logging.info('Validating')
            val_losses = []
            linear.eval()
            for _, eval_batch in enumerate(tqdm(valid_loader)):
                input_ids, token_type_ids, attention_mask, label = [t.to(DEVICE) for t in eval_batch[2:]]
                val_embedding = bert(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids)[0]
                val_predict = linear(val_embedding, biLSTM=args.bilstm)
                val_loss = loss_function(val_predict.view(-1,VOCAB_SIZE),label.view(-1).long())
                val_losses.append(val_loss.item())
        
        
        logging.info("Epoch: {}/{}...Loss: {:.6f}...Val Loss: {:.6f}".format(epoch, EPOCHS,np.mean(train_loss),np.mean(val_losses)))
        TRAIN_LOSS.append(np.mean(train_loss))
        VALID_LOSS.append(np.mean(val_losses))
        


        if np.mean(val_losses) <= valid_loss_min:
            logging.info('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,np.mean(val_losses)))
            valid_loss_min = np.mean(val_losses)
            checkpoint_path = f'{args.model_state_path}/ckpt.{epoch}.pt'
            save(linear, optimizer, checkpoint_path)


        with open(f'{args.model_state_path}/train_loss.pkl', 'wb') as f:
            pickle.dump(TRAIN_LOSS, f)    
        with open(f'{args.model_state_path}/valid_loss.pkl', 'wb') as f:
            pickle.dump(VALID_LOSS, f)
    


def process_command():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', dest="model_state_path", default="./model_state/")
    parser.add_argument('--train_path', dest="train_path", default="./data/train.txt")
    parser.add_argument('--dev_path', dest="dev_path", default="./data/dev.txt")
    parser.add_argument('--mask', dest='mask', action='store_true')
    parser.add_argument('--bilstm', dest='bilstm', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    setup_seed(666)
    args = process_command()
    loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')

    if not os.path.exists(args.model_state_path):
        os.makedirs(args.model_state_path)

    train(args)
