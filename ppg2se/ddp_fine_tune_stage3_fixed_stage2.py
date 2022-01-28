import random
from se2char.SE2Char import SE2Char
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import logging
import pickle
import argparse
import random
from datasets import PhoneDataset
from PhoneToEnbed import PhoneToEnbed
from transformers import BertModel

# for DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

cross_entropy = nn.CrossEntropyLoss(ignore_index=0)
BATCH_SIZE = 16
VOCAB_SIZE = 21128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 10
PPG2SE_PATH = './model_state/cosine_schedule_exp_reorg_entropy_eps/ckpt.32.pt'
SE2CHAE_PATH = '../se2char/without_punc_GRU/ckpt.28.pt'

def character_loss_with_ce(input_ids, vocab_prob):
    #input_ids = [batch, seq_len(511)]
    #vocab_prob = [batch, seq_len(511), VOCAB_SIZE]
    input_ids = input_ids.contiguous().view(-1)
    vocab_prob = vocab_prob.contiguous().view(-1,VOCAB_SIZE)
    assert input_ids.size(0) == vocab_prob.size(0)
    char_loss = cross_entropy(vocab_prob, input_ids.long())
    return char_loss

# save model
def save(model, optimizer, path):
    torch.save(
        {
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        },
        path
    )

def train(args):
    logging.info('Loading matbn dataset')
    train = PhoneDataset('train',eps=True)
    valid = PhoneDataset('dev',eps=True)
        
 
    # for DDP, add sampler to let every process use different data at training phase.
    train_sampler = torch.utils.data.distributed.DistributedSampler(train, shuffle=True)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid, shuffle=False)

    train_loader = DataLoader(train, batch_size=BATCH_SIZE,
                              collate_fn=train.collate_fn, num_workers=16,
                              sampler=train_sampler)    
    valid_loader = DataLoader(valid, batch_size=BATCH_SIZE,
                              collate_fn=valid.collate_fn, num_workers=16,
                              sampler=valid_sampler)

    logging.info('Loading phone2embed model from '+PPG2SE_PATH)
    phone2embed = PhoneToEnbed(num_encoder_layers=4,num_decoder_layers=3,activation='gelu', DEVICE = DEVICE, max_len=1024, phone_size=939).to(DEVICE)
    phone2embed.load_state_dict(torch.load(PPG2SE_PATH,map_location=torch.device(DEVICE))['model_state_dict'])    
    phone2embed.to(DEVICE)
    phone2embed.eval()

    logging.info('Loading BertModel')
    bert = BertModel.from_pretrained("bert-base-chinese")

    logging.info('Loading bert_decoder state from '+SE2CHAE_PATH)
    bert_decoder = SE2Char(hidden_size=bert.config.hidden_size,
                         vocab_size=bert.config.vocab_size,
                         biLSTM=args.bilstm)
    bert_decoder.load_state_dict(torch.load(SE2CHAE_PATH,map_location=torch.device(DEVICE))['model_state_dict'])    
    bert_decoder.to(DEVICE)

    logging.info(f"optimizer : Adam with lr {(1e-5)*2}")
    optimizer = torch.optim.Adam(bert_decoder.parameters(), lr=(1e-5)*2)
     
    phone2embed = torch.nn.SyncBatchNorm.convert_sync_batchnorm(phone2embed).to(DEVICE)
    local_rank = int(args.local_rank)
    phone2embed = DDP(phone2embed, device_ids=[local_rank], output_device=local_rank)
    bert_decoder = DDP(bert_decoder, device_ids=[local_rank], output_device=local_rank)
    
    valid_loss_min = np.Inf
    TRAIN_LOSS = {'loss':[]}
    VALID_LOSS = {'loss':[]}
 
    for epoch in range(1, EPOCHS+1):
        # for DDP, set sampler epoch to sync status of each process
        train_loader.sampler.set_epoch(epoch)
        valid_loader.sampler.set_epoch(epoch)

        bert_decoder.train()
        train_losses = []
        #train_mer_losses = []
        #train_ce_losses = []
        logging.info('Training Epoch : '+str(epoch))
            
        
        for i, batch in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            phone_posts, embedding, src_key_padding_masks, tgt_key_padding_masks, input_ids = [t.to(DEVICE) for t in batch[:-1]]

            #embedding_ids = batch[-1]
            tgt_input, tgt_output = embedding[:,:-1,:], embedding[:,1:,:]
            tgt_pad_input, tgt_pad_output = tgt_key_padding_masks[:,:-1], tgt_key_padding_masks[:,1:]
            
            with torch.no_grad():
                train_predict = phone2embed(src=phone_posts, tgt=tgt_input, \
                    src_key_padding_mask=src_key_padding_masks, \
                    tgt_key_padding_mask=tgt_pad_input, schedule = args.schedule, ratio = 0)
            # train_predict = [batch, 512, 768]
            assert tgt_output.size() == train_predict.size()
            
            vocab_prob = bert_decoder(last_hidden_state = train_predict, biLSTM=args.bilstm)
            
           

            loss = character_loss_with_ce(input_ids, vocab_prob)
            

            loss_tensor = torch.tensor(loss.item()).to(DEVICE)
            all_loss = [loss_tensor.clone() for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(all_loss, loss_tensor.clone())
            mean_all_loss = torch.mean(torch.tensor(all_loss)).to(DEVICE)
            train_losses.append(mean_all_loss.item())
            

            loss.backward()
            optimizer.step()
            
        with torch.no_grad():
            logging.info('Validating')
            bert_decoder.eval()
            val_losses = []
            #val_mer_losses = []
            #val_ce_losses = []
            
            for _, eval_batch in enumerate(tqdm(valid_loader)):
           
                phone_posts, embedding, src_key_padding_masks, tgt_key_padding_masks, input_ids = \
                [t.to(DEVICE) for t in eval_batch[:-1]]

                #embedding_ids = eval_batch[-1]

                tgt_input, tgt_output = embedding[:,:-1,:], embedding[:,1:,:]
                tgt_pad_input, tgt_pad_output = tgt_key_padding_masks[:,:-1], tgt_key_padding_masks[:,1:]
               
                
                eval_predict = phone2embed(src=phone_posts, tgt=tgt_input, \
                src_key_padding_mask=src_key_padding_masks, \
                tgt_key_padding_mask=tgt_pad_input, schedule = True, ratio = 0)
                
                assert tgt_output.size() == eval_predict.size()
                
                valid_vocab_prob = bert_decoder(last_hidden_state = eval_predict, biLSTM=args.bilstm)

                
                val_loss = character_loss_with_ce(input_ids, valid_vocab_prob)
                


                # broadcast loss to all process
                val_loss_tensor = torch.tensor(val_loss.item()).to(DEVICE)
                all_val_loss = [val_loss_tensor.clone() for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather(all_val_loss, val_loss_tensor.clone())
                mean_all_val_loss = torch.mean(torch.tensor(all_val_loss)).to(DEVICE)
                val_losses.append(mean_all_val_loss.item())
                
            
            
                
        
        logging.info("Epoch: {}/{}...Loss: {:.6f}...Val Loss: {:.6f}".format(epoch, EPOCHS,np.mean(train_losses),np.mean(val_losses)))
        TRAIN_LOSS['loss'].append(np.mean(train_losses))
        VALID_LOSS['loss'].append(np.mean(val_losses))
        
        # for DDP, only save model at master process
        if dist.get_rank() == 0:
            if np.mean(val_losses) <= valid_loss_min:
                logging.info('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,np.mean(val_losses)))
                valid_loss_min = np.mean(val_losses)
                save(bert_decoder, optimizer, f'{args.model_path}/ckpt.{epoch}.pt')
            
            with open(f'{args.model_path}/train_loss.pkl', 'wb') as f:
                pickle.dump(TRAIN_LOSS, f)    
            with open(f'{args.model_path}/valid_loss.pkl', 'wb') as f:
                pickle.dump(VALID_LOSS, f)
        
        
        
def process_command():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', dest="model_path")
    parser.add_argument('--schedule', dest="schedule", action='store_true')
    parser.add_argument('--bilstm', dest="bilstm", action='store_true')
    # for DDP
    parser.add_argument("--local_rank", default=-1)
    return parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
if __name__ == '__main__':
    args = process_command()
    local_rank = int(args.local_rank)

    # set up logging to only print info log on master process
    loglevel = logging.INFO if local_rank in [-1, 0] else logging.WARN
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')


    # init DDP
    local_rank = int(args.local_rank)
    DEVICE = torch.device("cuda", local_rank)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    
    # for DDP, set different seed for different process
    setup_seed(666 + local_rank)


    if not os.path.exists(args.model_path) and local_rank == 0:
        os.makedirs(args.model_path)
    
    dist.barrier()

    train(args)
