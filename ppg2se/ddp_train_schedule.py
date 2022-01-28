import random
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import torch.nn as nn
from tqdm import tqdm
import logging
import pickle
import argparse
import random
from datasets import PhoneDataset
from PhoneToEnbed import PhoneToEnbed
from transformers import AdamW
import math
import os

# for DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import matplotlib.pyplot as plt

BATCH_SIZE = 8



EPOCHS = 40
cos_loss = nn.CosineEmbeddingLoss(reduction = "mean")

# 'none' for calculate_all_output_loss
# 'mean' for calculate_mse
mse = nn.MSELoss(reduction = "mean")
cos = nn.CosineSimilarity(dim=2, eps=1e-6)

# save model
def save(model, optimizer, path):
    torch.save(
        {
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        },
        path
    )

def cal_linear_teacher_forcing_ratio(step,c):
    scheduled_ratio = 1 - (1/c) * step
    return scheduled_ratio

def sigmoid(gamma):
    if gamma < 0:
        return 1 - 1 / (1 + np.exp(-gamma))
    return 1 / (1 + np.exp(gamma))

def cal_inverse_sigmoid_teacher_forcing_ratio(iterations:int):
    steps = int((EPOCHS*iterations)/2)
    scheduled_ratios = [sigmoid(x) for x in range(-steps,steps)]
    assert len(scheduled_ratios) == EPOCHS*iterations
    return scheduled_ratios

def cal_exp_teacher_forcing_ratio(c):
    # Number : 0.0003
    # Root : 6567 (iters per epoch) * 40 (epochs) = 262680 0.9999691198
    # Root : 13133 (iters per epoch) * 40 (epochs) = 525320 0.9999845586
    return 0.9999845586**c

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def produce_pred_tgt_for_loss(tgt_output, train_predict, tgt_pad_output):
    '''
    tgt_output = [batch, seq_len(512), 768]
    train_predict = [batch, seq_len(512), 768]
    tgt_pad_output = [batch, seq_len(512)]
    '''
    tgt = tgt_output.contiguous().view(-1,768)
    output = train_predict.contiguous().view(-1,768)
    pad = tgt_pad_output.contiguous().view(-1)
    assert output.size() == tgt.size()
    tgt_embedding = tgt[~ pad]
    tgt_padding = tgt[pad]
    output_embedding = output[~ pad]
    output_padding = output[pad]
    assert tgt_padding.size() == output_padding.size()
    assert output_embedding.size() == tgt_embedding.size()
    assert (output_embedding.size(0) + output_padding.size(0)) == pad.size(0)
    return output_embedding, tgt_embedding, output_padding, tgt_padding
'''
def calculate_all_output_loss(tgt_output, predict):
    # tgt_output's 0th is embedding's 1th
    #tgt_output = [batch, seq_len(512), 768], embedding's seq len from 1 to 512
    #train_predict = [batch, seq_len(512), 768]
    # *_now : seq len from 2 to 512
    # *_before : seq len from 1 to 511 

    predict_now, predict_before, tgt_output_now, tgt_output_before = predict[:,1:], predict[:,:-1], tgt_output[:,1:], tgt_output[:,:-1]
    # cosine loss : loss = 1 - cos(x,y), weight = (1-cos(x_i,x_i-1))^2 + (1-cos(y_i,y_i-1))^2           
    #loss = 1 - cos(tgt_output ,predict)
    #weight = (1 - cos(predict_before,predict_now))**2 + (1 - cos(tgt_output_before,tgt_output_now))**2
    # mse loss : loss = mean(mse(x,y), dim=2) (divide by feature dim=768), weight = mean(mse(x_i,x_i-1), dim = 2)^2 + mean(mse(y_i,y_i-1), dim = 2)^2 (divide by feature dim=768)
    loss = torch.mean(mse(tgt_output ,predict), dim=2)
    weight = torch.mean(mse(predict_before,predict_now), dim=2)**2 + torch.mean(mse(tgt_output_before,tgt_output_now), dim=2)**2
   
    
    # add embedding's seq len (1th)'s  weight (=1)
    weight = torch.cat((torch.ones((weight.size(0))).unsqueeze(1).to(DEVICE),weight),dim=-1)
    sum_loss = sum([math.sqrt(w)*l for w, l in zip(weight.contiguous().view(-1), loss.contiguous().view(-1))])
    return sum_loss
'''
def calculate_cos_sim(tgt_output, predict, tgt_pad_output):
    output_embedding, tgt_embedding, _, _ = produce_pred_tgt_for_loss(tgt_output, predict, tgt_pad_output)
    loss = cos_loss(output_embedding, tgt_embedding, Variable(torch.Tensor(tgt_embedding.size(0)).to(DEVICE).fill_(1.0)))
    return loss


def calculate_mse(tgt_output, predict, tgt_pad_output):
    output_embedding, tgt_embedding, _, _ = produce_pred_tgt_for_loss(tgt_output, predict, tgt_pad_output)
    loss = mse(output_embedding, tgt_embedding)
    return loss

def train(args):
    
    logging.info('Loading matbn dataset')
    train = PhoneDataset('train', eps=True)
    valid = PhoneDataset('dev', eps=True)
        
    # for DDP, add sampler to let every process use different data at training phase.
    train_sampler = torch.utils.data.distributed.DistributedSampler(train, shuffle=True)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid, shuffle=False)

    train_loader = DataLoader(train, batch_size=BATCH_SIZE,
                              collate_fn=train.collate_fn, num_workers=16,
                              sampler=train_sampler)    
    valid_loader = DataLoader(valid, batch_size=BATCH_SIZE,
                              collate_fn=valid.collate_fn, num_workers=16,
                              sampler=valid_sampler)

    logging.info('Initial model')
    model = PhoneToEnbed(num_encoder_layers=12,num_decoder_layers=6,activation='gelu', DEVICE = DEVICE, max_len=1024, phone_size = 939).to(DEVICE)
    
    logging.info(f"optimizer : Adam with lr {4*(1e-5)}")
    optimizer = torch.optim.Adam(model.parameters(), lr=4*(1e-5))
    
    
    # for DDP 
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(DEVICE)
    local_rank = int(args.local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    valid_loss_min = np.Inf
    TRAIN_LOSS = {'loss':[]}
    VALID_LOSS = {'loss':[]}
    #RATIO = cal_inverse_sigmoid_teacher_forcing_ratio(len(train_loader))
    RATIO = []
    step = 0
    
    for epoch in range(1, EPOCHS+1):
        # for DDP, set sampler epoch to sync status of each process
        train_loader.sampler.set_epoch(epoch)
        valid_loader.sampler.set_epoch(epoch)

        model.train()
        train_loss = []
        
        logging.info('Training Epoch : '+str(epoch))
        for _, batch in enumerate(tqdm(train_loader)):
            '''
                phone_post = [batch, phone_seq_len(1024), 236]
                embedding = [batch, seq_len(513), 768], 包含start token
                input_ids = [batch, 512], 不包含start token
                src_key_padding_mask = [batch, phone_seq_len(1024)]
                tgt_key_padding_masks = [batch, seq_len(513)], 包含start token
                tgt_input = [batch, 0~511=(512), 768] 拿掉embedding最後一個seq (pad)
                tgt_output = [batch, 1~512=(512), 768] 拿掉embedding第一個seq (start)
                tgt_pad_input = [batch, 0~511=(512)] 
                tgt_pad_output = [batch, 1~512=(512)]
                train_predict = [batch, 512, 768]
            '''
            optimizer.zero_grad()
            
            phone_posts, embedding, src_key_padding_masks, tgt_key_padding_masks, input_ids = [t.to(DEVICE) for t in batch[:-1]]
            tgt_input, tgt_output = embedding[:,:-1,:], embedding[:,1:,:]
            tgt_pad_input, tgt_pad_output = tgt_key_padding_masks[:,:-1], tgt_key_padding_masks[:,1:]
             
            teacher_forcing_ratio = cal_exp_teacher_forcing_ratio(step)
            RATIO.append(teacher_forcing_ratio)
            

            
            train_predict = model(src=phone_posts, tgt=tgt_input, \
                src_key_padding_mask=src_key_padding_masks, \
                tgt_key_padding_mask=tgt_pad_input, \
                schedule = args.schedule, ratio = teacher_forcing_ratio)
            
            assert tgt_output.size() == train_predict.size()
           
            loss = calculate_cos_sim(tgt_output, train_predict, tgt_pad_output)
           
            
            # broadcast loss to all process
            loss_tensor = torch.tensor(loss.item()).to(DEVICE)
            all_loss = [loss_tensor.clone() for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(all_loss, loss_tensor.clone())
            mean_all_loss = torch.mean(torch.tensor(all_loss)).to(DEVICE)
            train_loss.append(mean_all_loss.item())
                        
            loss.backward()
            optimizer.step()
            
            step += 1
            
        with torch.no_grad():
            logging.info('Validating')
            val_losses = []
            model.eval()
            for _, eval_batch in enumerate(tqdm(valid_loader)):
                phone_posts, embedding, src_key_padding_masks, tgt_key_padding_masks, input_ids = [t.to(DEVICE) for t in eval_batch[:-1]]
               
                embedding_ids = eval_batch[-1]
                tgt_input, tgt_output = embedding[:,:-1,:], embedding[:,1:,:]
                tgt_pad_input, tgt_pad_output = tgt_key_padding_masks[:,:-1], tgt_key_padding_masks[:,1:]
               
                eval_predict = model(src=phone_posts, tgt=tgt_input, \
                    src_key_padding_mask=src_key_padding_masks, \
                    tgt_key_padding_mask=tgt_pad_input)
                
                assert tgt_output.size() == eval_predict.size()
                
                
                val_loss = calculate_cos_sim(tgt_output, eval_predict, tgt_pad_output)
                #val_loss = calculate_all_output_loss(tgt_output, eval_predict)
                #val_losses.append(val_loss.item())
                # broadcast loss to all process
                val_loss_tensor = torch.tensor(val_loss.item()).to(DEVICE)
                all_val_loss = [val_loss_tensor.clone() for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather(all_val_loss, val_loss_tensor.clone())
                mean_all_val_loss = torch.mean(torch.tensor(all_val_loss)).to(DEVICE)

                val_losses.append(mean_all_val_loss.item())
                
                
        logging.info("Epoch: {}/{}...Loss: {:.6f}...Val Loss: {:.6f}".format(epoch, EPOCHS,np.mean(train_loss),np.mean(val_losses)))
        TRAIN_LOSS['loss'].append(np.mean(train_loss))
        VALID_LOSS['loss'].append(np.mean(val_losses))
        
        # for DDP, only save model at master process
        if dist.get_rank() == 0:
            if np.mean(val_losses) <= valid_loss_min:
                logging.info('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,np.mean(val_losses)))
                valid_loss_min = np.mean(val_losses)
            checkpoint_path = f'{args.model_path}/ckpt.{epoch}.pt'
            save(model, optimizer, checkpoint_path)

            with open(f'{args.model_path}/train_loss.pkl', 'wb') as f:
                pickle.dump(TRAIN_LOSS, f)    
            with open(f'{args.model_path}/valid_loss.pkl', 'wb') as f:
                pickle.dump(VALID_LOSS, f)
            with open(f'{args.model_path}/ratio.pkl', 'wb') as f:
                pickle.dump(RATIO, f)
            
        
def process_command():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', dest="model_path")
    parser.add_argument('--schedule', dest='schedule', action='store_true')
    # for DDP
    parser.add_argument("--local_rank", default=-1)

    return parser.parse_args()

if __name__ == '__main__':
    
    args = process_command()
    local_rank = int(args.local_rank)

    # set up logging to only print info log on master process
    loglevel = logging.INFO if local_rank in [-1, 0] else logging.WARN
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')

    logging.info(f'Schedule: {args.schedule}')

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
    
    
    
    
    