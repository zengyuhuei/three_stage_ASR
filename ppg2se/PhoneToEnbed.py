import torch.nn as nn
from torch.nn.init import xavier_uniform_
import torch
import math
import random


class PhoneToEnbed(nn.Module):
    def __init__(self, DEVICE, d_model: int = 768, nhead: int = 8, num_encoder_layers: int = 3,
                 num_decoder_layers: int = 3, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", max_len=1024, phone_size = 236):
        super(PhoneToEnbed, self).__init__()
        print(f'Activation Function :{activation}')
        print(f'Encoder :{num_encoder_layers}')
        print(f'Decoder :{num_decoder_layers}')
        print(f'Phone size :{phone_size}')
        print(f'Dropout :{dropout}')
        self.fc = nn.Linear(phone_size, d_model)

        self.pos_encoder = PositionalEncoding(
            d_model, dropout, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.DEVICE = DEVICE

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, src, tgt,
                memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                schedule=False, ratio=None):
        '''
        If use multiple GPUs, batch_size will be divided by the number of GPUs (for example : #GPUs = 2)
        src = [batch/2, phone_seq_len(1024), 236] -> [phone_seq_len(1024), batch/2, 236] -> [phone_seq_len(1024), batch/2, 768]
        tgt = [batch/2, seq_len(511), 768] -> [seq_len(511), batch/2, 768]
        src_key_padding_mask = [batch/2, phone_seq_len(1024)]
        tgt_key_padding_masks = [batch/2, seq_len(511)]
        tgt_mask = [seq_len(511),seq_len(511)]
        '''

        src = src.transpose(1, 0)
        tgt = tgt.transpose(1, 0)

        # phone_embedding 236 to bert embedding 768
        src = self.fc(src)
        if src.size(1) != tgt.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")

        if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
            raise RuntimeError(
                "the feature number of src and tgt must be equal to d_model")

        tgt_mask = generate_square_subsequent_mask(tgt.size(0)).to(self.DEVICE)

        # add position embedding
        src = src * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        tgt = tgt * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)

        memory = self.encoder(
            src, mask=None, src_key_padding_mask=src_key_padding_mask)

        '''
        If use multiple GPUs, batch_size will be divided by the number of GPUs
        output = [seq_len(511), batch/2, 768] -> [batch/2, seq_len(511), 768]
        After return, the size will be [batch, seq_len(511), 768]
        '''

        if schedule:
            with torch.no_grad():
                # tgt = [seq_len(511), batch, 768] no sep
                # decode_output = [seq_len(511), batch, 768] no cls
                decode_output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                             tgt_key_padding_mask=tgt_key_padding_mask,
                                             memory_key_padding_mask=src_key_padding_mask)
                # output_no_sep = [seq_len(510), batch, 768] no cls no sep
                output_no_sep = decode_output[:-1]
                # tgt_emb = [seq_len(511), batch, 768]
                tgt_emb = torch.zeros(decode_output.size()).to(self.DEVICE)
                # get CLS
                tgt_emb[0] = tgt[0]

                for i in range(output_no_sep.size(0)):
                    tgt_emb[i+1] = tgt[i + 1] if random.random() <= ratio else output_no_sep[i]
                tgt_emb = tgt_emb * math.sqrt(self.d_model)
                tgt = self.pos_encoder(tgt_emb)
               
                

        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=src_key_padding_mask)

        output = output.transpose(1, 0)
        return output


def generate_square_subsequent_mask(sz: int):
    """
    Generate a square mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        print(f'PositionalEncoding Max Length: {max_len}')
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)





if __name__ == "__main__":
    model = PhoneToEnbed(activation='gelu',DEVICE='cpu',max_len=1024)
    print(model)
    #src = torch.zeros((3, 512, 236))
    #tgt = torch.rand((3, 512, 768))
    
    #tgt_mask = generate_square_subsequent_mask(len(tgt))
    #output = model(src=src, tgt=tgt)
    #pytorch_total_params = sum(p.numel() for p in model.parameters())
    #print(pytorch_total_params)