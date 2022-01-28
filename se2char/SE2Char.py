import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
class SE2Char(nn.Module):
    def __init__(self, hidden_size:int, vocab_size:int, biLSTM:bool=False, dropout:float=0.1):
        super(SE2Char, self).__init__()
        print(f'biLSTM :{biLSTM}')
        print(f'Dropout :{dropout}')
        if biLSTM:
            self.lstm = nn.GRU(
                input_size  = hidden_size,
                hidden_size = 2048,
                dropout=0.2,
                num_layers =  2,
                batch_first = True,
                bidirectional  = True,
            )
            self.dropout = nn.Dropout(p=dropout)
            self.linear = nn.Linear(2048*2, vocab_size)
        else:
            self.linear = torch.nn.Sequential(
                                torch.nn.Linear(hidden_size, 1024),
                                torch.nn.Dropout(0.2),           # drop 50% neurons
                                torch.nn.ReLU(),
                                torch.nn.Linear(1024, 2048),
                                torch.nn.Dropout(0.2),
                                torch.nn.ReLU(),
                                torch.nn.Linear(2048, vocab_size)
                        )
        
    def forward(self, last_hidden_state, biLSTM:bool=False):
        if biLSTM:
            self.lstm.flatten_parameters()
            # model的參數在gpu memory上的位置是連續的
            self.lstm.flatten_parameters()
            output, _ = self.lstm(last_hidden_state, None)
            tag_space = self.linear(output)
        else:
            tag_space = self.linear(last_hidden_state)
        return tag_space

if __name__ == "__main__":
    model = BertDecoder(768,21128, biLSTM=False,dropout=0.1)
    print(model)
    # 196,711,048 vs 46,177,928 (2 GRU l linear vs 3 linear)

    # 2 layer bi GRU with hidden size 2048, linear 2048*2 to 21128 : 196711048, 
    # 86,561,416 for linear
    # 110,149,632 for GRU

    # 2 layer bi GRU with hidden size 1024, linear 1024*2 to 21128 : 73,200,264
    
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    
