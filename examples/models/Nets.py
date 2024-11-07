import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

class RegressionModel(nn.Module):
    def __init__(self, args):
        super(RegressionModel, self).__init__()
        self.linear1 = nn.Linear(args.num_feature, 8)
        self.drop = nn.Dropout(0.1)
        self.linear2 = nn.Linear(8, args.num_classes)
        #self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.drop(x1)
        outputs = self.linear2(x2)
        return outputs
    
class LinearAttackModel(nn.Module):
    def __init__(self, num_feature):
        super(LinearAttackModel, self).__init__()
        self.linear1 = nn.Linear(num_feature, num_feature)
        self.hidden = nn.Linear(num_feature, 4*num_feature)
        self.linear2 = nn.Linear(4*num_feature, num_feature)
        #self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x1 = torch.sigmoid(self.linear1(x)) #self.linear1(x)#
        x2 = torch.sigmoid(self.hidden(x1)) #self.hidden(x1)#
        outputs = self.linear2(x2)
        return outputs
        
class CNN(nn.Module):
    def __init__(self, args, num_classes = None):
        super(CNN, self).__init__()
        self.args = args
        # block1
        self.conv1 = nn.Conv2d(args.num_channels, 32, kernel_size=5,padding=0,
                               stride=1,
                               bias=True)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # block2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=0,
                               stride=1,
                               bias=True)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # block3
        self.columns_fc1 = nn.Linear(1024, 512)
        
        # block4
        self.fc2 = nn.Linear(512, args.num_classes if num_classes==None else num_classes)
        
    def forward(self, inputs):
        
        x = self.pool1(F.relu(self.conv1(inputs)))
            
        x = self.pool2(F.relu(self.conv2(x)))
              
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.columns_fc1(x))
           
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        
        return x
    
class CNNCifar(nn.Module):
    def __init__(self, args, num_classes=None):
        super(CNNCifar, self).__init__()
        self.args = args
        self.all_layers = 5
            
        # block 1
        self.conv1 = nn.Conv2d(args.num_channels, 32, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # block 2
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # block 3
        self.fc3 = nn.Linear(64 * 5 * 5, 512)
        
        # block 4
        self.fc4 = nn.Linear(512, 128)
        
        # block 5
        self.fc5 = nn.Linear(
                128, args.num_classes if num_classes==None else num_classes
                )
        
        
    def forward(self, inputs):
        
        x = self.pool1(F.relu(self.conv1(inputs)))
        x = self.pool2(F.relu(self.conv2(x)))
        
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc3(x))
           
        x = F.relu(self.fc4(x))
        
        x = F.dropout(x, training=self.training)
        x = self.fc5(x)
        
        return x


class RNNModel(nn.Module):
    """encoder + rnn + decoder"""

    def __init__(self, args, num_classes=None):
        super(RNNModel, self).__init__()
        self.args = args
        
        v_size = args.ntoken if num_classes==None else num_classes 
        em_dim = args.ninp

        rnn_type = args.model
        hi_dim = args.nhid
        n_layers = args.num_layers
        dropout = args.dropout
        
        self.drop = nn.Dropout(dropout) 
        self.encoder1 = nn.Embedding(v_size, em_dim)  

        #print('rnn_type: ' + str(rnn_type))
        if rnn_type in ['RNN', 'LSTM', 'GRU']:
            self.rnn2 = getattr(nn, rnn_type)(em_dim, hi_dim, 1) 
            self.rnn3 = getattr(nn, rnn_type)(hi_dim, hi_dim, 1)
        else:
            raise ValueError("""'rnn_type' error, options are ['RNN', 'LSTM', 'GRU']""")

        self.decoder4 = nn.Linear(hi_dim, v_size) 

        
        if args.tie_weights:
            print('tie_weights!')
            if hi_dim != em_dim: 
                raise ValueError('When using the tied flag, hi_dim must be equal to em_dim')
            self.decoder4.weight = self.encoder1.weight
        
        self.init_weights() 

        self.rnn_type = rnn_type
        self.hi_dim = hi_dim
        self.n_layers = n_layers
        self.all_layers = 4
            
    def forward(self, inputs, hidden):
        
        # layer1
        emb = self.encoder1(inputs)  # encoder
        self.rnn2.flatten_parameters()
        self.rnn3.flatten_parameters()
        
             
        # layer2
        if self.rnn_type == 'LSTM':
            # output维度：(seq_len, batch_size, hidden_dim)
            output, (hidden1, cell1) = self.rnn2(
                    self.drop(emb), (hidden[0][0:1,:,:], hidden[1][0:1,:,:])
                    )
        else:
            output, hidden1 = self.rnn2(self.drop(emb), hidden[0:1,:,:])  
            # output维度：(seq_len, batch_size, hidden_dim)
               
        # layer3
        if self.rnn_type == 'LSTM':
            output, (hidden2, cell2) = self.rnn3(
                    self.drop(output), 
                    (hidden[0][1:,:,:], hidden[1][1:,:,:])
                    )
        else:
            output, hidden2 = self.rnn3(self.drop(output), hidden[1:, : , :])
        
        decoded = self.decoder4(
                self.drop(output.reshape(
                        output.size(0) * output.size(1), output.size(2)
                        )
                ))  # 展平，映射 output[-1,:,:]
        output = decoded.reshape(
                output.size(0), output.size(1), decoded.size(1)
                ) # decoded
        
        if self.rnn_type == 'LSTM':
            return output, \
                    (torch.cat((hidden1,hidden2),dim=0), 
                     torch.cat((cell1,cell2),dim=0))
        else:
            return output, torch.cat((hidden1,hidden2),dim=0)
    
    def init_weights(self):
        """权重初始化，如果tie_weights，则encoder和decoder权重是相同的"""
        init_range = 0.1
        self.encoder1.weight.data.uniform_(-init_range, init_range)
        self.decoder4.weight.data.uniform_(-init_range, init_range)
        self.decoder4.bias.data.fill_(0)

    def init_hidden(self, bsz):
        """初始化隐藏层，与batch_size相关"""
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':  # lstm：(h0, c0)
            return (Variable(weight.new(self.n_layers, 
                                        bsz, self.hi_dim).zero_()),
                    Variable(weight.new(self.n_layers, 
                                        bsz, self.hi_dim).zero_()))
        else:  # gru 和 rnn：h0
            return Variable(weight.new(self.n_layers, 
                                       bsz, self.hi_dim).zero_())
        
        