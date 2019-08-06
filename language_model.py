
import zipfile
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import numpy as np

#统计预测单词，可以直接对未解压的文件进行操作
def read_data_zip(path,flag=0):
    '''
    :param filename: lm_dev,lm_test,lm_train
    :param flag: 0:lm_dev,1:lm_test,2:lm_train,3:prevsent_dev,4:prevsent_test,5:prevsent_train
    '''
    with zipfile.ZipFile(path) as f:
        flist=[]
        for i in range(6):
            flist.append(f.namelist()[i])
        if flag==flag:
            context=[word.lower() for word in f.read(f.namelist()[flag]).decode('utf-8').split() if word !='<s>']
            return context
path_zip='language.zip'
print('dev predict_num:{},test predict_num:{}'.format(len(read_data_zip(path_zip,0)),len(read_data_zip(path_zip,1))))

import collections
#词到索引
def word2idx(path_zip,flag=2):
     with zipfile.ZipFile(path_zip) as f:
        #训练集的词频统计
        context=[word.lower() for word in f.read(f.namelist()[flag]).decode('utf-8').split() if word !='<s>']
        word_set=collections.Counter(context)
        word_set=sorted(word_set.items(), key=lambda x: x[1],reverse=True)#按频度降序排列
        vocab_dict={'unk':0,'<s>':1,'pad':2}
        idx_word=3
        for word,_ in word_set:
            vocab_dict[word]=idx_word
            idx_word+=1
        return vocab_dict
word_idx=word2idx(path_zip)
# 索引到词
def idx2word(word_idx):
    idx_word={}
    for i,word in enumerate(word_idx.keys()) :
        idx_word[i]=word
    return idx_word
idx_word=idx2word(word_idx)

#对解压后的文件进行操作
def read_data(flag=0):
    '''
    :param flag: 0:lm_train,1:lm_dev,2:lm_test,3:prevsent_dev,4:prevsent_test,5:prevsent_train
    
    '''
    if flag==0:
        filename='language/bobsue.lm.train.txt'
    elif flag==1:
        filename='language/bobsue.lm.dev.txt'
    else:
        filename='language/bobsue.lm.test.txt'
    max_len=0
    context=[]
    with open(filename,'r') as f:
        lines=f.readlines()
        for line in lines:
            line=line.lower().split()
            if max_len<len(line):
                max_len=len(line)
            context.append(line)
    return context,max_len

#定义dataset 生成训练集
class LangDataset(Dataset):
    def __init__(self,flag,time_steps,word_idx):
        self.datas,self.max_len=read_data(flag)
        self.t=time_steps
        self.word_idx=word_idx
        self.samples,self.labels,self.masks=self.gen_data(self.t)
    def gen_data(self,t):
        samples=[]
        labels=[]
        masks=[]
        for data in self.datas:
            for i in range(self.max_len-self.t):
                try: 
                    if data[i+1]:
                        sample=data[i:i+t]
                        label=data[i+1:i+t+1]
                        mask=[1]*len(label)+[0]*(t-len(label))
                        samples.append(sample+[0]*(t-len(sample)))
                        labels.append(label+[0]*(t-len(label)))
                        masks.append(mask)
                except IndexError:
                    break
        return samples,labels,masks 
    def __len__(self):
        return len(self.samples)   
    def __getitem__(self,idx):
        sample=np.array([word_idx.get(i,0) for i in self.samples[idx]]).reshape(-1)
        label=np.array([word_idx.get(i,0) for i in self.labels[idx]]).reshape(-1)
        mask=np.array([int(i) for i in self.masks[idx]],dtype=np.float32).reshape(-1)
        sample_label={'sample':sample,'label':label,'mask':mask}
        return sample_label 
 
#训练和测试

time_steps=4
batch_size=64
num_epoch=10
embedding_size=128
hidden_size=200
n_layers=2
vocab_size=len(word_idx.keys())
print(vocab_size)
drop_out=0.4

train_data= LangDataset(flag=0,time_steps=time_steps,word_idx=word_idx)
dataloader=DataLoader(train_data,batch_size=batch_size,num_workers=4,shuffle=False)

class RNN_Model(nn.Module):
    def __init__(self,rnn_type,vocab_size,embedding_size,hidden_size,n_layers,drop_out):
        super(RNN_Model, self).__init__()
        self.rnn_type = rnn_type
        self.vocab_size = vocab_size #1426
        self.embedding_size = embedding_size #128
        self.hidden_size = hidden_size #256
        self.n_layers = n_layers #2
        self.drop_out = drop_out #0.5
        self.drop = nn.Dropout(drop_out)
        self.encoder = nn.Embedding(vocab_size, embedding_size) #1426 128
        if rnn_type in ['LSTM', 'GRU']: # 下面代码以LSTM举例
            self.rnn = getattr(nn, rnn_type)(input_size=embedding_size, 
                                             hidden_size=hidden_size,
                                             num_layers=n_layers, 
                                             dropout=drop_out,
                                             batch_first=True,
                                            )
        self.decoder = nn.Linear(hidden_size, vocab_size)# 最后线性全连接隐藏层的维度(200,1421)
        self.init_weights()
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)   
    def forward(self, input):
        emb = self.drop(self.encoder(input)) #(5,1)->(5,128) 5是time_steps
        output,hidden = self.rnn(emb)#(5,128) output->(5,256) hidden,cell->(num_layers,256)
#         print(output.size())
#         print(hidden.size())
        output = self.drop(output)
#         print(output.size(0))#(64)
        decoded = self.decoder(output.contiguous().view(output.size(0)*output.size(1), output.size(2)))
        #output of shape (seq_len, batch, num_directions * hidden_size)
        #h_n of shape (num_layers * num_directions, batch, hidden_size)
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden
    def init_hidden(self, bsz, requires_grad=True):
        # 这步我们初始化下隐藏层参数
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros((self.n_layers, bsz, self.hidden_size), requires_grad=requires_grad),
                    weight.new_zeros((self.n_layers, bsz, self.hidden_size), requires_grad=requires_grad))           
        else:
            return weight.new_zeros((self.nlayers, bsz, self.hidden_size), requires_grad=requires_grad)
            # GRU神经网络把h层和c层合并了，所以这里只有一层。
            
import random
random.seed(53113)
np.random.seed(53113)
torch.manual_seed(53113)
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    torch.cuda.manual_seed(53113)
model=RNN_Model('LSTM',vocab_size,embedding_size,hidden_size,n_layers,drop_out=drop_out)
if USE_CUDA:
    try:
        model=model.cuda()
    except RuntimeError:
        model=model.cpu()
        
loss_fn = nn.CrossEntropyLoss(reduction='none') # 交叉熵损失
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#一次性预测
def evaluation_one_times(flag):
    model.eval()
    if flag==1:#dev
        dev_label=[word_idx.get(i,0) for i in read_data_zip(path_zip,0)]
        dev,_=read_data(flag)
        result=[]
        with torch.no_grad():
            for line in dev:
                pre_len=len(line)-1
                input_data=torch.LongTensor([word_idx.get(i,0) for i in line][:-1]).unsqueeze(0)
                output,(hidden,cell)=model(torch.LongTensor(input_data))
                result+=list(output.numpy().argmax(2).reshape(-1))
#                 print(input_data)
#                 print(output.numpy().argmax(2).shape)
            num=0
            for i in range(7957):
                if result[i]==dev_label[i]:
                    num=num+1
                else:
                    pass
        model.train()
        return num/7957
    elif flag==2:#dev
        test_label=[word_idx.get(i,0) for i in read_data_zip(path_zip,1)]
        test,_=read_data(flag)
        result=[]
        with torch.no_grad():
            for line in test:
                pre_len=len(line)-1
                input_data=torch.LongTensor([word_idx.get(i,0) for i in line][:-1]).unsqueeze(0)
                output,(hidden,cell)=model(torch.LongTensor(input_data))
                result+=list(output.numpy().argmax(2).reshape(-1))
#                 print(input_data)
#                 print(output.numpy().argmax(2).shape)
            num=0
            for i in range(8059):
                if result[i]==test_label[i]:
                    num=num+1
                else:
                    pass
        model.train()
        return num/8059
#训练

GRAD_CLIP=1
 # 训练模式
dev_acc_list=[]
for epoch in range(num_epoch):
    model.train()
    it=iter(dataloader)
#     hidden = model.init_hidden(batch_size)
    for i,data in enumerate(it):
#         model.init_hidden(len(data['sample']))
        if model.cpu():
            train_x, train_y,train_mask = data['sample'], data['label'],data['mask']
        else:
            train_x, train_y,train_mask = data['sample'].cuda(), data['label'].cuda(),data['mask'].cuda()
        model.zero_grad()
        output,(hidden,cell) = model(train_x)
        loss = loss_fn(output.view(-1, vocab_size), train_y.view(-1)).mul(train_mask.view(-1)).sum()/train_mask.view(-1).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        if i % 100 == 0:
            print("epoch", epoch, "iter", i, "loss", loss.item(),'dev acc',evaluation_one_times(1),'test acc',evaluation_one_times(2))
