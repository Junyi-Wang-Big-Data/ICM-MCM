import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from Transformer_data import final_data
import math
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# about hyperparameters
seq_len = 5             # 每个样本包含的时间步长
tar_len = 2             # 准备通过样本进行多个预测
position_len = 1000     # 最长序列长度
n_features = seq_len    # 输入的特征数，实际上为1
n_hidden = 64           # hidden_size,也就是d_model
n_layers = 2            # 层数
n_heads = 2             # 注意力头数
dropout = 0.4           # 随即失活数量
n_epochs = 500           # 训练轮数

# define the dataset about time series
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len,tar_len):
        self.data = data
        self.seq_len = seq_len
        self.tar_len = tar_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.tar_len + 1

    def __getitem__(self, idx):
        return self.data[idx:idx+self.seq_len], self.data[idx+self.seq_len:idx+self.seq_len+self.tar_len]

# 网上调整函数
def collate_fn(batch):
    # batch是一个列表，其中是一个一个的元组，每个元组是dataset中_getitem__的结果
    batch = list(zip(*batch))
    time_sery = batch[0]
    label = batch[1]
    # print(label)
    # label = label.astype(np.int32)
    # time_sery = time_sery.astype(np.int32)
    time_sery = torch.LongTensor(time_sery)
    label = torch.tensor(label,dtype=torch.float32)
    # print(label.shape)
    del batch
    return  time_sery,label

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=position_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerModel(nn.Module):
    def __init__(self, n_features, n_hidden, n_layers, n_heads, dropout):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Linear(n_features, n_hidden)
        self.position_encoding = PositionalEncoding(n_hidden)
        self.encoder_layer = nn.TransformerEncoderLayer(n_hidden, n_heads, 2048, dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, n_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(n_hidden, n_heads,dim_feedforward=2048, dropout=dropout)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, n_layers)
        self.fc = nn.Linear(n_hidden, tar_len)

    def forward(self, x):
        x = x.to(torch.float32)
        # print(x.shape)
        x = x.unsqueeze(1) 
        # print(x.shape)
        # x = torch.transpose(x)
        # print(x.shape)
        x = self.embedding(x)
        # print(x.shape)
        x = self.position_encoding(x)
        # print(x.shape)
        x = self.encoder(x)
        # print(x.shape)
        x = self.decoder(x, x)
        # print(x.shape)
        x = self.fc(x)
        # print(x.shape)
        x = x.squeeze(1)        # [batch_size, target_len]
        # print(x.shape)
        return x

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # print(len(inputs))
        inputs = torch.stack([torch.Tensor(i).to(device) for i in inputs])
        targets = torch.stack([torch.Tensor(i).to(torch.float32).to(device) for i in targets])
        # print(targets.shape)
        optimizer.zero_grad()
        outputs = model(inputs).to(torch.float32)
        # print(outputs.shape)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    return train_loss

def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = torch.stack([torch.Tensor(i).to(device) for i in inputs])
            targets = torch.stack([torch.Tensor(i).to(device) for i in targets])
            outputs = model(inputs).float()
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    return val_loss

# 加载数据
data = final_data
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
val_data = data[train_size:]

#创建数据集和数据加载器
train_dataset = TimeSeriesDataset(train_data, seq_len,tar_len)
val_dataset = TimeSeriesDataset(val_data, seq_len,tar_len)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16,collate_fn=collate_fn)


# 初始化模型和优化器
model = TransformerModel(n_features=n_features, n_hidden=n_hidden, n_layers=n_layers, n_heads=n_heads, dropout=dropout)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# 训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
train_loss_list = []
val_loss_list = []
for epoch in range(n_epochs):
    
    train_loss = train(model, train_loader, optimizer, criterion, device)
    val_loss = evaluate(model, val_loader, criterion, device)
    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)
    if epoch % 10 == 0:
        print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# plot train_loss and val_loss
plt.plot(val_loss_list,label='val')
plt.plot(train_loss_list,label='train')
plt.legend()
plt.show()

# 使用模型进行预测
model.eval()
# print(len(data))
pre_data = data
with torch.no_grad():
    for i in range(int(60/tar_len)):
        test_input = torch.Tensor(pre_data[len(pre_data)-seq_len:len(pre_data)]).unsqueeze(0).to(device)
        # print(len(pre_data))
        test_output = model(test_input)
        # print(test_output.cpu().numpy())
        res_list = test_output.cpu().numpy()[0]
        for res in res_list:
            pre_data.append(res)
        # print(test_output)
        
        # print(len(pre_data))  
# print(len(pre_data))
plt.plot(pre_data,label = 'pred')
plt.legend()
plt.show()
