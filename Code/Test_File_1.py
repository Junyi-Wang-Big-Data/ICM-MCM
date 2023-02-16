import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset



class IMDBDataset(Dataset):
    def __init__(self, train=True):
        # 注意vscode中须用r+\进行地址表示：
        self.train_path = r"D:\vscode\vscode_juypter\For Pytorch Learning\for nlp dataset\aclImdb\train"
        self.test_path = r'D:\vscode\vscode_juypter\For Pytorch Learning\for nlp dataset\aclImdb\test'
        data_path = self.train_path if train else self.test_path

        # 把所有的文件名放入列表
        temp_data_path = [os.path.join(data_path, 'pos'), os.path.join(data_path, 'neg')]
        self.total_file_path = []
        for path in temp_data_path:
            # 拿到path路径下的一堆名字：
            file_name_list = os.listdir(path)
            # 将名字拼起来之后得到完整路径：(只要以txt结尾的)
            file_path_list = [os.path.join(path, i) for i in file_name_list if i.endswith(".txt")]
            self.total_file_path.extend(file_path_list)

    # 拿地址：
    def __getitem__(self, index):
        # 获取地址
        file_path = self.total_file_path[index]
        # 获取label：
        label_str = file_path.split("\\")[-2]
        label = 0 if label_str == "neg" else 1
        # 获取内容：
        text = open(file_path, encoding='utf-8').read()
        #text = tokenize(text)
        # 本来的错误：too many dimension 'str'
        # 错误原因：return text, label
        return label, text

    def __len__(self):
        return len(self.total_file_path)


# 网上调整函数
def collate_fn(batch):
    # batch是一个列表，其中是一个一个的元组，每个元组是dataset中_getitem__的结果
    batch = list(zip(*batch))
    label = batch[0]
    text = batch[1]
    #sequence = [Word2seq.transform(content, max_len) for content in text]
    sequence = torch.LongTensor(sequence)
    label = torch.LongTensor(label)
    del batch
    return label, sequence


def get_dataloader(train=True):
    imdb_dataset = IMDBDataset(train)
    data_loader = DataLoader(imdb_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
    return data_loader


class LSTM_Movie_review(nn.Module):
    def __init__(self):
        super(LSTM_Movie_review, self).__init__()
        #self.embedding = nn.Embedding(len(Word2seq), 100)
        # 加入LSTM
        # self.lstm = nn.LSTM(input_size=100, hidden_size=hidden_size,
                            #num_layers=num_layers, batch_first=True,
                            #bidirectional=bidirectional,
                            #dropout=dropout)
        # self.fc = nn.Linear(hidden_size * 2, 2)

    def forward(self, input):
        x = self.embedding(input)
        # x:[batch_size, max_len, 2*hidden_size]
        # h_n,c_n:[2*2, batch_size, hidden_size]
        x, (h_n, c_n) = self.lstm(x)
        # 获取两个方向最后一次的output进行concat：
        # 正向最后一次的输出
        output_fw = h_n[-2, :, :]
        # 反向最后一次的输出
        output_bw = h_n[-1, :, :]
        # output形状为[batch_size,hidden_size*2]
        output = torch.cat([output_fw, output_bw], dim=-1)
        out = self.fc(output)
        return F.log_softmax(out, dim=-1)