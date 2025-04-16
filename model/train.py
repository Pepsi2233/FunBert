import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import pandas as pd
import numpy as np
import random
from sklearn.metrics import classification_report
from tqdm import tqdm
from torch import nn, optim
import math
from torch.utils.data import Dataset, random_split, DataLoader
import torch
import torch.nn.functional as F
from Module.kan import KAN


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() > 0:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed = 41
set_seed(seed)

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from transformers import AutoTokenizer, AutoModel


class MyDataset(Dataset):
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        inputs = self.sentences[index]
        label = self.labels[index]
        return inputs, label


def input_token(train_sentences, train_labels, test_sentences, test_labels):
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True)
    train_inputs = [tokenizer(sentence, return_tensors='pt')["input_ids"].squeeze() for sentence in train_sentences]
    test_inputs = [tokenizer(sentence, return_tensors='pt')["input_ids"].squeeze() for sentence in test_sentences]
    inputs = train_inputs + test_inputs
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=tokenizer.pad_token_id)
    train_inputs = inputs[:len(train_inputs)]
    test_inputs = inputs[len(train_inputs):]
    return train_inputs, train_labels, test_inputs, test_labels


class TransformerEncoder(nn.Module):
    def __init__(self, hidden_size, num_heads, num_layers, dropout):
        super(TransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=num_layers
        )

    def forward(self, src):
        # 转置源张量到 [seq_len, batch_size, hidden_size]
        src = src.permute(1, 0, 2)
        output = self.transformer_encoder(src)
        return output


class MyModel(nn.Module):
    def __init__(self, number):
        super(MyModel, self).__init__()
        self.bert = AutoModel.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True,output_attentions=True)
        self.encoder = TransformerEncoder(768, 8, 6, 0.1)
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=128, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.5)
        self.lin = nn.Sequential(
            KAN([768, 128, 512]),
            KAN([512, 256, number])
        )

    def forward(self, x):
        x = self.bert(x)[0]
        x = self.encoder(x)
        x = x.permute(1, 2, 0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x = F.max_pool1d(x, kernel_size=x.size(2))
        x = x.view(x.size(0), -1)
        x = self.lin(x)
        return x


# 训练和验证函数
def train_and_validate(model, train_loader, val_loader, num_epochs, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        best = 10

        for inputs, labels in tqdm(TrainLoader, desc='Epoch'):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            loss.backward() 
            optimizer.step()
            total_loss += loss.item()
        validate(model, val_loader, epoch, best)


def validate(model, val_loader, epoch, best):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    y_labels = []
    y_predicted = []

    with torch.no_grad():  
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            predicted = torch.argmax(outputs, dim=1)
            y_labels = y_labels + labels.tolist()
            y_predicted = y_predicted + predicted.tolist()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()
    return best


if __name__ == "__main__":
    train_file = "train.data"
    test_file = "test.data"
    train_data = pd.read_csv(train_file, sep=' ', header=None, names=['chr_id', 'sequence', 'label'])
    train_sentences = train_data["sequence"]
    train_labels = train_data["label"]
    test_data = pd.read_csv(test_file, sep=' ', header=None, names=['chr_id', 'sequence', 'label'])
    test_sentences = test_data["sequence"]
    test_labels = test_data["label"]
    train_inputs, train_labels, _, _ = input_token(train_sentences, train_labels, test_sentences, test_labels)
    ratio = 0.8
    batch_size = 32
    Train_Validate_Set = MyDataset(train_inputs, train_labels)
    Train_Set, Validate_Set = random_split(dataset=Train_Validate_Set,
                                           lengths=[math.ceil(len(Train_Validate_Set) * ratio),
                                                    len(Train_Validate_Set) -
                                                    math.ceil(len(Train_Validate_Set) * ratio)],
                                           generator=torch.Generator().manual_seed(seed))
    TrainLoader = DataLoader(dataset=Train_Set, batch_size=batch_size,
                             drop_last=True, shuffle=True, num_workers=0)
    ValidateLoader = DataLoader(dataset=Validate_Set, batch_size=batch_size,
                                drop_last=True, shuffle=True, num_workers=0)
    device = torch.device('cuda')
    model = MyModel(num).to(device)
    train_and_validate(model, TrainLoader, ValidateLoader, num_epochs=100, lr=1e-5)

