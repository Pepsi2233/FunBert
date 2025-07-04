import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, matthews_corrcoef, roc_curve, auc
from sklearn.preprocessing import label_binarize
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
        src = src.permute(1, 0, 2)
        output = self.transformer_encoder(src)
        return output


class MyModel(nn.Module):
    def __init__(self, number):
        super(MyModel, self).__init__()
        self.bert = AutoModel.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True)
        self.encoder = TransformerEncoder(768, 8, 6, 0.1)
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=128, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.5)
        self.lin = nn.Sequential(
            KAN([256, 128, 512]),
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
        x = F.max_pool1d(x, kernel_size=x.size(2)) 
        x = x.view(x.size(0), -1)
        x = self.lin(x)
        return x


def test(model, test_loader, device):
    model.to(device)
    model.eval()  
    correct = 0
    total = 0
    y_labels = []
    y_predicted = []
    y_probs = []

    with torch.no_grad(): 
        for inputs, labels in tqdm(test_loader, colour='red'):
            inputs, labels = inputs.to(device), labels.to(device) 
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1) 
            probs = torch.softmax(outputs, dim=1) 
            y_labels = y_labels + labels.tolist()
            y_predicted = y_predicted + predicted.tolist()
            y_probs.extend(probs.cpu().numpy())  
            total += labels.size(0) 
            correct += (predicted == labels).sum().item() 

    accuracy = correct / total 
    print(f'Test Accuracy: {accuracy}')
    report = classification_report(y_labels, y_predicted, zero_division=0, output_dict=True)
    print("Precision:", report['macro avg']['precision'])
    print("Recall:", report['macro avg']['recall'])
    print("F1-score:", report['macro avg']['f1-score'])
    mcc = matthews_corrcoef(y_labels, y_predicted)
    print(f'MCC: {mcc:.4f}')
    y_labels_bin = label_binarize(y_labels, classes=np.unique(y_labels)) 
    fpr, tpr, thresholds = roc_curve(y_labels_bin.ravel(), np.array(y_probs).ravel())
    roc_auc = auc(fpr, tpr)
    print(f'ROC AUC: {roc_auc:.4f}')

    return report, mcc, roc_auc


if __name__ == "__main__":
    train_file = "train.data"
    test_file = "test.data"

    train_data = pd.read_csv(train_file, sep=' ', header=None, names=['chr_id', 'sequence', 'label'])
    train_sentences = train_data["sequence"]
    train_labels = train_data["label"]

    # Test_dataset
    test_data = pd.read_csv(test_file, sep=' ', header=None, names=['chr_id', 'sequence', 'label'])
    test_sentences = test_data["sequence"]
    test_labels = test_data["label"]
    print(type(test_sentences))
    print(type(test_labels))
    print(test_sentences[:10])
    print(test_labels[:10])


    _, _, test_inputs, test_labels = input_token(train_sentences, train_labels, test_sentences, test_labels)

    batch_size = 32

    Test_Set = MyDataset(test_inputs, test_labels)
    TestLoader = DataLoader(dataset=Test_Set, batch_size=batch_size, shuffle=False, num_workers=0)

    num = 677
    device = torch.device('cuda')
    model = MyModel(num)
    model.load_state_dict(torch.load('model_path', weights_only=True))

    report, mcc, roc_auc = test(model, TestLoader, device)



