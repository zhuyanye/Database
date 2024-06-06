import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

# 数据处理代码
def read_data(file_path):
    sentences = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        sentence = []
        label = []
        # for line in f:
            # if line == '\n':
            #     sentences.append(sentence)
            #     labels.append(label)
            #     sentence = []
            #     label = []
            # else:
        for line in f.readlines():

                line = line.replace(' ', '\t')
                word, tag = line.replace('\n', '').split('\t')
                sentence.append(word)
                label.append(tag)
    return sentences, labels

def build_vocab(sentences):
    word2id = {'<PAD>': 0, '<UNK>': 1}
    for sentence in sentences:
        for word in sentence:
            if word not in word2id:
                word2id[word] = len(word2id)
    return word2id

def convert_to_ids(sentences, labels, word2id, label2id):
    X = [[word2id.get(word, 1) for word in sentence] for sentence in sentences]
    Y = [[label2id[label] for label in label_seq] for label_seq in labels]
    return X, Y

def pad_sequences(sequences, max_len=None, padding='post', truncating='post', value=0):
    print(sequences)
    max_len = max_len or max(len(seq) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        if len(seq) > max_len:
            if truncating == 'pre':
                padded_seq = seq[-max_len:]
            else:
                padded_seq = seq[:max_len]
        else:
            padded_seq = seq
        if padding == 'post':
            padded_seq = padded_seq + [value] * (max_len - len(padded_seq))
        else:
            padded_seq = [value] * (max_len - len(padded_seq)) + padded_seq
        padded_sequences.append(padded_seq)
    return padded_sequences

# 将数据集标注成BIO格式的函数
def label2BIO(labels):
    new_labels = []
    for label_seq in labels:
        new_label_seq = []
        for i, label in enumerate(label_seq):
            if label == 'O':
                new_label_seq.append('O')
            elif i == 0 or label_seq[i-1] != label:
                new_label_seq.append('B-' + label)
            else:
                new_label_seq.append('I-' + label)
        new_labels.append(new_label_seq)
    return new_labels

# 模型代码
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.crf = CRF(tagset_size)

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        tag_space = self.hidden2tag(lstm_out)
        return tag_space

class CRF(nn.Module):
    def __init__(self, tagset_size):
        super(CRF, self).__init__()
        self.tagset_size = tagset_size
        self.transitions = nn.Parameter(torch.randn(tagset_size, tagset_size))
        self.transitions.data[0, :] = -10000
        self.transitions.data[:, 0] = -10000
        self.transitions.data[:, -1] = -10000
        self.transitions.data[-1, :] = -10000

    def forward(self, emissions, tags):
        batch_size, seq_len, tagset_size = emissions.size()
        scores = torch.zeros(batch_size)
        tags = torch.cat([torch.tensor([[self.tagset_size-1]], dtype=torch.long)]*batch_size, dim=1)
        emissions = torch.cat([emissions, torch.zeros(batch_size, 1, tagset_size)], dim=1)
        for i in range(seq_len):
            scores = scores + self.transitions[tags[:, i+1], tags[:, i]] + emissions[:, i, tags[:, i+1]]
        scores = scores + self.transitions[self.tagset_size-1, tags[:, -1]]
        return -scores.mean()

    def decode(self, emissions):
        batch_size, seq_len, tagset_size = emissions.size()
        scores = torch.zeros(batch_size, tagset_size)
        tags = torch.zeros(batch_size, seq_len+1, tagset_size, dtype=torch.long)
        tags[:, 0, -1] = 1
        for i in range(seq_len):
            emission = emissions[:, i, :]
            score = scores.unsqueeze(2) + self.transitions.unsqueeze(0) + emission.unsqueeze(1)
            score, tag = score.max(dim=1)
            scores = score + emission
            tags[:, i+1, :] = tag
        scores = scores + self.transitions[self.tagset_size-1]
        path_score, best_last_tag = scores.max(dim=1)
        best_path = [best_last_tag.unsqueeze(1)]
        for i in range(seq_len-1, 0, -1):
            best_last_tag = tags[:, i+1, :].gather(1, best_last_tag.unsqueeze(1)).squeeze(1)
            best_path.insert(0, best_last_tag.unsqueeze(1))
        best_path = torch.cat(best_path, dim=1)
        return best_path

# 训练代码
def train(model, optimizer, train_data, dev_data, word2id, label2id, batch_size, num_epochs, model_path):
    train_sentences, train_labels = train_data
    print(train_sentences)
    dev_sentences, dev_labels = dev_data
    train_X, train_Y = convert_to_ids(train_sentences, train_labels, word2id, label2id)
    dev_X, dev_Y = convert_to_ids(dev_sentences, dev_labels, word2id, label2id)
    print(train_X)
    train_X = pad_sequences(train_X, padding='post')
    train_Y = pad_sequences(train_Y, padding='post')
    dev_X = pad_sequences(dev_X, padding='post')
    dev_Y = pad_sequences(dev_Y, padding='post')
    train_X = torch.tensor(train_X, dtype=torch.long)
    train_Y = torch.tensor(train_Y, dtype=torch.long)
    dev_X = torch.tensor(dev_X, dtype=torch.long)
    dev_Y = torch.tensor(dev_Y, dtype=torch.long)
    model.train()
    best_f1 = 0
    for epoch in range(num_epochs):
        for i in range(0, len(train_X), batch_size):
            batch_X = train_X[i:i+batch_size]
            batch_Y = train_Y[i:i+batch_size]
            optimizer.zero_grad()
            emissions = model(batch_X)
            loss = model.crf(emissions, batch_Y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            dev_emissions = model(dev_X)
            dev_pred = model.crf.decode(dev_emissions)
            dev_f1 = f1_score(dev_Y.view(-1).tolist(), dev_pred.view(-1).tolist(), average='macro')
            if dev_f1 > best_f1:
                best_f1 = dev_f1
                torch.save(model.state_dict(), model_path)
        model.train()

# 测试代码
def test(model, test_data, word2id, label2id, batch_size, model_path):
    test_sentences, test_labels = test_data
    test_X, test_Y = convert_to_ids(test_sentences, test_labels, word2id, label2id)
    test_X = pad_sequences(test_X, padding='post')
    test_Y = pad_sequences(test_Y, padding='post')
    test_X = torch.tensor(test_X, dtype=torch.long)
    test_Y = torch.tensor(test_Y, dtype=torch.long)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        test_emissions = model(test_X)
        test_pred = model.crf.decode(test_emissions)
        test_f1 = f1_score(test_Y.view(-1).tolist(), test_pred.view(-1).tolist(), average='macro')
    return test_f1

# 使用样例
train_data = read_data(r'E:\outwork\1110-nlp-electronic\torchcode\data\output.txt')
dev_data = read_data(r'E:\outwork\1110-nlp-electronic\torchcode\data\val_output.txt')
test_data = read_data(r'E:\outwork\1110-nlp-electronic\torchcode\data\output.txt')
sentences = train_data[0] + dev_data[0] + test_data[0]
word2id = build_vocab(sentences)
label2id = {'O': 0, 'B-LOC': 1, 'I-LOC': 2, 'B-PER': 3, 'I-PER': 4, 'B-ORG': 5, 'I-ORG': 6}
train_data = (label2BIO(train_data[1]), train_data[1])
print(f'1`11,{train_data}')
dev_data = (label2BIO(dev_data[1]), dev_data[1])
test_data = (label2BIO(test_data[1]), test_data[1])
model = BiLSTM_CRF(len(word2id), len(label2id), 100, 200)
optimizer = optim.Adam(model.parameters())
train(model, optimizer, train_data, dev_data, word2id, label2id, 32, 10, 'best_model.pth')
test_f1 = test(model, test_data, word2id, label2id, 32, 'best_model.pth')
