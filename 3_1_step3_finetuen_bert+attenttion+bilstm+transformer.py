from sentence_transformers import SentenceTransformer, InputExample, losses, models
from torch.utils.data import DataLoader
import pandas as pd
import torch
import torch.nn as nn


def collate_fn(batch):
    texts_a = [example.texts[0] for example in batch]
    texts_b = [example.texts[1] for example in batch]
    labels = [example.label for example in batch]
    return texts_a, texts_b, torch.tensor(labels)

# 生成句子对以及相关系数
def generate_pairs_and_scores(data):
    num_rows = data.shape[0]
    slist1 = []
    slist2 = []
    scores = []

    for i in range(num_rows):
        for j in range(i + 1, num_rows):  # 避免重复的配对和自配对
            slist1.append(data.iloc[i, 0])
            slist2.append(data.iloc[j, 0])

            if data.iloc[i, 1] == data.iloc[j, 1]:
                scores.append(0.8)
            else:
                scores.append(0.2)

    return slist1, slist2, scores

#生成微调输入实例
def create_input_examples(slist1, slist2, scores):
    input_examples = []
    for s1, s2, score in zip(slist1, slist2, scores):
        example = InputExample(texts=[s1, s2], label=score)
        input_examples.append(example)
    return input_examples


class CustomEvaluator:
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device

    def __call__(self, model, output_path=None, epoch=0, steps=0):
        model.eval()
        criterion = torch.nn.MSELoss()
        total_loss = 0
        with torch.no_grad():
            for batch in self.dataloader:
                sentence_pairs = [(example.texts[0], example.texts[1]) for example in batch]
                scores = torch.tensor([example.label for example in batch], device=self.device)

                # 前向传播
                embeddings = model(sentence_pairs)
                predictions = torch.cosine_similarity(embeddings[0], embeddings[1])

                # 计算损失
                loss = criterion(predictions, scores)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.dataloader)
        print(f"Epoch: {epoch}, Evaluation loss: {avg_loss}")

        # 如果需要，保存模型
        if output_path is not None:
            torch.save(model.state_dict(), output_path)

        return avg_loss

# 自定义模型类，集成了Sentence Transformer和LSTM层
class SBERTWithLSTM(nn.Module):
    def __init__(self, sbert_model_name, lstm_hidden_size, lstm_layers=1, bidirectional=True):
        super(SBERTWithLSTM, self).__init__()
        self.sbert = SentenceTransformer(sbert_model_name)
        self.lstm = nn.LSTM(self.sbert.get_sentence_embedding_dimension(),
                            lstm_hidden_size,
                            num_layers=lstm_layers,
                            bidirectional=bidirectional,
                            batch_first=True)
        # 如果LSTM是双向的，我们需要*2
        self.output_size = lstm_hidden_size if not bidirectional else lstm_hidden_size * 2

    def forward(self, texts_a, texts_b):
        embeddings_a = self.sbert.encode(texts_a, convert_to_tensor=True)
        embeddings_b = self.sbert.encode(texts_b, convert_to_tensor=True)
        lstm_output_a, _ = self.lstm(embeddings_a.unsqueeze(1))
        lstm_output_b, _ = self.lstm(embeddings_b.unsqueeze(1))
        lstm_output_a = lstm_output_a[:, -1, :]
        lstm_output_b = lstm_output_b[:, -1, :]
        predictions = torch.cosine_similarity(lstm_output_a, lstm_output_b)
        return predictions


# 微调函数，现在使用自定义的SBERTWithLSTM模型
def finetune_model():
    # 加载自定义模型
    best_loss = float('inf')
    lstm_hidden_size = 512
    model = SBERTWithLSTM('paraphrase-xlm-r-multilingual-v1', lstm_hidden_size)

    # 将模型转移到GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_data = pd.read_csv('dataset/cn4/finetune_data/finetune_train/0_output.txt', sep='\s+', header=None,
                             names=['Sentence', 'Group'])
    train_list1, train_list2, scores = generate_pairs_and_scores(train_data)
    # 创建 InputExample 实例列表
    train_examples = create_input_examples(train_list1, train_list2, scores)
    # 然后在创建DataLoader时传递这个collate_fn
    train_dataloader = DataLoader(train_examples, batch_size=16, shuffle=True, collate_fn=collate_fn)

    # 对于自定义模型，我们不能使用model.fit()，需要手动实现训练循环
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.MSELoss()  # 如果您的分数是在[0,1]之间，可以使用MSE损失

    val_data = pd.read_csv('dataset/cn4/finetune_data/finetune_val/0_output.txt', sep='\s+', header=None,
                           names=['Sentence', 'Group'])
    val_list1, val_list2, val_scores = generate_pairs_and_scores(val_data)
    val_examples = create_input_examples(val_list1, val_list2, val_scores)
    # Make sure to use collate_fn in your DataLoader
    val_dataloader = DataLoader(val_examples, batch_size=16, shuffle=False, collate_fn=collate_fn)

    val_evaluator = CustomEvaluator(val_dataloader, device)

    model.train()
    for epoch in range(5):
        for texts_a, texts_b, scores in train_dataloader:
            optimizer.zero_grad()

            # 将文本传递给模型进行前向传播
            output = model(texts_a, texts_b)
            # 将scores移动到GPU上
            scores = scores.to(device)
            # 计算损失
            loss = criterion(output, scores)

            # 反向传播
            loss.backward()
            optimizer.step()

        model_save_path = f'./output_sbert_lstm_attention_bert/best_model_epoch_{epoch}.pth'
        torch.save(model.state_dict(), model_save_path)  # 直接保存模型
        print(f"New best model saved at {model_save_path}")
# finetune_model()


#微调英文模型
def finetune_model_en():
    # 加载自定义模型
    best_loss = float('inf')
    pattern = r'(.+?)\s(\d+)$'
    lstm_hidden_size = 512
    model = SBERTWithLSTM('paraphrase-xlm-r-multilingual-v1', lstm_hidden_size)

    # 将模型转移到GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_data = pd.read_csv('dataset/en3/train/finetune_train/0_output_en_train.txt', sep=pattern, engine='python',
                             header=None,
                             names=['Sentence', 'Group'], usecols=[1, 2])
    train_list1, train_list2, scores = generate_pairs_and_scores(train_data)
    # 创建 InputExample 实例列表
    train_examples = create_input_examples(train_list1, train_list2, scores)
    # 然后在创建DataLoader时传递这个collate_fn
    train_dataloader = DataLoader(train_examples, batch_size=16, shuffle=True, collate_fn=collate_fn)

    # 对于自定义模型，我们不能使用model.fit()，需要手动实现训练循环
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.MSELoss()  # 如果您的分数是在[0,1]之间，可以使用MSE损失

    val_data = pd.read_csv('dataset/en3/train/finetune_val/0_output_en_val.txt', sep=pattern, engine='python',
                           header=None, names=['Sentence', 'Group'], usecols=[1, 2])
    val_list1, val_list2, val_scores = generate_pairs_and_scores(val_data)
    val_examples = create_input_examples(val_list1, val_list2, val_scores)
    # Make sure to use collate_fn in your DataLoader
    val_dataloader = DataLoader(val_examples, batch_size=16, shuffle=False, collate_fn=collate_fn)

    val_evaluator = CustomEvaluator(val_dataloader, device)

    model.train()
    for epoch in range(5):
        for texts_a, texts_b, scores in train_dataloader:
            optimizer.zero_grad()

            # 将文本传递给模型进行前向传播
            output = model(texts_a, texts_b)
            # 将scores移动到GPU上
            scores = scores.to(device)
            # 计算损失
            loss = criterion(output, scores)

            # 反向传播
            loss.backward()
            optimizer.step()

        model_save_path = f'./output_sbert_lstm_attention_bert_en/best_model_epoch_{epoch}.pth'
        torch.save(model.state_dict(), model_save_path)  # 直接保存模型
        print(f"New best model saved at {model_save_path}")
# finetune_model_en()