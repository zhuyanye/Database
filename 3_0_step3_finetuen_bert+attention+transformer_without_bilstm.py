# 微调sbert模型，使之更加贴合我们的需求
import time

from sentence_transformers import evaluation
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd


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


# 生成微调输入实例
def create_input_examples(slist1, slist2, scores):
    input_examples = []
    for s1, s2, score in zip(slist1, slist2, scores):
        # print(s1)
        # print(s2)
        example = InputExample(texts=[s1, s2], label=score)
        input_examples.append(example)
    return input_examples


# 微调中文模型
def finetune_model():
    val_data = pd.read_csv('dataset/cn4/finetune_data/finetune_val/0_output.txt', sep='\s+', header=None,
                           names=['Sentence', 'Group'])
    val_list1, val_list2, scores = generate_pairs_and_scores(val_data)
    evaluator = evaluation.EmbeddingSimilarityEvaluator(val_list1, val_list2, scores)

    # 输入预训练模型开始微调
    model = SentenceTransformer('paraphrase-xlm-r-multilingual-v1', device='cpu')

    # 假设你的数据保存在名为'data.txt'的文件中，并且是空格分隔的
    train_data = pd.read_csv('dataset/cn4/finetune_data/finetune_train/0_output.txt', sep='\s+', header=None,
                             names=['Sentence', 'Group'])

    train_list1, train_list2, scores = generate_pairs_and_scores(train_data)

    # 创建 InputExample 实例列表
    train_examples = create_input_examples(train_list1, train_list2, scores)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model)

    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100, evaluator=evaluator,
              evaluation_steps=500, output_path='./output', save_best_model=True)

# 微调中文模型
# finetune_model()

# 微调英文模型
def finetune_model_en():
    # 假设数字标签位于每行的末尾，并且前面只有一个空格
    pattern = r'(.+?)\s(\d+)$'
    # 使用正则表达式分隔符
    val_data = pd.read_csv('dataset/en3/train/finetune_val/0_output_en_val.txt', sep=pattern, engine='python',
                           header=None, names=['Sentence', 'Group'], usecols=[1, 2])
    val_list1, val_list2, scores = generate_pairs_and_scores(val_data)
    # print(val_list1)
    # print(val_list2)
    # print(scores)
    evaluator = evaluation.EmbeddingSimilarityEvaluator(val_list1, val_list2, scores)

    # 输入预训练模型开始微调
    model = SentenceTransformer('paraphrase-xlm-r-multilingual-v1', device='cpu')

    # 假设你的数据保存在名为'data.txt'的文件中，并且是空格分隔的
    train_data = pd.read_csv('dataset/en3/train/finetune_train/0_output_en_train.txt', sep=pattern, engine='python',
                             header=None,
                             names=['Sentence', 'Group'], usecols=[1, 2])

    train_list1, train_list2, scores = generate_pairs_and_scores(train_data)
    # print(train_list1)
    # print(train_list2)
    # print(scores)
    # time.sleep(100)
    # 创建 InputExample 实例列表
    train_examples = create_input_examples(train_list1, train_list2, scores)
    # print(train_examples)
    train_dataloader = DataLoader(train_examples, shuffle=False, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model)

    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100, evaluator=evaluator,
              evaluation_steps=500, output_path='./finetune_output_en', save_best_model=True)

# 微调英文模型
finetune_model_en()
