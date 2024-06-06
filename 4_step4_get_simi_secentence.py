import numpy as np
import torch
import json
from keys import relationship
from sentence_transformers import SentenceTransformer, util

def get_cn_data():
    f_list = ["./dataset/cn4/finetune_data/finetune_train/0_all_in_one.txt",
              "./dataset/cn4/finetune_data/finetune_val/0_all_in_one.txt",
              "./dataset/cn4/test_data/0_all_in_one.txt"]
    unique_lines = set()
    for file_path in f_list:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # 去掉每行的首尾空白字符后加入到set中
                stripped_line = line.strip()
                unique_lines.add(stripped_line)

    corpus = np.array(list(unique_lines))
    embedder = SentenceTransformer(r'finetune_output_cn', device='cpu')
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
    k = 60
    top_k = min(k, len(corpus))
    res_dict = {}
    for i in relationship:
        res_dict[i] = {}
        for j in relationship[i]['sub_obj']:
            res_dict[i][j] = set()
            query_embedding = embedder.encode(j, convert_to_tensor=True)
            cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
            top_results = torch.topk(cos_scores, k=top_k)
            for k in top_results[1]:
                res_dict[i][j].add(corpus[k])  # 添加到集合中
            res_dict[i][j] = list(res_dict[i][j])

    with open('5_step5_simi_cn_data.txt', 'w', encoding='utf-8') as file:
        file.write(json.dumps(res_dict, ensure_ascii=False, indent=4))

# get_cn_data()

def get_en_data():
    unique_lines = set()
    with open('./dataset/en3/all_cleaned_en.txt', 'r', encoding='utf-8') as file:
        for line in file:
            # 去掉每行的首尾空白字符后加入到set中
            stripped_line = line.strip()
            unique_lines.add(stripped_line)

    corpus = np.array(list(unique_lines))
    embedder = SentenceTransformer(r'./finetune_output_en', device='cpu')
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
    k = 60
    top_k = min(k, len(corpus))
    res_dict = {}
    for i in relationship:
        res_dict[i] = {}
        for j in relationship[i]['sub_obj']:
            res_dict[i][j] = set()
            # print(relationship[i]['sub_obj'][j]['en'])
            query_embedding = embedder.encode(relationship[i]['sub_obj'][j]['en'], convert_to_tensor=True)
            cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
            top_results = torch.topk(cos_scores, k=top_k)
            for k in top_results[1]:
                res_dict[i][j].add(corpus[k])  # 添加到集合中
            res_dict[i][j] = list(res_dict[i][j])

    with open('5_step5_simi_en_data.txt', 'w', encoding='utf-8') as file:
        file.write(json.dumps(res_dict, ensure_ascii=False, indent=4))

get_en_data()
