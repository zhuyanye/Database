from matplotlib import pyplot as plt
from sentence_transformers import SentenceTransformer, util
from sklearn.decomposition import PCA
import numpy as np
from keys import relationship
import json



def get_final_data(sentences,center_sentences):
    model = SentenceTransformer(r'finetune_output_cn', device='cpu')
    # 将句子和中心点句子转换为向量
    sentence_vectors = model.encode(sentences)
    center_vectors = model.encode(center_sentences)


    # 使用PCA进行降维到三维
    pca = PCA(n_components=3)
    sentence_vectors_3d = pca.fit_transform(sentence_vectors)
    center_vectors_3d = pca.transform(center_vectors)

    # 计算每个句子到所有中心点的距离之和
    distance_sums = np.sum(np.sqrt(np.sum((sentence_vectors_3d[:, np.newaxis] - center_vectors_3d) ** 2, axis=2)), axis=1)

    # 创建句子和距离之和的配对数据
    sentence_distance_pairs = [(sentence, distance) for sentence, distance in zip(sentences, distance_sums)]

    # 按距离从小到大排序配对数据
    sorted_sentence_distance_pairs = sorted(sentence_distance_pairs, key=lambda pair: pair[1])

    # for i in range(len(center_sentences)):
    #     center_sentences[i] = connect(center_sentences[i])
    # 输出排序后的结果
    for pair in sorted_sentence_distance_pairs:
        print(pair)
        # 将中心句子加入到返回的列表中
    final_list = center_sentences+[pair[0] for pair in sorted_sentence_distance_pairs]
    return final_list

def run_cn():
    # 打开文件
    with open('./5_step5_simi_cn_data.txt', 'r', encoding='utf-8') as file:
        # 解析文件内容为字典
        data_dict = json.load(file)
    final_res = {}
    for i in data_dict:
        final_res[i] = {}
        for j in data_dict[i]:

            sentences = data_dict[i][j]
            key = relationship[i]['sub_obj'][j]['key']
            # print(i)
            # print(j)
            # print(key)
            # print(sentences)
            path = './dataset/cn4/finetune_data/finetune_train/0_output.txt'
            # 初始化一个空列表来保存匹配的行
            matching_lines = []
            # 打开文件并逐行读取
            with open(path, 'r', encoding='utf-8') as file:
                for line in file:
                    # 移除行尾的换行符并分割字符串和数字
                    parts = line.strip().split(' ')
                    # 检查最后一个元素是否是数字n
                    if int(parts[-1]) == int(key):
                        # 如果匹配，加入到列表中
                        matching_lines.append(parts[0])
            # print(matching_lines)
            final_res[i][j] = get_final_data(sentences, matching_lines)
    with open('6_step6_final_cn_data.txt', 'w', encoding='utf-8') as file:
        file.write(json.dumps(final_res, ensure_ascii=False, indent=4))
    # print(final_res)
run_cn()

def run_en():
    # 打开文件
    with open('./5_step5_simi_en_data.txt', 'r', encoding='utf-8') as file:
        # 解析文件内容为字典
        data_dict = json.load(file)
    final_res = {}
    for i in data_dict:
        final_res[i] = {}
        for j in data_dict[i]:
            sentences = data_dict[i][j]
            key = relationship[i]['sub_obj'][j]['key']
            # print(i)
            # print(j)
            # print(key)
            # print(sentences)
            path = './dataset/en3/train/finetune_train/0_output_en_train.txt'
            # 初始化一个空列表来保存匹配的行
            matching_lines = []
            # 打开文件并逐行读取
            with open(path, 'r', encoding='utf-8') as file:
                for line in file:
                    # print(line)
                    # 移除行尾的换行符并分割字符串和数字
                    parts = line.split(' ')
                    # print(parts)
                    # 检查最后一个元素是否是数字n
                    if int(parts[-1]) == int(key):
                        # 如果匹配，加入到列表中
                        matching_lines.append(" ".join(parts[:-1]))
            # print(matching_lines)
            final_res[i][j] = get_final_data(sentences, matching_lines)
    with open('6_step6_final_en_data.txt', 'w', encoding='utf-8') as file:
        file.write(json.dumps(final_res, ensure_ascii=False, indent=4))
    # print(final_res)
# run_en()