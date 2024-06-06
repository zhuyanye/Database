
import requests
from bs4 import BeautifulSoup

def fetch_text(url):
    # 发送HTTP请求
    response = requests.get(url)
    # 检查请求是否成功
    response.raise_for_status()
    # 解析HTML内容
    soup = BeautifulSoup(response.content, 'html.parser')
    # 查找所有的<p>标签
    p_tags = soup.find_all('p')
    # 遍历每个<p>标签，并获取其及其子标签的文本
    res = ''
    for i, p_tag in enumerate(p_tags, 1):
        if i>1:
            text = p_tag.get_text()
            res +=text
    return res



import pandas as pd

# Load the Excel file into a DataFrame
df = pd.read_excel('./中国水电站资料/水电站社会责任新闻1033条语料.xlsx')

# Generate the dictionary format
data_dict = {}
for index, row in df.iterrows():
    data_dict[index] = {"分类": row["分类"], "链接": row["链接"]}
# print(data_dict)

import os

# Ensure the directory exists
output_dir = './dataset/cn2/'
os.makedirs(output_dir, exist_ok=True)

for key, value in data_dict.items():
    # Skip entries where '分类' is NaN
    if pd.isna(value['分类']):
        continue

    file_name = f"{key}_{value['分类']}.txt"
    file_path = os.path.join(output_dir, file_name)

    with open(file_path, 'w', encoding='utf-8') as f:
        print(value['链接'])
        response = requests.get(value['链接'])
        html_content = response.content

        # 使用BeautifulSoup解析HTML
        soup = BeautifulSoup(html_content, 'html.parser')

        # 提取网页中的所有文字
        text = soup.get_text()
        # 将换行符替换为句号
        text = text.replace('\n', '.')

        # 按句号分割文本
        sentences = text.split('.')

        # 去除空文本
        sentences = [s.strip() for s in sentences if s.strip() != '' and len(s.strip()) >= 10 and '微信' not in s and '。' in s ]

        f.write('\n'.join(sentences))