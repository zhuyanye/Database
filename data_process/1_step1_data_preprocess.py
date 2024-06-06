# 检查哪些txt有问题

import glob
import math
import os
import re
import shutil
import time
import openpyxl
import requests
from bs4 import BeautifulSoup

# 假设你的Excel文件名是固定的，这里用 your_excel_file.xlsx 代替
# excel_file = r'E:\outwork\1110-nlp-electronic\中国水电站资料\水电站社会责任新闻1033条语料.xlsx'
# wb = openpyxl.load_workbook(excel_file)
# sheet = wb.active


def fetch_text(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)

    # 检查请求是否成功
    response.raise_for_status()
    # 解析HTML内容
    soup = BeautifulSoup(response.content, 'html.parser')
    # 查找所有的<p>和<span>标签
    text_tags = soup.find_all(['p', 'span'])
    # 遍历每个标签，并获取其文本内容
    res = ''
    for tag in text_tags:
        # 获取标签的文本并添加到结果字符串
        text = tag.get_text()
        res += text + ' '  # 添加一个空格作为文本分隔，可以根据需要移除
    return res


# 检查cn2有没有空文件并下载
def check_empty_and_download():
    path = r'E:\outwork\1110-nlp-electronic\dataset\cn2'

    # 使用glob模式匹配来获取所有txt文件
    txt_files = glob.glob(os.path.join(path, '*.txt'))

    # 初始化一个空列表来保存所有空的txt文件名
    empty_txt_files = []

    # 遍历文件列表，检查文件大小
    for file_path in txt_files:
        # 如果文件大小为0，则添加文件名到列表中
        if os.path.getsize(file_path) == 0:
            empty_txt_files.append(os.path.basename(file_path))

    # 打印所有空的txt文件名
    # print(len(empty_txt_files))
    # print(empty_txt_files)

    values_by_id = {}

    for filename in empty_txt_files:
        # 分割字符串获取编号部分
        file_id = filename.split('_')[0]

        row = sheet[file_id]

        # 获取C列的值，这里假设C列是第三列
        c_value = row[2].value
        values_by_id[filename] = c_value
    # print(len(values_by_id))
    # print(values_by_id)

    for i in values_by_id:
        filename = i
        url = values_by_id[i]
        res = fetch_text(url)
        print(i)
        print(url)
        # print(res)
        if res:
            # 构建完整的文件路径
            file_path = os.path.join(".", "dataset", "cn2", filename)

            # 将文本内容写入文件，这里使用'w'模式覆盖写入
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(res)

            # 防止被网站限制，每次请求后暂停一秒
            time.sleep(1)
        else:
            break


# 下载所有文件到cn2
def download_all():
    for row in sheet.iter_rows(min_row=2):
        idx = str(row[0].value)
        name = str(row[1].value)
        url = str(row[2].value)
        print(idx, name, url)
        filename = idx + '_' + name + '.txt'
        file_path = os.path.join(".", "dataset", "cn2", filename)
        res = fetch_text(url)
        if res:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(res)
            # 防止被网站限制，每次请求后暂停一秒
            time.sleep(1)
        else:
            break


# 预处理文件到cn3
def clean_txt():
    patha = '../dataset/cn2'  # 原始文件的路径
    pathb = '../dataset/cn3'  # 清理后的文件保存路径

    # 确保保存清理后文件的路径存在
    if not os.path.exists(pathb):
        os.makedirs(pathb)
    # 正则表达式，匹配只包含英文字符的行
    english_line_pattern = re.compile(r'^[A-Za-z\s.,!?-_]*$')
    # 用于匹配以数字开头的模式，如 "1.", "（1）", "1、"
    numbered_item_pattern = re.compile(r'^\d+[\.\）\、]|^（\d+）|^\(\d+\)|^【\d+】|^\d+，')
    # 用于移除所有连续空格（包括全角空格）的正则表达式
    spaces_pattern = re.compile(r'\s+')
    # 定义一个列表，其中包含要屏蔽的关键词
    keywords_to_exclude = ['本文', '投稿信箱', '编辑部电话', 'Email', '访问www', '电子邮箱', '请联系', 'http',
                           '投稿邮箱', '点击分享', '公众号', '点击下方', 'E小水电', '.com', '阅读原文', '点击浏览器',
                           '公网安备', '许可证', '主办单位', '举报电话', '.html', '原标题']

    # 列出所有txt文件
    for filename in os.listdir(patha):
        if filename.endswith('.txt'):
            # 读取原始文件内容
            with open(os.path.join(patha, filename), 'r', encoding='utf-8') as file:
                lines = file.readlines()
            seen = set()
            cleaned_lines = []
            for line in lines:
                # 移除\xa0并去除两边空白
                stripped_line = line.strip().replace('\xa0', '').replace('◆', '').replace('▲', '').replace('●',
                                                                                                           '').replace(
                    '⚡', '').replace('‍', '').replace('☞', '').replace('▶', '')
                # 使用空格对每行进行断句
                sentences = stripped_line.split(' ')
                # 处理每个句子
                for sentence in sentences:
                    # 进一步按照'。'进行断句
                    sub_sentences = sentence.split('。')
                    # 检查句子是否不为空，不是全英文，长度超过15，并且之前未见过
                    for sub_sentence in sub_sentences:
                        sub_sentence = numbered_item_pattern.sub('', sub_sentence)
                        sub_sentence = spaces_pattern.sub('', sub_sentence)
                        sub_sentence = numbered_item_pattern.sub('', sub_sentence)
                        sub_sentence = spaces_pattern.sub('', sub_sentence)
                        # 检查句子是否包含任何排除关键词
                        if any(keyword in sub_sentence for keyword in keywords_to_exclude):
                            continue
                            # 检查子句是否不为空，不是全英文，长度超过15，并且之前未见过
                        if (sub_sentence and
                                not english_line_pattern.match(sub_sentence) and
                                len(sub_sentence) >= 20 and
                                sub_sentence not in seen):
                            seen.add(sub_sentence)  # 将子句添加到seen集合中
                            cleaned_lines.append(sub_sentence)  # 将子句添加到cleaned_lines列表中
            # 如果清理后的内容不为空，则写入到新路径下的同名文件
            if cleaned_lines:
                # for i in cleaned_lines:
                # print(i)
                with open(os.path.join(pathb, filename), 'w', encoding='utf-8') as file:
                    file.write('\n'.join(cleaned_lines))
        # break
        # time.sleep(5)


# clean_txt()

# 生成train，val文件夹数据集
def get_train_val():
    src_directory = '../dataset/cn3/'
    # 文件夹路径
    train_directory = '../dataset/cn4/finetune_data/finetune_train'
    val_directory = '../dataset/cn4/finetune_data/finetune_val'
    test_directory = '../dataset/cn4/test_data'

    # 确保目标文件夹存在
    os.makedirs(train_directory, exist_ok=True)
    os.makedirs(val_directory, exist_ok=True)
    os.makedirs(test_directory, exist_ok=True)

    # 将文件按分类名字分类
    category_files = {}

    for filename in os.listdir(src_directory):
        if filename.endswith(".txt"):
            num = int(filename.split('_')[0])
            category = filename.split('_')[1].replace('.txt', '')

            if num % 2 != 0:  # 选择奇数开头的文件
                if category not in category_files:
                    category_files[category] = []
                category_files[category].append(filename)
            else:
                shutil.copy(os.path.join(src_directory, filename), test_directory)

    # 对每个分类，将80%的文件复制到train_directory，剩余的文件复制到val_directory
    for category, files in category_files.items():
        train_count = math.floor(0.8 * len(files))
        train_files = files[:train_count]
        val_files = files[train_count:]

        for file in train_files:
            shutil.copy(os.path.join(src_directory, file), train_directory)

        for file in val_files:
            shutil.copy(os.path.join(src_directory, file), val_directory)


# get_train_val()

# 将 train，val文件夹中的数据，合并成一个txt
# path1 = '../dataset/cn4/finetune_data/finetune_train'
# path2 = '../dataset/cn4/finetune_data/finetune_val'
# path3 = '../dataset/cn4/test_data'


def get_all_in_one(path):
    output_path = os.path.join(path, '0_all_in_one.txt')
    merged_content = []  # 用于保存所有处理过的内容
    seen_lines = set()  # 用于跟踪已经见过的行，以避免重复

    # 遍历文件夹中的所有.txt文件
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            with open(os.path.join(path, filename), 'r', encoding='utf-8') as file:
                lines = file.readlines()  # 读取所有行

            # 处理每一行
            for i, line in enumerate(lines):
                # 移除行首尾的空格
                line = line.strip()
                # 如果行以逗号开头，并且不是第一行，且前一行没有结束于句号，则与前一行合并
                if i > 0 and (line.startswith('，') or line.startswith(',')) and not merged_content[-1].endswith('。'):
                    merged_content[-1] += line
                else:
                    # 检查行是否已经出现过
                    if line not in seen_lines:
                        seen_lines.add(line)
                        merged_content.append(line)  # 添加新的唯一行到merged_content

    # 保存处理后的内容到新文件
    with open(output_path, 'w', encoding='utf-8') as output_file:
        output_file.write('\n'.join(merged_content))

# get_all_in_one(path1)
# get_all_in_one(path2)
# get_all_in_one(path3)

def clean_en_txt():
    patha = '../dataset/en2'  # 原始文件的路径
    pathb = '../dataset/en3'  # 清理后的文件保存路径

    # 确保保存清理后文件的路径存在
    if not os.path.exists(pathb):
        os.makedirs(pathb)

    # 正则表达式，匹配只包含英文字符的行
    # english_line_pattern = re.compile(r'^[A-Za-z\s.,!?-_]*$')
    # 用于匹配以数字开头的模式，如 "1.", "（1）", "1、"
    numbered_item_pattern = re.compile(r'^\d+[\.\）\、]|^（\d+）|^\(\d+\)|^&#8203;``【oaicite:0】``&#8203;|^\d+，')
    # 用于移除所有连续空格（包括全角空格）的正则表达式
    # spaces_pattern = re.compile(r'\s+')

    # 定义一个列表，其中包含要屏蔽的关键词
    keywords_to_exclude = ['http','cid']

    # 新建一个文件用于写入所有清理后的内容
    cleaned_all_filename = 'all_cleaned_en.txt'
    with open(os.path.join(pathb, cleaned_all_filename), 'w', encoding='utf-8') as cleaned_all_file:
        # 列出所有txt文件
        for filename in os.listdir(patha):
            if filename.endswith('.txt'):
                # 读取原始文件内容
                with open(os.path.join(patha, filename), 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                    # print(lines)

                seen = set()
                cleaned_lines = []
                for line in lines:
                    # 移除\xa0并去除两边空白
                    stripped_line = line.strip().replace('\xa0', '').replace('◆', '').replace('▲', '').replace('●',
                                                                                                               '').replace(
                        '⚡', '').replace('‍', '').replace('☞', '').replace('▶', '').replace('• ','').replace('..','').replace('_','')
                    # 使用空格对每行进行断句
                    sentences = stripped_line.split('\n')
                    # 处理每个句子
                    for sentence in sentences:
                            sub_sentence = numbered_item_pattern.sub('', sentence)
                            # 检查句子是否包含任何排除关键词
                            if any(keyword in sub_sentence for keyword in keywords_to_exclude):
                                continue
                                # 检查子句是否不为空，不是全英文，长度超过15，并且之前未见过
                            if (sub_sentence  and
                                    len(sub_sentence.split(' ')) >= 5 and
                                    sub_sentence not in seen):
                                seen.add(sub_sentence)  # 将子句添加到seen集合中
                                cleaned_lines.append(sub_sentence)  # 将子句添加到cleaned_lines列表中将子句添加到cleaned_lines列表中

                # 如果清理后的内容不为空，则写入到新文件
                if cleaned_lines:
                    cleaned_all_file.write('\n'.join(cleaned_lines))
                    cleaned_all_file.write('\n')  # 在不同文件的内容间加入换行符以分隔

clean_en_txt()