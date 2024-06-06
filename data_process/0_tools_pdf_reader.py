import glob
import pdfplumber
import os

# 定义要遍历的文件夹路径，包括子文件夹
root_folder = '../美国水电'

# # 使用glob获取所有PDF文件的路径
# pdf_files = glob.glob(root_folder + '/*.pdf', recursive=True)
# print(pdf_files)

# 提取PDF文本内容
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# 保存文本到同名的txt文件
def save_text_as_txt(pdf_path, text):
    txt_path = os.path.splitext(pdf_path)[0] + '.txt'
    with open(txt_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write(text)


# 打印或做其他操作，例如保存到文件
# for pdf_path in pdf_files:
#     print(pdf_path)
#     # 提取文本并保存为同名txt文件
#     pdf_text = extract_text_from_pdf(pdf_path)
#     save_text_as_txt(pdf_path, pdf_text)

import shutil
#
# 源文件夹路径
source_dir = '../美国水电'

# 目标文件夹路径
target_dir = r'E:\outwork\1110-nlp-electronic\dataset\en'

# 确保目标文件夹存在
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 遍历源文件夹
for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.endswith('.txt'):
            # 构建源文件路径
            source_file = os.path.join(root, file)
            # 构建目标文件路径
            target_file = os.path.join(target_dir, file)
            # 移动文件
            shutil.move(source_file, target_file)
            print(f"移动文件: {source_file} 到 {target_file}")