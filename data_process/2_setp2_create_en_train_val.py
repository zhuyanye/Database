import random

# 读取文件并按最后的数字分类
with open('en_train+val.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

classified_lines = {}
for line in lines:
    sentence, number = line.rsplit(' ', 1)
    number = number.strip()  # 移除数字后的换行符
    if number in classified_lines:
        classified_lines[number].append(line)
    else:
        classified_lines[number] = [line]

# 分割数据并写入新文件
with open('0_output_en_train.txt', 'w',encoding='utf-8') as file_b, open('0_output_en_val.txt', 'w',encoding='utf-8') as file_c:
    for number, lines in classified_lines.items():
        # 随机打乱顺序
        random.shuffle(lines)
        # 计算80%的行数
        split_idx = int(0.8 * len(lines))
        # 写入b.txt
        for line in lines[:split_idx]:
            file_b.write(line)
        # 写入c.txt
        for line in lines[split_idx:]:
            file_c.write(line)

print("分类和分割完成！")