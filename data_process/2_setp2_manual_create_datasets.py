#输入关键词，构建dataset
import os.path

path1 = '../dataset/cn4/finetune_data/finetune_train'
def secentence_finder(path):
    file_path = os.path.join(path,'0_all_in_one.txt')
    output_path = os.path.join(path,'0_output.txt')
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        lines = content.split('\n')
        sentences = [sentence for line in lines for sentence in line.split('.')]
    # 获取用户输入
    user_input = input("请输入要搜索的关键词或短语：").lower()  # 将用户输入转换为小写,中文不变
    matched_sentences = [s for s in sentences if user_input in s.lower()][:100]
    if len(matched_sentences):
        # 再次获取用户输入的数字
        num = input("请输入标签：")
        num_list = [num] * len(matched_sentences)
        # 将结果追加写入output.txt
        with open(output_path, 'a', encoding='utf-8') as output_file:
            for s, n in zip(matched_sentences, num_list):
                # 翻译句子
                # translated_text = translate_text(s)
                # 写入原始句子和数字
                output_file.write(s + ' ' + n + '\n')
                # 写入翻译后的句子
                # output_file.write(translated_text + '\n')
    else:
        print('没有')

secentence_finder(path1)