import os
import sys
if __name__ == '__main__':
    text_in = r'E:\outwork\1110-nlp-electronic\torchcode\data\output.txt'
    vocab_out = 'vocab1.txt'
    lexicon = {}
    with open(text_in, 'r', encoding='utf-8') as f:#读取 训练集文本文件 character.txt
        for line in f:
            parts = line.strip().split()#去掉空格
            idx = parts[0] #id
            text= parts[1:] # 将汉字存入 text
            for p in text:
                if p not in lexicon:
                    lexicon[p] = 1
                else:
                    lexicon[p] += 1# 统计词频
    print('There are %d label in lexicon!' % len(lexicon))#输出一共有多少个字
    vocab = sorted(lexicon.items(), key=lambda x: x[1], reverse=True) # 按词频 排序
    print(vocab)
    index = 3
    with open(vocab_out, 'w') as w:
        w.write('<PAD> 0\n')
        w.write('<S/E> 1\n')
        w.write('<UNK> 2\n')
        for (l, n) in vocab:
            w.write(l+' '+str(index)+'\n')# 一个汉字 对应一个标签
            index += 1
    print('Done!')