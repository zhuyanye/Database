

if __name__ == "__main__":
    # path = '../../data/THUCNews/data/train.txt'
    path=r'E:\outwork\1110-nlp-electronic\torchcode\data\outputt.txt'
    input_data= open(path, 'r', encoding='utf-8')
    train_data = set()
    for item in input_data.readlines():
        item = item.replace('\t',' ')
        print(item)
        item_str, _= item.split(' ')
        for item_char in item_str:
            train_data.add(item_char)

    vocab = {'<PAD>': 0, '<NUL>': 1}
    index = 2
    for item in train_data:
        vocab[item] = index
        index += 1
    output = open('vocab.txt', 'w', encoding='utf-8')
    for key, value in enumerate(vocab):
        output.writelines(f'{value}\t{key}\n')
    print('build vocab success')
