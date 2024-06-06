# -*- coding: utf-8 -*-
import json
import sys
import uuid
import requests
import hashlib
import time
from importlib import reload

import time

reload(sys)

YOUDAO_URL = 'https://openapi.youdao.com/api'
APP_KEY = '5b8cd4a0b39f20fd'
APP_SECRET = 'HDxNpYLcQHNAV5HyMNNxh5uonlcX7y0p'


def encrypt(signStr):
    hash_algorithm = hashlib.sha256()
    hash_algorithm.update(signStr.encode('utf-8'))
    return hash_algorithm.hexdigest()


def truncate(q):
    if q is None:
        return None
    size = len(q)
    return q if size <= 20 else q[0:10] + str(size) + q[size - 10:size]


def do_request(data):
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    return requests.post(YOUDAO_URL, data=data, headers=headers)


def connect(q):
    data = {}
    data['from'] = 'en'
    data['to'] = 'zh-CHS'
    data['signType'] = 'v3'
    curtime = str(int(time.time()))
    data['curtime'] = curtime
    salt = str(uuid.uuid1())
    signStr = APP_KEY + truncate(q) + salt + curtime + APP_SECRET
    sign = encrypt(signStr)
    data['appKey'] = APP_KEY
    data['q'] = q
    data['salt'] = salt
    data['sign'] = sign

    response = do_request(data)
    result = json.loads(response.content.decode(encoding="utf-8"))
    try:
        return result['translation'][0]
    except:
        print(q)
        print(result)
        return "error"
    # print(result['translation'][0])

# 请将5_step5_simi_en_data.txt的内容复制到keys这个文件，作为a这个变量的值
from keys import a

b_list = ['事故预防','接受监督','政府关系','沟通管理','社会责任披露','低碳','技术投资','清洁能源','能源审查和评估','节能减排','生态保护','绿色供应链','资源管理','废物利用','教育培训','社区沟通']
for i in a:
    # print(i)
    for j in a[i]:
        if j not in b_list:
            # print(j)
            for k in a[i][j]:
                # print(a[i][j])
                # print(k)
                filename = i + '_' + j + '.txt'
                with open(filename, 'a', encoding="utf-8") as file:
                    file.write(connect(k))
                    file.writelines("\n")
                time.sleep(1)
