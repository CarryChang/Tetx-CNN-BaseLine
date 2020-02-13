#!/user/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/2/13 0013 13:10
# @Author  : CarryChang
# @Software: PyCharm
# @email: coolcahng@gmail.com
# @web ：CarryChang.top
from data.processing import file_read
from snownlp import SnowNLP
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    import time
    st = time.time()
    path_list = ['data/1-1.txt', 'data/5-1.txt']
    number = 1000
    y_pred = []
    y_true = []
    for path in path_list:
        for content in file_read(number, path):
            source = content.split('\t')
            y_true.append(int(source[0]))
            eval_content = source[1]
            score = SnowNLP(content).sentiments
            if score >= 0.5:
                y_pred.append(5)
            else:
                y_pred.append(1)
    # print(y_true)
    # print(y_pred)
    print('acc:{}'.format(accuracy_score(y_true, y_pred)))
    print('time used :{}'.format(time.time()-st))
    # acc:0.9375
    # time used :23.121785879135132