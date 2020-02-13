#!/user/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/2/13 0013 13:10
# @Author  : CarryChang
# @Software: PyCharm
# @email: coolcahng@gmail.com
# @web ：CarryChang.top
from data.processing import file_read
import requests
from sklearn.metrics import accuracy_score
import json
if __name__ == "__main__":
    import time
    st = time.time()
    path_list = ['data/1-1.txt', 'data/5-1.txt']
    number = 1000
    y_pred = []
    y_true = []
    api_url = "http://127.0.0.1:5000/sentiment_analysis_api"
    for path in path_list:
        for content in file_read(number, path):
            source = content.split('\t')
            y_true.append(int(source[0]))
            eval_content = source[1]
            para = {"content": eval_content}
            score = requests.post(api_url, data=json.dumps(para)).json()['sa']
            if float(score) >= 0.5:
                y_pred.append(5)
            else:
                y_pred.append(1)
    # print(y_true)
    # print(y_pred)
    print('acc:{}'.format(accuracy_score(y_true, y_pred)))
    print('time used :{}'.format(time.time()-st))
    # snownlp : acc:0.9375
    # time used :14.206809282302856
    # acc:0.941
    # time used :13.918262243270874
    #


