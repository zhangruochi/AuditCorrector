import sys
import os
import json
from tqdm import tqdm

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pycorrector"))

from pycorrector.macbert.macbert_corrector import MacBertCorrector

m = MacBertCorrector("./macbert4csc")
path = "./reality_test.json"
# 读取文件数据
with open(path, "r") as f:
    raw_data = json.load(f)

# 读取每一条json数据
pre_acc = 0
for d in tqdm(raw_data):
    line = d['original_text']
    predict_text, err = m.macbert_correct(line)

    # print("正在打印第{}条数据".format(correct_sum))
    if predict_text == d['correct_text']:
        pre_acc += 1
    else:
        print("original: " + d['original_text'])
        print("predict: " + predict_text)
        print("correct: " + d["correct_text"])
        print("==================================================")

# 0.9479507313920927
sentence_acc = pre_acc / len(raw_data)
with open('reality_test.txt', 'w') as f:
    f.write(str(sentence_acc))
print(sentence_acc)
