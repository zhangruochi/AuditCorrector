# import sys
import json
from tqdm import tqdm
# sys.path.append("..")
from pycorrector.macbert.macbert_corrector import MacBertCorrector

m = MacBertCorrector("./macbert4csc")
path = "./reality_test.json"
# 读取文件数据
with open(path, "r") as f:
    row_data = json.load(f)

# 读取每一条json数据
correct_sum = 0
for d in tqdm(row_data):
    line = d['original_text']
    correct_sent, err = m.macbert_correct(line)
    d['predict_text'] = correct_sent

# 读取每一条json数据
pre_acc = 0
for dd in tqdm(row_data):
    correct_sum += 1
    # print("正在打印第{}条数据".format(correct_sum))
    if dd['predict_text'] == dd['correct_text']:
        pre_acc += 1
    else:
        print("predict: " + dd["predict_text"])
        print("correct: " + dd["correct_text"])
        print("==================================================")

# 0.9479507313920927
sentence_acc = pre_acc / correct_sum
with open('reality_test.txt', 'w') as f:
    f.write(str(sentence_acc))
print(sentence_acc)
