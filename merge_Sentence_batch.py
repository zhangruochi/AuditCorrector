import sys
import re
import pandas as pd
from tqdm import tqdm
import os
# sys.path.append("/Users/admin/Documents/yangqian/projects/AuditCorrector/pycorrector")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pycorrector"))
from pycorrector.macbert.macbert_corrector import MacBertCorrector
from preprocess import rm_dups


def load_dataset(path):
    # excel
    df = pd.read_excel(path)
    return df


def read_data(data):
    original_txt = data["FACT_SUMMARY"]
    correct_text = data["CORRECTED_TEXT"]  # 正确文本
    total_col = data["CORRECTED_TEXT"].shape[0]  # 数据总数
    predict_corr_total = 0
    m = load_model()
    for i, text in enumerate(original_txt.tolist()):
        all_dicts, all_sents = process_sent(text)  # 处理句子不必要元素，切分句子
        predict_sent, all_errs = merge_sentence(all_dicts, all_sents, m)  # 模型预测句子并合并
        # correct_txt = re.sub(" ", "", correct_text[i])
        flag = predict_acc(predict_sent, correct_text[i])
        if flag:
            predict_corr_total += 1
        else:
            print("=======================================================")
            print("original_txt: " + original_txt[i])
            print("predict_text: " + predict_sent)
            print("correct_text: " + correct_text[i])

    sentence_acc = predict_corr_total / total_col
    print(sentence_acc)
    with open('reality_test.txt', 'w') as f:
        f.write(str(sentence_acc))
    return predict_sent, all_errs


def process_sent(data):
    # 需要去除的元素
    rm_delimiters = ["，", "。", "；", "！", "、", ";", ",", "\n"]
    # rm_delimiters_null = ["，", "。", "；", "！", "、", ";", ","]
    rp_delimiters = ["\d, \d"]
    rm_regex_pattern = re.compile('|'.join(map(re.escape, rm_delimiters)))
    # rm_regex_pattern_null = re.compile('|'.join(map(re.escape, rm_delimiters_null)))
    rp_regex_pattern = re.compile('|'.join(rp_delimiters))

    # all_dicts保存句子中所有去掉的元素
    # data = re.sub(" ", "", data)  # 将所有的空格去掉
    all_sents = rm_regex_pattern.findall(data)
    all_index = [r.start() for r in re.finditer(rm_regex_pattern, data)]   # r.span() 返回起始和终止元组
    all_dicts = {}
    for i in range(len(all_sents)):
        all_dicts[all_index[i]] = all_sents[i]

    # 切分句子
    text = rm_dups(data, rp_regex_pattern)  # 去掉数字与数字之间的逗号
    all_sents = re.split(rm_regex_pattern, text)  # 根据rm_regex_pattern进行分割
    all_sents = [_.strip() for _ in all_sents if _.strip()]
    return all_dicts, all_sents


def load_model():
    m = MacBertCorrector("./macbert4csc")
    return m


def merge_sentence(all_dicts, all_sents, m):
    predict_sent = ""
    all_errs = []
    for i, sent in enumerate(all_sents):
        if sent != [] or sent != " ":
            predict_text, err = m.macbert_correct(sent)
            predict_sent = predict_sent + predict_text
            for e in err:
                all_errs.append(e)
    for key in all_dicts.keys():
        predict_sent = predict_sent[:key] + all_dicts[key] + predict_sent[key:]
    return predict_sent, all_errs


def predict_acc(predict_sent, correct_text):

    if predict_sent == correct_text:
        return True
    else:
        return False


if __name__ == '__main__':
    path = "/Users/admin/Documents/chinasoft/nlp/NLP_SUMMARY_CACHE_202206271753_test.xls"
    data = load_dataset(path)
    predict_sent, all_errs = read_data(data)
