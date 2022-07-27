import sys
import re
sys.path.append("/Users/admin/Documents/yangqian/projects/AuditCorrector/pycorrector")
# sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pycorrector"))
from pycorrector.macbert.macbert_corrector import MacBertCorrector
from preprocess import rm_dups


def process_sent(data):
    # 需要去除的元素
    rm_delimiters = ["，", "。", "；", "！", "、", ";", ","]
    rp_delimiters = ["\d,\d"]
    rm_regex_pattern = re.compile('|'.join(map(re.escape, rm_delimiters)))
    rp_regex_pattern = re.compile('|'.join(rp_delimiters))

    # all_dicts保存句子中所有去掉的元素
    all_sents = rm_regex_pattern.findall(data)
    all_index = [r.start() for r in re.finditer(rm_regex_pattern, data)]  #  r.span() 返回起始和终止元组
    all_dicts = {}
    for i in range(len(all_sents)):
        all_dicts[all_index[i]] = all_sents[i]

    # 切分句子
    text = rm_dups(data, rp_regex_pattern)
    all_sents = re.split(rm_regex_pattern, text)
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


if __name__ == '__main__':
    data = "1）经对全省税收征关系统数据晒查发现，2012年度全省市、区级地方税务部门征收营业税2,462,945.85万源，征收与营业税关联成市维护建设税219,143.50万元，是营业稅的8.9%，大于规定税率7%的1.9百分点，差额46,795.97万元；全省县级地方税务部门征收营业税933,498.86万元，征收与营业税关联城市维护建设税64,376.47万元，是营业税的6.9%，大于贵定最高税率5%的1.9百分点，差额17,736.48万元。"
    all_dicts, all_sents = process_sent(data)
    m = load_model()
    predict_sent, all_errs = merge_sentence(all_dicts, all_sents, m)
    print(predict_sent)
    print(all_errs)



