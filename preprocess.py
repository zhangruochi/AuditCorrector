import pandas as pd
from tqdm import tqdm
import re
from matplotlib import pyplot as plt
import re
import json

import jieba
import random

from pypinyin import pinyin, lazy_pinyin, Style
from Pinyin2Hanzi import DefaultHmmParams
from Pinyin2Hanzi import viterbi
from Pinyin2Hanzi import is_pinyin
import copy

MAX_POS = 10
MAX_REP = 5


rm_delimiters = ["，", "。", "；", "！", "、", ";", ","]
rp_delimiters = ["\d,\d"]

rm_regex_pattern = re.compile('|'.join(map(re.escape, rm_delimiters)))
rp_regex_pattern = re.compile('|'.join(rp_delimiters))


def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False


def rm_dups(text, rp_regex_pattern):

    while True:
        m = re.search(rp_regex_pattern, text)
        if m:
            mm = m.group()
            text = text.replace(mm, mm.replace(',', ''))
        else:
            break

    return text

def rep_tokens(text):

    res = []
    all_chars = list(text)

    hmmparams = DefaultHmmParams()

    # single error
    pos_ids = random.sample(list(range(len(all_chars))), k=min(MAX_POS,len(all_chars)))
    for pos in pos_ids:
        row_token = all_chars[pos]
        if is_chinese(row_token):
            token_pinyin = lazy_pinyin(all_chars[pos])[0]

            if not is_pinyin(token_pinyin):
                continue

            result = viterbi(hmm_params=hmmparams,
                            observations=(token_pinyin, ),
                            path_num=MAX_REP)
            for item in result:
                rep_token = item.path[0]
                _ = copy.copy(all_chars)
                _[pos] = rep_token
                res.append(([pos], "".join(_)))
                # print("row: {} | pinyin: {} | rep: {}".format(row_token, token_pinyin, rep_token))

    # multi error
    pos_ids = random.sample(list(range(len(all_chars))),
                            k=random.randint(a = min(MAX_POS, len(all_chars))//2, b = min(MAX_POS, len(all_chars)) ) )
    dict_ = {id: [] for id in pos_ids}

    for pos in pos_ids:

        row_token = all_chars[pos]
        if is_chinese(row_token):
            token_pinyin = lazy_pinyin(all_chars[pos])[0]

            if not is_pinyin(token_pinyin):
                continue

            result = viterbi(hmm_params=hmmparams,
                             observations=(token_pinyin, ),
                             path_num=MAX_REP)

            for id_, item in enumerate(result):
                rep_token = item.path[0]
                dict_[pos].append(rep_token)

                # print("row: {} | pinyin: {} | rep: {}".format(
                #     row_token, token_pinyin, rep_token))

    all_reps = [(k, v) for k,v in dict_.items() if v]
    # print(all_reps)

    if all_reps:

        min_length = min([len(_[-1]) for _ in all_reps])

        for i in range(min_length):
            chars_lists = copy.copy(all_chars)
            pos_ = []
            for k,v in all_reps:
                chars_lists[k] = v[i]
                pos_.append(k)
            res.append((pos_, "".join(chars_lists)))

    # pos_,v = res[-1]

    # for p in pos_:
    #     print(p, v[p], all_chars[p])

    # print(res)
    return res


if __name__ == "__main__":
    all_data = pd.read_csv("make_data.csv")
    lengths = []

    train_json_data = []
    valid_json_data = []
    test_json_data = []


    train_ids = int(all_data.shape[0] * 0.8)
    valid_ids = int(all_data.shape[0] * 0.9)

    for i, text in tqdm(enumerate(all_data["correct_txt"].tolist()), total = all_data.shape[0]):

        if i <  train_ids:
            json_data = train_json_data
        elif i < valid_ids:
            json_data = valid_json_data
        else:
            json_data = test_json_data

        text = rm_dups(text, rp_regex_pattern)

        all_sents = re.split(rm_regex_pattern, text)
        all_sents = [_.strip() for _ in all_sents if _.strip()]

        for j, sent in enumerate(all_sents):
            length = len(sent)
            lengths.append(length)
            rep_sents = rep_tokens(sent)

            for k, rep_ in enumerate(rep_sents):

                json_data.append({
                    "id": "{}-{}-{}".format(i,j,k),
                    "original_text": rep_[-1],
                    "wrong_ids": rep_[0],
                    "correct_text": sent,
                })
        # if i >=1:
        #     break

    plt.hist(lengths, bins = 500, range = (0,500))
    plt.savefig("dist.png")

    with open("train.json", "w") as f:
        json.dump(train_json_data, f, ensure_ascii=False)
    with open("dev.json", "w") as f:
        json.dump(valid_json_data, f, ensure_ascii=False)
    with open("test.json", "w") as f:
        json.dump(test_json_data, f, ensure_ascii=False)