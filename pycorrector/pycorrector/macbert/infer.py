# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com), Abtion(abtion@outlook.com), okcd00(okcd00@qq.com)
@description: 
"""
import sys
import torch
import argparse
from transformers import BertTokenizer
from loguru import logger
sys.path.append('../..')

from pycorrector.macbert.macbert4csc import MacBert4Csc
from pycorrector.macbert.softmaskedbert4csc import SoftMaskedBert4Csc
from pycorrector.macbert.macbert_corrector import get_errors
from pycorrector.macbert.defaults import _C as cfg

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class Inference:
    def __init__(self, ckpt_path='output/macbert4csc/epoch=09-val_loss=0.01.ckpt',
                 vocab_path='output/macbert4csc/vocab.txt',
                 cfg_path='train_macbert4csc.yml'):
        logger.debug("device: {}".format(device))
        self.tokenizer = BertTokenizer.from_pretrained(vocab_path)
        cfg.merge_from_file(cfg_path)

        if 'macbert4csc' in cfg_path:
            self.model = MacBert4Csc.load_from_checkpoint(checkpoint_path=ckpt_path,
                                                          cfg=cfg,
                                                          map_location=device,
                                                          tokenizer=self.tokenizer)
        elif 'softmaskedbert4csc' in cfg_path:
            self.model = SoftMaskedBert4Csc.load_from_checkpoint(checkpoint_path=ckpt_path,
                                                                 cfg=cfg,
                                                                 map_location=device,
                                                                 tokenizer=self.tokenizer)
        else:
            raise ValueError("model not found.")
        self.model.to(device)
        self.model.eval()

    def predict(self, sentence_list):
        """
        文本纠错模型预测
        Args:
            sentence_list: list
                输入文本列表
        Returns: tuple
            corrected_texts(list)
        """
        is_str = False
        if isinstance(sentence_list, str):
            is_str = True
            sentence_list = [sentence_list]
        corrected_texts = self.model.predict(sentence_list)
        if is_str:
            return corrected_texts[0]
        return corrected_texts

    def predict_with_error_detail(self, sentence_list):
        """
        文本纠错模型预测，结果带错误位置信息
        Args:
            sentence_list: list
                输入文本列表
        Returns: tuple
            corrected_texts(list), details(list)
        """
        details = []
        is_str = False
        if isinstance(sentence_list, str):
            is_str = True
            sentence_list = [sentence_list]
        corrected_texts = self.model.predict(sentence_list)

        for corrected_text, text in zip(corrected_texts, sentence_list):
            corrected_text, sub_details = get_errors(corrected_text, text)
            details.append(sub_details)
        if is_str:
            return corrected_texts[0], details[0]
        return corrected_texts, details


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="infer")
    parser.add_argument(
        "--ckpt_path",
        default="output/macbert4csc/epoch=01-val_loss=0.03.ckpt",
        help="path to config file",
        type=str)
    parser.add_argument("--vocab_path", default="output/macbert4csc/vocab.txt", help="path to config file", type=str)
    parser.add_argument("--config_file", default="train_macbert4csc.yml", help="path to config file", type=str)
    args = parser.parse_args()
    m = Inference(args.ckpt_path, args.vocab_path, args.config_file)
    inputs = [
        "故东出资部到位10300万元", "研究原制定的提成将理制度为经有关部门批准",
        "总监理工程师符合召标文件资质要囚地只有陕西建案工承监理有限公司和陕西汇源环境功程监理有限功司", "安约定范为内公成造价报价比率确定",
        "导致预算收入编制不完争", "保正各项负债在规定期限内完成", "签订工程造价协意书",
        "研究元除在管理费中列支业务招待费16000元外", "研究远除在管理费中列支业务招待费16000元外",
        "致使成际公司对中铁十七具公成指挥部申报的机场站工程第二期已完成工程量合渭河特大桥工程第二", "积极完善了相官内控制度",
        "基础教与数字成果评审", "部分固定资产实物与卡片信息部符", "不符合《中华人民共和国合童法》第八条依法成立的合同"
    ]
    outputs = m.predict(inputs)
    for a, b in zip(inputs, outputs):
        print('input  :', a)
        print('predict:', b)
        print()

    # # 在sighan2015 test数据集评估模型
    # # macbert4csc Sentence Level: acc:0.7845, precision:0.8174, recall:0.7256, f1:0.7688, cost time:10.79 s
    # # softmaskedbert4csc Sentence Level: acc:0.6964, precision:0.8065, recall:0.5064, f1:0.6222, cost time:16.20 s
    # from pycorrector.utils.eval import eval_sighan2015_by_model

    # eval_sighan2015_by_model(m.predict_with_error_detail)
