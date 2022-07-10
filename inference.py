import sys

sys.path.append("..")
from pycorrector.macbert.macbert_corrector import MacBertCorrector

if __name__ == '__main__':
    error_sentences = [
        "故东出资部到位10300万元", "研究原制定的提成将理制度为经有关部门批准",
        "总监理工程师符合召标文件资质要囚地只有陕西建案工承监理有限公司和陕西汇源环境功程监理有限功司", "安约定范为内公成造价报价比率确定",
        "导致预算收入编制不完争", "保正各项负债在规定期限内完成", "签订工程造价协意书",
        "研究元除在管理费中列支业务招待费16000元外", "研究远除在管理费中列支业务招待费16000元外",
        "致使成际公司对中铁十七具公成指挥部申报的机场站工程第二期已完成工程量合渭河特大桥工程第二", "积极完善了相官内控制度",
        "基础教与数字成果评审", "部分固定资产实物与卡片信息部符", "不符合《中华人民共和国合童法》第八条依法成立的合同"
    ]

    m = MacBertCorrector("./macbert4csc")
    for line in error_sentences:
        correct_sent, err = m.macbert_correct(line)
        print("query:{} => predict: {}".format(
            line,
            correct_sent))
        print("err:{}\n".format(err))
