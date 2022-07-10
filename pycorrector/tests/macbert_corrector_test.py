# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com), Abtion(abtion@outlook.com)
@description:
"""
import sys
import time

sys.path.append("../")
from pycorrector.macbert import macbert_corrector

error_sentences = [
    '首金得主杨倩、三跳满分的全红婵、举重纪录创造者李雯雯……“00后”选手闪曜奥运舞台。',
    '首金得主银色的k2p, ，。，mM是MTKkE2.iiKz还是舞台。',
    '真麻烦你了。希望你们好好的跳无',
    '少先队员因该为老人让坐',
    '少 先  队 员 因 该 为 老人让坐',
    '机七学习是人工智能领遇最能体现智能的一个分知',
    '今天心情很好',
    '汽车新式在这条路上',
    '中国人工只能布局很不错',
    '想不想在来一次比赛',
    '你不觉的高兴吗',
    '权利的游戏第八季',
    '美食美事皆不可辜负，这场盛会你一定期待已久',
    '点击咨询痣疮是什么原因?咨询医师痣疮原因',
    '附睾焱的症状?要引起注意!',
    '外阴尖锐涅疣怎样治疗?-济群解析',
    '洛阳大华雅思 30天突破雅思7分',
    '男人不育少靖子症如何治疗?专业男科,烟台京城医院',
    '疝気医院那好 为老人让坐，疝気专科百科问答',
    '成都医院治扁平苔鲜贵吗_国家2甲医院',
    '少先队员因该为老人让坐',
    '服装店里的衣服各试各样',
    '一只小鱼船浮在平净的河面上',
    '我的家乡是有明的渔米之乡',
    ' _ ,',
    '我对于宠物出租得事非常认同，因为其实很多人喜欢宠物',  # 出租的事
    '有了宠物出租地方另一方面还可以题高人类对动物的了解，因为那些专业人氏可以指导我们对于动物的习惯。',  # 题高 => 提高 专业人氏 => 专业人士
    '三个凑皮匠胜过一个诸葛亮也有道理。',  # 凑
    '还有广告业是只要桌子前面坐者工作未必产生出来好的成果。',
]

badcase = ['这个跟 原木纯品 那个啥区别？不是原木纸浆做的?',
           '能充几次呢？',
           '这是酸奶还是像饮料一样的奶？',
           '现在银色的K2P是MTK还是博通啊？',
           '是浓稠的还是稀薄的？',
           '这个到底有多辣',
           'U盘有送挂绳吗 ',
           '果子酸吗？有烂的吗？',
           '刚下单买了一箱，需要明天到货，先问下味道如何',
           '2周岁22斤宝宝用多大的啊？',
           '请问这茶是一条装的吗',
           '有坏的果吗',
           '生产日期怎么样 新嘛',
           '插上去的时候是驱蚊液放下面的吗？',
           '橄榄的和这款哪个比较好用？味道都是一样的么？',
           '妈妈说：”以后做错了要主动人错哦，别赖。“，我说：”我知道了。“',
           '他在唱”听妈妈的话“，真的很好听呢',
           '我最爱看的是《巴黎圣母院》这书，里面思想深邃，值得回味。',
           ]
error_sentences.extend(badcase)
start = time.time()
bertCorrector = macbert_corrector.MacBertCorrector()
for line in error_sentences:
    correct_sent, err = bertCorrector.macbert_correct(line)
    print("original sentence:{} => {} err:{}".format(line, correct_sent, err))
print('time spend:', time.time() - start, ' sentence count:', len(error_sentences))
