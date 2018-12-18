#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 12:10:35 2018

@author: tom
"""

import sys, os
import opencc
from pyltp import Postagger,Segmentor,SentenceSplitter,NamedEntityRecognizer,Parser,SementicRoleLabeller
t2s = opencc.OpenCC('t2s')
s2t = opencc.OpenCC('s2t')


# sentence 分詞
inputs = '我想要去台北。謝謝。'
inputs = t2s.convert(inputs)
sentence = SentenceSplitter.split(inputs)[0]
segmentor = Segmentor()
segmentor.load("./ltp_data_v3.4.0/cws.model")
words = segmentor.segment(sentence)
print("\t".join(words))





# Parts of Speech(POS) Tagging  詞性標注
postagger = Postagger()
postagger.load('./ltp_data_v3.4.0/pos.model')
postags = postagger.postag(words)
print("\t".join(postags))

# 依存句法分析
parser = Parser()
parser.load("./ltp_data_v3.4.0/parser.model")
arcs = parser.parse(words, postags)
print("\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs))

# 實體命名識別 地名(Ns) 機構名(Ni) 人名(Nh)
recognizer = NamedEntityRecognizer()
recognizer.load("./ltp_data_v3.4.0/ner.model")
netags = recognizer.recognize(words, postags)
print("\t".join(netags))


labeller = SementicRoleLabeller()
labeller.load("./ltp_data_v3.4.0/pisrl.model")
roles = labeller.label(words, postags, arcs)

for role in roles:
    print(role.index, "".join(["%s:(%d,%d)" % (arg.name, arg.range.start, arg.range.end) for arg in role.arguments]))
    