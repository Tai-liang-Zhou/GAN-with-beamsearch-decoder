#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 17:31:16 2018
@author: tom
from : https://github.com/tensorflow/nmt/blob/master/nmt/scripts/bleu.py
Python implementation of BLEU and smooth-BLEU.
This module provides a Python implementation of BLEU and smooth-BLEU.
Smooth BLEU is computed following the method outlined in the paper:
Chin-Yew Lin, Franz Josef Och. ORANGE: a method for evaluating automatic
evaluation metrics for machine translation. COLING 2004.
"""
from nltk.translate.bleu_score import sentence_bleu

#reference_corpus = [['我', '也', '喜歡', '這', '條', '魚','，', '但', '我', '最', '喜歡', '甜點' ,'。']]
#translation_corpus = ['我', '也', '喜歡', '這', '條', '魚', '，', '什麼', '？']
reference_corpus = [['the', 'quick', 'brown','fox', 'jumped', 'over', 'the', 'lazy', 'dog']]

translation_corpus = ['the', 'fast', 'brown','fox', 'jumped', 'over', 'the', 'lazy', 'dog']
print('Cumulative 1-gram: %f' %
      sentence_bleu(reference_corpus, translation_corpus, weights=(1, 0, 0, 0)))
print('Cumulative 2-gram: %f' % sentence_bleu(reference_corpus,translation_corpus, weights=(0.5, 0.5, 0, 0)))
print('Cumulative 3-gram: %f' % sentence_bleu(reference_corpus,translation_corpus, weights=(0.33, 0.33, 0.33, 0)))
print('Cumulative 4-gram: %f' % sentence_bleu(reference_corpus,translation_corpus, weights=(0.25, 0.25, 0.25, 0.25)))

references = [['The', 'cat', 'is', 'on', 'the', 'mat']]
candidate = ['The', 'dog', 'is', 'on', 'the', 'mat']
score = sentence_bleu(references, candidate)
print(round(score, 5))



references = [['嗨瑪麗', '我們', '真的', '很', '期待', '這裡', '美食', '。']]
candidate1 = ['謝謝', '。', '請', '知道', '到', '岩石', '上', '是', '你', '了', '。']
score = sentence_bleu(references, candidate1, weights=(0.25, 0.25, 0.25, 0.25))
print(round(score, 5))


references = [['你', '的', '酒單', '在', '哪裡', '？']]
candidate = ['當然', '有', '，', '好', '。']
score = sentence_bleu(references, candidate, weights=(0.25, 0.25, 0.25, 0.25))
print(score)


references = [['我', '喜歡', '很多', '東西', '，', '但', '泰國', '或','日本' ,'人','會','很好','。']]
candidate = ['我', '喜歡', '很多', '東西', '，', '但', '泰國', '或','日本' ,'人','時','。']
score = sentence_bleu(references, candidate, weights=(0.25, 0.25, 0.25, 0.25))
print(score)










from nltk.translate.bleu_score import sentence_bleu
reference = [['this', 'is', 'small', 'test']]
candidate = ['this', 'is', 'a', 'test']
score = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
print(score)


reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['the', 'fast', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']
score = sentence_bleu(reference, candidate)
print(score)

# two words different
from nltk.translate.bleu_score import sentence_bleu
reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['the', 'fast', 'brown', 'fox', 'jumped', 'over', 'the', 'sleepy', 'dog']
score = sentence_bleu(reference, candidate)
print(score)