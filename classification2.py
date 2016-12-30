# coding: utf-8
from __future__ import division
import nltk
import codecs
from collections import defaultdict
from math import log
from nltk.corpus import stopwords


def train(samples):
    classes, freq = defaultdict(lambda: 0), defaultdict(lambda: 0)

    for label, feats in samples:
        classes[label] += 1  # count classes frequencies
        for feat in feats:
            freq[label, feat] += 1  # count features frequencies
    print(freq)
    for label, feat in freq:  # normalize features frequencies
        freq[label, feat] /= classes[label]
    for c in classes:  # normalize classes frequencies
        classes[c] /= len(samples)

    return classes, freq  # return P(C) and P(O|C)


def classify(classifier, feats):
    classes, prob = classifier
    return min(classes.keys(),  # calculate argmin(-log(C|O))
               key=lambda cl: -log(classes[cl]) + \
                              sum(-log(prob.get((cl, feat), 10 ** (-7))) for feat in feats))


def get_features(sample): return [('ll: %s' % line) for line in sample]

def stop_w(line):
    stop_words = stopwords.words('russian')  # обозначаем стоп слова
    stop_words.extend(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', '—', 'к', 'на'])
    line = [i for i in line if (i not in stop_words)]
    return line

def good_form(line):  # line treatment
    line.replace("(", "").replace(")", "").replace("«", "").replace("»", "").replace("'", "").replace("'", "") \
        .replace("`", "").replace("`", "").replace(".", " ").replace("-", " ").replace("—", " ").lower()
    line = nltk.word_tokenize(line)
    line = stop_w(line)
    return line


samples = (good_form(line) for line in codecs.open('news_train3.txt', 'r', "utf_8"))
features = [(line.pop(0), get_features(line)) for line in samples]
classifier = train(features)


filename = u'news_test.txt'
f = open(u'Out.txt', 'w', encoding='UTF-8')
try:  # Для обработки отсутствия файла
    with codecs.open(filename, encoding='UTF-8') as file_object:  # with закроет файл, если что-то пойдет не так
        for line in file_object:  # такой подход как раз позволяет считывать по строкам его
            line_test = line.rstrip('\n')
            f.write(classify(classifier, get_features(good_form(line_test))) + '\n')
except IOError as er:  # Обработка отсутствия файла
    print(u'Can\'t open the "{0}" file'.format(
        er.filename))
f.close()
