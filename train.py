#coding=utf-8
import csv
import os
import numpy as np
import pandas as pd
import jieba
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_classif
from tqdm import tqdm
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def stopwordslist():
    stopwords = [line.strip() for line in open('null.txt', encoding='UTF-8').readlines()]
    return stopwords

def seg_depart(sentence, stp):
    # 对文档中的每一行进行中文分词
    # print("正在分词")
    sentence_depart = jieba.cut(str(sentence).strip())
    # 创建一个停用词列表
    stopwords = stp
    # 输出结果为outstr
    outstr = ''
    # 去停用词
    for word in sentence_depart:
        if word not in stopwords:
            outstr += word+" "

    return outstr

def data_get():
    data_0 = pd.read_csv("体育.csv")
    data_1 = pd.read_csv("彩票.csv")
    data_2 = pd.read_csv("房产.csv")
    data_3 = pd.read_csv("时政.csv")
    data_4 = pd.read_csv("社会.csv")
    data_5 = pd.read_csv("股票.csv")
    data_6 = pd.read_csv("财经.csv")
    data = pd.concat([data_0, data_1, data_2, data_3, data_4, data_5, data_6]).reset_index(drop = True)
    return data

def train():
    x_train,x_test,y_train,y_test = train_test_split(seg_depart(data['text']), data.class_name, test_size=0.2, random_state=2020)

    pipe = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer(binary=True)),
                     ('model', LogisticRegression())])

    model = pipe.fit(x_train, y_train)
    prediction = model.predict(x_test)
    print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))

if __name__ == '__main__':
    data = data_get()
    newData = pd.Series(len(data.text))
    # newData[0] = seg_depart(data.text[0])
    # newData[1] = seg_depart(data.text[1])
    # print(newData)
    stp = stopwordslist()
    for i in tqdm(range(len(data.text))):
        newData[i] = seg_depart(data.text[i],stp)
    x_train, x_test, y_train, y_test = train_test_split(newData, data.class_name, test_size=0.2,
                                                        random_state=2020)

    pipe = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('model', MultinomialNB())])

    model = pipe.fit(x_train, y_train)
    prediction = model.predict(x_test)
    print("accuracy: {}%".format(round(accuracy_score(y_test, prediction) * 100, 2)))
    print(confusion_matrix(y_test, prediction))
    print(classification_report(y_test, prediction))
    # print("\n")
    # print(pd.concat([data.class_name[:10], newData],axis=1))
    # with open('null.txt', 'rb') as fp:
    #     stopword = fp.read().decode('utf-8')
    # stpwrdlst = stopword.splitlines()
    #
    # X, Y = newData, data.class_name
    # cv = CountVectorizer(
    #                      max_features=20,
    #                      stop_words = stpwrdlst)
    # X_vec = cv.fit_transform(X)
    #
    # res = dict(zip(cv.get_feature_names(),
    #                mutual_info_classif(X_vec, Y, discrete_features=True)
    #                ))
    # print(res)




#words = seg_depart(df)

#df = df.append(df1)
#df.columns = ["123","456"]  # 添加表头
#df.to_csv("test.csv", index=None,mode='a',encoding='utf-8')
#print(words)

#word_count = []
#count = Counter(str(words).split())
#word_count.append(count)

#print(word_count)
#count = CountVectorizer()
#tf=count.fit_transform(words)
#sorted_item=sorted(count.vocabulary_.items(), key= lambda d:d[1], reverse=False)
#print(sorted_item)