

import jieba
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
#读入停用词表
from sklearn.naive_bayes import MultinomialNB


def load_stop_words():
    path="stop_words.txt"
    file=open(path,'rb').read().decode('utf-8').split('\r\n')
    return list(file)
#去除停用词
def rm_stop_words(word_list):
    stop_words=load_stop_words()
    for i in range(word_list.__len__())[::-1]:
        if word_list[i] in stop_words:
            word_list.pop(i)
    return word_list
# 去除频率低的词
def rm_word_freq_so_little(dictionary, freq_thred):
    small_freq_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq < freq_thred]
    dictionary.filter_tokens(bad_ids=small_freq_ids)
    dictionary.compactify()
def TextProcessing(filepath,test_size):
    data_list=[]
    class_list=[]
    for file in filepath:
        data=pd.read_csv(file)
        for text in data['data']:
            text=list(jieba.cut(str(text)))
            text=rm_stop_words(text)
            data_list.append(text)
            class_list.append(file)
    data_class_list=list(zip(data_list,class_list))
    random.shuffle(data_class_list)
    index=int(len(data_class_list)*test_size)+1
    train_list=data_class_list[index:]
    test_list=data_class_list[:index]
    train_data_list,train_class_list=zip(*train_list)
    test_data_list,test_class_list=zip(*test_list)

    #all_words_dict = {}
    # for word_list in train_data_list:
    #     for word in word_list:
    #         if word in all_words_dict:
    #             all_words_dict[word] += 1
    #         else:
    #             all_words_dict[word] = 1
    # # key函数利用词频进行降序排序
    # all_words_tuple_list = sorted(all_words_dict.items(), key=lambda f: f[1], reverse=True)  # 内建函数sorted参数需为list
    # all_words_list = list(list(zip(*all_words_tuple_list))[0])
    return  all_words_list,train_data_list,test_data_list,train_class_list,test_class_list
def word_dict(all_word_list,deleteN):
    feature_words=[]
    n=1
    for t in range(deleteN,len(all_word_list),1):
        if n>4000:
            break
        feature_words.append(all_word_list[t])
        n+n+1
        return  feature_words
def TextFeatures(train_data_list, test_data_list, feature_words):
    def text_features(text, feature_words):
        text_words = set(text)
        features = [1 if word in text_words else 0 for word in feature_words]
        return features
    train_feature_list = [text_features(text, feature_words) for text in train_data_list]
    test_feature_list = [text_features(text, feature_words) for text in test_data_list]
    return train_feature_list, test_feature_list
def TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list):
    classifier = MultinomialNB().fit(train_feature_list, train_class_list)
    test_accuracy = classifier.score(test_feature_list, test_class_list)
    return test_accuracy
filepath=['data/sport.csv','data/fannao.csv','data/computer.csv','data/education.csv','data/game.csv','data/yule.csv','data/health.csv']
all_word_list,train_data_list,test_data_list,train_class_list,test_class_list=TextProcessing(filepath,test_size=0.4)
deleteNs=range(0,100,20)
test_accuracy_list=[]
for deleteN in deleteNs:
    feature_words=word_dict(all_word_list,deleteN)
    train_feature_list,test_feature_list=TextFeatures(train_data_list,test_data_list,feature_words)
    test_accuracy=TextClassifier(train_feature_list,test_feature_list,train_class_list,test_class_list)
    test_accuracy_list.append(test_accuracy)
print(test_accuracy_list)