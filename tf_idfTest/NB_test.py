import jieba
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
#读入停用词表
def load_stop_words():
    path="stop_words.txt"
    file=open(path,'rb').read().decode('utf-8').split('\r\n')
    return list(file)
#文本处理
def TextProcessing(filepath,test_size):
    data_list=[]
    class_list=[]
    for file in filepath:
        data=pd.read_csv(file)
        for text in data['data']:
            data_list.append(text)
            class_list.append(file)
    seg_list=[' '.join(jieba.cut(str(text)))for text in data_list]
    tfidf_vec=TfidfVectorizer(stop_words=load_stop_words())
    tfidf=tfidf_vec.fit_transform(seg_list)
    train_data_list, test_data_list, train_class_list, test_class_list = train_test_split(tfidf, class_list, test_size=test_size)
    return  seg_list,train_data_list,test_data_list,train_class_list,test_class_list
#模型训练，测试
def TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list):
    classifier = MultinomialNB().fit(train_feature_list, train_class_list)
    test_accuracy = classifier.score(test_feature_list, test_class_list)
    return test_accuracy
filepath=['data/sport.csv','data/fannao.csv','data/game.csv','data/shangye.csv','data/daily.csv','data/education.csv']
all_word_list,train_data_list,test_data_list,train_class_list,test_class_list=TextProcessing(filepath,test_size=0.4)
# print(train_data_list.shape())
# print(test_data_list.shape())
deleteNs=range(0,100,20)
test_accuracy_list=[]
for deleteN in deleteNs:
    test_accuracy=TextClassifier(train_data_list,test_data_list,train_class_list,test_class_list)
    test_accuracy_list.append(test_accuracy)
print(test_accuracy_list)