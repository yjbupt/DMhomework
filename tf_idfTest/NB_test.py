import jieba
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from sklearn.naive_bayes import MultinomialNB
def load_file_path(path):
    file_list=[]
    list_dir=os.listdir(path)
    for file in list_dir:
        file_path=os.path.join(path,file)
        file_list.append(file_path)
    return file_list
#读入停用词表
def load_stop_words():
    path="stop_words.txt"
    file=open(path,'rb').read().decode('utf-8').split('\r\n')
    return list(file)
#文本处理
def TextProcessing(filepath,test_size,max_df_test):
    data_list=[]
    class_list=[]
    for file in filepath:
        data=pd.read_csv(file)
        for text in data['data']:
            data_list.append(text)
            class_list.append(file)
    seg_list=[' '.join(jieba.cut(str(text)))for text in data_list]
    tfidf_vec=TfidfVectorizer(stop_words=load_stop_words(),max_df=max_df_test)
    tfidf=tfidf_vec.fit_transform(seg_list)
    train_data_list, test_data_list, train_class_list, test_class_list = train_test_split(tfidf, class_list, test_size=test_size)
    return  seg_list,train_data_list,test_data_list,train_class_list,test_class_list
#模型训练，测试
def TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list):
    classifier = MultinomialNB().fit(train_feature_list, train_class_list)
    test_accuracy = classifier.score(test_feature_list, test_class_list)
    return test_accuracy
path='data'
filepath=load_file_path(path)
# print(train_data_list.shape())
# print(test_data_list.shape())
test_accuracy_list=[]
for max_df_test in [0.5,0.6,0.7,0.8,0.9,1.0]:
    all_word_list, train_data_list, test_data_list, train_class_list, test_class_list = TextProcessing(filepath, test_size=0.4,max_df_test=max_df_test)
    test_accuracy=TextClassifier(train_data_list,test_data_list,train_class_list,test_class_list)
    test_accuracy_list.append(test_accuracy)
print(test_accuracy_list)