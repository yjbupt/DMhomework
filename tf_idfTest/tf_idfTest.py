from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import  jieba
import  pandas as pd
# a ="自然语言处理是计算机科学领域与人工智能领域中的一个重要方向。它研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法。自然语言处理是一门融语言学、计算机科学、数学于一体的科学"
# b = "因此，这一领域的研究将涉及自然语言，即人们日常使用的语言，所以它与语言学的研究有着密切的联系，但又有重要的区别。自然语言处理并不是一般地研究自然语言，而在于研制能有效地实现自然语言通信的计算机系统，特别是其中的软件系统。"
# c ="因而它是计算机科学的一部分。自然语言处理（NLP）是计算机科学，人工智能，语言学关注计算机和人类（自然）语言之间的相互作用的领域。"
# d="'英格兰你让我很失望我真的十分失望啊怎摸会这样一场友谊赛不用太当真'"
# str_list=[a,b,c,d]
# print(type(a))
# all_list= ['  '.join(jieba.cut(s,cut_all = False)) for s in str_list]
# print(all_list)
# tfidf_vec=TfidfVectorizer()
# tfidf=tfidf_vec.fit_transform(all_list).toarray()
# print(tfidf.todense)
data=pd.read_csv('data/sport.csv')
text_data=[]
seg_text=[]
for text in data['data']:
    text_data.append(text)
all_list=[' '.join(jieba.cut(s))for s in text_data]
#print(all_list)
def load_stop_words():
    path="stop_words.txt"
    file=open(path,'rb').read().decode('utf-8').split('\r\n')
    return list(file)
stopwords=load_stop_words()
#with open('stop_words.txt','rb') as fp:
    #stopword=fp.read().decode('utf-8')
#stopwordlist=stopword.splitlines()
tfidf_vec=TfidfVectorizer(stop_words=stopwords)
tfidf=tfidf_vec.fit_transform(all_list).toarray()
word=tfidf_vec.get_feature_names()
for j in range(len(word)):
    if tfidf[0][j]>0:
        print(word[j],tfidf[0][j])

