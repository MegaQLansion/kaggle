# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 13:08:14 2018

@author: Lansion
"""
import pandas as pd
imdb_train=pd.read_csv('C:/Users/Administrator/ml_python/ex1/Datasets/imdb_kaggle/labeledTrainData.tsv',delimiter='\t')
imdb_test=pd.read_csv('C:/Users/Administrator/ml_python/ex1/Datasets/imdb_kaggle/testData.tsv',delimiter='\t')

#initialization
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

#preprocessing function
def review_to_text(review,remove_stopwords):
    #remove html marker
    raw_text=BeautifulSoup(review,'html').get_text()
    #remove non-character sign
    letters=re.sub('[^a-zA-Z]','',raw_text)
    words=letters.lower().split()
    #remove stop words in review
    if remove_stopwords:
        stop_words=set(stopwords.words('english'))
        words=[w for w in words if w not in stop_words]
    return words

#preprocessing to train and test raw data
#==============================================================================
# X_train=[]
# for review in imdb_train['review']:
#     X_train.append(''.join(review_to_text(review,True)))
# X_test=[]
# for review in imdb_test['review']:
#     X_test.append(''.join(review_to_text(review,True)))
#==============================================================================

y_train=imdb_train['sentiment']
#setting pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
pip_CV=Pipeline([('CV',CountVectorizer(analyzer='word')),('MNB',MultinomialNB())])
pip_TV=Pipeline([('TV',TfidfVectorizer(analyzer='word')),('MNB',MultinomialNB())])

#setting parameters
params_CV={'CV__binary':[True,False],'CV__ngram_range':[(1,1),(1,2)],'MNB__alpha':[0.1,1.0,10.0]}
params_TV={'TV__binary':[True,False],'TV__ngram_range':[(1,1),(1,2)],'MNB__alpha':[0.1,1.0,10.0]}

#==============================================================================
# #Grid Search(CV)
# GS_CV=GridSearchCV(pip_CV,params_CV,cv=4,verbose=1)
# GS_CV.fit(X_train,y_train)
# print 'The best accuracy gained by grid seach with CV is',GS_CV.best_score_,'with a parameter of',GS_CV.best_params_
# #Grid Search(TV)
# GS_TV=GridSearchCV(pip_TV,params_TV,cv=4,verbose=1)
# GS_TV.fit(X_train,y_train)
# print 'The best accuracy gained by grid seach with TV is',GS_TV.best_score_,'with a parameter of',GS_TV.best_params_
# 
# #prediction
# CV_y_predict=GS_CV.predict(X_test)
# TV_y_predict=GS_TV.predict(X_test)
#==============================================================================

submission_CV=pd.DataFrame({'id':imdb_test['id'],'sentiment':CV_y_predict})
submission_TV=pd.DataFrame({'id':imdb_test['id'],'sentiment':TV_y_predict})
submission_CV.to_csv('C:/Users/Administrator/ml_python/ex1/Imdb/CV_418.csv',index=False)
submission_TV.to_csv('C:/Users/Administrator/ml_python/ex1/Imdb/TV_418.csv',index=False)
unlabeled_train=pd.read_csv('C:/Users/Administrator/ml_python/ex1/Datasets/imdb_kaggle/unlabeledTrainData.tsv',delimiter='\t',quoting=3)
import nltk.data
tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')

#define sentence generator
def review_to_sentences(review,tokenizer):
    raw_sentence=tokenizer.tokenize(review.strip())
    sentences=[]
    for raw_sentence in raw_sentence:
        if len(raw_sentence)>0:
            sentences.append(review_to_text(raw_sentence,False))
    return sentences

#==============================================================================
# #preparing data for w2v
# corpora=[]
# for review in unlabeled_train['review']:
#     corpora+=review_to_sentences(review.decode('utf8'),tokenizer)
#==============================================================================
    
#hyperparameters for w2v
num_features=300
min_word_count=20
context=10
downsampling=1e-3
#w2v
from gensim.models import word2vec
model=word2vec.Word2Vec(corpora,size=num_features,min_count=min_word_count,window=context,sample=downsampling)
#==============================================================================
# model.init_sims(replace=True)
# model_name="C:/Users/Administrator/ml_python/ex1/Imdb/F300_M20_C10"
# model.save(model_name)
#==============================================================================
#data cleaning
import numpy as np
def makeFeatureVec(words,model,num_features):
    featureVec=np.zeros((num_features,),dtype='float32')
    nwords=0
    index2word_set=set(model.wv.index2word)
    for word in words:
        if word in index2word_set:
            nwords=nwords+1
            featureVec=np.add(featureVec,model[word])
    featureVec=np.divide(featureVec,nwords)
    return featureVec
#
#==============================================================================
# clean_train_reviews=[]
# for review in imdb_train["review"]:
#     clean_train_reviews.append(review_to_text(review,remove_stopwords=True))
#==============================================================================
def getAvgFeatureVecs(reviews,model,num_features):
    counter=0
    reviewFeatureVecs=np.zeros((len(reviews),num_features),dtype='float32')
    for review in reviews:
        reviewFeatureVecs[counter]=makeFeatureVec(review,model,num_features)
        counter+=1
    return reviewFeatureVecs
Train_Vec=getAvgFeatureVecs(clean_train_reviews,model,num_features) 
#hyperparameters searching
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
GBC=GradientBoostingClassifier()
params_GBC={'n_estimators':[10,100,500],'learning_rate':[0.01,0.1,1.0],'max_depth':[2,3,4]}
GS=GridSearchCV(GBC,params_GBC,cv=4,verbose=1)
#==============================================================================
# GS.fit(Train_Vec,y_train)
# print 'The best accuracy is',GS.best_score_,'with params of',GS.best_params_
#==============================================================================
