
###1
import pandas as pd
import numpy as np
titanic_train=pd.read_csv('C:/Users/Administrator/ml_python/ex1/Datasets/titanic_kaggle/train.csv')
titanic_test=pd.read_csv('C:/Users/Administrator/ml_python/ex1/Datasets/titanic_kaggle/test.csv')
titanic_train['FamilySize'] = titanic_train['SibSp'] + titanic_train['Parch']
titanic_train['Age'].fillna(titanic_train['Age'].median(),inplace=True)
titanic_train['Embarked'].fillna('S',inplace=True)
titanic_train['Fare'].fillna(titanic_train['Fare'].mean(),inplace=True)
X_train=titanic_train[['Pclass','Age','Sex','FamilySize','Fare','Embarked']]
y_train=titanic_train['Survived']
#feature extraction
from sklearn.feature_extraction import DictVectorizer
DV=DictVectorizer(sparse=False)
X_train=DV.fit_transform(X_train.to_dict(orient='record'))
#test sets
titanic_test=pd.read_csv('C:/Users/Administrator/ml_python/ex1/Datasets/titanic_kaggle/test.csv')
titanic_test['FamilySize'] = titanic_test['SibSp'] + titanic_test['Parch']
titanic_test['Age'].fillna(titanic_train['Age'].median(),inplace=True)
titanic_test['Embarked'].fillna('S',inplace=True)
titanic_test['Fare'].fillna(titanic_test['Fare'].mean(),inplace=True)
X_test=titanic_test[['Pclass','Age','Sex','FamilySize','Fare','Embarked']]
X_test=DV.transform(X_test.to_dict(orient='record'))

#initialize training environment
from sklearn.ensemble import GradientBoostingClassifier
GBDT=GradientBoostingClassifier()



GBDT.fit(X_train,y_train)
GBDT_y_predict=GBDT.predict(X_test)
titanic_test['Survived'] = GBDT_y_predict


#submission
titanic_test[['PassengerId', 'Survived']].to_csv('C:/Users/Administrator/ml_python/ex1/Datasets/titanic_kaggle/GBDT_419.csv', index=False)
    