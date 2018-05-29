import warnings
warnings.filterwarnings('ignore')
import numpy as np 
import pandas as pd 
train_sets = pd.read_csv("C:/Users/Administrator/ml_python/ex1/kaggle/Project/Proj5_MB/train.csv")
test_sets = pd.read_csv("C:/Users/Administrator/ml_python/ex1/kaggle/Project/Proj5_MB/test.csv")


upper_bound = 175
feature_train= train_sets.drop(['ID','y'], axis=1)
feature_test= test_sets.drop(['ID'], axis=1)
train_sets['y'].loc[train_sets['y']>upper_bound] = upper_bound
drop_list=[]
for features in feature_train.columns:
    if features not in ['X0','X1','X2','X3','X4','X5','X6','X8']:
        if sum(feature_train[features])<=train_sets.shape[0]*0.002:
            drop_list.append(features)
            train_sets= train_sets.drop([features], axis=1)
            test_sets= test_sets.drop([features], axis=1)
        if sum(feature_train[features])>=train_sets.shape[0]*0.998:
            drop_list.append(features)
            train_sets= train_sets.drop([features], axis=1)
            test_sets= test_sets.drop([features], axis=1)
for features in ['X0','X1','X2','X3','X4','X5','X6','X8']:
    feature_classification = np.sort(train_sets[features].unique()).tolist()
    drop_list=[]
    label=[]
    for cls in feature_classification:
        temp = train_sets.loc[train_sets[features] == cls]
        if temp.shape[0]<=train_sets.shape[0]*0.001:
            drop_list.append(cls)
    for drops in drop_list:
        train_sets[features][train_sets.loc[(train_sets[features]==drops)==True][features].index]='zero'
        test_sets[features][test_sets.loc[(test_sets[features]==drops)==True][features].index]='zero'
def classification(sets):
    for ones in ['t','ak','z','j','y','aj']:
        sets['X0'][sets.loc[(sets['X0']==ones)==True][features].index]='one'
   

classification(train_sets)
classification(test_sets)

##

X_train=train_sets.drop(['y','ID'], axis=1)
y_train=train_sets['y']
X_test=test_sets.drop(['ID'], axis=1)



from sklearn.feature_extraction import DictVectorizer
DV=DictVectorizer(sparse=False)
X_train=DV.fit_transform(X_train.to_dict(orient='record'))
X_test=DV.transform(X_test.to_dict(orient='recored'))

#feature scaling
from sklearn.preprocessing import StandardScaler
ss_X=StandardScaler()
ss_y=StandardScaler()
X_train=ss_X.fit_transform(X_train)
X_test=ss_X.transform(X_test)
y_train=np.log1p(y_train)

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor
XGBR=XGBRegressor()
RFR=RandomForestRegressor()
ETR=ExtraTreesRegressor()
GBR=GradientBoostingRegressor()

XGBR.fit(X_train,y_train)
RFR.fit(X_train,y_train)
ETR.fit(X_train,y_train)
GBR.fit(X_train,y_train)

XGBR_y_predict=XGBR.predict(X_test)
RFR_y_predict=RFR.predict(X_test)
ETR_y_predict=ETR.predict(X_test)
GBR_y_predict=GBR.predict(X_test)


XGBR_y_predict=np.expm1(XGBR_y_predict)
RFR_y_predict=np.expm1(RFR_y_predict)
ETR_y_predict=np.expm1(ETR_y_predict)
GBR_y_predict=np.expm1(GBR_y_predict)

params=[0.8,0.1,0.05,0.05]
test_sets['y']=params[0]*XGBR_y_predict+params[1]*RFR_y_predict+params[2]*ETR_y_predict+params[3]*GBR_y_predict
XGBR_submission=test_sets[['ID','y']]
XGBR_submission[['ID','y']].to_csv('C:/Users/Administrator/ml_python/ex1/kaggle/Project/Proj5_MB/submission.csv', index=False)
print ('Done')
