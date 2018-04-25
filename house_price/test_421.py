

import pandas as pd
import numpy as np
hp_train=pd.read_csv('C:/Users/Administrator/ml_python/ex1/Datasets/house_pricing/train.csv')
hp_test=pd.read_csv('C:/Users/Administrator/ml_python/ex1/Datasets/house_pricing/test.csv')

hp_train['PorchSF']=hp_train['OpenPorchSF']+hp_train['EnclosedPorch']
hp_train= hp_train.drop(['LotArea','LotShape','HouseStyle','OverallCond','BsmtFinSF2','TotRmsAbvGrd','GarageType','GarageFinish','GarageCond','Street','Alley','Utilities','Condition1','Condition2','BldgType','RoofMatl','BsmtFinType1','BsmtFinSF1','BsmtFinType2','Heating','Functional','Fireplaces','PoolQC','Fence','MiscFeature','SaleCondition','RoofStyle','BedroomAbvGr','FireplaceQu','MoSold','BsmtFullBath','BsmtHalfBath','MiscVal','LowQualFinSF','3SsnPorch','ScreenPorch','PoolArea','OpenPorchSF','EnclosedPorch','LandContour','LotConfig'], axis=1)
hp_train['YrSold'] = 2018-hp_train['YrSold'] 
hp_train['GarageYrBlt'] = 2018-hp_train['GarageYrBlt'] 
hp_train['YearBuilt'] = 2018-hp_train['YearBuilt'] 
hp_train['YearRemodAdd'] = 2018-hp_train['YearRemodAdd'] 
hp_train['LotFrontage'].fillna(hp_train['LotFrontage'].median(),inplace=True)
hp_train['GarageYrBlt'].fillna(hp_train['GarageYrBlt'].median(),inplace=True)
hp_train['BsmtUnfSF'].fillna(hp_train['BsmtUnfSF'].mean(),inplace=True)
hp_train['TotalBsmtSF'].fillna(hp_train['TotalBsmtSF'].mean(),inplace=True)
hp_train['GarageArea'].fillna(hp_train['MasVnrArea'].mean(),inplace=True)
hp_train['MasVnrArea'].fillna(hp_train['MasVnrArea'].mean(),inplace=True)


hp_train['Exterior1st'][hp_train.loc[(hp_train['Exterior1st']!='VinylSd')==True]['Exterior1st'].index]='Other'
hp_train['Exterior2nd'][hp_train.loc[(hp_train['Exterior2nd']!='VinylSd')==True]['Exterior2nd'].index]='Other'
hp_train['ExterCond'][hp_train.loc[(hp_train['ExterCond']!='Fa')==True]['ExterCond'].index]='Other'
hp_train['Foundation'][hp_train.loc[(hp_train['Foundation']!='PConc')==True]['Foundation'].index]='Other'
hp_train['BsmtExposure'][hp_train.loc[(hp_train['BsmtExposure']!='Gd')==True]['BsmtExposure'].index]='Other'
hp_train['HeatingQC'][hp_train.loc[(hp_train['HeatingQC']!='Ex')==True]['HeatingQC'].index]='Other'
hp_train['Electrical'][hp_train.loc[(hp_train['Electrical']!='SBrkr')==True]['Electrical'].index]='Other'
hp_train['SaleType'][hp_train.loc[(hp_train['SaleType']!='New')==True]['SaleType'].index]='Other'
hp_train['LandSlope'][hp_train.loc[(hp_train['LandSlope']!='Gtl')==True]['LandSlope'].index]='Other'

hp_train['MSZoning'].fillna('oo',inplace=True)
hp_train['MSZoning'][hp_train.loc[(hp_train['MSZoning']=='FV')==True]['MSZoning'].index]='o'
hp_train['MSZoning'][hp_train.loc[(hp_train['MSZoning']=='RL')==True]['MSZoning'].index]='oo'
hp_train['MSZoning'][hp_train.loc[(hp_train['MSZoning']=='RP')==True]['MSZoning'].index]='oo'
hp_train['MSZoning'][hp_train.loc[(hp_train['MSZoning']=='A')==True]['MSZoning'].index]='oo'
hp_train['MSZoning'][hp_train.loc[(hp_train['MSZoning']=='C (all)')==True]['MSZoning'].index]='oo'
hp_train['MSZoning'][hp_train.loc[(hp_train['MSZoning']=='I')==True]['MSZoning'].index]='oo'
hp_train['MSZoning'][hp_train.loc[(hp_train['MSZoning']=='RH')==True]['MSZoning'].index]='ooo'
hp_train['MSZoning'][hp_train.loc[(hp_train['MSZoning']=='RM')==True]['MSZoning'].index]='ooo'

hp_train['MasVnrType'].fillna('ooo',inplace=True)
hp_train['MasVnrType'][hp_train.loc[(hp_train['MasVnrType']=='Stone')==True]['MasVnrType'].index]='o'
hp_train['MasVnrType'][hp_train.loc[(hp_train['MasVnrType']=='BrkFace')==True]['MasVnrType'].index]='oo'
hp_train['MasVnrType'][hp_train.loc[(hp_train['MasVnrType']=='None')==True]['MasVnrType'].index]='ooo'
hp_train['MasVnrType'][hp_train.loc[(hp_train['MasVnrType']=='CBlock')==True]['MasVnrType'].index]='ooo'
hp_train['MasVnrType'][hp_train.loc[(hp_train['MasVnrType']=='BrkCmn')==True]['MasVnrType'].index]='ooo'

hp_train['ExterQual'][hp_train.loc[(hp_train['ExterQual']=='Ex')==True]['ExterQual'].index]='o'
hp_train['ExterQual'][hp_train.loc[(hp_train['ExterQual']=='Gd')==True]['ExterQual'].index]='oo'
hp_train['ExterQual'][hp_train.loc[(hp_train['ExterQual']=='TA')==True]['ExterQual'].index]='ooo'
hp_train['ExterQual'][hp_train.loc[(hp_train['ExterQual']=='Fa')==True]['ExterQual'].index]='ooo'

hp_train['BsmtQual'].fillna('oooo',inplace=True)
hp_train['BsmtQual'][hp_train.loc[(hp_train['BsmtQual']=='Ex')==True]['BsmtQual'].index]='o'
hp_train['BsmtQual'][hp_train.loc[(hp_train['BsmtQual']=='Gd')==True]['BsmtQual'].index]='oo'
hp_train['BsmtQual'][hp_train.loc[(hp_train['BsmtQual']=='TA')==True]['BsmtQual'].index]='ooo'
hp_train['BsmtQual'][hp_train.loc[(hp_train['BsmtQual']=='Fa')==True]['BsmtQual'].index]='ooo'

hp_train['BsmtCond'].fillna('oooo',inplace=True)
hp_train['BsmtCond'][hp_train.loc[(hp_train['BsmtCond']=='Ex')==True]['BsmtCond'].index]='o'
hp_train['BsmtCond'][hp_train.loc[(hp_train['BsmtCond']=='Gd')==True]['BsmtCond'].index]='oo'
hp_train['BsmtCond'][hp_train.loc[(hp_train['BsmtCond']=='TA')==True]['BsmtCond'].index]='ooo'
hp_train['BsmtCond'][hp_train.loc[(hp_train['BsmtCond']=='Fa')==True]['BsmtCond'].index]='ooo'

hp_train['KitchenQual'].fillna('ooo',inplace=True)
hp_train['KitchenQual'][hp_train.loc[(hp_train['KitchenQual']=='Ex')==True]['KitchenQual'].index]='o'
hp_train['KitchenQual'][hp_train.loc[(hp_train['KitchenQual']=='Gd')==True]['KitchenQual'].index]='oo'
hp_train['KitchenQual'][hp_train.loc[(hp_train['KitchenQual']=='TA')==True]['KitchenQual'].index]='ooo'
hp_train['KitchenQual'][hp_train.loc[(hp_train['KitchenQual']=='Fa')==True]['KitchenQual'].index]='ooo'

hp_train['GarageQual'].fillna('ooo',inplace=True)
hp_train['GarageQual'][hp_train.loc[(hp_train['GarageQual']=='Ex')==True]['GarageQual'].index]='o'
hp_train['GarageQual'][hp_train.loc[(hp_train['GarageQual']=='Gd')==True]['GarageQual'].index]='o'
hp_train['GarageQual'][hp_train.loc[(hp_train['GarageQual']=='TA')==True]['GarageQual'].index]='oo'
hp_train['GarageQual'][hp_train.loc[(hp_train['GarageQual']=='Fa')==True]['GarageQual'].index]='ooo'
hp_train['GarageQual'][hp_train.loc[(hp_train['GarageQual']=='Po')==True]['GarageQual'].index]='ooo'
hp_train['GarageQual'].fillna('oooo',inplace=True)

hp_train['GarageCars'].fillna('o',inplace=True)
hp_train['GarageCars'][hp_train.loc[(hp_train['GarageCars']<=1)==True]['GarageCars'].index]='o'
hp_train['GarageCars'][hp_train.loc[(hp_train['GarageCars']==2)==True]['GarageCars'].index]='oo'
hp_train['GarageCars'][hp_train.loc[(hp_train['GarageCars']>=3)==True]['GarageCars'].index]='ooo'

hp_train['FullBath'][hp_train.loc[(hp_train['FullBath']==0)==True]['FullBath'].index]='o'

hp_train['OverallQual'][hp_train.loc[(hp_train['OverallQual']>=8)==True]['OverallQual'].index]='o'
hp_train['OverallQual'][hp_train.loc[(hp_train['OverallQual']==7)==True]['OverallQual'].index]='tw'
hp_train['OverallQual'][hp_train.loc[(hp_train['OverallQual']==6)==True]['OverallQual'].index]='t'
hp_train['OverallQual'][hp_train.loc[(hp_train['OverallQual']==5)==True]['OverallQual'].index]='f'
hp_train['OverallQual'][hp_train.loc[(hp_train['OverallQual']<=4)==True]['OverallQual'].index]='fi'

hp_train['MSSubClass'][hp_train.loc[(hp_train['MSSubClass']==30)==True]['MSSubClass'].index]='o'
hp_train['MSSubClass'][hp_train.loc[(hp_train['MSSubClass']==45)==True]['MSSubClass'].index]='o'
hp_train['MSSubClass'][hp_train.loc[(hp_train['MSSubClass']==180)==True]['MSSubClass'].index]='o'
hp_train['MSSubClass'][hp_train.loc[(hp_train['MSSubClass']==50)==True]['MSSubClass'].index]='oo'
hp_train['MSSubClass'][hp_train.loc[(hp_train['MSSubClass']==90)==True]['MSSubClass'].index]='oo'
hp_train['MSSubClass'][hp_train.loc[(hp_train['MSSubClass']==85)==True]['MSSubClass'].index]='oo'
hp_train['MSSubClass'][hp_train.loc[(hp_train['MSSubClass']==80)==True]['MSSubClass'].index]='oo'
hp_train['MSSubClass'][hp_train.loc[(hp_train['MSSubClass']==190)==True]['MSSubClass'].index]='oo'
hp_train['MSSubClass'][hp_train.loc[(hp_train['MSSubClass']==70)==True]['MSSubClass'].index]='ooo'
hp_train['MSSubClass'][hp_train.loc[(hp_train['MSSubClass']==20)==True]['MSSubClass'].index]='ooo'
hp_train['MSSubClass'][hp_train.loc[(hp_train['MSSubClass']==150)==True]['MSSubClass'].index]='ooo'
hp_train['MSSubClass'][hp_train.loc[(hp_train['MSSubClass']==40)==True]['MSSubClass'].index]='oooo'
hp_train['MSSubClass'][hp_train.loc[(hp_train['MSSubClass']==75)==True]['MSSubClass'].index]='oooo'
hp_train['MSSubClass'][hp_train.loc[(hp_train['MSSubClass']==60)==True]['MSSubClass'].index]='ooooo'
hp_train['MSSubClass'][hp_train.loc[(hp_train['MSSubClass']==120)==True]['MSSubClass'].index]='ooooo'

hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='Edwards')==True]['Neighborhood'].index]='o'
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='OldTown')==True]['Neighborhood'].index]='o'
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='BrkSide')==True]['Neighborhood'].index]='o'
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='BrDale')==True]['Neighborhood'].index]='o'
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='IDOTRR')==True]['Neighborhood'].index]='o'
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='NWAmes')==True]['Neighborhood'].index]='oo'
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='NAmes')==True]['Neighborhood'].index]='oo'
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='Gilbert')==True]['Neighborhood'].index]='oo'
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='Blmngtn')==True]['Neighborhood'].index]='oo'
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='SawyerW')==True]['Neighborhood'].index]='oo'
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='CollgCr')==True]['Neighborhood'].index]='ooo'
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='ClearCr')==True]['Neighborhood'].index]='ooo'
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='Crawfor')==True]['Neighborhood'].index]='ooo'
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='Somerst')==True]['Neighborhood'].index]='ooo'
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='NridgHt')==True]['Neighborhood'].index]='oooo'
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='NoRidge')==True]['Neighborhood'].index]='oooo'
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='StoneBr')==True]['Neighborhood'].index]='oooo'
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='Sawyer')==True]['Neighborhood'].index]='ooooo'
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='Blueste')==True]['Neighborhood'].index]='ooooo'
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='Mitchel')==True]['Neighborhood'].index]='ooooo'
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='SWISU')==True]['Neighborhood'].index]='ooooo'
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='Timber')==True]['Neighborhood'].index]='oooooo'
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='Veenker')==True]['Neighborhood'].index]='oooooo'
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='MeadowV')==True]['Neighborhood'].index]='ooooooo'
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='NPkVill')==True]['Neighborhood'].index]='oooooooo'

hp_train['FullBath'][hp_train.loc[(hp_train['FullBath']==1)==True]['FullBath'].index]='o'
hp_train['FullBath'][hp_train.loc[(hp_train['FullBath']==2)==True]['FullBath'].index]='oo'
hp_train['FullBath'][hp_train.loc[(hp_train['FullBath']==3)==True]['FullBath'].index]='ooo'

hp_train['MasVnrAreaExist'] ='oo'
hp_train['MasVnrAreaExist'][hp_train.loc[(hp_train['MasVnrArea']!=0)==True]['MasVnrArea'].index]='o'
hp_train['TotalBsmtSFExist'] ='oo'
hp_train['TotalBsmtSFExist'][hp_train.loc[(hp_train['TotalBsmtSF']!=0)==True]['TotalBsmtSF'].index]='o'
hp_train['2ndFlrSFExist'] ='oo'
hp_train['2ndFlrSFExist'][hp_train.loc[(hp_train['2ndFlrSF']!=0)==True]['2ndFlrSF'].index]='o'
hp_train['WoodDeckSFExist'] ='oo'
hp_train['WoodDeckSFExist'][hp_train.loc[(hp_train['WoodDeckSF']!=0)==True]['WoodDeckSF'].index]='o'
hp_train['GarageAreaExist'] ='oo'
hp_train['GarageAreaExist'][hp_train.loc[(hp_train['GarageArea']!=0)==True]['GarageArea'].index]='o'
hp_train['PorchSFExist'] ='oo'
hp_train['PorchSFExist'][hp_train.loc[(hp_train['PorchSF']!=0)==True]['PorchSF'].index]='o'

hp_train['TotalSF']=hp_train['PorchSF']+hp_train['GarageArea']+hp_train['GrLivArea']+hp_train['TotalBsmtSF']
hp_train.to_csv('C:/Users/Administrator/ml_python/ex1/house_price/hp_train_421.csv', index=False)

X=hp_train
X=X.drop(['SalePrice','Id'], axis=1)
y=hp_train['SalePrice']
#split data sets
time=[1,1,1,1,1,1,1,1,1,1]
for i in time:
    
    from sklearn.cross_validation import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=33)
    #Vectorizer
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
    y_train=ss_y.fit_transform(y_train.reshape(-1,1))
    y_test=ss_y.transform(y_test.reshape(-1,1))
    
    
    from sklearn.linear_model import LinearRegression,SGDRegressor 
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor
    from sklearn.tree import DecisionTreeRegressor
    RSVR=SVR(kernel='rbf')
    RFR=RandomForestRegressor()
    ETR=ExtraTreesRegressor()
    GBR=GradientBoostingRegressor()
    
    #training & prediction
    
    RSVR.fit(X_train,y_train)
    RSVR_y_predict=RSVR.predict(X_test)
    RFR.fit(X_train,y_train)
    RFR_y_predict=RFR.predict(X_test)
    ETR.fit(X_train,y_train)
    ETR_y_predict=ETR.predict(X_test)
    GBR.fit(X_train,y_train)
    GBR_y_predict=GBR.predict(X_test)
    
    #rating
    from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
    print r2_score(y_test,GBR_y_predict)
