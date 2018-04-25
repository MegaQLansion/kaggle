import pandas as pd
import numpy as np
hp_train=pd.read_csv('C:/Users/Administrator/ml_python/ex1/Datasets/house_pricing/train.csv')
hp_train['PorchSF']=hp_train['OpenPorchSF']+hp_train['EnclosedPorch']
hp_train= hp_train.drop(['Street','Alley','Utilities','Condition1','Condition2','BldgType','RoofMatl','BsmtFinType1','BsmtFinSF1','BsmtFinType2','Heating','Functional','Fireplaces','PoolQC','Fence','MiscFeature','SaleCondition','RoofStyle','BedroomAbvGr','FireplaceQu','MoSold','BsmtFullBath','BsmtHalfBath','MiscVal','LowQualFinSF','3SsnPorch','ScreenPorch','PoolArea','OpenPorchSF','EnclosedPorch'], axis=1)
hp_train['YrSold'] = 2018-hp_train['YrSold'] 
hp_train['GarageYrBlt'] = 2018-hp_train['GarageYrBlt'] 
hp_train['YearBuilt'] = 2018-hp_train['YearBuilt'] 
hp_train['YearRemodAdd'] = 2018-hp_train['YearRemodAdd'] 
hp_train['LotFrontage'].fillna(hp_train['LotFrontage'].median(),inplace=True)
hp_train['GarageYrBlt'].fillna(hp_train['GarageYrBlt'].median(),inplace=True)


hp_train['HouseStyle'][hp_train.loc[(hp_train['HouseStyle']!='2Story')==True]['HouseStyle'].index]='Other'
hp_train['Exterior1st'][hp_train.loc[(hp_train['Exterior1st']!='VinylSd')==True]['Exterior1st'].index]='Other'
hp_train['Exterior2nd'][hp_train.loc[(hp_train['Exterior2nd']!='VinylSd')==True]['Exterior2nd'].index]='Other'
hp_train['ExterCond'][hp_train.loc[(hp_train['ExterCond']!='Fa')==True]['ExterCond'].index]='Other'
hp_train['Foundation'][hp_train.loc[(hp_train['Foundation']!='PConc')==True]['Foundation'].index]='Other'
hp_train['BsmtExposure'][hp_train.loc[(hp_train['BsmtExposure']!='Gd')==True]['BsmtExposure'].index]='Other'
hp_train['HeatingQC'][hp_train.loc[(hp_train['HeatingQC']!='Ex')==True]['HeatingQC'].index]='Other'
hp_train['Electrical'][hp_train.loc[(hp_train['Electrical']!='SBrkr')==True]['Electrical'].index]='Other'
hp_train['GarageFinish'][hp_train.loc[(hp_train['GarageFinish']!='RFn')==True]['GarageFinish'].index]='Other'
hp_train['SaleType'][hp_train.loc[(hp_train['SaleType']!='New')==True]['SaleType'].index]='Other'
hp_train['LotConfig'][hp_train.loc[(hp_train['LotConfig']!='CuIDSac')==True]['LotConfig'].index]='Other'
hp_train['LandSlope'][hp_train.loc[(hp_train['LandSlope']!='Gtl')==True]['LandSlope'].index]='Other'


hp_train['TotRmsAbvGrd'][hp_train.loc[(hp_train['TotRmsAbvGrd']<=6)==True]['TotRmsAbvGrd'].index]='L6'
hp_train['TotRmsAbvGrd'][hp_train.loc[(hp_train['TotRmsAbvGrd']!='L6')==True]['TotRmsAbvGrd'].index]='B6'
#
hp_train['GarageCond'][hp_train.loc[(hp_train['GarageCond']=='Fa')==True]['GarageCond'].index]='Bd'
hp_train['GarageCond'][hp_train.loc[(hp_train['GarageCond']=='Po')==True]['GarageCond'].index]='Bd'
hp_train['GarageCond'][hp_train.loc[(hp_train['GarageCond']!='Bd')==True]['GarageCond'].index]='Gd'

hp_train['LandContour'][hp_train.loc[(hp_train['LandContour']=='HLS')==True]['LandContour'].index]=0
hp_train['LandContour'][hp_train.loc[(hp_train['LandContour']=='Low')==True]['LandContour'].index]=0
hp_train['LandContour'][hp_train.loc[(hp_train['LandContour']!=0)==True]['LandContour'].index]=1

hp_train['MSZoning'][hp_train.loc[(hp_train['MSZoning']=='FV')==True]['MSZoning'].index]=1
hp_train['MSZoning'][hp_train.loc[(hp_train['MSZoning']=='RL')==True]['MSZoning'].index]=2
hp_train['MSZoning'][hp_train.loc[(hp_train['MSZoning']=='RP')==True]['MSZoning'].index]=2
hp_train['MSZoning'][hp_train.loc[(hp_train['MSZoning']=='A')==True]['MSZoning'].index]=2
hp_train['MSZoning'][hp_train.loc[(hp_train['MSZoning']=='C (all)')==True]['MSZoning'].index]=2
hp_train['MSZoning'][hp_train.loc[(hp_train['MSZoning']=='I')==True]['MSZoning'].index]=2
hp_train['MSZoning'][hp_train.loc[(hp_train['MSZoning']=='RH')==True]['MSZoning'].index]=3
hp_train['MSZoning'][hp_train.loc[(hp_train['MSZoning']=='RM')==True]['MSZoning'].index]=3

hp_train['LotShape'][hp_train.loc[(hp_train['LotShape']=='Reg')==True]['LotShape'].index]=1
hp_train['LotShape'][hp_train.loc[(hp_train['LotShape']=='IR1')==True]['LotShape'].index]=2
hp_train['LotShape'][hp_train.loc[(hp_train['LotShape']=='IR2')==True]['LotShape'].index]=3
hp_train['LotShape'][hp_train.loc[(hp_train['LotShape']=='IR3')==True]['LotShape'].index]=3


hp_train['OverallCond'][hp_train.loc[(hp_train['OverallCond']>=6)==True]['OverallCond'].index]='o'
hp_train['OverallCond'][hp_train.loc[(hp_train['OverallCond']==5)==True]['OverallCond'].index]='tw'
hp_train['OverallCond'][hp_train.loc[(hp_train['OverallCond']==4)==True]['OverallCond'].index]='t'
hp_train['OverallCond'][hp_train.loc[(hp_train['OverallCond']<=3)==True]['OverallCond'].index]='f'

hp_train['MasVnrType'][hp_train.loc[(hp_train['MasVnrType']=='Stone')==True]['MasVnrType'].index]=1
hp_train['MasVnrType'][hp_train.loc[(hp_train['MasVnrType']=='BrkFace')==True]['MasVnrType'].index]=2
hp_train['MasVnrType'][hp_train.loc[(hp_train['MasVnrType']=='None')==True]['MasVnrType'].index]=3
hp_train['MasVnrType'][hp_train.loc[(hp_train['MasVnrType']=='CBlock')==True]['MasVnrType'].index]=3
hp_train['MasVnrType'][hp_train.loc[(hp_train['MasVnrType']=='BrkCmn')==True]['MasVnrType'].index]=3

hp_train['ExterQual'][hp_train.loc[(hp_train['ExterQual']=='Ex')==True]['ExterQual'].index]=1
hp_train['ExterQual'][hp_train.loc[(hp_train['ExterQual']=='Gd')==True]['ExterQual'].index]=2
hp_train['ExterQual'][hp_train.loc[(hp_train['ExterQual']=='TA')==True]['ExterQual'].index]=3
hp_train['ExterQual'][hp_train.loc[(hp_train['ExterQual']=='Fa')==True]['ExterQual'].index]=3


hp_train['BsmtQual'][hp_train.loc[(hp_train['BsmtQual']=='Ex')==True]['BsmtQual'].index]=1
hp_train['BsmtQual'][hp_train.loc[(hp_train['BsmtQual']=='Gd')==True]['BsmtQual'].index]=2
hp_train['BsmtQual'][hp_train.loc[(hp_train['BsmtQual']=='TA')==True]['BsmtQual'].index]=3
hp_train['BsmtQual'][hp_train.loc[(hp_train['BsmtQual']=='Fa')==True]['BsmtQual'].index]=3
hp_train['BsmtQual'][hp_train.loc[(hp_train['BsmtQual']=='Po')==True]['BsmtQual'].index]=3

hp_train['BsmtCond'].fillna(3,inplace=True)
hp_train['BsmtCond'][hp_train.loc[(hp_train['BsmtCond']=='Ex')==True]['BsmtCond'].index]=1
hp_train['BsmtCond'][hp_train.loc[(hp_train['BsmtCond']=='Gd')==True]['BsmtCond'].index]=2
hp_train['BsmtCond'][hp_train.loc[(hp_train['BsmtCond']=='TA')==True]['BsmtCond'].index]=3
hp_train['BsmtCond'][hp_train.loc[(hp_train['BsmtCond']=='Fa')==True]['BsmtCond'].index]=3
hp_train['BsmtCond'][hp_train.loc[(hp_train['BsmtCond']=='Po')==True]['BsmtCond'].index]=3

hp_train['KitchenQual'][hp_train.loc[(hp_train['KitchenQual']=='Ex')==True]['KitchenQual'].index]=1
hp_train['KitchenQual'][hp_train.loc[(hp_train['KitchenQual']=='Gd')==True]['KitchenQual'].index]=2
hp_train['KitchenQual'][hp_train.loc[(hp_train['KitchenQual']=='TA')==True]['KitchenQual'].index]=3
hp_train['KitchenQual'][hp_train.loc[(hp_train['KitchenQual']=='Fa')==True]['KitchenQual'].index]=3

hp_train['GarageType'].fillna(4,inplace=True)
hp_train['GarageType'][hp_train.loc[(hp_train['GarageType']=='BuiltIn')==True]['GarageType'].index]=1
hp_train['GarageType'][hp_train.loc[(hp_train['GarageType']=='Attchd')==True]['GarageType'].index]=2
hp_train['GarageType'][hp_train.loc[(hp_train['GarageType']=='2Types')==True]['GarageType'].index]=3
hp_train['GarageType'][hp_train.loc[(hp_train['GarageType']=='Basment')==True]['GarageType'].index]=3
hp_train['GarageType'][hp_train.loc[(hp_train['GarageType']=='Detchd')==True]['GarageType'].index]=3
hp_train['GarageType'][hp_train.loc[(hp_train['GarageType']=='CarPort')==True]['GarageType'].index]=3


hp_train['GarageQual'][hp_train.loc[(hp_train['GarageQual']=='Ex')==True]['GarageQual'].index]=1
hp_train['GarageQual'][hp_train.loc[(hp_train['GarageQual']=='Gd')==True]['GarageQual'].index]=1
hp_train['GarageQual'][hp_train.loc[(hp_train['GarageQual']=='TA')==True]['GarageQual'].index]=2
hp_train['GarageQual'][hp_train.loc[(hp_train['GarageQual']=='Fa')==True]['GarageQual'].index]=3
hp_train['GarageQual'][hp_train.loc[(hp_train['GarageQual']=='Po')==True]['GarageQual'].index]=3
hp_train['GarageQual'].fillna(4,inplace=True)

hp_train['GarageCars'][hp_train.loc[(hp_train['GarageCars']<=1)==True]['GarageCars'].index]=1
hp_train['GarageCars'][hp_train.loc[(hp_train['GarageCars']>=3)==True]['GarageCars'].index]=3



hp_train['FullBath'][hp_train.loc[(hp_train['FullBath']==0)==True]['FullBath'].index]=1



hp_train['OverallQual'][hp_train.loc[(hp_train['OverallQual']>=8)==True]['OverallQual'].index]='o'
hp_train['OverallQual'][hp_train.loc[(hp_train['OverallQual']==7)==True]['OverallQual'].index]='tw'
hp_train['OverallQual'][hp_train.loc[(hp_train['OverallQual']==6)==True]['OverallQual'].index]='t'
hp_train['OverallQual'][hp_train.loc[(hp_train['OverallQual']==5)==True]['OverallQual'].index]='f'
hp_train['OverallQual'][hp_train.loc[(hp_train['OverallQual']<=4)==True]['OverallQual'].index]='fi'

hp_train['MSSubClass'][hp_train.loc[(hp_train['MSSubClass']==30)==True]['MSSubClass'].index]=1
hp_train['MSSubClass'][hp_train.loc[(hp_train['MSSubClass']==45)==True]['MSSubClass'].index]=1
hp_train['MSSubClass'][hp_train.loc[(hp_train['MSSubClass']==180)==True]['MSSubClass'].index]=1
hp_train['MSSubClass'][hp_train.loc[(hp_train['MSSubClass']==50)==True]['MSSubClass'].index]=2
hp_train['MSSubClass'][hp_train.loc[(hp_train['MSSubClass']==90)==True]['MSSubClass'].index]=2
hp_train['MSSubClass'][hp_train.loc[(hp_train['MSSubClass']==85)==True]['MSSubClass'].index]=2
hp_train['MSSubClass'][hp_train.loc[(hp_train['MSSubClass']==80)==True]['MSSubClass'].index]=2
hp_train['MSSubClass'][hp_train.loc[(hp_train['MSSubClass']==190)==True]['MSSubClass'].index]=2
hp_train['MSSubClass'][hp_train.loc[(hp_train['MSSubClass']==70)==True]['MSSubClass'].index]=3
hp_train['MSSubClass'][hp_train.loc[(hp_train['MSSubClass']==20)==True]['MSSubClass'].index]=3
hp_train['MSSubClass'][hp_train.loc[(hp_train['MSSubClass']==160)==True]['MSSubClass'].index]=3
hp_train['MSSubClass'][hp_train.loc[(hp_train['MSSubClass']==40)==True]['MSSubClass'].index]=4
hp_train['MSSubClass'][hp_train.loc[(hp_train['MSSubClass']==75)==True]['MSSubClass'].index]=4
hp_train['MSSubClass'][hp_train.loc[(hp_train['MSSubClass']==60)==True]['MSSubClass'].index]=5
hp_train['MSSubClass'][hp_train.loc[(hp_train['MSSubClass']==120)==True]['MSSubClass'].index]=5

hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='Edwards')==True]['Neighborhood'].index]=1
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='OldTown')==True]['Neighborhood'].index]=1
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='BrkSide')==True]['Neighborhood'].index]=1
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='BrDale')==True]['Neighborhood'].index]=1
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='IDOTRR')==True]['Neighborhood'].index]=1
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='NWAmes')==True]['Neighborhood'].index]=2
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='NAmes')==True]['Neighborhood'].index]=2
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='Gilbert')==True]['Neighborhood'].index]=2
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='Blmngtn')==True]['Neighborhood'].index]=2
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='SawyerW')==True]['Neighborhood'].index]=2
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='CollgCr')==True]['Neighborhood'].index]=3
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='ClearCr')==True]['Neighborhood'].index]=3
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='Crawfor')==True]['Neighborhood'].index]=3
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='Somerst')==True]['Neighborhood'].index]=3
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='NridgHt')==True]['Neighborhood'].index]=4
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='NoRidge')==True]['Neighborhood'].index]=4
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='StoneBr')==True]['Neighborhood'].index]=4
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='Sawyer')==True]['Neighborhood'].index]=5
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='Blueste')==True]['Neighborhood'].index]=5
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='Mitchel')==True]['Neighborhood'].index]=5
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='SWISU')==True]['Neighborhood'].index]=5
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='Timber')==True]['Neighborhood'].index]=6
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='Veenker')==True]['Neighborhood'].index]=6
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='MeadowV')==True]['Neighborhood'].index]=7
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='NPkVill')==True]['Neighborhood'].index]=8

hp_train['MasVnrAreaExist'] =0
hp_train['MasVnrAreaExist'][hp_train.loc[(hp_train['MasVnrArea']!=0)==True]['MasVnrArea'].index]=1
hp_train['TotalBsmtSFExist'] =0
hp_train['TotalBsmtSFExist'][hp_train.loc[(hp_train['TotalBsmtSF']!=0)==True]['TotalBsmtSF'].index]=1
hp_train['2ndFlrSFExist'] =0
hp_train['2ndFlrSFExist'][hp_train.loc[(hp_train['2ndFlrSF']!=0)==True]['2ndFlrSF'].index]=1
hp_train['WoodDeckSFExist'] =0
hp_train['WoodDeckSFExist'][hp_train.loc[(hp_train['WoodDeckSF']!=0)==True]['WoodDeckSF'].index]=1
hp_train['GarageAreaExist'] =0
hp_train['GarageAreaExist'][hp_train.loc[(hp_train['GarageArea']!=0)==True]['GarageArea'].index]=1
hp_train['PorchSFExist'] =0
hp_train['PorchSFExist'][hp_train.loc[(hp_train['PorchSF']!=0)==True]['PorchSF'].index]=1
hp_train['TotalSF']=hp_train['PorchSF']+hp_train['GarageArea']+hp_train['GrLivArea']+hp_train['TotalBsmtSF']+hp_train['LotArea']
#submission
hp_train.to_csv('C:/Users/Administrator/ml_python/ex1/Datasets/test.csv', index=False)
    