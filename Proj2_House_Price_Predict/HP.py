

import pandas as pd
import numpy as np
hp_train=pd.read_csv('C:/Users/Administrator/ml_python/ex1/Datasets/house_pricing/train.csv')
hp_test=pd.read_csv('C:/Users/Administrator/ml_python/ex1/Datasets/house_pricing/test.csv')

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
hp_train['BsmtFinSF2'].fillna(0,inplace=True)
hp_train['BsmtFinSF1'].fillna(hp_test['BsmtFinSF1'].mean(),inplace=True)

hp_train['PorchSF']=hp_train['OpenPorchSF']+hp_train['EnclosedPorch']
hp_train['TotalHouse'] = hp_train['TotalBsmtSF'] + hp_train['1stFlrSF'] + hp_train['2ndFlrSF']   
hp_train['TotalArea'] = hp_train['TotalBsmtSF'] + hp_train['1stFlrSF'] + hp_train['2ndFlrSF'] + hp_train['GarageArea']
hp_train['LotArea_OverallQual'] = hp_train['LotArea'] * hp_train['OverallQual']
hp_train['TotalHouse_LotArea'] = hp_train['TotalHouse'] + hp_train['LotArea']
hp_train['Bsmt'] = hp_train['BsmtFinSF1'] + hp_train['BsmtFinSF2'] + hp_train['BsmtUnfSF']
hp_train['PorchArea'] = hp_train['PorchSF']+hp_train['3SsnPorch']+hp_train['ScreenPorch']
hp_train['TotalPlace'] = hp_train['TotalBsmtSF'] + hp_train['1stFlrSF'] + hp_train['2ndFlrSF'] + hp_train['GarageArea'] + hp_train['PorchSF']+hp_train['3SsnPorch']+hp_train['ScreenPorch']

hp_train= hp_train.drop(['Street','Alley','Utilities','Condition2','RoofMatl','BsmtFinType1','BsmtFinSF1','BsmtFinType2','Fireplaces','PoolQC','Fence','MiscFeature','RoofStyle','BedroomAbvGr','MoSold','BsmtFullBath','BsmtHalfBath','MiscVal','LowQualFinSF','3SsnPorch','ScreenPorch','PoolArea','OpenPorchSF','EnclosedPorch','LandContour','LotConfig','LandSlope','ExterCond','BsmtExposure','CentralAir','Electrical','GarageCars','SaleType'], axis=1)


hp_train['SaleCondition'].fillna('oo',inplace=True)
hp_train['SaleCondition'][hp_train.loc[(hp_train['SaleCondition']=='AdjLand')==True]['SaleCondition'].index]='o'
hp_train['SaleCondition'][hp_train.loc[(hp_train['SaleCondition']=='Normal')==True]['SaleCondition'].index]='oo'
hp_train['SaleCondition'][hp_train.loc[(hp_train['SaleCondition']=='Partial')==True]['SaleCondition'].index]='ooo'
hp_train['SaleCondition'][hp_train.loc[(hp_train['SaleCondition']=='Abnorml')==True]['SaleCondition'].index]='oooo'
hp_train['SaleCondition'][hp_train.loc[(hp_train['SaleCondition']=='Alloca')==True]['SaleCondition'].index]='oooo'
hp_train['SaleCondition'][hp_train.loc[(hp_train['SaleCondition']=='Family')==True]['SaleCondition'].index]='oooo'

hp_train['FireplaceQu'].fillna('ooo',inplace=True)
hp_train['FireplaceQu'][hp_train.loc[(hp_train['FireplaceQu']=='None')==True]['FireplaceQu'].index]='o'
hp_train['FireplaceQu'][hp_train.loc[(hp_train['FireplaceQu']=='Po')==True]['FireplaceQu'].index]='o'
hp_train['FireplaceQu'][hp_train.loc[(hp_train['FireplaceQu']=='Fa')==True]['FireplaceQu'].index]='oo'
hp_train['FireplaceQu'][hp_train.loc[(hp_train['FireplaceQu']=='TA')==True]['FireplaceQu'].index]='ooo'
hp_train['FireplaceQu'][hp_train.loc[(hp_train['FireplaceQu']=='Gd')==True]['FireplaceQu'].index]='oooo'  
hp_train['FireplaceQu'][hp_train.loc[(hp_train['FireplaceQu']=='Ex')==True]['FireplaceQu'].index]='ooooo'

hp_train['BldgType'].fillna('oo',inplace=True)
hp_train['BldgType'][hp_train.loc[(hp_train['BldgType']=='TwnhsE')==True]['BldgType'].index]='oo'
hp_train['BldgType'][hp_train.loc[(hp_train['BldgType']=='1Fam')==True]['BldgType'].index]='oo'
hp_train['BldgType'][hp_train.loc[(hp_train['BldgType']!='oo')==True]['BldgType'].index]='o '

hp_train['Heating'].fillna('o',inplace=True)
hp_train['Heating'][hp_train.loc[(hp_train['Heating']=='Floor')==True]['Heating'].index]='o'
hp_train['Heating'][hp_train.loc[(hp_train['Heating']=='Grav')==True]['Heating'].index]='o'
hp_train['Heating'][hp_train.loc[(hp_train['Heating']=='Wall')==True]['Heating'].index]='oo'
hp_train['Heating'][hp_train.loc[(hp_train['Heating']=='OthW')==True]['Heating'].index]='ooo'
hp_train['Heating'][hp_train.loc[(hp_train['Heating']=='GasW')==True]['Heating'].index]='oooo'  
hp_train['Heating'][hp_train.loc[(hp_train['Heating']=='GasA')==True]['Heating'].index]='ooooo'

hp_train['Condition1'].fillna('ooo',inplace=True)
hp_train['Condition1'][hp_train.loc[(hp_train['Condition1']=='Artery')==True]['Condition1'].index]='o'
hp_train['Condition1'][hp_train.loc[(hp_train['Condition1']=='Feedr')==True]['Condition1'].index]='oo'
hp_train['Condition1'][hp_train.loc[(hp_train['Condition1']=='RRAe')==True]['Condition1'].index]='oo'
hp_train['Condition1'][hp_train.loc[(hp_train['Condition1']=='Norm')==True]['Condition1'].index]='ooo'
hp_train['Condition1'][hp_train.loc[(hp_train['Condition1']=='RRAn')==True]['Condition1'].index]='ooo'  
hp_train['Condition1'][hp_train.loc[(hp_train['Condition1']=='PosN')==True]['Condition1'].index]='oooo'
hp_train['Condition1'][hp_train.loc[(hp_train['Condition1']=='RRNe')==True]['Condition1'].index]='oooo'
hp_train['Condition1'][hp_train.loc[(hp_train['Condition1']=='PosA')==True]['Condition1'].index]='ooooo'
hp_train['Condition1'][hp_train.loc[(hp_train['Condition1']=='RRNn')==True]['Condition1'].index]='ooooo'

hp_train['HouseStyle'].fillna('oo',inplace=True)
hp_train['HouseStyle'][hp_train.loc[(hp_train['HouseStyle']=='1.5Unf')==True]['HouseStyle'].index]='o'
hp_train['HouseStyle'][hp_train.loc[(hp_train['HouseStyle']=='1.5Fin')==True]['HouseStyle'].index]='oo'
hp_train['HouseStyle'][hp_train.loc[(hp_train['HouseStyle']=='2.5Unf')==True]['HouseStyle'].index]='oo'
hp_train['HouseStyle'][hp_train.loc[(hp_train['HouseStyle']=='SFoyer')==True]['HouseStyle'].index]='oo'
hp_train['HouseStyle'][hp_train.loc[(hp_train['HouseStyle']=='1Story')==True]['HouseStyle'].index]='ooo'  
hp_train['HouseStyle'][hp_train.loc[(hp_train['HouseStyle']=='SLvl')==True]['HouseStyle'].index]='ooo'
hp_train['HouseStyle'][hp_train.loc[(hp_train['HouseStyle']=='2Story')==True]['HouseStyle'].index]='oooo'
hp_train['HouseStyle'][hp_train.loc[(hp_train['HouseStyle']=='2.5Fin')==True]['HouseStyle'].index]='oooo'

hp_train['Exterior1st'].fillna('ooo',inplace=True)
hp_train['Exterior1st'][hp_train.loc[(hp_train['Exterior1st']=='BrkComm')==True]['Exterior1st'].index]='o'
hp_train['Exterior1st'][hp_train.loc[(hp_train['Exterior1st']=='AsphShn')==True]['Exterior1st'].index]='oo'
hp_train['Exterior1st'][hp_train.loc[(hp_train['Exterior1st']=='CBlock')==True]['Exterior1st'].index]='oo'
hp_train['Exterior1st'][hp_train.loc[(hp_train['Exterior1st']=='AsbShng')==True]['Exterior1st'].index]='oo'
hp_train['Exterior1st'][hp_train.loc[(hp_train['Exterior1st']=='WdShing')==True]['Exterior1st'].index]='ooo'  
hp_train['Exterior1st'][hp_train.loc[(hp_train['Exterior1st']=='Wd Sdng')==True]['Exterior1st'].index]='ooo'
hp_train['Exterior1st'][hp_train.loc[(hp_train['Exterior1st']=='MetalSd')==True]['Exterior1st'].index]='ooo'
hp_train['Exterior1st'][hp_train.loc[(hp_train['Exterior1st']=='Stucco')==True]['Exterior1st'].index]='ooo'
hp_train['Exterior1st'][hp_train.loc[(hp_train['Exterior1st']=='HdBoard')==True]['Exterior1st'].index]='ooo'
hp_train['Exterior1st'][hp_train.loc[(hp_train['Exterior1st']=='BrkFace')==True]['Exterior1st'].index]='oooo'
hp_train['Exterior1st'][hp_train.loc[(hp_train['Exterior1st']=='Plywood')==True]['Exterior1st'].index]='oooo'
hp_train['Exterior1st'][hp_train.loc[(hp_train['Exterior1st']=='VinylSd')==True]['Exterior1st'].index]='ooooo'  
hp_train['Exterior1st'][hp_train.loc[(hp_train['Exterior1st']=='CemntBd')==True]['Exterior1st'].index]='oooooo'
hp_train['Exterior1st'][hp_train.loc[(hp_train['Exterior1st']=='Stone')==True]['Exterior1st'].index]='ooooooo'
hp_train['Exterior1st'][hp_train.loc[(hp_train['Exterior1st']=='ImStucc')==True]['Exterior1st'].index]='ooooooo'


hp_train['Exterior1st'][hp_train.loc[(hp_train['Exterior1st']!='VinylSd')==True]['Exterior1st'].index]='Other'
hp_train['Exterior2nd'][hp_train.loc[(hp_train['Exterior2nd']!='VinylSd')==True]['Exterior2nd'].index]='Other'
hp_train['Foundation'][hp_train.loc[(hp_train['Foundation']!='PConc')==True]['Foundation'].index]='Other'
hp_train['HeatingQC'][hp_train.loc[(hp_train['HeatingQC']!='Ex')==True]['HeatingQC'].index]='Other'
hp_train['GarageFinish'][hp_train.loc[(hp_train['GarageFinish']!='RFn')==True]['GarageFinish'].index]='Other'


hp_train['TotRmsAbvGrd'][hp_train.loc[(hp_train['TotRmsAbvGrd']<=6)==True]['TotRmsAbvGrd'].index]='L6'
hp_train['TotRmsAbvGrd'][hp_train.loc[(hp_train['TotRmsAbvGrd']!='L6')==True]['TotRmsAbvGrd'].index]='B6'
#
hp_train['GarageCond'][hp_train.loc[(hp_train['GarageCond']=='Fa')==True]['GarageCond'].index]='Bd'
hp_train['GarageCond'][hp_train.loc[(hp_train['GarageCond']=='Po')==True]['GarageCond'].index]='Bd'
hp_train['GarageCond'][hp_train.loc[(hp_train['GarageCond']!='Bd')==True]['GarageCond'].index]='Gd'


hp_train['MSZoning'].fillna('oo',inplace=True)
hp_train['MSZoning'][hp_train.loc[(hp_train['MSZoning']=='FV')==True]['MSZoning'].index]='o'
hp_train['MSZoning'][hp_train.loc[(hp_train['MSZoning']=='RL')==True]['MSZoning'].index]='oo'
hp_train['MSZoning'][hp_train.loc[(hp_train['MSZoning']=='RP')==True]['MSZoning'].index]='oo'
hp_train['MSZoning'][hp_train.loc[(hp_train['MSZoning']=='A')==True]['MSZoning'].index]='oo'
hp_train['MSZoning'][hp_train.loc[(hp_train['MSZoning']=='C (all)')==True]['MSZoning'].index]='oo'
hp_train['MSZoning'][hp_train.loc[(hp_train['MSZoning']=='I')==True]['MSZoning'].index]='oo'
hp_train['MSZoning'][hp_train.loc[(hp_train['MSZoning']=='RH')==True]['MSZoning'].index]='ooo'
hp_train['MSZoning'][hp_train.loc[(hp_train['MSZoning']=='RM')==True]['MSZoning'].index]='ooo'

hp_train['LotShape'][hp_train.loc[(hp_train['LotShape']=='Reg')==True]['LotShape'].index]='o'
hp_train['LotShape'][hp_train.loc[(hp_train['LotShape']=='IR1')==True]['LotShape'].index]='oo'
hp_train['LotShape'][hp_train.loc[(hp_train['LotShape']=='IR2')==True]['LotShape'].index]='ooo'
hp_train['LotShape'][hp_train.loc[(hp_train['LotShape']=='IR3')==True]['LotShape'].index]='ooo'


hp_train['OverallCond'][hp_train.loc[(hp_train['OverallCond']>=6)==True]['OverallCond'].index]='o'
hp_train['OverallCond'][hp_train.loc[(hp_train['OverallCond']==5)==True]['OverallCond'].index]='tw'
hp_train['OverallCond'][hp_train.loc[(hp_train['OverallCond']==4)==True]['OverallCond'].index]='t'
hp_train['OverallCond'][hp_train.loc[(hp_train['OverallCond']<=3)==True]['OverallCond'].index]='f'

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

hp_train['GarageType'].fillna('oooo',inplace=True)
hp_train['GarageType'][hp_train.loc[(hp_train['GarageType']=='BuiltIn')==True]['GarageType'].index]='o'
hp_train['GarageType'][hp_train.loc[(hp_train['GarageType']=='Attchd')==True]['GarageType'].index]='oo'
hp_train['GarageType'][hp_train.loc[(hp_train['GarageType']=='2Types')==True]['GarageType'].index]='ooo'
hp_train['GarageType'][hp_train.loc[(hp_train['GarageType']=='Basment')==True]['GarageType'].index]='ooo'
hp_train['GarageType'][hp_train.loc[(hp_train['GarageType']=='Detchd')==True]['GarageType'].index]='ooo'
hp_train['GarageType'][hp_train.loc[(hp_train['GarageType']=='CarPort')==True]['GarageType'].index]='ooo'

hp_train['GarageQual'].fillna('ooo',inplace=True)
hp_train['GarageQual'][hp_train.loc[(hp_train['GarageQual']=='Ex')==True]['GarageQual'].index]='o'
hp_train['GarageQual'][hp_train.loc[(hp_train['GarageQual']=='Gd')==True]['GarageQual'].index]='o'
hp_train['GarageQual'][hp_train.loc[(hp_train['GarageQual']=='TA')==True]['GarageQual'].index]='oo'
hp_train['GarageQual'][hp_train.loc[(hp_train['GarageQual']=='Fa')==True]['GarageQual'].index]='ooo'
hp_train['GarageQual'][hp_train.loc[(hp_train['GarageQual']=='Po')==True]['GarageQual'].index]='ooo'
hp_train['GarageQual'].fillna('oooo',inplace=True)

hp_train['FullBath'][hp_train.loc[(hp_train['FullBath']==0)==True]['FullBath'].index]='o'


hp_train['MSSubClass'][hp_train.loc[(hp_train['MSSubClass']==30)==True]['MSSubClass'].index]='o'
hp_train['MSSubClass'][hp_train.loc[(hp_train['MSSubClass']==45)==True]['MSSubClass'].index]='o'
hp_train['MSSubClass'][hp_train.loc[(hp_train['MSSubClass']==180)==True]['MSSubClass'].index]='oooo'
hp_train['MSSubClass'][hp_train.loc[(hp_train['MSSubClass']==50)==True]['MSSubClass'].index]='oo'
hp_train['MSSubClass'][hp_train.loc[(hp_train['MSSubClass']==90)==True]['MSSubClass'].index]='oo'
hp_train['MSSubClass'][hp_train.loc[(hp_train['MSSubClass']==190)==True]['MSSubClass'].index]='oo'
hp_train['MSSubClass'][hp_train.loc[(hp_train['MSSubClass']==70)==True]['MSSubClass'].index]='ooo'
hp_train['MSSubClass'][hp_train.loc[(hp_train['MSSubClass']==80)==True]['MSSubClass'].index]='ooo'
hp_train['MSSubClass'][hp_train.loc[(hp_train['MSSubClass']==20)==True]['MSSubClass'].index]='ooo'
hp_train['MSSubClass'][hp_train.loc[(hp_train['MSSubClass']==75)==True]['MSSubClass'].index]='ooo'
hp_train['MSSubClass'][hp_train.loc[(hp_train['MSSubClass']==60)==True]['MSSubClass'].index]='ooooo'
hp_train['MSSubClass'][hp_train.loc[(hp_train['MSSubClass']==120)==True]['MSSubClass'].index]='ooooo'
hp_train['MSSubClass'][hp_train.loc[(hp_train['MSSubClass']==85)==True]['MSSubClass'].index]='oo'
hp_train['MSSubClass'][hp_train.loc[(hp_train['MSSubClass']==160)==True]['MSSubClass'].index]='oooooo'
hp_train['MSSubClass'][hp_train.loc[(hp_train['MSSubClass']==150)==True]['MSSubClass'].index]='oooooo'
hp_train['MSSubClass'][hp_train.loc[(hp_train['MSSubClass']==40)==True]['MSSubClass'].index]='oooooo'

hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='Edwards')==True]['Neighborhood'].index]='o'
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='OldTown')==True]['Neighborhood'].index]='o'
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='BrkSide')==True]['Neighborhood'].index]='o'
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='BrDale')==True]['Neighborhood'].index]='ooooooo'
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='IDOTRR')==True]['Neighborhood'].index]='ooooooo'
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='NWAmes')==True]['Neighborhood'].index]='oo'
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='Gilbert')==True]['Neighborhood'].index]='oo'
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='SawyerW')==True]['Neighborhood'].index]='oo'
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='CollgCr')==True]['Neighborhood'].index]='ooo'
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='ClearCr')==True]['Neighborhood'].index]='ooo'
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='Crawfor')==True]['Neighborhood'].index]='ooo'
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='Blmngtn')==True]['Neighborhood'].index]='ooo'
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='NridgHt')==True]['Neighborhood'].index]='oooo'
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='NoRidge')==True]['Neighborhood'].index]='oooo'
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='Blueste')==True]['Neighborhood'].index]='e'
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='Mitchel')==True]['Neighborhood'].index]='ooooo'
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='SWISU')==True]['Neighborhood'].index]='e'
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='Timber')==True]['Neighborhood'].index]='oooooo'
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='Veenker')==True]['Neighborhood'].index]='oooooo'
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='NPkVill')==True]['Neighborhood'].index]='ooooo'
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='Somerst')==True]['Neighborhood'].index]='oooooo'
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='NAmes')==True]['Neighborhood'].index]='e'
hp_train['Neighborhood'][hp_train.loc[(hp_train['Neighborhood']=='Sawyer')==True]['Neighborhood'].index]='e'

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
hp_train['TotalSF']=hp_train['PorchSF']+hp_train['GarageArea']+hp_train['GrLivArea']+hp_train['TotalBsmtSF']+hp_train['LotArea']


skewed_sets=['BsmtFinSF2','LotFrontage','LotArea','MasVnrArea','TotalBsmtSF','1stFlrSF','2ndFlrSF','GrLivArea','GarageArea','WoodDeckSF']
for train in skewed_sets:
    hp_train[train] = np.log1p(hp_train[train])

hp_train = pd.get_dummies(hp_train)

hp_train.to_csv('C:/Users/Administrator/ml_python/ex1/house_price/hp_train_422.csv', index=False)
hp_test['YrSold'] = 2018-hp_test['YrSold'] 
hp_test['GarageYrBlt'] = 2018-hp_test['GarageYrBlt'] 
hp_test['YearBuilt'] = 2018-hp_test['YearBuilt'] 
hp_test['YearRemodAdd'] = 2018-hp_test['YearRemodAdd'] 
hp_test['LotFrontage'].fillna(hp_test['LotFrontage'].median(),inplace=True)
hp_test['GarageYrBlt'].fillna(hp_test['GarageYrBlt'].median(),inplace=True)
hp_test['BsmtUnfSF'].fillna(hp_test['BsmtUnfSF'].mean(),inplace=True)
hp_test['TotalBsmtSF'].fillna(hp_test['TotalBsmtSF'].mean(),inplace=True)
hp_test['GarageArea'].fillna(hp_test['MasVnrArea'].mean(),inplace=True)
hp_test['MasVnrArea'].fillna(hp_test['MasVnrArea'].mean(),inplace=True)
hp_test['BsmtFinSF2'].fillna(0,inplace=True)
hp_test['BsmtFinSF1'].fillna(hp_test['BsmtFinSF1'].mean(),inplace=True)

hp_test['PorchSF']=hp_test['OpenPorchSF']+hp_test['EnclosedPorch']
hp_test['TotalHouse'] = hp_test['TotalBsmtSF'] + hp_test['1stFlrSF'] + hp_test['2ndFlrSF']   
hp_test['TotalArea'] = hp_test['TotalBsmtSF'] + hp_test['1stFlrSF'] + hp_test['2ndFlrSF'] + hp_test['GarageArea']
hp_test['LotArea_OverallQual'] = hp_test['LotArea'] * hp_test['OverallQual']
hp_test['TotalHouse_LotArea'] = hp_test['TotalHouse'] + hp_test['LotArea']
hp_test['Bsmt'] = hp_test['BsmtFinSF1'] + hp_test['BsmtFinSF2'] + hp_test['BsmtUnfSF']
hp_test['PorchArea'] = hp_test['PorchSF']+hp_test['3SsnPorch']+hp_test['ScreenPorch']
hp_test['TotalPlace'] = hp_test['TotalBsmtSF'] + hp_test['1stFlrSF'] + hp_test['2ndFlrSF'] + hp_test['GarageArea'] + hp_test['PorchSF']+hp_test['3SsnPorch']+hp_test['ScreenPorch']

hp_test= hp_test.drop(['Street','Alley','Utilities','Condition2','RoofMatl','BsmtFinType1','BsmtFinSF1','BsmtFinType2','Fireplaces','PoolQC','Fence','MiscFeature','RoofStyle','BedroomAbvGr','MoSold','BsmtFullBath','BsmtHalfBath','MiscVal','LowQualFinSF','3SsnPorch','ScreenPorch','PoolArea','OpenPorchSF','EnclosedPorch','LandContour','LotConfig','LandSlope','ExterCond','BsmtExposure','CentralAir','Electrical','GarageCars','SaleType'], axis=1)


hp_test['SaleCondition'].fillna('oo',inplace=True)
hp_test['SaleCondition'][hp_test.loc[(hp_test['SaleCondition']=='AdjLand')==True]['SaleCondition'].index]='o'
hp_test['SaleCondition'][hp_test.loc[(hp_test['SaleCondition']=='Normal')==True]['SaleCondition'].index]='oo'
hp_test['SaleCondition'][hp_test.loc[(hp_test['SaleCondition']=='Partial')==True]['SaleCondition'].index]='ooo'
hp_test['SaleCondition'][hp_test.loc[(hp_test['SaleCondition']=='Abnorml')==True]['SaleCondition'].index]='oooo'
hp_test['SaleCondition'][hp_test.loc[(hp_test['SaleCondition']=='Alloca')==True]['SaleCondition'].index]='oooo'
hp_test['SaleCondition'][hp_test.loc[(hp_test['SaleCondition']=='Family')==True]['SaleCondition'].index]='oooo'

hp_test['FireplaceQu'].fillna('ooo',inplace=True)
hp_test['FireplaceQu'][hp_test.loc[(hp_test['FireplaceQu']=='None')==True]['FireplaceQu'].index]='o'
hp_test['FireplaceQu'][hp_test.loc[(hp_test['FireplaceQu']=='Po')==True]['FireplaceQu'].index]='o'
hp_test['FireplaceQu'][hp_test.loc[(hp_test['FireplaceQu']=='Fa')==True]['FireplaceQu'].index]='oo'
hp_test['FireplaceQu'][hp_test.loc[(hp_test['FireplaceQu']=='TA')==True]['FireplaceQu'].index]='ooo'
hp_test['FireplaceQu'][hp_test.loc[(hp_test['FireplaceQu']=='Gd')==True]['FireplaceQu'].index]='oooo'  
hp_test['FireplaceQu'][hp_test.loc[(hp_test['FireplaceQu']=='Ex')==True]['FireplaceQu'].index]='ooooo'

hp_test['BldgType'].fillna('oo',inplace=True)
hp_test['BldgType'][hp_test.loc[(hp_test['BldgType']=='TwnhsE')==True]['BldgType'].index]='oo'
hp_test['BldgType'][hp_test.loc[(hp_test['BldgType']=='1Fam')==True]['BldgType'].index]='oo'
hp_test['BldgType'][hp_test.loc[(hp_test['BldgType']!='oo')==True]['BldgType'].index]='o '

hp_test['Heating'].fillna('o',inplace=True)
hp_test['Heating'][hp_test.loc[(hp_test['Heating']=='Floor')==True]['Heating'].index]='o'
hp_test['Heating'][hp_test.loc[(hp_test['Heating']=='Grav')==True]['Heating'].index]='o'
hp_test['Heating'][hp_test.loc[(hp_test['Heating']=='Wall')==True]['Heating'].index]='oo'
hp_test['Heating'][hp_test.loc[(hp_test['Heating']=='OthW')==True]['Heating'].index]='ooo'
hp_test['Heating'][hp_test.loc[(hp_test['Heating']=='GasW')==True]['Heating'].index]='oooo'  
hp_test['Heating'][hp_test.loc[(hp_test['Heating']=='GasA')==True]['Heating'].index]='ooooo'

hp_test['Condition1'].fillna('ooo',inplace=True)
hp_test['Condition1'][hp_test.loc[(hp_test['Condition1']=='Artery')==True]['Condition1'].index]='o'
hp_test['Condition1'][hp_test.loc[(hp_test['Condition1']=='Feedr')==True]['Condition1'].index]='oo'
hp_test['Condition1'][hp_test.loc[(hp_test['Condition1']=='RRAe')==True]['Condition1'].index]='oo'
hp_test['Condition1'][hp_test.loc[(hp_test['Condition1']=='Norm')==True]['Condition1'].index]='ooo'
hp_test['Condition1'][hp_test.loc[(hp_test['Condition1']=='RRAn')==True]['Condition1'].index]='ooo'  
hp_test['Condition1'][hp_test.loc[(hp_test['Condition1']=='PosN')==True]['Condition1'].index]='oooo'
hp_test['Condition1'][hp_test.loc[(hp_test['Condition1']=='RRNe')==True]['Condition1'].index]='oooo'
hp_test['Condition1'][hp_test.loc[(hp_test['Condition1']=='PosA')==True]['Condition1'].index]='ooooo'
hp_test['Condition1'][hp_test.loc[(hp_test['Condition1']=='RRNn')==True]['Condition1'].index]='ooooo'

hp_test['HouseStyle'].fillna('oo',inplace=True)
hp_test['HouseStyle'][hp_test.loc[(hp_test['HouseStyle']=='1.5Unf')==True]['HouseStyle'].index]='o'
hp_test['HouseStyle'][hp_test.loc[(hp_test['HouseStyle']=='1.5Fin')==True]['HouseStyle'].index]='oo'
hp_test['HouseStyle'][hp_test.loc[(hp_test['HouseStyle']=='2.5Unf')==True]['HouseStyle'].index]='oo'
hp_test['HouseStyle'][hp_test.loc[(hp_test['HouseStyle']=='SFoyer')==True]['HouseStyle'].index]='oo'
hp_test['HouseStyle'][hp_test.loc[(hp_test['HouseStyle']=='1Story')==True]['HouseStyle'].index]='ooo'  
hp_test['HouseStyle'][hp_test.loc[(hp_test['HouseStyle']=='SLvl')==True]['HouseStyle'].index]='ooo'
hp_test['HouseStyle'][hp_test.loc[(hp_test['HouseStyle']=='2Story')==True]['HouseStyle'].index]='oooo'
hp_test['HouseStyle'][hp_test.loc[(hp_test['HouseStyle']=='2.5Fin')==True]['HouseStyle'].index]='oooo'

hp_test['Exterior1st'].fillna('ooo',inplace=True)
hp_test['Exterior1st'][hp_test.loc[(hp_test['Exterior1st']=='BrkComm')==True]['Exterior1st'].index]='o'
hp_test['Exterior1st'][hp_test.loc[(hp_test['Exterior1st']=='AsphShn')==True]['Exterior1st'].index]='oo'
hp_test['Exterior1st'][hp_test.loc[(hp_test['Exterior1st']=='CBlock')==True]['Exterior1st'].index]='oo'
hp_test['Exterior1st'][hp_test.loc[(hp_test['Exterior1st']=='AsbShng')==True]['Exterior1st'].index]='oo'
hp_test['Exterior1st'][hp_test.loc[(hp_test['Exterior1st']=='WdShing')==True]['Exterior1st'].index]='ooo'  
hp_test['Exterior1st'][hp_test.loc[(hp_test['Exterior1st']=='Wd Sdng')==True]['Exterior1st'].index]='ooo'
hp_test['Exterior1st'][hp_test.loc[(hp_test['Exterior1st']=='MetalSd')==True]['Exterior1st'].index]='ooo'
hp_test['Exterior1st'][hp_test.loc[(hp_test['Exterior1st']=='Stucco')==True]['Exterior1st'].index]='ooo'
hp_test['Exterior1st'][hp_test.loc[(hp_test['Exterior1st']=='HdBoard')==True]['Exterior1st'].index]='ooo'
hp_test['Exterior1st'][hp_test.loc[(hp_test['Exterior1st']=='BrkFace')==True]['Exterior1st'].index]='oooo'
hp_test['Exterior1st'][hp_test.loc[(hp_test['Exterior1st']=='Plywood')==True]['Exterior1st'].index]='oooo'
hp_test['Exterior1st'][hp_test.loc[(hp_test['Exterior1st']=='VinylSd')==True]['Exterior1st'].index]='ooooo'  
hp_test['Exterior1st'][hp_test.loc[(hp_test['Exterior1st']=='CemntBd')==True]['Exterior1st'].index]='oooooo'
hp_test['Exterior1st'][hp_test.loc[(hp_test['Exterior1st']=='Stone')==True]['Exterior1st'].index]='ooooooo'
hp_test['Exterior1st'][hp_test.loc[(hp_test['Exterior1st']=='ImStucc')==True]['Exterior1st'].index]='ooooooo'


hp_test['Exterior1st'][hp_test.loc[(hp_test['Exterior1st']!='VinylSd')==True]['Exterior1st'].index]='Other'
hp_test['Exterior2nd'][hp_test.loc[(hp_test['Exterior2nd']!='VinylSd')==True]['Exterior2nd'].index]='Other'
hp_test['Foundation'][hp_test.loc[(hp_test['Foundation']!='PConc')==True]['Foundation'].index]='Other'
hp_test['HeatingQC'][hp_test.loc[(hp_test['HeatingQC']!='Ex')==True]['HeatingQC'].index]='Other'
hp_test['GarageFinish'][hp_test.loc[(hp_test['GarageFinish']!='RFn')==True]['GarageFinish'].index]='Other'


hp_test['TotRmsAbvGrd'][hp_test.loc[(hp_test['TotRmsAbvGrd']<=6)==True]['TotRmsAbvGrd'].index]='L6'
hp_test['TotRmsAbvGrd'][hp_test.loc[(hp_test['TotRmsAbvGrd']!='L6')==True]['TotRmsAbvGrd'].index]='B6'
#
hp_test['GarageCond'][hp_test.loc[(hp_test['GarageCond']=='Fa')==True]['GarageCond'].index]='Bd'
hp_test['GarageCond'][hp_test.loc[(hp_test['GarageCond']=='Po')==True]['GarageCond'].index]='Bd'
hp_test['GarageCond'][hp_test.loc[(hp_test['GarageCond']!='Bd')==True]['GarageCond'].index]='Gd'


hp_test['MSZoning'].fillna('oo',inplace=True)
hp_test['MSZoning'][hp_test.loc[(hp_test['MSZoning']=='FV')==True]['MSZoning'].index]='o'
hp_test['MSZoning'][hp_test.loc[(hp_test['MSZoning']=='RL')==True]['MSZoning'].index]='oo'
hp_test['MSZoning'][hp_test.loc[(hp_test['MSZoning']=='RP')==True]['MSZoning'].index]='oo'
hp_test['MSZoning'][hp_test.loc[(hp_test['MSZoning']=='A')==True]['MSZoning'].index]='oo'
hp_test['MSZoning'][hp_test.loc[(hp_test['MSZoning']=='C (all)')==True]['MSZoning'].index]='oo'
hp_test['MSZoning'][hp_test.loc[(hp_test['MSZoning']=='I')==True]['MSZoning'].index]='oo'
hp_test['MSZoning'][hp_test.loc[(hp_test['MSZoning']=='RH')==True]['MSZoning'].index]='ooo'
hp_test['MSZoning'][hp_test.loc[(hp_test['MSZoning']=='RM')==True]['MSZoning'].index]='ooo'

hp_test['LotShape'][hp_test.loc[(hp_test['LotShape']=='Reg')==True]['LotShape'].index]='o'
hp_test['LotShape'][hp_test.loc[(hp_test['LotShape']=='IR1')==True]['LotShape'].index]='oo'
hp_test['LotShape'][hp_test.loc[(hp_test['LotShape']=='IR2')==True]['LotShape'].index]='ooo'
hp_test['LotShape'][hp_test.loc[(hp_test['LotShape']=='IR3')==True]['LotShape'].index]='ooo'


hp_test['OverallCond'][hp_test.loc[(hp_test['OverallCond']>=6)==True]['OverallCond'].index]='o'
hp_test['OverallCond'][hp_test.loc[(hp_test['OverallCond']==5)==True]['OverallCond'].index]='tw'
hp_test['OverallCond'][hp_test.loc[(hp_test['OverallCond']==4)==True]['OverallCond'].index]='t'
hp_test['OverallCond'][hp_test.loc[(hp_test['OverallCond']<=3)==True]['OverallCond'].index]='f'

hp_test['MasVnrType'].fillna('ooo',inplace=True)
hp_test['MasVnrType'][hp_test.loc[(hp_test['MasVnrType']=='Stone')==True]['MasVnrType'].index]='o'
hp_test['MasVnrType'][hp_test.loc[(hp_test['MasVnrType']=='BrkFace')==True]['MasVnrType'].index]='oo'
hp_test['MasVnrType'][hp_test.loc[(hp_test['MasVnrType']=='None')==True]['MasVnrType'].index]='ooo'
hp_test['MasVnrType'][hp_test.loc[(hp_test['MasVnrType']=='CBlock')==True]['MasVnrType'].index]='ooo'
hp_test['MasVnrType'][hp_test.loc[(hp_test['MasVnrType']=='BrkCmn')==True]['MasVnrType'].index]='ooo'

hp_test['ExterQual'][hp_test.loc[(hp_test['ExterQual']=='Ex')==True]['ExterQual'].index]='o'
hp_test['ExterQual'][hp_test.loc[(hp_test['ExterQual']=='Gd')==True]['ExterQual'].index]='oo'
hp_test['ExterQual'][hp_test.loc[(hp_test['ExterQual']=='TA')==True]['ExterQual'].index]='ooo'
hp_test['ExterQual'][hp_test.loc[(hp_test['ExterQual']=='Fa')==True]['ExterQual'].index]='ooo'

hp_test['BsmtQual'].fillna('oooo',inplace=True)
hp_test['BsmtQual'][hp_test.loc[(hp_test['BsmtQual']=='Ex')==True]['BsmtQual'].index]='o'
hp_test['BsmtQual'][hp_test.loc[(hp_test['BsmtQual']=='Gd')==True]['BsmtQual'].index]='oo'
hp_test['BsmtQual'][hp_test.loc[(hp_test['BsmtQual']=='TA')==True]['BsmtQual'].index]='ooo'
hp_test['BsmtQual'][hp_test.loc[(hp_test['BsmtQual']=='Fa')==True]['BsmtQual'].index]='ooo'

hp_test['BsmtCond'].fillna('oooo',inplace=True)
hp_test['BsmtCond'][hp_test.loc[(hp_test['BsmtCond']=='Ex')==True]['BsmtCond'].index]='o'
hp_test['BsmtCond'][hp_test.loc[(hp_test['BsmtCond']=='Gd')==True]['BsmtCond'].index]='oo'
hp_test['BsmtCond'][hp_test.loc[(hp_test['BsmtCond']=='TA')==True]['BsmtCond'].index]='ooo'
hp_test['BsmtCond'][hp_test.loc[(hp_test['BsmtCond']=='Fa')==True]['BsmtCond'].index]='ooo'

hp_test['KitchenQual'].fillna('ooo',inplace=True)
hp_test['KitchenQual'][hp_test.loc[(hp_test['KitchenQual']=='Ex')==True]['KitchenQual'].index]='o'
hp_test['KitchenQual'][hp_test.loc[(hp_test['KitchenQual']=='Gd')==True]['KitchenQual'].index]='oo'
hp_test['KitchenQual'][hp_test.loc[(hp_test['KitchenQual']=='TA')==True]['KitchenQual'].index]='ooo'
hp_test['KitchenQual'][hp_test.loc[(hp_test['KitchenQual']=='Fa')==True]['KitchenQual'].index]='ooo'

hp_test['GarageType'].fillna('oooo',inplace=True)
hp_test['GarageType'][hp_test.loc[(hp_test['GarageType']=='BuiltIn')==True]['GarageType'].index]='o'
hp_test['GarageType'][hp_test.loc[(hp_test['GarageType']=='Attchd')==True]['GarageType'].index]='oo'
hp_test['GarageType'][hp_test.loc[(hp_test['GarageType']=='2Types')==True]['GarageType'].index]='ooo'
hp_test['GarageType'][hp_test.loc[(hp_test['GarageType']=='Basment')==True]['GarageType'].index]='ooo'
hp_test['GarageType'][hp_test.loc[(hp_test['GarageType']=='Detchd')==True]['GarageType'].index]='ooo'
hp_test['GarageType'][hp_test.loc[(hp_test['GarageType']=='CarPort')==True]['GarageType'].index]='ooo'

hp_test['GarageQual'].fillna('ooo',inplace=True)
hp_test['GarageQual'][hp_test.loc[(hp_test['GarageQual']=='Ex')==True]['GarageQual'].index]='o'
hp_test['GarageQual'][hp_test.loc[(hp_test['GarageQual']=='Gd')==True]['GarageQual'].index]='o'
hp_test['GarageQual'][hp_test.loc[(hp_test['GarageQual']=='TA')==True]['GarageQual'].index]='oo'
hp_test['GarageQual'][hp_test.loc[(hp_test['GarageQual']=='Fa')==True]['GarageQual'].index]='ooo'
hp_test['GarageQual'][hp_test.loc[(hp_test['GarageQual']=='Po')==True]['GarageQual'].index]='ooo'
hp_test['GarageQual'].fillna('oooo',inplace=True)

hp_test['FullBath'][hp_test.loc[(hp_test['FullBath']==0)==True]['FullBath'].index]='o'


hp_test['MSSubClass'][hp_test.loc[(hp_test['MSSubClass']==30)==True]['MSSubClass'].index]='o'
hp_test['MSSubClass'][hp_test.loc[(hp_test['MSSubClass']==45)==True]['MSSubClass'].index]='o'
hp_test['MSSubClass'][hp_test.loc[(hp_test['MSSubClass']==180)==True]['MSSubClass'].index]='oooo'
hp_test['MSSubClass'][hp_test.loc[(hp_test['MSSubClass']==50)==True]['MSSubClass'].index]='oo'
hp_test['MSSubClass'][hp_test.loc[(hp_test['MSSubClass']==90)==True]['MSSubClass'].index]='oo'
hp_test['MSSubClass'][hp_test.loc[(hp_test['MSSubClass']==190)==True]['MSSubClass'].index]='oo'
hp_test['MSSubClass'][hp_test.loc[(hp_test['MSSubClass']==70)==True]['MSSubClass'].index]='ooo'
hp_test['MSSubClass'][hp_test.loc[(hp_test['MSSubClass']==80)==True]['MSSubClass'].index]='ooo'
hp_test['MSSubClass'][hp_test.loc[(hp_test['MSSubClass']==20)==True]['MSSubClass'].index]='ooo'
hp_test['MSSubClass'][hp_test.loc[(hp_test['MSSubClass']==75)==True]['MSSubClass'].index]='ooo'
hp_test['MSSubClass'][hp_test.loc[(hp_test['MSSubClass']==60)==True]['MSSubClass'].index]='ooooo'
hp_test['MSSubClass'][hp_test.loc[(hp_test['MSSubClass']==120)==True]['MSSubClass'].index]='ooooo'
hp_test['MSSubClass'][hp_test.loc[(hp_test['MSSubClass']==85)==True]['MSSubClass'].index]='oo'
hp_test['MSSubClass'][hp_test.loc[(hp_test['MSSubClass']==160)==True]['MSSubClass'].index]='oooooo'
hp_test['MSSubClass'][hp_test.loc[(hp_test['MSSubClass']==150)==True]['MSSubClass'].index]='oooooo'
hp_test['MSSubClass'][hp_test.loc[(hp_test['MSSubClass']==40)==True]['MSSubClass'].index]='oooooo'

hp_test['Neighborhood'][hp_test.loc[(hp_test['Neighborhood']=='Edwards')==True]['Neighborhood'].index]='o'
hp_test['Neighborhood'][hp_test.loc[(hp_test['Neighborhood']=='OldTown')==True]['Neighborhood'].index]='o'
hp_test['Neighborhood'][hp_test.loc[(hp_test['Neighborhood']=='BrkSide')==True]['Neighborhood'].index]='o'
hp_test['Neighborhood'][hp_test.loc[(hp_test['Neighborhood']=='BrDale')==True]['Neighborhood'].index]='ooooooo'
hp_test['Neighborhood'][hp_test.loc[(hp_test['Neighborhood']=='IDOTRR')==True]['Neighborhood'].index]='ooooooo'
hp_test['Neighborhood'][hp_test.loc[(hp_test['Neighborhood']=='NWAmes')==True]['Neighborhood'].index]='oo'
hp_test['Neighborhood'][hp_test.loc[(hp_test['Neighborhood']=='Gilbert')==True]['Neighborhood'].index]='oo'
hp_test['Neighborhood'][hp_test.loc[(hp_test['Neighborhood']=='SawyerW')==True]['Neighborhood'].index]='oo'
hp_test['Neighborhood'][hp_test.loc[(hp_test['Neighborhood']=='CollgCr')==True]['Neighborhood'].index]='ooo'
hp_test['Neighborhood'][hp_test.loc[(hp_test['Neighborhood']=='ClearCr')==True]['Neighborhood'].index]='ooo'
hp_test['Neighborhood'][hp_test.loc[(hp_test['Neighborhood']=='Crawfor')==True]['Neighborhood'].index]='ooo'
hp_test['Neighborhood'][hp_test.loc[(hp_test['Neighborhood']=='Blmngtn')==True]['Neighborhood'].index]='ooo'
hp_test['Neighborhood'][hp_test.loc[(hp_test['Neighborhood']=='NridgHt')==True]['Neighborhood'].index]='oooo'
hp_test['Neighborhood'][hp_test.loc[(hp_test['Neighborhood']=='NoRidge')==True]['Neighborhood'].index]='oooo'
hp_test['Neighborhood'][hp_test.loc[(hp_test['Neighborhood']=='Blueste')==True]['Neighborhood'].index]='e'
hp_test['Neighborhood'][hp_test.loc[(hp_test['Neighborhood']=='Mitchel')==True]['Neighborhood'].index]='ooooo'
hp_test['Neighborhood'][hp_test.loc[(hp_test['Neighborhood']=='SWISU')==True]['Neighborhood'].index]='e'
hp_test['Neighborhood'][hp_test.loc[(hp_test['Neighborhood']=='Timber')==True]['Neighborhood'].index]='oooooo'
hp_test['Neighborhood'][hp_test.loc[(hp_test['Neighborhood']=='Veenker')==True]['Neighborhood'].index]='oooooo'
hp_test['Neighborhood'][hp_test.loc[(hp_test['Neighborhood']=='NPkVill')==True]['Neighborhood'].index]='ooooo'
hp_test['Neighborhood'][hp_test.loc[(hp_test['Neighborhood']=='Somerst')==True]['Neighborhood'].index]='oooooo'
hp_test['Neighborhood'][hp_test.loc[(hp_test['Neighborhood']=='NAmes')==True]['Neighborhood'].index]='e'
hp_test['Neighborhood'][hp_test.loc[(hp_test['Neighborhood']=='Sawyer')==True]['Neighborhood'].index]='e'

hp_test['MasVnrAreaExist'] ='oo'
hp_test['MasVnrAreaExist'][hp_test.loc[(hp_test['MasVnrArea']!=0)==True]['MasVnrArea'].index]='o'
hp_test['TotalBsmtSFExist'] ='oo'
hp_test['TotalBsmtSFExist'][hp_test.loc[(hp_test['TotalBsmtSF']!=0)==True]['TotalBsmtSF'].index]='o'
hp_test['2ndFlrSFExist'] ='oo'
hp_test['2ndFlrSFExist'][hp_test.loc[(hp_test['2ndFlrSF']!=0)==True]['2ndFlrSF'].index]='o'
hp_test['WoodDeckSFExist'] ='oo'
hp_test['WoodDeckSFExist'][hp_test.loc[(hp_test['WoodDeckSF']!=0)==True]['WoodDeckSF'].index]='o'
hp_test['GarageAreaExist'] ='oo'
hp_test['GarageAreaExist'][hp_test.loc[(hp_test['GarageArea']!=0)==True]['GarageArea'].index]='o'
hp_test['PorchSFExist'] ='oo'
hp_test['PorchSFExist'][hp_test.loc[(hp_test['PorchSF']!=0)==True]['PorchSF'].index]='o'
hp_test['TotalSF']=hp_test['PorchSF']+hp_test['GarageArea']+hp_test['GrLivArea']+hp_test['TotalBsmtSF']+hp_test['LotArea']


skewed_sets=['BsmtFinSF2','LotFrontage','LotArea','MasVnrArea','TotalBsmtSF','1stFlrSF','2ndFlrSF','GrLivArea','GarageArea','WoodDeckSF']
for train in skewed_sets:
    hp_test[train] = np.log1p(hp_test[train])

hp_test = pd.get_dummies(hp_test)

hp_test.to_csv('C:/Users/Administrator/ml_python/ex1/house_price/hp_test_422.csv', index=False)


X=hp_train
X_train=X.drop(['SalePrice','Id'], axis=1)
y_train=hp_train['SalePrice']
X_test=hp_test.drop(['Id'], axis=1)

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
y_train=np.log1p(y_train)
#initialization

from sklearn.linear_model import LinearRegression,SGDRegressor 
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

PSVR=SVR(kernel='poly')
RSVR=SVR(kernel='rbf')
ETR=ExtraTreesRegressor()
GBR=GradientBoostingRegressor()

#training & prediction
PSVR.fit(X_train,y_train)
PSVR_y_predict=PSVR.predict(X_test)
RSVR.fit(X_train,y_train)
RSVR_y_predict=RSVR.predict(X_test)
ETR.fit(X_train,y_train)
GBR.fit(X_train,y_train)
ETR_y_predict=ETR.predict(X_test)
GBR_y_predict=GBR.predict(X_test)

ETR_y_predict=expm1(ETR_y_predict)
GBR_y_predict=expm1(GBR_y_predict)
hp_test['SalePrice']=PSVR_y_predict
PSVR_submission=hp_test[['Id','SalePrice']]
PSVR_submission[['Id','SalePrice']].to_csv('C:/Users/Administrator/ml_python/ex1/house_price/PSVR_422.csv', index=False)

hp_test['SalePrice']=RSVR_y_predict
RSVR_submission=hp_test[['Id','SalePrice']]
RSVR_submission[['Id','SalePrice']].to_csv('C:/Users/Administrator/ml_python/ex1/house_price/RSVR_422.csv', index=False)

hp_test['SalePrice']=ETR_y_predict
ETR_submission=hp_test[['Id','SalePrice']]
ETR_submission[['Id','SalePrice']].to_csv('C:/Users/Administrator/ml_python/ex1/house_price/ETR_422.csv', index=False)

hp_test['SalePrice']=GBR_y_predict
GBR_submission=hp_test[['Id','SalePrice']]
GBR_submission[['Id','SalePrice']].to_csv('C:/Users/Administrator/ml_python/ex1/house_price/GBR_422.csv', index=False)
