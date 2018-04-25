import pandas as pd
hp_train=pd.read_csv('C:/Users/Administrator/ml_python/ex1/Datasets/house_pricing/train.csv')
#==============================================================================
# 
# X_6=hp_train.loc[hp_train['BedroomAbvGr']=='FV']['BedroomAbvGr']
# print '1',len(X_6)
# 
# X_2=hp_train.loc[hp_train['BedroomAbvGr']=='RL']['BedroomAbvGr']
# print '2',len(X_2)
# 
# X_3=hp_train.loc[hp_train['BedroomAbvGr']=='RM']['BedroomAbvGr']
# print '3',len(X_3)
# print len(X_6)+len(X_2)+len(X_3)
# X_3=hp_train.loc[hp_train['BedroomAbvGr']=='RH']['BedroomAbvGr']
# print '3',len(X_3)
#==============================================================================




import matplotlib.pyplot as plt
fig=plt.figure()
fig.set(alpha=0.2)
hp_train.SalePrice[hp_train.BedroomAbvGr==1].plot(kind='kde')
hp_train.SalePrice[hp_train.BedroomAbvGr==2].plot(kind='kde')
hp_train.SalePrice[hp_train.BedroomAbvGr==3].plot(kind='kde')
hp_train.SalePrice[hp_train.BedroomAbvGr==4].plot(kind='kde')

plt.legend(('Lv1','RL','RM','RH'),loc='best')
fig.savefig("BedroomAbvGr.png")
#==============================================================================
# test=['GarageYrBlt','YrSold','YearBuilt','YearRemodAdd','BedroomAbvGr','LotArea','MasVnrArea','TotalBsmtSF','1stFlrSF','2ndFlrSF','GrLivArea','GarageArea','WoodDeckSF','MiscVal','BsmtFinSF2','BsmtUnfSF','LowQualFinSF','3SsnPorch','ScreenPorch']
# for set in test:
#     X=hp_train[[set,'SalePrice']]
#     X_nonzero=X.loc[X[set]!=0]
#     print set,len(X_nonzero)
#==============================================================================
#==============================================================================
# X=hp_train[['BedroomAbvGr','SalePrice']]
# X_nonzero=X.loc[X['BedroomAbvGr']!=0]
# X=X_nonzero['BedroomAbvGr']
# y=X_nonzero['SalePrice']
# plt.scatter(X,y)
# plt.show()
#==============================================================================
