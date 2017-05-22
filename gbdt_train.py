import pandas as pd
import numpy as np
import scipy as sc
import xgboost as xgb
from joblib import dump
import  matplotlib.pyplot as plt
train_data = '/home/administrator/Limingpan/pre/merged_train.csv'

train_raw = pd.read_csv(train_data,header=0)
train_Y = train_raw['label'].as_matrix()
train_X_creID = train_raw['creativeID'].as_matrix()
train_X_userID = train_raw['userID'].as_matrix()
train_X_posID = train_raw['positionID'].as_matrix()
train_X_siteID = train_raw['sitesetID'].as_matrix()
train_X_adID = train_raw['adID'].as_matrix()
train_X_camID = train_raw['camgaignID'].as_matrix()
train_X_advID = train_raw['advertiserID'].as_matrix()
train_X_appID = train_raw['appID'].as_matrix()
train_X = train_raw.as_matrix()
train_X = np.delete(train_X,[0,1,2,3,4,5,8,9,10,11,20],1)
# print train_X
# print train_X.shape
def modelfilt(X,Y,params,useTrainCV=True,cv_fold=5,model_name='train',gen_new_feature=True,new_fea_filename="train_new_file.txt",n_trees=1,cvfolds=2):
	print X.shape,Y.shape
	dtrain = xgb.DMatrix(X,label=Y)
	plst = params.items()
	if useTrainCV:
		cv = xgb.cv(plst,dtrain,num_boost_round=params['n_estimators'],nfold=cvfolds)
		params['n_estimators']=cv.shape[0]
		
	plst = params.items() 
	bst = xgb.train(plst,dtrain,n_trees)
	bst.save_model(model_name)	
	# xgb.plot_importance(bst)
	# xgb.plot_tree(bst)
	# plt.show()
	# xgb.to_graghviz(bst,num_trees=n_trees)
	if gen_new_feature:
		train_new_feature = bst.predict(dtrain,pred_leaf=True)
		print train_new_feature.shape
		np.savetxt(new_fea_filename,train_new_feature,delimiter=',')
		# train_new_feature.head()
params = {'booster':'gbtree','max_depth':7,'eta':.02,'objective':'binary:logistic','verbose':0,
			'subsample':1.0,'early_stoppping_rounds':100000,'seed':999,'eval_metric':'logloss','nthread':16,
			'colsample_bytree':1,'min_child_weight':10,'gamma':4,'alpha':0.6,'max_delta_step':2,'scale_pos_weight':10,'n_estimators':30}
modelfilt(train_X,train_Y,params,model_name='none_id_fea.model',new_fea_filename='none_id_fea.txt',n_trees=100,cvfolds=10)

train_X_creID = np.reshape(train_X_creID,(train_X_creID.shape[0],1))
train_X_userID = np.reshape(train_X_userID,(train_X_userID.shape[0],1))
train_X_posID = np.reshape(train_X_posID,(train_X_posID.shape[0],1))
train_X_siteID = np.reshape(train_X_siteID,(train_X_siteID.shape[0],1))
train_X_adID = np.reshape(train_X_adID,(train_X_adID.shape[0],1))
train_X_camID = np.reshape(train_X_camID,(train_X_camID.shape[0],1))
train_X_advID = np.reshape(train_X_advID,(train_X_advID.shape[0],1))
train_X_appID = np.reshape(train_X_appID,(train_X_appID.shape[0],1))


print train_X_creID.shape
modelfilt(train_X_creID,train_Y,params,model_name='cre_id_fea.model',new_fea_filename='cre_id_fea.txt',n_trees=10,cvfolds=5)
modelfilt(train_X_userID,train_Y,params,model_name='user_id_fea.model',new_fea_filename='user_id_fea.txt',n_trees=10,cvfolds=5)
modelfilt(train_X_posID,train_Y,params,model_name='pos_id_fea.model',new_fea_filename='pos_id_fea.txt',n_trees=10,cvfolds=5)
modelfilt(train_X_siteID,train_Y,params,model_name='site_id_fea.model',new_fea_filename='site_id_fea.txt',n_trees=10,cvfolds=5)
modelfilt(train_X_adID,train_Y,params,model_name='ad_id_fea.model',new_fea_filename='ad_id_fea.txt',n_trees=10,cvfolds=5)
modelfilt(train_X_camID,train_Y,params,model_name='cam_id_fea.model',new_fea_filename='cam_id_fea.txt',n_trees=10,cvfolds=5)
modelfilt(train_X_advID,train_Y,params,model_name='adv_id_fea.model',new_fea_filename='adv_id_fea.txt',n_trees=10,cvfolds=5)
modelfilt(train_X_appID,train_Y,params,model_name='app_id_fea.model',new_fea_filename='app_id_fea.txt',n_trees=10,cvfolds=5)

