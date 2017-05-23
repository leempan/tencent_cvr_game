from pandas import read_csv
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot
# load data
def gbdt_cv(X,y,n_estimators,maxdepth):
	print X.shape,y.shape
	# encode string class values as integers
	label_encoded_y = LabelEncoder().fit_transform(y)
	print label_encoded_y
	# grid search
	model = XGBClassifier()
	param_grid = dict(n_estimators=n_estimators,max_depth = maxdepth)
	kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
	grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold,verbose=1)
	grid_result = grid_search.fit(X, label_encoded_y)
	# summarize results
	print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
	means = grid_result.cv_results_['mean_test_score']
	stds = grid_result.cv_results_['std_test_score']
	params = grid_result.cv_results_['params']
	for mean, stdev, param in zip(means, stds, params):
		print("%f (%f) with: %r" % (mean, stdev, param))

train_data = 'merged_train.csv'
train_raw = read_csv(train_data,header=0)
train_Y = train_raw['label'].as_matrix()[0:train_raw.shape[0]/10]
train_X_creID = train_raw['creativeID'].as_matrix()[0:train_raw.shape[0]/10]
train_X_userID = train_raw['userID'].as_matrix()[0:train_raw.shape[0]/10]
train_X_posID = train_raw['positionID'].as_matrix()[0:train_raw.shape[0]/10]
train_X_siteID = train_raw['sitesetID'].as_matrix()[0:train_raw.shape[0]/10]
train_X_adID = train_raw['adID'].as_matrix()[0:train_raw.shape[0]/10]
train_X_camID = train_raw['camgaignID'].as_matrix()[0:train_raw.shape[0]/10]
train_X_advID = train_raw['advertiserID'].as_matrix()[0:train_raw.shape[0]/10]
train_X_appID = train_raw['appID'].as_matrix()[0:train_raw.shape[0]/10]
train_X = train_raw.as_matrix()[0:train_raw.shape[0]/10,:]
train_X = np.delete(train_X,[0,1,2,3,4,5,8,9,10,11,20],1)

n_estimators = range(10, 100, 10)
maxdepth = range(3,10,1)
gbdt_cv(train_X_creID.reshape(len(train_X_creID),1),train_Y,n_estimators,maxdepth)
gbdt_cv(train_X_userID.reshape(len(train_X_userID),1),train_Y,n_estimators,maxdepth)
gbdt_cv(train_X_posID.reshape(len(train_X_posID),1),train_Y,n_estimators,maxdepth)
gbdt_cv(train_X_siteID.reshape(len(train_X_siteID),1),train_Y,n_estimators,maxdepth)
gbdt_cv(train_X_adID.reshape(len(train_X_adID),1),train_Y,n_estimators,maxdepth)
gbdt_cv(train_X_camID.reshape(len(train_X_camID),1),train_Y,n_estimators,maxdepth)
gbdt_cv(train_X_advID.reshape(len(train_X_advID),1),train_Y,n_estimators,maxdepth)
gbdt_cv(train_X_appID.reshape(len(train_X_appID),1),train_Y,n_estimators,maxdepth)
n_estimators = range(20, 300, 20)
gbdt_cv(train_X,train_Y,n_estimators,maxdepth)

