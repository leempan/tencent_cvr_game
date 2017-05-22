import pandas as pd
import numpy as np
import scipy as sc
import xgboost as xgb

print 'preparing data...'
test_data = '/home/administrator/Limingpan/pre/merged_test.csv'
test_raw = pd.read_csv(test_data,header=0)
test_X_creID = test_raw['creativeID'].as_matrix()
test_X_userID = test_raw['userID'].as_matrix()
test_X_posID = test_raw['positionID'].as_matrix()
test_X_siteID = test_raw['sitesetID'].as_matrix()
test_X_adID = test_raw['adID'].as_matrix()
test_X_camID = test_raw['camgaignID'].as_matrix()
test_X_advID = test_raw['advertiserID'].as_matrix()
test_X_appID = test_raw['appID'].as_matrix()
test_X = test_raw.as_matrix()
test_X_noneID = np.delete(test_X,[0,1,2,3,4,5,8,9,10,11,20],1)
print 'preparing data done.'
d_noneid_test = xgb.DMatrix(test_X_noneID)
print 'loading noneid model'
bst_none_id = xgb.Booster({'nthread':16})
bst_none_id.load_model('none_id_fea.model')
print 'load noneid model finished'
test_noneid_newfea = bst_none_id.predict(d_noneid_test,pred_leaf=True)
np.savetxt('none_id_fea_test.txt',test_noneid_newfea.astype(int),delimiter=',')

test_X_creID = np.reshape(test_X_creID,(test_X_creID.shape[0],1))
test_X_userID = np.reshape(test_X_userID,(test_X_userID.shape[0],1))
test_X_posID = np.reshape(test_X_posID,(test_X_posID.shape[0],1))
test_X_siteID = np.reshape(test_X_siteID,(test_X_siteID.shape[0],1))
test_X_adID = np.reshape(test_X_adID,(test_X_adID.shape[0],1))
test_X_camID = np.reshape(test_X_camID,(test_X_camID.shape[0],1))
test_X_advID = np.reshape(test_X_advID,(test_X_advID.shape[0],1))
test_X_appID = np.reshape(test_X_appID,(test_X_appID.shape[0],1))

d_creid_test = xgb.DMatrix(test_X_creID)
print 'loading creid model'
bst_cre_id = xgb.Booster({'nthread':16})
bst_cre_id.load_model('cre_id_fea.model')
print 'load creid model finished'
test_creid_newfea = bst_cre_id.predict(d_creid_test,pred_leaf=True)
np.savetxt('cre_id_fea_test.txt',test_creid_newfea.astype(int),delimiter=',')

d_userid_test = xgb.DMatrix(test_X_userID)
print 'loading userid model'
bst_user_id = xgb.Booster({'nthread':16})
bst_user_id.load_model('user_id_fea.model')
print 'load model finished'
test_userid_newfea = bst_user_id.predict(d_userid_test,pred_leaf=True)
np.savetxt('user_id_fea_test.txt',test_userid_newfea.astype(int),delimiter=',')

d_posid_test = xgb.DMatrix(test_X_posID)
print 'loading posid model'
bst_pos_id = xgb.Booster({'nthread':16})
bst_pos_id.load_model('pos_id_fea.model')
print 'load posid model finished'
test_posid_newfea = bst_pos_id.predict(d_posid_test,pred_leaf=True)
np.savetxt('pos_id_fea_test.txt',test_posid_newfea.astype(int),delimiter=',')

d_siteid_test = xgb.DMatrix(test_X_siteID)
print 'loading siteid model'
bst_site_id = xgb.Booster({'nthread':16})
bst_site_id.load_model('site_id_fea.model')
print 'load siteid model finished'
test_siteid_newfea = bst_site_id.predict(d_siteid_test,pred_leaf=True)
np.savetxt('site_id_fea_test.txt',test_siteid_newfea.astype(int),delimiter=',')

d_adid_test = xgb.DMatrix(test_X_adID)
print 'loading adid model'
bst_ad_id = xgb.Booster({'nthread':16})
bst_ad_id.load_model('ad_id_fea.model')
print 'load adid model finished'
test_adid_newfea = bst_ad_id.predict(d_adid_test,pred_leaf=True)
np.savetxt('ad_id_fea_test.txt',test_adid_newfea.astype(int),delimiter=',')

d_camid_test = xgb.DMatrix(test_X_camID)
print 'loading camid model'
bst_cam_id = xgb.Booster({'nthrecam':16})
bst_cam_id.load_model('cam_id_fea.model')
print 'load camid model finished'
test_camid_newfea = bst_cam_id.predict(d_camid_test,pred_leaf=True)
np.savetxt('cam_id_fea_test.txt',test_camid_newfea.astype(int),delimiter=',')

d_advid_test = xgb.DMatrix(test_X_advID)
print 'loading advid model'
bst_adv_id = xgb.Booster({'nthreadv':16})
bst_adv_id.load_model('adv_id_fea.model')
print 'load advid model finished'
test_advid_newfea = bst_adv_id.predict(d_advid_test,pred_leaf=True)
np.savetxt('adv_id_fea_test.txt',test_advid_newfea.astype(int),delimiter=',')

d_appid_test = xgb.DMatrix(test_X_appID)
print 'loading appid model'
bst_app_id = xgb.Booster({'nthreapp':16})
bst_app_id.load_model('app_id_fea.model')
print 'load appid model finished'
test_appid_newfea = bst_app_id.predict(d_appid_test,pred_leaf=True)
np.savetxt('app_id_fea_test.txt',test_appid_newfea.astype(int),delimiter=',')

