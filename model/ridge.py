
from model.MultiAttn_BLS import DataProcess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from mpl_toolkits.mplot3d import Axes3D
# from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error,r2_score
import sklearn.metrics
from IPython import display
from lib.metrics import *
from  Attn_BL import get_data
x, x_broad, y = DataProcess(type='seaclutter').load_data()
train_x, train_y, val_x, val_y, test_x, test_y, idx1, idx2 ,val_x_0, val_y_0= get_data(x,y) #
train_x = train_x.cpu().numpy()
train_y = train_y.cpu().numpy()
# val_x = val_x.cpu().numpy()
# val_y = val_y.cpu().numpy()
val_x = val_x_0.cpu().numpy()
val_y = val_y_0.cpu().numpy()
test_x = test_x.cpu().numpy()
test_y = test_y.cpu().numpy()

# all_data = DataProcess(type='rossler').generate_dataset(save=False)
# train_x = all_data['train']['x']  # (2100,5)
# train_y = all_data['train']['y']
#
# val_x = all_data['val']['x']  # (900,5)
# val_y = all_data['val']['y']
#
# test_x = all_data['test']['x']
# test_y = all_data['test']['y']


if __name__ == '__main__':
    ridge = Ridge()
    test_y_true = test_y # numpy
    model = ridge.fit(train_x, train_y)
    val_prediction = model.predict(val_x)
    test_prediction = model.predict(test_x)
    from lib.metrics import mae, masked_mape_np, rmse, masked_rmspe_np
    mae = mae(test_y_true, test_prediction)
    mape = masked_mape_np(test_y_true, test_prediction, 0)
    rmse = rmse(test_y_true, test_prediction)
    rmspe = masked_rmspe_np(test_y_true, test_prediction, 0)
    print('MAE: %.3f' % (mae))
    print('MAPE: %.3f' % (mape))
    print('RMSE: %.3f' % (rmse))
    print('RMSPE: %.3f' % (rmspe))
    all_model_val_prediction_truth = np.load(
        '/home/zw100/Multi-Attn BLS/experiments/seaclutter/seaclutter_lr0.0001_batchsize_100/all_model_val_prediction_truth.npy',
        allow_pickle=True).tolist()
    all_model_test_prediction_truth = np.load(
        '/home/zw100/Multi-Attn BLS/experiments/seaclutter/seaclutter_lr0.0001_batchsize_100/all_model_test_prediction_truth.npy',
        allow_pickle=True).tolist()
    all_model_val_prediction_truth['Ridge Regression']['prediction'] = [value[0] for value in val_prediction.tolist()]
    all_model_val_prediction_truth['Ridge Regression']['truth'] = [value[0] for value in val_y.tolist()]
    all_model_test_prediction_truth['Ridge Regression']['prediction'] = [value[0] for value in test_prediction.tolist()]
    all_model_test_prediction_truth['Ridge Regression']['truth'] = [value[0] for value in test_y.tolist()]
    np.save('/home/zw100/Multi-Attn BLS/experiments/seaclutter/seaclutter_lr0.0001_batchsize_100/all_model_val_prediction_truth.npy', all_model_val_prediction_truth)
    np.save('/home/zw100/Multi-Attn BLS/experiments/seaclutter/seaclutter_lr0.0001_batchsize_100/all_model_test_prediction_truth.npy', all_model_test_prediction_truth)
    print('finished')

