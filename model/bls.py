from model.MultiAttn_BLS import DataProcess
from sklearn.linear_model import Ridge
from lib.metrics import *
from  Attn_BL import get_data
type = 'seaclutter'
x, x_broad, y = DataProcess(type='seaclutter').load_data()
train_x, train_y, val_x, val_y, test_x, test_y, idx1, idx2,val_x_0, val_y_0 = get_data(x_broad,y) #
train_x = train_x.cpu().numpy()
train_y = train_y.cpu().numpy()
# val_x = val_x.cpu().numpy()
# val_y = val_y.cpu().numpy()
val_x = val_x_0.cpu().numpy()
val_y = val_y_0.cpu().numpy()
test_x = test_x.cpu().numpy()
test_y = test_y.cpu().numpy()

# all_data = DataProcess(type = 'rossler').generate_dataset(save=False)
# train_x = all_data['train']['x']  # (2100,5)
# train_y = all_data['train']['y']
#
# val_x = all_data['val']['x']  # (900,5)
# val_y = all_data['val']['y']
#
# test_x = all_data['test']['x']
# test_y = all_data['test']['y']
if __name__ == '__main__':
    # bls = BlsLayer()
    ridge = Ridge()
    test_y_true = test_y
    # train_x = bls(train_x).cpu().numpy()
    # test_x = bls(test_x).cpu().numpy()
    model = ridge.fit(train_x, train_y)
    val_prediction = model.predict(val_x)
    test_prediction = model.predict(test_x)
    from lib.metrics import mae, masked_mape_np, rmse, masked_rmspe_np
    mae = mae(test_y, test_prediction)
    mape = masked_mape_np(test_y_true, test_prediction, 0)
    rmse = rmse(test_y_true, test_prediction)
    rmspe = masked_rmspe_np(test_y_true, test_prediction, 0)
    print('MAE: %.3f' % (mae))
    print('MAPE: %.3f' % (mape))
    print('RMSE: %.3f' % (rmse))
    print('RMSPE: %.3f' % (rmspe))
    # # 加载数据
    all_model_val_prediction_truth = np.load('/home/zw100/Multi-Attn BLS/experiments/seaclutter/seaclutter_lr0.0001_batchsize_100/all_model_val_prediction_truth.npy', allow_pickle=True).tolist()
    all_model_test_prediction_truth = np.load('/home/zw100/Multi-Attn BLS/experiments/seaclutter/seaclutter_lr0.0001_batchsize_100/all_model_test_prediction_truth.npy',allow_pickle=True).tolist()
    all_model_val_prediction_truth['BL']['prediction'] = [value[0] for value in val_prediction.tolist()]
    all_model_val_prediction_truth['BL']['truth'] = [value[0] for value in val_y.tolist()]
    all_model_test_prediction_truth['BL']['prediction'] = [value[0] for value in test_prediction.tolist()]
    all_model_test_prediction_truth['BL']['truth'] = [value[0] for value in test_y.tolist()]
    np.save('/home/zw100/Multi-Attn BLS/experiments/seaclutter/seaclutter_lr0.0001_batchsize_100/all_model_val_prediction_truth.npy', all_model_val_prediction_truth)
    np.save('/home/zw100/Multi-Attn BLS/experiments/seaclutter/seaclutter_lr0.0001_batchsize_100/all_model_test_prediction_truth.npy', all_model_test_prediction_truth)
    print('finished')