
import torch.nn as nn
from model.MultiAttn_BLS import DataProcess
from sklearn.linear_model import Ridge
from lib.metrics import *
import torch
def transform_numpy_to_tensor(data, device=torch.device("cuda")):
    data = torch.from_numpy(data).type(torch.FloatTensor).to(device)
    return data
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# x, _, y = DataProcess().load_data()
type = 'seaclutter'
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
# all_data = DataProcess(type = 'rossler').generate_dataset(save=False)
# train_x = all_data['train']['x']  # (2100,5)
# train_y = all_data['train']['y']
#
# val_x = all_data['val']['x']  # (900,5)
# val_y = all_data['val']['y']
#
# test_x = all_data['test']['x']
# test_y = all_data['test']['y']


# ------train_loader------
train_x_tensor = transform_numpy_to_tensor(train_x).reshape(-1,1,5)
train_y_tensor = transform_numpy_to_tensor(train_y).reshape(-1,1,1)

# ------val_loader------
val_x_tensor = transform_numpy_to_tensor(val_x).reshape(-1,1,5)
val_y_tensor = transform_numpy_to_tensor(val_y).reshape(-1,1,1)

# ------test_loader------
test_x_tensor = transform_numpy_to_tensor(test_x).reshape(-1,1,5)
test_y_tensor = transform_numpy_to_tensor(test_y).reshape(-1,1,1)



class LSTM_Regression(nn.Module):
    """
        使用LSTM进行回归

        参数：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """

    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s * b, h)
        x = self.fc(x)
        x = x.view(s, b, -1)  # 把形状改回来
        return x
if __name__ == "__main__":
    model = LSTM_Regression(5, 30, output_size=1, num_layers=2).to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    output = []
    loss_epoch = []
    for i in range(1000):
        out = model(train_x_tensor)
        loss = loss_function(out, train_y_tensor)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if (i + 1) % 100 == 0:
            print('Epoch: {}, Loss:{:.5f}'.format(i + 1, loss.item()))
        output.append(out)
        loss_epoch.append(loss)
    # 测试集
    model = model.eval()  # 转换成测试模式
    # pred_test = model(valid_seq)  # 全量训练集的模型输出 (seq_size, batch_size, output_size)
    # pred_test = pred_test.cpu().view(-1).detach().numpy()
    test_y_true = test_y_tensor.cpu().view(-1).numpy()

    test_prediction = model(test_x_tensor)
    val_prediction = model(val_x_tensor)
    test_prediction = test_prediction.cpu().view(-1).detach().numpy()
    val_prediction = val_prediction.cpu().view(-1).detach().numpy()
    from lib.metrics import mae, masked_mape_np, rmse, masked_rmspe_np

    mae = mae(test_y_true, test_prediction)
    mape = masked_mape_np(test_y_true, test_prediction, 0)
    rmse = rmse(test_y_true, test_prediction)
    rmspe = masked_rmspe_np(test_y_true, test_prediction, 0)
    print('MAE: %.3f' % (mae))
    print('MAPE: %.3f' % (mape))
    print('RMSE: %.3f' % (rmse))
    print('RMSPE: %.3f' % (rmspe))
    all_model_val_prediction_truth = np.load('/home/zw100/Multi-Attn BLS/experiments/seaclutter/seaclutter_lr0.0001_batchsize_100/all_model_val_prediction_truth.npy', allow_pickle=True).tolist()
    all_model_test_prediction_truth = np.load('/home/zw100/Multi-Attn BLS/experiments/lorenz/lorenz_lr0.0001_batchsize_100/all_model_test_prediction_truth.npy',allow_pickle=True).tolist()
    all_model_val_prediction_truth['Lstm']['prediction'] = val_prediction.tolist()
    all_model_val_prediction_truth['Lstm']['truth'] = [value[0] for value in val_y.tolist()]
    all_model_test_prediction_truth['Lstm']['prediction'] = test_prediction.tolist()
    all_model_test_prediction_truth['Lstm']['truth'] = [value[0] for value in test_y.tolist()]
    np.save('/home/zw100/Multi-Attn BLS/experiments/seaclutter/seaclutter_lr0.0001_batchsize_100/all_model_val_prediction_truth.npy', all_model_val_prediction_truth)
    np.save('/home/zw100/Multi-Attn BLS/experiments/seaclutter/seaclutter_lr0.0001_batchsize_100/all_model_test_prediction_truth.npy', all_model_test_prediction_truth)



