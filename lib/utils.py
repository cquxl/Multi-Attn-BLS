import os
import numpy as np
import torch
import torch.utils.data

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from lib.metrics import *
import sys
sys.path.append('E:\MultiAttn-BLS')

from data.data_process import *
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# epochs = 50
# 反归一化
def re_normalization(x, _min, _max):
    x = np.array(x) * (_max-_min) + _min
    return x

def max_min_normalization(x, _max, _min):
    x = 1. * (x - _min)/(_max - _min)
    x = x * 2. - 1.
    return x
def re_max_min_normalization(x, _max, _min):
    x = (x + 1.) / 2.
    x = 1. * x * (_max - _min) + _min
    return x
def transform_numpy_to_tensor(data, device=torch.device("cuda")):
    data = torch.from_numpy(data).type(torch.FloatTensor).to(device)
    return data
# 以下作为输入，dataloader,加载每个batch的x,y要把它转化为numpy形式输入到bls,用torch.numpy()进行转换，比较快，共享相同内存
def get_data_loader(root_path='E:/MultiAttn-BLS/data', data_type='lorenz',
                    shuffle=True, is_load=True, batch_size=100):
    if is_load == False:
        all_data = DataProcess(root_path=root_path, data_type=data_type).generate_dataset(save=False)
    if is_load:
        filename = root_path+'/%s/%s_multiattn_bls.npy' % (data_type, data_type)
        all_data = np.load(filename, allow_pickle=True).tolist()
    train_x = all_data['train']['x'] #(2100,5)
    train_y = all_data['train']['y']

    val_x = all_data['val']['x']    #(900,5)
    val_y = all_data['val']['y']

    test_x = all_data['test']['x']
    test_y = all_data['test']['y']

    _min = all_data['stats']['min'] #(5,)
    _max  = all_data['stats']['max']

    # ------train_loader------
    train_x_tensor = transform_numpy_to_tensor(train_x)
    train_y_tensor = transform_numpy_to_tensor(train_y)
    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_y_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # ------val_loader------
    val_x_tensor = transform_numpy_to_tensor(val_x)
    val_y_tensor = transform_numpy_to_tensor(val_y)
    val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_y_tensor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ------test_loader------
    test_x_tensor = transform_numpy_to_tensor(test_x)
    test_y_tensor = transform_numpy_to_tensor(test_y)
    test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_y_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ------print size------
    print('train size', train_x_tensor.size(), train_y_tensor.size())
    print('val size', val_x_tensor.size(), val_y_tensor.size())
    print('test size', test_x_tensor.size(), test_y_tensor.size())

    all_data_loader = {
        'train': {
            'x_tensor': train_x_tensor,
            'y_tensor': train_y_tensor,
            'data_loader': train_loader
        },
        'val': {
            'x_tensor': val_x_tensor,
            'y_tensor': val_y_tensor,
            'data_loader': val_loader
        },
        'test': {
            'x_tensor': test_x_tensor,
            'y_tensor': test_y_tensor,
            'data_loader': test_loader
        },
        'stats': {
            'min': _min,
            'std': _max,
            'batch_size': batch_size
        }
    }
    np.save(root_path+'/%s' % (data_type)+ '/batch_size'+str(batch_size) + '_loader_multiattn_bls.npy', all_data_loader)
    print('data loader已保存' + root_path+'/%s' % (data_type)+ '/batch_size'+str(batch_size) + '_loader_multiattn_bls.npy')
    # return all_data_loader
    return train_loader, train_y_tensor, val_loader, val_y_tensor, test_loader, test_y_tensor, _min, _max

def compute_val_loss_multi_attn_bls(net, val_loader, criterion, sw, epoch, epochs,limit=None):
    '''

    Args:
        net: model
        val_loader: 验证集torch.utils.data.utils.DataLoader
        criterion: torch.nn.MSELoss
        sw: tensorboardX.SummaryWriter
        epoch: 训练迭代第几轮
        limit: int,对验证集损失的打印是否有限制

    Returns:val_loss

    '''
    net.train(False) #确保没在训练时，dropout为0
    with torch.no_grad(): #不计入梯度计算，减少内存损耗
        val_loader_length= len(val_loader) # 多少个batch,比如val总长度为900，batch_size =20 那么有，45个batch
        tmp = [] # 记录验证集每个batch的平均损失
        for batch_index, batch_data in enumerate(val_loader): #注意bls的输入必须是numpy形式
            x, y = batch_data #输入x必须是numpy,都具有device,需cpu().numpy()
            outputs = net(x.cpu().numpy())
            loss = criterion(outputs, y) #计算每个batch的平均损失
            tmp.append(loss)
            if batch_index % 10 == 0: #每10个batch打印一次结果
                print('validation batch %s / %s, loss: %.8f' % (batch_index+1, val_loader_length, loss.item()))
            if (limit is not None) and batch_index >= limit:
                break
        validation_loss = sum(tmp)/len(tmp) #记录单个epoch验证集的平均损失
        print('validation_loss epoch %s / %s, loss: %.8f' % (epoch, epochs, validation_loss))
        sw.add_scalar('validation_loss', validation_loss, epoch)
        return validation_loss

# 测试集评估
def evaluate_on_test_multi_attn_bls(net, test_loader, test_y_tensor, sw, epoch):
    '''

    Args:
        net: model
        test_loader: torch.utils.data.utils.DataLoader
        test_target_tensor: torch.tensor，shape(N,1)
        sw: tensorboardX.SummaryWriter
        epoch: 训练迭代第几轮
        mean: (5,)
        std: (5,)

    Returns:

    '''
    net.train(False)
    with torch.no_grad():
        test_loader_length = len(test_loader)
        test_y_true = test_y_tensor.cpu().numpy()
        prediction = [] # 存储所有batch的output
        for batch_index, batch_data in enumerate(test_loader):
            x, y = batch_data
            outputs = net(x.cpu().numpy())
            prediction.append(outputs.detach().cpu().numpy()) #切断反向传播，并转化numpy便于操作

            if batch_index % 10 == 0:
                print('predicting testing set batch %s / %s' % (batch_index + 1, test_loader_length))
        prediction = np.concatenate(prediction, 0) # (n_test,1)
        # prediction_length = prediction.shape[0]
        # 评估
        from lib.metrics import mae, masked_mape_np, rmse, masked_rmspe_np
        mae = mae(test_y_true,prediction)
        mape = masked_mape_np(test_y_true,prediction,0)
        rmse = rmse(test_y_true,prediction)
        rmspe = masked_rmspe_np(test_y_true,prediction,0)
        print('MAE: %.2f' % (mae))
        print('MAPE: %.2f' % (mape))
        print('RMSE: %.2f' % (rmse))
        print('RMSPE: %.2f' % (rmspe))
        if sw:
            sw.add_scalar('MAE', mae, epoch)
            sw.add_scalar('MAPE', mape, epoch)
            sw.add_scalar('RMSE', rmse, epoch)
            sw.add_scalar('RMSPE', rmspe, epoch)
        return mae, mape, rmse, rmspe
def predict_and_save_results_multi_attn_bls(net, data_loader, data_y_tensor, _min, _max,
                                            global_step, params_path, type='predict'):
    '''

    Args:
        net: model
        data_loader: torch.utils.data.utils.DataLoader
        data_y_tensor: tensor
        global_step:
        mean:
        std:
        params_path: the path for saving the results
        type:

    Returns:

    '''
    net.train(False)
    with torch.no_grad():
        data_y = data_y_tensor.cpu().numpy()
        loader_length = len(data_loader)
        prediction = []
        in_put = []
        for batch_index, batch_data in enumerate(data_loader):
            x, y = batch_data
            in_put.append(x.cpu().numpy())
            outputs = net(x.cpu().numpy())
            prediction.append(outputs.detach().cpu().numpy())
        if batch_index % 10 == 0:
            print('predicting data set batch %s / %s' % (batch_index + 1, loader_length))
        in_put =  np.concatenate(in_put, 0)

        # in_put = re_normalization(in_put, _min, _max)
        prediction = np.concatenate(prediction, 0)
        print('input:', in_put.shape)
        print('prediction:', prediction.shape)
        print('data_y:', data_y.shape)
        output_filename = os.path.join(params_path, 'output_epoch_%s_%s' % (global_step, type))
        np.savez(output_filename, input=in_put, prediction=prediction, data_y=data_y)
        prediction_data = {
            'input': in_put,
            'prediction': prediction,
            'true': data_y
        }
        # np.save(root_path+'/%s' % (data_type)+ '/%s_prediction_multiattn_bls.npy' % (type), prediction_data)


        #计算误差
        excel_list = []
        from lib.metrics import mae, masked_mape_np, rmse, masked_rmspe_np
        mae = mae(data_y, prediction)
        mape = masked_mape_np(data_y, prediction, 0)
        rmse = rmse(data_y, prediction)
        rmspe = masked_rmspe_np(data_y, prediction, 0)
        print('MAE: %.2f' % (mae))
        print('MAPE: %.2f' % (mape))
        print('RMSE: %.2f' % (rmse))
        print('RMSPE: %.2f' % (rmspe))
        excel_list.append([mae,mape, rmse,rmspe])
        excel_data = {
            'evaluation record': excel_list
        }
        # np.save(root_path + '/%s' % (data_type) + '/%s_evaluation_multiattn_bls.npy' % (type), excel_data)
        # print(excel_list)









if __name__ == '__main__':
    get_data_loader(is_load=False, batch_size=16)
