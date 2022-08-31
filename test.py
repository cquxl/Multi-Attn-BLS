import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lib.metrics import mae, masked_mape_np, rmse, masked_rmspe_np
all_model_val_prediction_truth = np.load('/home/zw100/Multi-Attn BLS/experiments/seaclutter/seaclutter_lr0.0001_batchsize_100/all_model_val_prediction_truth.npy', allow_pickle=True).tolist()
all_model_test_prediction_truth = np.load('/home/zw100/Multi-Attn BLS/experiments/seaclutter/seaclutter_lr0.0001_batchsize_100/all_model_test_prediction_truth.npy',allow_pickle=True).tolist()
all_model_val_indexs ={
    'Multi-Attn BLS': {
        'mae': 0,
        'mape': 0,
        'rmse': 0,
        'rmspe': 0
    },
    'BL': {
        'mae': 0,
        'mape': 0,
        'rmse': 0,
        'rmspe': 0
    },
    'Ridge Regression': {
        'mae': 0,
        'mape': 0,
        'rmse': 0,
        'rmspe': 0
    },
    'Lstm': {
        'mae': 0,
        'mape': 0,
        'rmse': 0,
        'rmspe': 0
    }
}
all_model_test_indexs = {
    'Multi-Attn BLS': {
        'mae': 0,
        'mape': 0,
        'rmse': 0,
        'rmspe': 0
    },
    'BL': {
        'mae': 0,
        'mape': 0,
        'rmse': 0,
        'rmspe': 0
    },
    'Ridge Regression': {
        'mae': 0,
        'mape': 0,
        'rmse': 0,
        'rmspe': 0
    },
    'Lstm': {
        'mae': 0,
        'mape': 0,
        'rmse': 0,
        'rmspe': 0
    }
}
def renormalization(_min, _max, data, a=-1, b=1): #data-->numpy a=-1, b=1
    data = np.array(data)
    k = (b-a)/(_max-_min)
    return (data-a)/k+_min
def cal_indexes(test_y_true, test_prediction):
    from lib.metrics import mae, masked_mape_np, rmse, masked_rmspe_np
    test_y_true = np.array(test_y_true)
    test_prediction = np.array(test_prediction)    
    _mae = mae(test_y_true, test_prediction)
    _mape = masked_mape_np(test_y_true, test_prediction, 0)
    _rmse = rmse(test_y_true, test_prediction)
    _rmspe = masked_rmspe_np(test_y_true, test_prediction, 0)
    return _mae, _mape, _rmse, _rmspe
def plot_prediction_truth(data=all_model_val_prediction_truth , type='seaclutter',
                          root_path = '/home/zw100/Multi-Attn BLS/experiments',
                          fig_type='val', figsize=(14,4.3)):
    figure_path = root_path +'/%s/all_model_%s_prediction_truth.png' % (type,fig_type)
    indexs_path = root_path +'/%s/all_model_%s_indexes.npy' % (type,fig_type)
    # 读取各类型的数据,ffanguiyihua1
    _min = np.array(data['stats']['_min']).tolist()[0]
    _max = np.array(data['stats']['_max']).tolist()[0]
    # _min = np.array(all_model_val_prediction_truth['stats']['_min']).tolist()[0]
    # _max = np.array(all_model_val_prediction_truth['stats']['_max']).tolist()[0]
    y = data['Multi-Attn BLS']['truth']
    _y = renormalization(_min,_max,y)
    y1 = data['Multi-Attn BLS']['prediction']
    _y1 = renormalization(_min,_max,y1)
    y2 = data['BL']['prediction']
    _y2 = renormalization(_min, _max, y2)
    y3 = data['Ridge Regression']['prediction']
    _y3 = renormalization(_min, _max, y3)
    y4 = data['Lstm']['prediction']
    _y4 = renormalization(_min, _max, y4)
    _mae1, _mape1, _rmse1, _rmspe1 =  cal_indexes(y,y1)
    _mae2, _mape2, _rmse2, _rmspe2 = cal_indexes(y, y2)
    _mae3, _mape3, _rmse3, _rmspe3 = cal_indexes(y, y3)
    _mae4, _mape4, _rmse4, _rmspe4 = cal_indexes(y, y4)
    if fig_type == 'val':
        all_model_val_indexs['Multi-Attn BLS']['mae'] = _mae1
        all_model_val_indexs['Multi-Attn BLS']['mape'] = _mape1
        all_model_val_indexs['Multi-Attn BLS']['rmse'] = _rmse1
        all_model_val_indexs['Multi-Attn BLS']['rmspe'] = _rmspe1

        all_model_val_indexs['BL']['mae'] = _mae2
        all_model_val_indexs['BL']['mape'] = _mape2
        all_model_val_indexs['BL']['rmse'] = _rmse2
        all_model_val_indexs['BL']['rmspe'] = _rmspe2

        all_model_val_indexs['Ridge Regression']['mae'] = _mae3
        all_model_val_indexs['Ridge Regression']['mape'] = _mape3
        all_model_val_indexs['Ridge Regression']['rmse'] = _rmse3
        all_model_val_indexs['Ridge Regression']['rmspe'] = _rmspe3

        all_model_val_indexs['Lstm']['mae'] = _mae4
        all_model_val_indexs['Lstm']['mape'] = _mape4
        all_model_val_indexs['Lstm']['rmse'] = _rmse4
        all_model_val_indexs['Lstm']['rmspe'] = _rmspe4
        np.save(indexs_path, all_model_val_indexs)
    else:
        all_model_test_indexs['Multi-Attn BLS']['mae'] = _mae1
        all_model_test_indexs['Multi-Attn BLS']['mape'] = _mape1
        all_model_test_indexs['Multi-Attn BLS']['rmse'] = _rmse1
        all_model_test_indexs['Multi-Attn BLS']['rmspe'] = _rmspe1

        all_model_test_indexs['BL']['mae'] = _mae2
        all_model_test_indexs['BL']['mape'] = _mape2
        all_model_test_indexs['BL']['rmse'] = _rmse2
        all_model_test_indexs['BL']['rmspe'] = _rmspe2

        all_model_test_indexs['Ridge Regression']['mae'] = _mae3
        all_model_test_indexs['Ridge Regression']['mape'] = _mape3
        all_model_test_indexs['Ridge Regression']['rmse'] = _rmse3
        all_model_test_indexs['Ridge Regression']['rmspe'] = _rmspe3

        all_model_test_indexs['Lstm']['mae'] = _mae4
        all_model_test_indexs['Lstm']['mape'] = _mape4
        all_model_test_indexs['Lstm']['rmse'] = _rmse4
        all_model_test_indexs['Lstm']['rmspe'] = _rmspe4
        np.save(indexs_path, all_model_test_indexs)

    plt.figure(figsize=figsize)
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False

    plt.plot(_y, '-', markersize=4, markerfacecolor='#222831')
    plt.plot(_y1, '-', markersize=4, markerfacecolor='#6a2c70')
    plt.plot(_y2, '-', markersize=4, markerfacecolor='#b83b5e')
    plt.plot(_y3, '-', markersize=4, markerfacecolor='#f08a5d')
    plt.plot(_y4, '-', markersize=4, markerfacecolor='#f9ed69')
    plt.xlabel('time-step')
    plt.ylabel('x(t)')
    plt.legend(['true_state','Multi-Attn BLS','BL', 'Ridge Regression','Lstm'])
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    plt.savefig(figure_path, dpi=300,bbox_inches='tight')
    # 计算评价指标
def plot_indexes(root_path='/home/zw100/Multi-Attn BLS/experiments',type='seaclutter',
                 index_type='val', figsize=(14,4.3)):
    data = np.load(root_path+'/%s/all_model_%s_indexes.npy' % (type, index_type), allow_pickle=True).tolist()
    # 读取各指标并绘图
    # mae1 = data['Multi-Attn BLS']['mae']
    # mae2 = data['BL']['mae']
    # mae3 = data['Ridge Regression']['mae']
    # mae4 = data['Lstm']['mae']
    # x_dict = {}
    # df = pd.DataFrame(data)
    plt.figure(figsize=figsize)
    width = 0.75
    bin_width=width/4
    ind = np.arange(0,len(data))
    # sns.set_theme(style="whitegrid")
    index = 0
    # 6a2c70
    # b83b5e
    # f08a5d
    # f9ed69
    colors = ['#6a2c70', '#b83b5e', '#f08a5d', '#f9ed69']
    for key, value in data.items():
        indexes_values = [round(value,5) for value in list(value.values())] # 获取各指标值
        indexes_keys = list(value.keys())
        xs = ind-(bin_width)*(1.5-index)
        plt.bar(xs, indexes_values, width=bin_width, label=key, color=colors[index])
        # for indexes, x in zip(indexes_values, xs):
        #     plt.annotate(indexes, xy=(x, indexes), xytext=(x-0.07,indexes+0.001))
        index += 1
    plt.legend()
    # plt.ylabel('')
    plt.xticks(ind, indexes_keys)

    plt.savefig(root_path+'/%s/all_model_%s_indexes.png' % (type, index_type),dpi=300,bbox_inches='tight')





if __name__ == '__main__':
    # print(all_model_val_prediction_truth)
    # print(all_model_test_prediction_truth)![](experiments/lorenz/all_model_test_prediction_truth.png)
    # print('finished')
    plot_prediction_truth(data=all_model_test_prediction_truth, fig_type='test')
    plot_indexes(index_type='test')
    print('finished')