import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import math
import json
import matplotlib.pyplot as plt
# import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# from tensorboardX import SummaryWriter
import os
from pprint import pprint
# from IPython import display
# display.set_matplotlib_formats('svg')
# 相关设置
torch.manual_seed(42)
calculate_loss_over_all_values = False
output_window = 1
batch_size = 50
start_epoch = 0
lr = 0.0001
step_size = 3
gamma =0.96
criterion = nn.MSELoss()
epochs = 1000
best_model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
all_model_val_prediction_truth = {
    'Multi-Attn BLS': {
        'prediction':[],
        'truth': []
    },
    'BL': {
        'prediction':[],
        'truth': []
    },
    'Ridge Regression': {
        'prediction': [],
        'truth' : []
    },
    'Lstm': {
        'prediction': [],
        'truth': []
    }
}
all_model_test_prediction_truth = all_model_val_prediction_truth
def transform_numpy_to_tensor(data, device=device):
    data = torch.from_numpy(data).type(torch.FloatTensor).to(device)
    return data
#------------------数据处理-------------------
class DataProcess:
    def __init__(self, root='/home/zw100/Multi-Attn BLS/data', type='lorenz'):
        self.root = root
        self.type = type
    def load_data(self):
        filename = self.root+'/'+ self.type
        # 宽度特征
        x_broad = pd.read_csv(filename+'/%s宽度特征4000.csv' % (self.type), header=None)
        x = pd.read_csv(filename+'/x-%s.csv' % (self.type), header=None)
        y = pd.read_csv(filename + '/y-%s.csv' % (self.type), header=None)
        print('重构序列X维度', x.shape)
        print('加宽后的序列X维度', x_broad.shape)
        print('y维度', y.shape)
        return x, x_broad, y

    def generate_dataset(self, save=True, part_length=3000, train_ratio=0.7):
        _, data_x, data_y = self.load_data()
        data_x_part1 = data_x.iloc[:part_length, :]
        scaler = MinMaxScaler(feature_range=(-1, 1))
        data_x_part1 = scaler.fit_transform(data_x_part1)

        data_x_part2 = data_x.iloc[part_length:, :]
        data_x_part2 = scaler.fit_transform(data_x_part2)

        data_y_part1 = data_y.iloc[:part_length, :]
        _min = data_y_part1.min()
        _max = data_y_part1.max()
        data_y_part1 = scaler.fit_transform(data_y_part1)

        data_y_part2 = data_y.iloc[part_length:, :]
        data_y_part2 = scaler.fit_transform(data_y_part2)
        indices = np.arange(len(data_x_part1))

        # 拆分训练集,验证集,测试集
        test_time_order = np.arange(part_length, len(data_x))
        # 不用打乱，因为在加载data_loader时会打乱，训练集：part1前70%,验证集part1后30%
        train_x, train_y = np.array(data_x_part1)[:int(train_ratio * part_length)], np.array(data_y_part1)[:int(train_ratio * part_length)]
        idx1 = np.arange(0, int(train_ratio * part_length))
        val_x, val_y = np.array(data_x_part1)[int(train_ratio * part_length):], np.array(data_y_part1)[int(train_ratio * part_length):]
        idx2 = np.arange(int(train_ratio * part_length), part_length)

        train_x = np.array(train_x)
        train_y = np.array(train_y)
        val_x = np.array(val_x)
        val_y = np.array(val_y)
        test_x = np.array(data_x_part2)
        test_y = np.array(data_y_part2)

        all_data = {
            'train': {
                'x': train_x,
                'y': train_y,
                'time_order': idx1
            },
            'val': {
                'x': val_x,
                'y': val_y,
                'time_order': idx2
            },
            'test': {
                'x': test_x,
                'y': test_y,
                'time_order': test_time_order
            },
            'stats': {
                'min': _min,
                'max': _max
            }
        }
        print('train x:', all_data['train']['x'].shape)
        print('val x:', all_data['val']['x'].shape)
        print('test x:', all_data['test']['x'].shape)
        if save:
            filename = self.root + '/%s/%s_train_valid_test.npy' % (self.type, self.type)
            # with open(filename,'w') as json_file:
            #     json.dump(all_data, json_file, ensure_ascii=False)
            #     json_file.close()
            np.save(filename, all_data)
            print('划分数据集已保存：', filename)
            # return train_x
            # print(train_x_norm)
        return all_data
    def get_data_loader(self, save=True):
        all_data = self.generate_dataset()
        train_x = all_data['train']['x']  # (2100,5)
        train_y = all_data['train']['y']

        val_x = all_data['val']['x']  # (900,5)
        val_y = all_data['val']['y']

        test_x = all_data['test']['x']
        test_y = all_data['test']['y']

        _min = all_data['stats']['min']  # (5,)
        _max = all_data['stats']['max']

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
        if save:
            filename = self.root + '/%s/%s_train_valid_test_loader_%s.npy' % (self.type, self.type, batch_size)
            np.save(filename, all_data_loader)
        return all_data_loader

#------------------建立模型-------------------

# 位置编码
#输入batchsize,嵌入维数, 经过bls，维度为71

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, max_len=71):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model) # 初始化位置编码全部为0
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) #（100，1）
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # 维度为(1,100,512)
        self.register_buffer('pe', pe) # 定义参数在训练时不被更新
        # self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        return x.unsqueeze(-1) + self.pe[:x.size(0), :]# 维度为_,71,512

class MultiAttn_BLS(nn.Module):
    '''
    Transformer一层encoder，8个头
    input:batch_size,100-->位置编码-->batch_size,100,512-->encoder-->维度不变-->decoder-->

    '''
    def __init__(self, d_model=512, max_len=71, num_layers=1, n_head=8, dropout=0):
        super(MultiAttn_BLS, self).__init__()
        self.src_mask = None # 不需要掩码
        self.pos_encoder = PositionalEncoding(d_model,max_len)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                        nhead=n_head,
                                                        dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        # 展品再进行线性连接
        self.decoder = nn.Linear(d_model*max_len, 1) #用全连接作为最后的输出
        self._init_weights() # 对decoder的参数进行初始化
        self.src_key_padding_mask = None

    def _init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_() # 偏置初始化为0
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        x_pe = self.pos_encoder(x) #维度：batch_size,100,512
        x_en = self.transformer_encoder(x_pe, self.src_mask, self.src_key_padding_mask)
        x_de_in = x_en.flatten(start_dim=1) # batch_size,512*71
        output = self.decoder(x_de_in) # 返回，batch_size,1
        return output

#------------------验证集损失，用来找到最好的模型参数-------------------
def compute_val_loss_and_prediciton(net, val_loader): #单个epoch
    net.train(False)  # 确保没在训练时，dropout为0
    with torch.no_grad():  # 不计入梯度计算，减少内存损耗
        val_loader_length = len(val_loader)  # 多少个batch,比如val总长度为900，batch_size =20 那么有，45个batch
        tmp = []  # 记录验证集每个batch的平均损失
        prediction = torch.Tensor()
        for batch_index, batch_data in enumerate(val_loader):
            x, y = batch_data
            outputs = net(x)
            loss = criterion(outputs, y) #单个batch的损失
            tmp.append(loss)
            prediction = torch.cat((prediction, outputs.cpu())) #默认是行叠加
        validation_loss = sum(tmp) / len(tmp)  # 记录单个epoch验证集的平均损失
    return prediction, validation_loss
#------------------验证集预测与真实值预测-------------------
def plot_prediction_truth(prediction,truth, best_epoch,
                          figure_path, figsize=(7,4.3)):
    plt.figure(figsize=figsize)
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(truth, 's-', markersize=4, markerfacecolor='#e29c45')
    plt.plot(prediction, '*-',markersize=4, markerfacecolor='#44cef6')
    plt.plot(np.array(prediction)-np.array(truth), color="green")
    plt.axhline(y=0, color='k')
    plt.xlabel('time-step')
    plt.ylabel('x(t)')
    plt.legend(['truth','prediction','error'])
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    plt.savefig(figure_path+'/%s_val_graph' % best_epoch, dpi=300,bbox_inches='tight')



#------------------训练模型-------------------
def train_main(net, type='lorenz'):
    # global figure_path
    fold_dir = '%s_lr%s_batchsize_%s' % (type, lr, batch_size)
    params_path = os.path.join('/home/zw100/Multi-Attn BLS/experiments', type, fold_dir)
    # 产生保存参数的文件夹
    if (start_epoch == 0) and (not os.path.exists(params_path)):
        os.makedirs(params_path)
        figure_path = os.path.join(params_path,'figure')
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)
        print('create params directory %s' % (params_path))
    # elif (start_epoch > 0) and (os.path.exists(params_path)):
    #     print('train from params directory %s' % (params_path))
    # else:
    #     raise SystemExit('Wrong type of model!')
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)#优化器对象，多少轮更新学习率，lr的系数
    # sw = SummaryWriter(logdir=params_path, flush_secs=5)

    _,best_val_loss = compute_val_loss_and_prediciton(net, val_loader) #初始化
    best_epoch = 0
    # best_epoch_list =[0]
    torch.save(net.state_dict(), os.path.join(params_path, 'epoch_%s.params' % start_epoch))
    for epoch in range(start_epoch, epochs):
        net.train()
        train_loss_epoch_total = 0
        start_time = time.time()
        for batch_index, batch_data in enumerate(train_loader):
            x, y = batch_data
            optimizer.zero_grad()  # 梯度清零
            outputs = net(x)
            loss = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()
            train_loss_epoch_total += loss.item() #计算单个epoch训练损失和
        # 单个epoch训练集的平均损失
        train_loss = train_loss_epoch_total/len(train_loader)
        # 单个epoch验证集的平均损失
        val_prediction, val_loss = compute_val_loss_and_prediciton(net, val_loader)
        if val_loss < best_val_loss:
            params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)
            best_val_loss = val_loss.item()
            best_epoch = epoch + 1 #是梯度传播结束后的net所对应
            best_params_filename = os.path.join(params_path, 'best.params')
            # 保存参数
            torch.save(net.state_dict(), params_filename)
            torch.save(net.state_dict(), best_params_filename)
            # 绘制相应的验证集曲线图
            # val_prediction, val_y_tensor都是张量，且都带有device
            if best_epoch % 10 > 0: # 绘制的图像保存在figure_path中
                val_prediction = [value[0] for value in val_prediction.cpu().numpy().tolist()]
                val_y = [value[0] for value in val_y_tensor.cpu().numpy().tolist()]
                figure_path = os.path.join(params_path,'figure')
                plot_prediction_truth(val_prediction, val_y, best_epoch,
                                       figure_path, figsize=(7, 4.3))
                print('epoch %d,val_loss:%.5f, lr:%.4f, time:%.2f' % (best_epoch, best_val_loss, optimizer.state_dict()['param_groups'][0]['lr'], time.time()-start_time))
                # 保存验证集的预测值
            # 记录最后一次的预测值并保存
            all_model_val_prediction_truth['Multi-Attn BLS']['prediction'] = val_prediction
            all_model_val_prediction_truth['Multi-Attn BLS']['truth'] = val_y
            np.save('/home/zw100/Multi-Attn BLS/all_model_val_prediction_truth.npy', all_model_val_prediction_truth)























if __name__ == "__main__":
    # DataProcess().load_data()
    # all_data = DataProcess().generate_dataset()
    # data = all_data['train']['x']
    all_data_loader = DataProcess().get_data_loader()
    train_loader, val_loader, test_loader = all_data_loader['train']['data_loader'], all_data_loader['val']['data_loader'], all_data_loader['test']['data_loader']
    train_y_tensor, val_y_tensor, test_y_tensor = all_data_loader['train']['y_tensor'], all_data_loader['val']['y_tensor'] , all_data_loader['test']['y_tensor']
    net = MultiAttn_BLS().to(device)
    train_main(net, type='lorenz')