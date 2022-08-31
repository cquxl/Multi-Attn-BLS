#导入基本块
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import time
import math
# from matplotlib.pyplot import pyplot
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from IPython import display
display.set_matplotlib_formats('svg')

#环境设置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.manual_seed(42)
np.random.seed(42)
# This concept is also called teacher forceing.
# The flag decides if the loss will be calculted over all
# or just the predicted values.
calculate_loss_over_all_values = False
#相空间重构
#延迟时间14，嵌入维数为5
#lorenz:嵌入维数m=5,延迟时间tau=16
#rosser:嵌入维数m=5,延迟时间tau=38
#sea:嵌入维数m=5,延迟时间tau=16
input_window = 100
output_window = 1
batch_size = 100  # batch size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#使用gpu
train_ratio  = 0.7
part_length = 3000
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
    },
    'stats': {
        '_min': 0,
        "_max": 0,
    }
}
all_model_test_prediction_truth = {
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
    },
    'stats': {
        '_min': 0,
        "_max": 0,
    }
}
#先加载3种数据
#加了宽度特征的数据
lorenz_filename='/home/zw100/Multi-Attn BLS/data/lorenz'
rossler_filename='/home/zw100/Multi-Attn BLS/data/rossler'
sea_filename='/home/zw100/Multi-Attn BLS/data/seaclutter'
#lorenz对应的宽度特征。重构特征，y值
lorenz_broad=pd.read_csv(lorenz_filename+'/lorenz宽度特征4000.csv',header=None)
lorenz_x=pd.read_csv(lorenz_filename+'/x-lorenz.csv',header=None)
lorenz_y=pd.read_csv(lorenz_filename+'/y-lorenz.csv',header=None)

# # #rosser对应的宽度特征。重构特征，y值
rossler_broad=pd.read_csv(rossler_filename+'/rossler宽度特征4000.csv',header=None)
rossler_x=pd.read_csv(rossler_filename+'/x-rossler.csv',header=None)
rossler_y=pd.read_csv(rossler_filename+'/y-rossler.csv',header=None)

# #sea对应的宽度特征。重构特征，y值
sea_broad=pd.read_csv(sea_filename+'/seaclutter宽度特征4000.csv',header=None)
sea_x=pd.read_csv(sea_filename+'/x-seaclutter.csv',header=None)
sea_y=pd.read_csv(sea_filename+'/y-seaclutter.csv',header=None)

def renormalization(_min, _max, data, a=-1, b=1): #data-->numpy a=-1, b=1
    k = (b-a)/(_max-_min)
    return (data-a)/k+_min

def get_data(X_broad, y_broad):
    # 取前3000的数据进行训练和验证，后面的数据进行测试
    # 训练和验证
    data = X_broad.iloc[:3000, :]
    test_data = X_broad.iloc[3000:, :]
    #     # 标准化为-1到1
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data = scaler.fit_transform(data)
    test_data = scaler.fit_transform(test_data)
    # 训练和验证标签
    label = y_broad.iloc[:3000, :]
    val_min = label.min()
    val_max = label.max()
    all_model_val_prediction_truth['stats']['_min'] = val_min
    all_model_val_prediction_truth['stats']['_max'] = val_max
    label = scaler.fit_transform(label)

    test_label = y_broad.iloc[3000:, :]
    test_min = test_label.min()
    test_max = test_label.max()
    all_model_test_prediction_truth['stats']['_min'] = test_min
    all_model_test_prediction_truth['stats']['_max'] = test_max
    test_label = scaler.fit_transform(test_label)
    indices = np.arange(len(data))
    # 拆分训练集和验证集
    train_X, valid_X, train_y, valid_y, idx1, idx2 = train_test_split(data, label, indices, train_size=0.7,
                                                                      random_state=42)
    # train_X, train_y = np.array(data)[:int(train_ratio * part_length)], np.array(label)[:int(train_ratio * part_length)]
    # idx1 = np.arange(0, int(train_ratio * part_length))
    valid_X_0, valid_y_0 = np.array(data)[int(train_ratio * part_length):], np.array(label)[int(train_ratio * part_length):]
    #
    # idx2 = np.arange(int(train_ratio * part_length), part_length)

    # 将数据转化为Tensor
    # 训练
    train_seq = torch.from_numpy(np.array(train_X)).type(torch.FloatTensor)
    train_label = torch.from_numpy(np.array(train_y)).type(torch.FloatTensor)
    # 验证集
    valid_seq = torch.from_numpy(np.array(valid_X)).type(torch.FloatTensor)
    valid_label = torch.from_numpy(np.array(valid_y)).type(torch.FloatTensor)
    valid_seq_0 = torch.from_numpy(np.array(valid_X_0)).type(torch.FloatTensor)
    valid_label_0 = torch.from_numpy(np.array(valid_y_0)).type(torch.FloatTensor)
    # 测试集
    test_seq = torch.from_numpy(np.array(test_data)).type(torch.FloatTensor)
    test_label = torch.from_numpy(np.array(test_label)).type(torch.FloatTensor)

    return train_seq.to(device), train_label.to(device), valid_seq.to(device), valid_label.to(device), test_seq.to(
        device), test_label.to(device), idx1, idx2, valid_seq_0.to(device), valid_label_0.to(device)


def get_batch(seq, label, i, batch_size):
    seq_len = min(batch_size, len(seq) - i)
    data_x = seq[i:i + seq_len]
    data_y = label[i:i + seq_len]

    input = torch.stack(torch.stack([item for item in data_x]).chunk(input_window, 1))
    target = torch.stack(torch.stack([item for item in data_y]).chunk(input_window, 1))
    return input, target

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransAm(nn.Module):
    def __init__(self, feature_size=512, num_layers=1, dropout=0, ffn_size=71):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=8, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, 1)
        self.ffn = nn.Linear(ffn_size , 1)
        # self.ffn=nn.Linear(71,1)
        self.init_weights()
        self.src_key_padding_mask = None

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        #         # if self.src_mask is None or self.src_mask.size(0) != len(src):
        #         #     device = src.device
        #         #     mask = self._generate_square_subsequent_mask(len(src)).to(device)
        #         #     self.src_mask = mask
        #         if self.src_key_padding_mask is None:
        #             mask_key = src_padding.bool()
        #             self.src_key_padding_mask = mask_key

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)  # , self.src_mask)
        output = self.decoder(output)

        output = output.transpose(0, 2)  # 1x50x71
        output = self.ffn(output)

        return output

#训练和记录数据
def train(train_seq,train_label):
    model.train()  # Turn on the train mode
    total_loss = 0.
    start_time = time.time()

    for batch, i in enumerate(range(0, len(train_seq) , batch_size)):
        data, targets = get_batch(train_seq,train_label,i, batch_size)
        optimizer.zero_grad()
        output = model(data)

        if calculate_loss_over_all_values:
            loss = criterion(output, targets)
        else:
            loss = criterion(output[-output_window:], targets[-output_window:])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = int(len(train_seq) / batch_size/5 )
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            # print('| epoch {:3d} | {:5d}/{:5d} batches | '
            #       'lr {:02.6f} | {:5.2f} ms | '
            #       'loss {:5.5f}'.format(
            #     epoch, batch, len(train_seq) // batch_size, scheduler.get_lr()[0],
            #                   elapsed * 1000 / log_interval,
            #     cur_loss))  # , math.exp(cur_loss)
            total_loss = 0
            start_time = time.time()

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

def evaluate(eval_model, data_source_x,data_source_y):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    eval_batch_size = 50
    result = torch.Tensor(0)
    truth = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(data_source_x) - 1, eval_batch_size):
            data, targets= get_batch(data_source_x,data_source_y,i, eval_batch_size)
            output = eval_model(data)
            if calculate_loss_over_all_values:
                total_loss +=  len(data)*criterion(output, targets).cpu().item()
            else:
                total_loss +=  len(data)*criterion(output[-output_window:], targets[-output_window:]).cpu().item()
            result = torch.cat((result, output[-1].squeeze(1).view(-1).cpu()),
                                    0)  # todo: check this. -> looks good to me
            truth = torch.cat((truth, targets[-1].squeeze(1).view(-1).cpu()), 0)
    return total_loss / len(data_source_x),result,truth

if __name__ == '__main__':
    type = 'seaclutter'
    train_seq, train_label, valid_seq, valid_label, test_seq, test_label, idx1, idx2,valid_seq_0, valid_label_0= get_data(sea_broad, sea_y)
    #128 best
    model = TransAm(feature_size=128, ffn_size=train_seq.size(1)).to(device)

    criterion = nn.MSELoss()
    lr = 0.0001
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.96)

    best_val_loss = float("inf")
    epochs = 1000  # The number of epochs
    best_model = None
    valid_loss_epoch = []
    train_loss_epoch = []
    test_loss_epoch = []
    train_output_epoch = []
    valid_output_epoch = []
    test_output_epoch = []
    train_target_epoch = []
    valid_target_epoch = []
    test_target_epoch = []
    # output_epoch=[]
    fold_dir = '%s_lr%s_batchsize_%s' % (type, lr, batch_size)
    params_path = os.path.join('/home/zw100/Multi-Attn BLS/experiments', type, fold_dir)
    # 产生保存参数的文件夹
    if not os.path.exists(params_path):
        os.makedirs(params_path)
        figure_path = os.path.join(params_path, 'figure')
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        epoch_start_time = time.time()
        train(train_seq, train_label)
        train_loss, train_output, train_target = evaluate(model, train_seq, train_label)
        # valid_loss, valid_output, valid_target = evaluate(model, valid_seq, valid_label)
        valid_loss, valid_output, valid_target = evaluate(model, valid_seq_0, valid_label_0)
        test_loss, test_output, test_target = evaluate(model, test_seq, test_label)
        # 保存较好的模型
        if valid_loss < best_val_loss:
            params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)
            best_val_loss = valid_loss
            best_epoch = epoch + 1  # 是梯度传播结束后的net所对应
            best_params_filename = os.path.join(params_path, 'best.params')
            # 保存参数
            torch.save(model.state_dict(), params_filename)
            torch.save(model.state_dict(), best_params_filename)
            if best_epoch //20 >0 and best_epoch % 20 > 0: # 绘制的图像保存在figure_path中
                val_prediction = valid_output.cpu().numpy()
                val_y = valid_target.cpu().numpy()
                test_prediction = test_output.cpu().numpy()
                test_y = test_target.cpu().numpy()
                figure_path = os.path.join(params_path,'figure')
                plot_prediction_truth(val_prediction, val_y, best_epoch,
                                       figure_path, figsize=(7, 4.3))
                print('epoch %d,val_loss:%.8f, lr:%.6f, time:%.2f' % (best_epoch, best_val_loss, optimizer.state_dict()['param_groups'][0]['lr'], time.time() - start_time))
                all_model_val_prediction_truth['Multi-Attn BLS']['prediction'] = val_prediction.tolist()
                all_model_val_prediction_truth['Multi-Attn BLS']['truth'] = val_y.tolist()
                all_model_test_prediction_truth['Multi-Attn BLS']['prediction']= test_prediction.tolist()
                all_model_test_prediction_truth['Multi-Attn BLS']['truth'] = test_y.tolist()
                np.save(params_path+ '/all_model_val_prediction_truth.npy', all_model_val_prediction_truth)
                np.save(params_path+'/all_model_test_prediction_truth.npy', all_model_test_prediction_truth)
        #     if (epoch % 10 == 0):
        #         val_loss1,output1, true1= plot_and_loss(model,valid_seq,valid_label, epoch)
        # #         # predict_future(model, val_data, 200)
        #     else:
        #         val_loss1 = evaluate(model, valid_seq,valid_label)

        # print('-' * 89)
        # print(
        #     '| end of epoch {:3d} | time: {:5.2f}s | valid loss {:.10f} | train loss {:.10f}| test loss {:.10f} '.format(
        #         epoch, (time.time() - epoch_start_time),
        #         valid_loss, train_loss, test_loss))  # , math.exp(val_loss) | valid ppl {:8.2f}
        # print('-' * 89)
        # # output_epoch.append(output1)
        # valid_loss_epoch.append(valid_loss)
        # train_loss_epoch.append(train_loss)
        # test_loss_epoch.append(test_loss)
        # train_output_epoch.append(train_output)
        # valid_output_epoch.append(valid_output)
        # test_output_epoch.append(test_output)
        # train_target_epoch.append(train_target)
        # valid_target_epoch.append(valid_target)
        # test_target_epoch.append(test_target)

                scheduler.step()