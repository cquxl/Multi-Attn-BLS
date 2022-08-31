import numpy as np


# MAPE,返回的是MAPE的分子数值，如5%返回的是5
def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val): #TRUE,y_true为nan-->,mask=0,y_true非nan-->mask=1
            mask = ~np.isnan(y_true)
        else:#null_val非null:y_true为nan-->mask =1,
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        ape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
                                y_true))
        ape = np.nan_to_num(mask * ape) #nan返回0，无穷大返回较大的数值
        return np.mean(ape)

# MAE
def mae(y_true,y_pred):
    mae = np.abs(np.subtract(y_pred, y_true).astype('float32'))
    return np.mean(mae)

# RMSE
def rmse(y_true, y_pred):
    mse = np.power(y_true-y_pred, 2).astype('float32')
    return np.sqrt(np.mean(mse))

# RMSPE
def masked_rmspe_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val): #TRUE,y_true为nan-->,mask=0,y_true非nan-->mask=1
            mask = ~np.isnan(y_true)
        else:#null_val非null:y_true为nan-->mask =1,
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
                                y_true))
        mspe = np.power(mape, 2)
        mspe = np.nan_to_num(mask * mape) #nan返回0，无穷大返回较大的数值
        return np.sqrt(np.mean(mspe))

