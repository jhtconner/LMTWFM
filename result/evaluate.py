import numpy as np
import torch

def weighted_rmse(y_pred,y_true):
    # y_pred shape: (samples, time, channels, WIDTH, HEIGHT) = (1452, 24, 1, 29, 17)
    # Height dimension (17) is LAST, so latitude is the last dimension
    lat = np.linspace(49.5, 59.5, num=17, dtype=float, endpoint=True)
    RMSE = np.empty([y_pred.size(1)])  # y_pred.size(1) = time dimension
    weights_lat = np.cos(np.deg2rad(lat))
    weights_lat /= weights_lat.mean()
    for i in range(y_pred.size(1)):
        # y_pred[:,i,0,:,:] gives (samples, width, height) = (1452, 29, 17)
        diff = y_pred[:,i,0,:,:] - y_true[:,i,0,:,:]  # Shape: (samples, 29, 17)
        # No permute needed - latitude weights apply to last dimension already
        RMSE[i] = np.sqrt((diff**2*weights_lat).mean([-2,-1])).mean(axis=0)
    return RMSE

def weighted_mae(y_pred,y_true):
    lat = np.linspace(49.5, 59.5, num=17, dtype=float, endpoint=True)
    MAE = np.empty([y_pred.size(1)])
    weights_lat = np.cos(np.deg2rad(lat))
    weights_lat /= weights_lat.mean()
    for i in range(y_pred.size(1)):
        diff_abs = abs(y_pred[:, i, 0, :, :] - y_true[:, i, 0, :, :])  # (samples, 29, 17)
        # No permute needed
        MAE[i] = (diff_abs * weights_lat).mean([0, -2, -1])
    return MAE

def weighted_acc(y_pred,y_true):
    lat = np.linspace(49.5, 59.5, num=17, dtype=float, endpoint=True)
    ACC = np.empty([y_pred.size(1)])
    weights_lat = np.cos(np.deg2rad(lat))
    weights_lat /= weights_lat.mean()
    w = torch.tensor(weights_lat)
    for i in range(y_pred.size(1)):
        pred_t = y_pred[:,i,0,:,:]  # (samples, 29, 17) - width, height
        true_t = y_true[:,i,0,:,:]
        clim = true_t.mean(0)
        a = true_t - clim
        a_prime = (a - a.mean())  # No permute - keep as (samples, 29, 17)
        fa = pred_t - clim
        fa_prime = (fa - fa.mean())  # No permute
        # Apply weights to last dimension (height/latitude)
        ACC[i] = (
                torch.sum(w * fa_prime * a_prime) /
                torch.sqrt(
                    torch.sum(w * fa_prime ** 2) * torch.sum(w * a_prime ** 2)
                )
        )
    return ACC

###输入预测值与真实值
y_pred = np.load(r'w_pred.npy')
y_pred = y_pred[:,:,None]  # Shape: (samples, time, 1, height, width)
y_true = np.load(r'w_true.npy')
y_true = y_true[:,:,None]

print(f'Loaded data shapes: pred={y_pred.shape}, true={y_true.shape}')

# Pass entire arrays - functions will iterate over time internally
y_pred_tensor = torch.tensor(y_pred)
y_true_tensor = torch.tensor(y_true)

rmse_results = weighted_rmse(y_pred_tensor, y_true_tensor)
mae_results = weighted_mae(y_pred_tensor, y_true_tensor)
acc_results = weighted_acc(y_pred_tensor, y_true_tensor)

for i in range(72):
    print(i+1)
    print('RMSE:', rmse_results[i])
    print('MAE: ', mae_results[i])
    print('ACC: ', acc_results[i])