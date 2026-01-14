import numpy as np

# ENGLAND NORMALIZATION PARAMETERS
# Calculated from: lmtwfm_data/npy_files/train_wind.npy
u_mean = 0.06038649380207062
u_std = 1.0029470920562744
v_mean = -0.062388960272073746
v_std = 0.9924566149741609

print('Loading predictions...')
uv_true = np.load(r'uv_true.npy')
uv_pred = np.load(r'uv_pred.npy')

print('Denormalizing u and v components...')
u_true = uv_true[:, :, 0, :, :] * u_std + u_mean
v_true = uv_true[:, :, 1, :, :] * v_std + v_mean

u_pred = uv_pred[:, :, 0, :, :] * u_std + u_mean
v_pred = uv_pred[:, :, 1, :, :] * v_std + v_mean

print('Converting to wind speed...')
w_true = np.sqrt(u_true ** 2 + v_true ** 2)
w_pred = np.sqrt(u_pred ** 2 + v_pred ** 2)

print('Saving wind speed predictions...')
np.save(r"w_pred.npy", arr=w_pred)
np.save(r"w_true.npy", arr=w_true)

print('Done, Files saved:')
print('  - result/w_pred.npy')
print('  - result/w_true.npy')