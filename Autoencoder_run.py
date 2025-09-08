import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from Autoencoder_dev import (
    AudioDataset, 
    M_Band_Wavelet_Autoencoder, 
    MODEL_SAVE_PATH, 
    SIGNAL_LENGTH, 
    SAMPLE_RATE, 
    FILTER_LENGTH, 
    M,
    NORMAL_SOUND_DIR
)

if not os.path.exists(MODEL_SAVE_PATH):
    print(f"エラー: 保存されたモデルファイル '{MODEL_SAVE_PATH}' が見つかりません。")
    exit()
        
print(f"'{MODEL_SAVE_PATH}' からモデルを読み込みます...")

validation_model = M_Band_Wavelet_Autoencoder(M=M, filter_length=FILTER_LENGTH, signal_length=SIGNAL_LENGTH)
validation_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
validation_model.eval()

try:
    validation_dataset = AudioDataset(folder_path=NORMAL_SOUND_DIR, signal_length=SIGNAL_LENGTH, sample_rate=SAMPLE_RATE)
    normal_input_tensor = validation_dataset[0].unsqueeze(0)
except Exception as e:
    print(f"検証データの読み込みエラー: {e}")
    exit()

criterion = nn.MSELoss()

abnormal_input_numpy = normal_input_tensor.squeeze().numpy().copy()
noise_start = SIGNAL_LENGTH // 2
abnormal_input_numpy[noise_start:noise_start+50] += np.random.randn(50) * 0.5
abnormal_input_tensor = torch.tensor(abnormal_input_numpy).unsqueeze(0)

with torch.no_grad():
    reconstructed_normal = validation_model(normal_input_tensor)
    loss_normal = criterion(reconstructed_normal, normal_input_tensor)
        
    reconstructed_abnormal = validation_model(abnormal_input_tensor)
    loss_abnormal = criterion(reconstructed_abnormal, abnormal_input_tensor)

print("\n--- 異常検知結果 ---")
print(f"正常データの再構成誤差 (MSE): {loss_normal.item():.8f}")
print(f"異常データの再構成誤差 (MSE): {loss_abnormal.item():.8f}")
if loss_abnormal > loss_normal * 2:
    print(">>> 異常を検知しました！")
else:
    print(">>> 正常と判断しました。")
        
plt.figure(figsize=(12, 8))
time_axis = np.linspace(0, SIGNAL_LENGTH / SAMPLE_RATE, SIGNAL_LENGTH)
    
plt.subplot(2, 1, 1)
plt.title(f"Normal Sound Reconstruction (MSE: {loss_normal.item():.6f})")
plt.plot(time_axis, normal_input_tensor.squeeze().numpy(), 'b-', label='Original Normal Audio', alpha=0.7)
plt.plot(time_axis, reconstructed_normal.squeeze().numpy(), 'r--', label='Reconstructed Audio')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
    
plt.subplot(2, 1, 2)
plt.title(f"Abnormal Sound Reconstruction (MSE: {loss_abnormal.item():.6f})")
plt.plot(time_axis, abnormal_input_tensor.squeeze().numpy(), 'b-', label='Original Abnormal Audio', alpha=0.7)
plt.plot(time_axis, reconstructed_abnormal.squeeze().numpy(), 'r--', label='Reconstructed Audio')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()

plt.tight_layout()
plt.show()
