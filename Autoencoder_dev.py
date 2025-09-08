import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import librosa


class AudioDataset(Dataset):
    """
    指定されたフォルダから音声ファイルを読み込み、
    固定長のセグメントにして返すデータセット。
    """

    def __init__(self, folder_path, signal_length, sample_rate, train=False, hop_ratio=0.5):
        super().__init__()
        self.folder_path = folder_path
        self.signal_length = signal_length
        self.sample_rate = sample_rate
        self.train = train
        self.hop_ratio = hop_ratio

        self.file_paths = [os.path.join(folder_path, f) for f in os.listdir(
            folder_path) if f.endswith('.wav')]
        if not self.file_paths:
            raise ValueError(f"指定されたフォルダ '{folder_path}' にWAVファイルが見つかりません。")

        self.segments = []
        for path in self.file_paths:
            audio, _ = librosa.load(path, sr=self.sample_rate, mono=True)
            audio = librosa.util.normalize(audio)

            hop = max(1, int(self.signal_length * self.hop_ratio))
            for start in range(0, len(audio) - self.signal_length + 1, hop):
                segment = audio[start:start + self.signal_length]

        if not self.segments:
            raise ValueError("どの音声ファイルも指定された信号長より短いため、学習データを作成できませんでした。")
        print(f"{len(self.file_paths)}個のファイルから、{len(self.segments)}個の学習セグメントを作成しました。")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        return torch.tensor(self.segments[idx], dtype=torch.float32)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        seg = self.segments[idx].copy()
        if self.train:
            g = np.random.uniform(0.8, 1.2)
            seg *= g
            shift = np.random.randint(-64, 65)
            seg = np.roll(seg, shift)
            if np.random.rand() < 0.3:
                s = np.random.randint(0, len(seg)-128)
                seg[s:s+128] = 0.0
        return torch.tensor(seg, dtype=torch.float32)


class M_Band_Wavelet_Autoencoder(nn.Module):
    def __init__(self, M, filter_length, signal_length):
        super(M_Band_Wavelet_Autoencoder, self).__init__()
        self.M = M
        self.filter_length = filter_length
        self.signal_length = signal_length
        self.analysis_filters = nn.Parameter(torch.randn(M, 1, filter_length))
        self.synthesis_filters = nn.Parameter(torch.randn(M, 1, filter_length))
        self.padding = (self.filter_length - 1) // 2

        l_subband = math.floor(
            (self.signal_length + 2 * self.padding - self.filter_length) / self.M) + 1
        l_reconstructed = (l_subband - 1) * self.M - 2 * \
            self.padding + self.filter_length
        self.output_padding = self.signal_length - l_reconstructed

    def decompose(self, signal):
        if signal.dim() == 2:
            signal = signal.unsqueeze(1)
        return torch.nn.functional.conv1d(signal, self.analysis_filters, stride=self.M, padding=self.padding)

    def reconstruct(self, subbands):
        return torch.nn.functional.conv_transpose1d(
            subbands, self.synthesis_filters, stride=self.M, padding=self.padding, output_padding=self.output_padding
        ).squeeze(1)

    def forward(self, x):
        original_len = x.shape[-1]
        subbands = self.decompose(x)
        reconstructed_x = self.reconstruct(subbands)
        return reconstructed_x[:, :original_len]


def stft_l1_loss(x, y, n_fft=512, hop=128, win=512):
    X = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win,
                   return_complex=True, center=True)
    Y = torch.stft(y, n_fft=n_fft, hop_length=hop, win_length=win,
                   return_complex=True, center=True)
    return (X.abs() - Y.abs()).abs().mean()


lambda_spec = 0.5

# ハイパーパラメータ設定・学習用フォルダの指定
NORMAL_SOUND_DIR = os.path.expanduser(
    '~/DATABASE/AirCompressorDataset/Healthy')
MODEL_SAVE_PATH = 'm_band_wavelet_model.pth'
SAMPLE_RATE = 16000
SIGNAL_LENGTH = 4096
M = 8
FILTER_LENGTH = 64
EPOCHS = 1000
BATCH_SIZE = 32
PATIENCE = 10  # 10エポック連続で改善が見られなければ終了
MIN_DELTA = 1e-6  # 改善とみなす最小の変化量
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

full_tmp = AudioDataset(NORMAL_SOUND_DIR, SIGNAL_LENGTH,
                        SAMPLE_RATE, train=False, hop_ratio=0.5)
num_segments = len(full_tmp)

indices = np.arange(num_segments)
rng = np.random.default_rng(42)
rng.shuffle(indices)
n_val = max(1, int(0.2 * num_segments))
val_idx = indices[:n_val]
train_idx = indices[n_val:]

# 学習用と検証用で別インスタンス（学習側のみAugを有効化）
train_ds = AudioDataset(NORMAL_SOUND_DIR, SIGNAL_LENGTH,
                        SAMPLE_RATE, train=True,  hop_ratio=0.5)
val_ds = AudioDataset(NORMAL_SOUND_DIR, SIGNAL_LENGTH,
                      SAMPLE_RATE, train=False, hop_ratio=0.5)

train_set = torch.utils.data.Subset(train_ds, train_idx.tolist())
val_set = torch.utils.data.Subset(val_ds,   val_idx.tolist())

_pin = (device.type == 'cuda')
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=2, pin_memory=_pin)
val_loader = DataLoader(val_set,   batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=2, pin_memory=_pin)

model = M_Band_Wavelet_Autoencoder(
    M=M, filter_length=FILTER_LENGTH, signal_length=SIGNAL_LENGTH).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True)

best_val = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    for x in train_loader:
        x = x.to(device)
        optimizer.zero_grad(set_to_none=True)
        y = model(x)
        loss_mse = criterion(y, x)
        loss_spec = stft_l1_loss(y, x)
        loss = loss_mse + lambda_spec * loss_spec
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x in val_loader:
            x = x.to(device)
            y = model(x)
            val_mse = criterion(y, x)
            val_spec = stft_l1_loss(y, x)
            val_loss += (val_mse + lambda_spec * val_spec).item()

    val_loss /= len(val_loader)
    scheduler.step(val_loss)

    if val_loss + MIN_DELTA < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= PATIENCE:
        break

print("学習が完了しました。")
print(f"\n最も性能の良かったモデル（誤差: {best_val:.8f}）が '{MODEL_SAVE_PATH}' に保存されています。")
