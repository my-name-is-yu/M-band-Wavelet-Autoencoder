import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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
    def __init__(self, folder_path, signal_length, sample_rate):
        super().__init__()
        self.folder_path = folder_path
        self.signal_length = signal_length
        self.sample_rate = sample_rate
        
        self.file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.wav')]
        if not self.file_paths:
            raise ValueError(f"指定されたフォルダ '{folder_path}' にWAVファイルが見つかりません。")

        self.segments = []
        for path in self.file_paths:
            audio, _ = librosa.load(path, sr=self.sample_rate, mono=True)
            audio = librosa.util.normalize(audio)
            
            num_segments = len(audio) // self.signal_length
            for i in range(num_segments):
                segment = audio[i * self.signal_length : (i + 1) * self.signal_length]
                self.segments.append(segment)

        if not self.segments:
            raise ValueError("どの音声ファイルも指定された信号長より短いため、学習データを作成できませんでした。")
        print(f"{len(self.file_paths)}個のファイルから、{len(self.segments)}個の学習セグメントを作成しました。")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        return torch.tensor(self.segments[idx], dtype=torch.float32)

class M_Band_Wavelet_Autoencoder(nn.Module):
    def __init__(self, M, filter_length, signal_length):
        super(M_Band_Wavelet_Autoencoder, self).__init__()
        self.M = M
        self.filter_length = filter_length
        self.signal_length = signal_length
        self.analysis_filters = nn.Parameter(torch.randn(M, 1, filter_length))
        self.synthesis_filters = nn.Parameter(torch.randn(M, 1, filter_length))
        self.padding = (self.filter_length - 1) // 2
        
        l_subband = math.floor((self.signal_length + 2 * self.padding - self.filter_length) / self.M) + 1
        l_reconstructed = (l_subband - 1) * self.M - 2 * self.padding + self.filter_length
        self.output_padding = self.signal_length - l_reconstructed
        
    def decompose(self, signal):
        if signal.dim() == 2: signal = signal.unsqueeze(1)
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

# ハイパーパラメータ設定・学習用フォルダの指定
NORMAL_SOUND_DIR = os.path.expanduser('~/DATABASE/AirCompressorDataset/Healthy')
MODEL_SAVE_PATH = 'm_band_wavelet_model.pth'
SAMPLE_RATE = 16000
SIGNAL_LENGTH = 4096
M = 8
FILTER_LENGTH = 64
EPOCHS = 1000
BATCH_SIZE = 32
PATIENCE = 10  # 10エポック連続で改善が見られなければ終了
MIN_DELTA = 1e-6 # 改善とみなす最小の変化量

if __name__ == '__main__':
    
    try:
        dataset = AudioDataset(folder_path=NORMAL_SOUND_DIR, signal_length=SIGNAL_LENGTH, sample_rate=SAMPLE_RATE)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    except Exception as e:
        print(f"データ読み込みエラー: {e}")
        exit()

    model = M_Band_Wavelet_Autoencoder(M=M, filter_length=FILTER_LENGTH, signal_length=SIGNAL_LENGTH)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Early Stopping用の変数を初期化
    best_loss = float('inf')
    patience_counter = 0

    print("\n学習を開始します...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_data in dataloader:
            model.train()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.8f}')

        # Early Stoppingのチェック
        if best_loss - avg_loss > MIN_DELTA:
            # 誤差が改善した
            best_loss = avg_loss
            patience_counter = 0
            # 最も性能の良いモデルを保存
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            # print(f"  -> Best model saved with loss: {best_loss:.8f}") # 保存したことを確認したい場合はコメントアウトを外す
        else:
            # 誤差が改善しなかった
            patience_counter += 1
        
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping. {PATIENCE}エポック連続で誤差が改善しなかったため、学習を終了します。")
            break

    print("学習が完了しました。")
    print(f"\n最も性能の良かったモデル（誤差: {best_loss:.8f}）が '{MODEL_SAVE_PATH}' に保存されています。")

