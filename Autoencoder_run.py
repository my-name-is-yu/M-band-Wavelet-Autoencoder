import os
import glob
import torch
import torch.nn as nn
import numpy as np
import librosa
from Autoencoder_dev import (
    AudioDataset,
    M_Band_Wavelet_Autoencoder,
)
from config import (
    MODEL_SAVE_PATH,
    SIGNAL_LENGTH,
    SAMPLE_RATE,
    FILTER_LENGTH,
    M,
    EVAL_SOUND_DIR,
    EVAL_HOP_RATIO,
    THRESHOLD_MODE,
    RELATIVE_FACTOR,
    PERCENTILE,
    ABSOLUTE_THRESHOLD,
    LAMBDA_SPEC,
)

if not os.path.exists(MODEL_SAVE_PATH):
    print(f"エラー: 保存されたモデルファイル '{MODEL_SAVE_PATH}' が見つかりません。")
    raise SystemExit(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"'{MODEL_SAVE_PATH}' からモデルを読み込みます...")
model = M_Band_Wavelet_Autoencoder(
    M=M, filter_length=FILTER_LENGTH, signal_length=SIGNAL_LENGTH).to(device)
state = torch.load(MODEL_SAVE_PATH, map_location=device)
model.load_state_dict(state)
model.eval()


def stft_l1_loss(x, y, n_fft=512, hop=128, win=512):
    window = torch.hann_window(win, device=x.device)
    X = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win,
                   window=window, return_complex=True, center=True)
    Y = torch.stft(y, n_fft=n_fft, hop_length=hop, win_length=win,
                   window=window, return_complex=True, center=True)
    return (X.abs() - Y.abs()).abs().mean(dim=(-2, -1))  # per-sample


def segment_audio(audio: np.ndarray, seg_len: int, hop_ratio: float) -> np.ndarray:
    audio = librosa.util.normalize(audio)
    hop = max(1, int(seg_len * hop_ratio))
    if len(audio) < seg_len:
        return np.empty((0, seg_len), dtype=np.float32)
    segs = []
    for start in range(0, len(audio) - seg_len + 1, hop):
        segs.append(audio[start:start + seg_len])
    return np.stack(segs, axis=0).astype(np.float32)


def file_anomaly_score(path: str) -> float:
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    segs = segment_audio(audio, SIGNAL_LENGTH, EVAL_HOP_RATIO)
    if segs.shape[0] == 0:
        return float('nan')
    with torch.no_grad():
        x = torch.from_numpy(segs).to(device)
        y = model(x)
        # MSE per segment
        mse = ((y - x) ** 2).mean(dim=1)
        # STFT mag L1 per segment (match training objective)
        spec = stft_l1_loss(y, x)
        loss = mse + LAMBDA_SPEC * spec
        return loss.mean().item()


wav_paths = sorted(glob.glob(os.path.join(EVAL_SOUND_DIR, '*.wav')))
if not wav_paths:
    print(f"エラー: フォルダ '{EVAL_SOUND_DIR}' にWAVファイルが見つかりません。")
    raise SystemExit(1)

print(f"評価フォルダ: {EVAL_SOUND_DIR}")
scores = []
for p in wav_paths:
    s = file_anomaly_score(p)
    scores.append((p, s))

# しきい値の決定
valid_scores = [s for _, s in scores if not np.isnan(s)]
if not valid_scores:
    print("エラー: 有効な音声長のファイルがありません（全てが短すぎます）。")
    raise SystemExit(1)

threshold = None
mode = THRESHOLD_MODE.lower()
if mode == 'relative':
    med = float(np.median(valid_scores))
    threshold = med * float(RELATIVE_FACTOR)
elif mode == 'percentile':
    threshold = float(np.percentile(valid_scores, PERCENTILE))
elif mode == 'absolute':
    if ABSOLUTE_THRESHOLD is None:
        print("エラー: THRESHOLD_MODE='absolute' ですが ABSOLUTE_THRESHOLD が未設定です。")
        raise SystemExit(1)
    threshold = float(ABSOLUTE_THRESHOLD)
else:
    print(f"エラー: 不正なTHRESHOLD_MODE '{THRESHOLD_MODE}'")
    raise SystemExit(1)

print("\n--- ファイル別 異常判定 ---")
print(f"しきい値: {threshold:.6f}  (mode={THRESHOLD_MODE})")

num_abn = 0
for p, s in scores:
    if np.isnan(s):
        print(f"SKIP: {os.path.basename(p)}  (短すぎて解析不可)")
        continue
    label = 'ABNORMAL' if s > threshold else 'NORMAL '
    if label == 'ABNORMAL':
        num_abn += 1
    print(f"{label}  score={s:.6f}  file={os.path.basename(p)}")

print("\n--- サマリ ---")
print(f"総ファイル数: {len(scores)}  有効: {len(valid_scores)}  異常判定: {num_abn}")
