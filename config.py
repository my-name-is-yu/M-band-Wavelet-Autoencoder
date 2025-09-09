import os

# Paths
MODEL_SAVE_PATH = 'm_band_wavelet_model.pth'  # モデルの保存先
NORMAL_SOUND_DIR = os.path.expanduser(
    '~/DATABASE/AirCompressorDataset/Healthy')  # 学習用フォルダ
EVAL_SOUND_DIR = NORMAL_SOUND_DIR  # ここを評価対象のフォルダに変更可

# 音声・モデルパラメータ
SAMPLE_RATE = 16000
SIGNAL_LENGTH = 4096
M = 8
FILTER_LENGTH = 64

# 学習パラメータ
EPOCHS = 10
BATCH_SIZE = 32
PATIENCE = 3  # Early Stopping用
MIN_DELTA = 1e-6  # 改善とみなす最小値
LAMBDA_SPEC = 0.5  # 0で時間領域の誤差のみで学習、1でスペクトル誤差のみで学習

# 評価パラメータ
EVAL_HOP_RATIO = 0.5  # フレーム分割時の重なり 0.5で50%オーバーラップ
THRESHOLD_MODE = 'relative'
'''
'relative': 正常スコアの中央値 * RELATIVE_FACTOR をしきい値にする。RELATIVE_FACTORはこの時有効
'percentile': 正常スコア分布のパーセンタイル値をしきい値にする。PERCENTILEはこの時有効
'absolute': 固定の数値をしきい値にする。ABSOLUTE_THRESHOLDはこの時有効
'''
RELATIVE_FACTOR = 2.0        # 正常スコアのRELATIVE_FACTOR倍を超えたら異常とみなす
PERCENTILE = 95              # 正常スコア分布のPERCENTILE値を閾値とする
ABSOLUTE_THRESHOLD = None    # 再構成誤差スコアがこの値を超えたら異常
