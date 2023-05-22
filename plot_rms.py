#
# 計算機科学実験及演習 4「音響信号処理」
# サンプルソースコード
#
# 音声ファイルを読み込み，各フレーム毎のRMSを表示
#

# ライブラリの読み込み
import matplotlib.pyplot as plt
import numpy as np
import librosa

# サンプリングレート
SR = 16000

# 音声ファイルの読み込み
x, _ = librosa.load('a.wav', sr=SR)

# フレームサイズ
size_frame = 512

# フレームサイズに合わせてハミング窓を作成
hamming_window = np.hamming(size_frame)

# シフトサイズ
size_shift = 16000 / 100

# RMSを保存するリスト
rms = []

# size_shift分ずらしながらsize_frame分のデータを取得
# np.arange関数はfor文で辿りたい数値のリストを返す
# 通常のrange関数と違うのは3つ目の引数で間隔を指定できるところ
# (初期位置, 終了位置, 1ステップで進める間隔)
for i in np.arange(0, len(x)-size_frame, size_shift):

  # 該当フレームのデータを取得
  idx = int(i)	# arangeのインデクスはfloatなのでintに変換
  x_frame = x[idx : idx+size_frame]

  # 【補足】
  # 配列（リスト）のデータ参照
  # list[A:B] listのA番目からB-1番目までのデータを取得

  # 窓掛けしたデータをFFT
  # np.fft.rfftを使用するとFFTの前半部分のみが得られる
  fft_spec = np.fft.rfft(x_frame * hamming_window)

  # np.fft.fft / np.fft.fft2 を用いた場合
  # 複素スペクトログラムの前半だけを取得
  #fft_spec_first = fft_spec[:int(size_frame/2)]

  # 【補足】
  # 配列（リスト）のデータ参照
  # list[:B] listの先頭からB-1番目までのデータを取得

  # 計算した対数振幅スペクトログラムを配列に保存
  fft_spec = np.frombuffer(fft_spec, dtype= "int16")
  rms_frame = np.sqrt((1/size_frame) * (np.sum(fft_spec))*(np.sum(fft_spec)))
  rms.append(rms_frame)


db = 20 * np.log10(rms)
t = np.arange(0, len(db)/SR, 1/SR)

plt.plot(t, db, label='signal')
plt.show()

	


