#
# 計算機科学実験及演習 4「音響信号処理」
# サンプルソースコード
#
# 音声ファイルを読み込み，フーリエ変換を行う．
#

# ライブラリの読み込み
import matplotlib.pyplot as plt
import numpy as np
import librosa

# サンプリングレート
SR = 16000

# 音声ファイルの読み込み
x, _ = librosa.load('viv.wav', sr=SR)

# 高速フーリエ変換
# np.fft.rfftを使用するとFFTの前半部分のみが得られる
fft_spec = np.fft.rfft(x)

# 複素スペクトログラムを対数振幅スペクトログラムに
fft_log_abs_spec = np.log(np.abs(fft_spec))

#
# スペクトルを画像に表
#

# スペクトログラムを描画
plt.xlabel('frequency [Hz]')		# x軸のラベルを設定
plt.ylabel('amplitude')				# y軸のラベルを設定
plt.xlim([0, SR/2])					# x軸の範囲を設定
# x軸のデータを生成（元々のデータが0~8000Hzに対応するようにする）
x_data = np.linspace((SR/2)/len(fft_log_abs_spec), SR/2, len(fft_log_abs_spec))
plt.plot(x_data, fft_log_abs_spec)			# 描画
plt.xlim([0, 1000])
# 【補足】
# 縦軸の最大値はサンプリング周波数の半分 = 16000 / 2 = 8000 Hz となる

# 表示
plt.show()

