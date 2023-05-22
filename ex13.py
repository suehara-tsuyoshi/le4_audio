#
# 計算機科学実験及演習 4「音響信号処理」
# サンプルソースコード
#
# 音声ファイルを読み込み，フーリエ変換を行う．
#

# ライブラリの読み込み
import matplotlib.pyplot as plt
import numpy as np
from cepstrum import cepstrum
import librosa

# サンプリングレート
SR = 16000

def cepstrum(x):

  # 高速フーリエ変換
  # np.fft.rfftを使用するとFFTの前半部分のみが得られる
  fft_spec = np.fft.rfft(x)

  # 複素スペクトログラムを対数振幅スペクトログラムに
  fft_log_abs_spec = np.log(np.abs(fft_spec))

  # 対数振幅スペクトルをケプストラム係数に
  fft_log_abs_ceps = np.fft.fft(fft_log_abs_spec)

  dimension = 13

  fft_log_abs_ceps[dimension:len(fft_log_abs_ceps)-dimension] = 0

  spectrum_envelope = (np.fft.ifft(fft_log_abs_ceps)).real

  #
  # スペクトルを画像に表示・保存
  #

  # 画像として保存するための設定
  fig = plt.figure()

  # スペクトログラムを描画
  plt.xlabel('frequency [Hz]')		# x軸のラベルを設定
  plt.ylabel('amplitude')				# y軸のラベルを設定
  plt.xlim([0, 8000])					# x軸の範囲を設定
  # x軸のデータを生成（元々のデータが0~8000Hzに対応するようにする）
  x_data = np.linspace((SR/2)/len(fft_log_abs_spec), SR/2, len(fft_log_abs_spec))
  plt.plot(x_data, fft_log_abs_spec, color = "blue")			# 描画
  # 【補足】
  # 縦軸の最大値はサンプリング周波数の半分 = 16000 / 2 = 8000 Hz となる

  x_data2 = np.linspace((SR/2)/len(spectrum_envelope), SR/2, len(spectrum_envelope))
  print(len(spectrum_envelope))

  plt.plot(x_data2, spectrum_envelope, color = "red")


  # 表示

  # 横軸を0~2000Hzに拡大
  # xlimで表示の領域を変えるだけ
  plt.xlabel('frequency [Hz]')
  plt.ylabel('amplitude')
  plt.xlim([0, 2000])



  # 表示
  plt.show()

# 音声ファイルの読み込み
y, _ = librosa.load('aiueo.wav', sr=SR)

y_a = y[22000:38000]
y_i = y[45000:61000]
y_u = y[74000:90000]
y_e = y[102000:118000]
y_o = y[132000:148000]

cepstrum(y_a)
cepstrum(y_i)
cepstrum(y_u)
cepstrum(y_e)
cepstrum(y_o)
