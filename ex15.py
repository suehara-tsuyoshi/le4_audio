#
# 計算機科学実験及演習 4「音響信号処理」
# サンプルソースコード
#
# 音声ファイルを読み込み，波形を図示する．
#

# ライブラリの読み込み
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.function_base import cov
import librosa

# aiueo.wav
# x_a = x[20000:30000]
# x_i = x[55000:65000]
# x_u = x[95000:105000]
# x_e = x[140000:150000]
# x_o = x[185000:195000]

# aiueo2_self.wav
# x_a = x[22000:38000]
# x_i = x[45000:61000]
# x_u = x[74000:90000]
# x_e = x[102000:118000]
# x_o = x[132000:148000]

# aiueo2_mic.wav
# x_a = x[12000:22000]
# x_i = x[40000:50000]
# x_u = x[70000:80000]
# x_e = x[100000:110000]
# x_o = x[130000:140000]

def cepstrum(a):
  # フレームサイズ
  size_frame = 512 # 2のべき乗

  # フレームサイズに合わせてハミング窓を作成
  hamming_window = np.hamming(size_frame)

  # シフトサイズ
  size_shift = 16000 / 100	# 0.01 秒 (10 msec)

  # スペクトログラムを保存するlist
  cepstrogram = []

  # size_shift分ずらしながらsize_frame分のデータを取得
  for i in np.arange(0, len(a)-size_frame, size_shift):

    # 該当フレームのデータを取得
    idx = int(i)	# arangeのインデクスはfloatなのでintに変換
    a_frame = a[idx : idx+size_frame]
    fft_spec = np.fft.rfft(a_frame * hamming_window)

    # 複素スペクトログラムを対数振幅スペクトログラムに
    fft_log_abs_spec = np.log(np.abs(fft_spec))

    # 対数振幅スペクトルをケプストラム係数に
    fft_log_abs_ceps = np.fft.fft(fft_log_abs_spec)

    # ケプストラム係数をリストに追加
    cepstrogram.append(np.real(fft_log_abs_ceps[0:dimension]))

  return cepstrogram

def log_likelihood(x_vowel, a):
  # 各母音に対応するケプストラム係数、その平均、その共分散を格納するlist
  ceps_vowel = []
  mean_vowel = []
  cov_vowel = []
  log_cov_vowel = []

  # 計算して格納
  for i in range(5):
    ceps_vowel.append(cepstrum(x_vowel[i]))
    mean_vowel.append(np.mean(np.array(ceps_vowel[i]), axis = 0))
    cov_vowel.append(np.var(np.array(ceps_vowel[i]), axis = 0))
    log_cov_vowel.append(np.log(cov_vowel[i]) // 2)

  # 対数尤度のリスト
  L = [0.0, 0.0, 0.0, 0.0, 0.0]

  # 「あ」～「お」の対数尤度を計算
  for d in range(dimension):
    for i in range(5):
      L[i] -= log_cov_vowel[i][d]
      L[i] -= ((a[d]-mean_vowel[i][d]) ** 2) / (2 * cov_vowel[i][d])

  return (L.index(max(L)))

def speech_recognition(x_model, a):
  # フレームサイズ
  size_frame = 512# 2のべき乗

  # フレームサイズに合わせてハミング窓を作成
  hamming_window = np.hamming(size_frame)

  # シフトサイズ
  size_shift = 16000 / 100	# 0.01 秒 (10 msec)

  recognition_list = []

  # size_shift分ずらしながらsize_frame分のデータを取得
  for i in np.arange(0, len(x)-size_frame, size_shift):
    
    # 該当フレームのデータを取得
    idx = int(i)	# arangeのインデクスはfloatなのでintに変換
    a_frame = a[idx : idx+size_frame]
    fft_spec = np.fft.rfft(a_frame * hamming_window)
    # 複素スペクトログラムを対数振幅スペクトログラムに
    fft_log_abs_spec = np.log(np.abs(fft_spec))

    # 対数振幅スペクトルをケプストラム係数に
    fft_log_abs_ceps = np.fft.fft(fft_log_abs_spec)

    # 母音推定を行いリストに追加
    recognition_id = log_likelihood(x_model, (np.real(fft_log_abs_ceps[0:dimension])))

    recognition_list.append(recognition_id)

  return recognition_list

def smoothing(a):
  # XYXならXXXに変換
  for i in range(len(a)-2):
    if a[i]!=a[i+1] and a[i+1]!=a[i+2]:
      a[i+1]=a[i]
     
  
  # XYYXならXXXXに変換
  for i in range(len(a)-3):
    if a[i]!=a[i+1] and a[i+1]==a[i+2] and a[i+2]!=a[i+3]:
      a[i+1]=a[i]
      a[i+2]=a[i]

  # XYYYXならXXXXXに変換
  for i in range(len(a)-4):
    if a[i]!=a[i+1] and a[i+1]==a[i+2] and a[i+2]==a[i+3] and a[i+3]!=a[i+4]:
      a[i+1]=a[i]
      a[i+2]=a[i]
      a[i+3]=a[i]

  return a


# 次元
dimension = 13
# サンプリングレート
SR = 16000
# 音声ファイルの読み込み
# xに波形データが保存される
# 第二戻り値はサンプリングレート（ここでは必要ないので _ としている）
x, _ = librosa.load('aiueo2_mic.wav', sr=SR)
x_aiueo = np.array([x[12000:22000], x[40000:50000], x[70000:80000], x[100000:110000], x[130000:140000]]) 
# 音声ファイルの読み込み
y, _ = librosa.load('aiueo_mic.wav', sr=SR)
data = speech_recognition(x_aiueo, y)

#
# スペクトログラムを画像に表示・保存
#
data = smoothing(data)


def spec(a):
  # サンプリングレート
  SR = 16000
  #
  # 短時間フーリエ変換
  #

  # フレームサイズ
  size_frame = 2048	# 2のべき乗

  # フレームサイズに合わせてハミング窓を作成
  hamming_window = np.hamming(size_frame)

  # シフトサイズ
  size_shift = 16000 / 100	# 0.01 秒 (10 msec)

  # スペクトログラムを保存するlist
  spectrogram = []

  # size_shift分ずらしながらsize_frame分のデータを取得
  # np.arange関数はfor文で辿りたい数値のリストを返す
  # 通常のrange関数と違うのは3つ目の引数で間隔を指定できるところ
  # (初期位置, 終了位置, 1ステップで進める間隔)
  for i in np.arange(0, len(a)-size_frame, size_shift):

    # 該当フレームのデータを取得
    idx = int(i)	# arangeのインデクスはfloatなのでintに変換
    a_frame = a[idx : idx+size_frame]

    # 【補足】
    # 配列（リスト）のデータ参照
    # list[A:B] listのA番目からB-1番目までのデータを取得

    # 窓掛けしたデータをFFT
    # np.fft.rfftを使用するとFFTの前半部分のみが得られる
    fft_spec = np.fft.rfft(a_frame * hamming_window)

    # np.fft.fft / np.fft.fft2 を用いた場合
    # 複素スペクトログラムの前半だけを取得
    #fft_spec_first = fft_spec[:int(size_frame/2)]

    # 【補足】
    # 配列（リスト）のデータ参照
    # list[:B] listの先頭からB-1番目までのデータを取得

    # 複素スペクトログラムを対数振幅スペクトログラムに
    fft_log_abs_spec = np.log(np.abs(fft_spec))


    # 計算した対数振幅スペクトログラムを配列に保存
    spectrogram.append(fft_log_abs_spec[0:128])
  
  return spectrogram



# スペクトログラムを描画
# plt.xlabel('time [s]')					# x軸のラベルを設定
# plt.ylabel('frequency [Hz]')		# y軸のラベルを設定
# plt.imshow(
# 	np.flipud(np.array(spec(y)).T),		# 画像とみなすために，データを転地して上下反転
# 	extent=[0, len(y), 0, 1000],			# (横軸の原点の値，横軸の最大値，縦軸の原点の値，縦軸の最大値)
# 	aspect='auto',
# 	interpolation='nearest'
# )
plt.plot(data, color = "blue")
plt.yticks([0, 1, 2, 3, 4], ['a','i','u','e','o'])
plt.show()






