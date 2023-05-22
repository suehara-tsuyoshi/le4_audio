#
# 計算機科学実験及演習 4「音響信号処理」
# サンプルソースコード
#
# 音声ファイルを読み込み，スペクトログラムを計算して図示する．
#

# ライブラリの読み込み
import matplotlib.pyplot as plt
import numpy as np
import math
import librosa
# 配列 a の index 番目の要素がピーク（両隣よりも大きい）であれば True を返す関数
# 自己相関を求める際に使用
def is_peak(a, index):
	# 配列の端は例外的に除く
	if index == 0 or index == len(a) - 1:
		return False
  # 左右の大きいほうの値より大きければ True
	else:
		return  a[index] >= max(a[index-1], a[index+1])

# 基本周波数を求める関数
def correlate(a):
  # 自己相関が格納された，長さが len(x)*2-1 の対称な配列を得る
  autocorr = np.correlate(a, a, 'full')

  # 不要な前半を捨てる
  autocorr = autocorr [len (autocorr ) // 2 : ]

  # ピークのインデックスを抽出する
  peakindices = [i for i in range (len (autocorr )) if is_peak (autocorr, i)]

  # インデックス0 がピークに含まれていれば捨てる
  # インデックス0が 通常最大のピークになる
  peakindices = [i for i in peakindices if i != 0]

  # 返り値の初期化
  max_peak_frequency = 8000.0

  # ピークがなければ最大周波数を返す
  # ピークがあれば自己相関が最大となる(0を含めると2番目に大きい)周期を得る
  if len(peakindices) != 0:
    # 自己相関が最大となるインデックスを得る
    max_peak_index = max (peakindices , key=lambda index: autocorr [index])
    # インデックスに対応する周波数を得る
    max_peak_frequency = SR / max_peak_index
  
  return max_peak_frequency

# 音声波形データを受け取り，ゼロ交差数を計算する関数
def zero_cross(waveform):
  # ゼロ交差数を格納する変数
	zc = 0

  # 配列内で連続する2つの値の正負が異なれば交差しているとし，zcの値を１増やす
	for i in range(len(waveform) - 1):
		if(
			(waveform[i] > 0.0 and waveform[i+1] < 0.0) or
			(waveform[i] < 0.0 and waveform[i+1] > 0.0)
		):
			zc += 1
	
	return zc

def nn2hz(notenum):
  if notenum <= 40 or notenum >= 70:
    return 0.0
  else : 
    return 440.0 * (2.0 ** ((notenum - 69) / 12.0))

def hz2nn(frequency):
  return int (round (12.0 * (math.log((frequency + 0.001) / 440.0) / math.log(2.0)))) + 69


# スムージングを行う関数
def smoothing(a):
  for i in range(len(a)):
    a[i] = nn2hz(hz2nn(a[i]))

  for i in range(len(a)-2):
    if a[i] == a[i+2]:
      a[i+1] = a[i]
  
  for i in range(len(a)-10):
    if a[i] == a[i+10]:
      a[i+5] = a[i]

  for i in range(len(a)-4):
    if a[i]!=a[i+1] and a[i+1]==a[i+2] and a[i+2]==a[i+3] and a[i+3]!=a[i+4]:
      a[i+1]=a[i]

  for i in range(len(a)-3):
    if a[i]!=a[i+1] and a[i+1]==a[i+2] and a[i+2]!=a[i+3]:
      a[i+1]=a[i]

  for i in range(len(a)-2):
    if a[i]!=a[i+1] and a[i+1]!=a[i+2]:
      a[i+1]=a[i]
  
  res = np.empty(len(a))
  for i in range(len(a)-1):
    if a[i]==a[i+1]:
      res[i] = a[i]

  return res


# サンプリングレート
SR = 16000

# 音声ファイルの読み込み
x, _ = librosa.load('aiueo_mic.wav', sr=SR)

#
# 短時間フーリエ変換
#

# フレームサイズ
size_frame = 2048	# 2のべき乗

# フレームサイズに合わせてハミング窓を作成
hamming_window = np.hamming(size_frame)

# シフトサイズ
size_shift = 16000 / 100	# 0.001 秒 (10 msec)

# 基本周波数を保存するlist
frequency_zero = []

# size_shift分ずらしながらsize_frame分のデータを取得
# np.arange関数はfor文で辿りたい数値のリストを返す
# 通常のrange関数と違うのは3つ目の引数で間隔を指定できるところ
# (初期位置, 終了位置, 1ステップで進める間隔)
for i in np.arange(0, len(x)-size_frame, size_shift):

  # 該当フレームのデータを取得
  idx = int(i)	# arangeのインデクスはfloatなのでintに変換
  x_frame = x[idx : idx+size_frame]
  
  # 
  # 基本周波数の推定
  # 
  # 自己相関から基本周波数を推定
  x_f0 = correlate(x_frame)

  # ゼロ交差数の計算
  x_zero_cross = zero_cross(x_frame) 

  # ゼロ交差数が閾値より大きければ無声音とみなしf0を0とする
  if x_zero_cross > 300:
    frequency_zero.append(0)
  else:
    frequency_zero.append(x_f0)

  # 計算した対数振幅スペクトログラムを配列に保存
  frequency_zero.append(x_f0)


frequency_zero = smoothing(frequency_zero)

#
# スペクトログラムを画像に表示・保存
#
plt.xlabel('sample')					# x軸のラベルを設定
plt.ylabel('frequency [Hz]')		# y軸のラベルを設定

plt.plot(frequency_zero, linewidth=10, color="0.9")
plt.ylim([0, 500])
plt.show()




