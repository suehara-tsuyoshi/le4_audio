#
# 計算機科学実験及演習 4「音響信号処理」
# サンプルソースコード
#
# 音声ファイルを読み込み，スペクトログラムを計算して図示する．
#

# ライブラリの読み込み
import matplotlib.pyplot as plt
import numpy as np
import librosa

# 配列 a の index 番目の要素がピーク（両隣よりも大きい）であれば True を返す
def is_peak(a, index):
	# 
	if index == 0 or index == len(a) -1:
		return False
	else:
		return  a[index] >= max(a[index-1], a[index+1])

def correlate(a):
  # 自己相関が格納された，長さが len(x)*2-1 の対称な配列を得る
  autocorr = np.correlate(a, a, 'full')

  # 不要な前半を捨てる
  autocorr = autocorr [len (autocorr ) // 2 : ]

  # ピークのインデックスを抽出する
  peakindices = [i for i in range (len (autocorr )) if is_peak (autocorr, i)]

  # インデックス0 がピークに含まれていれば捨てる
  peakindices = [i for i in peakindices if i != 0]

  
  # 返り値の初期化
  max_peak_frequency = 8000.0

  if len(peakindices) != 0:
    # 自己相関が最大となるインデックスを得る
    max_peak_index = max (peakindices , key=lambda index: autocorr [index])
    # インデックスに対応する周波数を得る
    max_peak_frequency = SR / max_peak_index
  
  return max_peak_frequency

# サンプリングレート
SR = 16000

# 音声ファイルの読み込み
x, _ = librosa.load('aiueo.wav', sr=SR)

#
# 短時間フーリエ変換
#

# フレームサイズ
size_frame = 512	# 2のべき乗

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

  # 自己相関から基本周波数を推定
  x_f0 = correlate(x_frame)

  # 計算した対数振幅スペクトログラムを配列に保存
  frequency_zero.append(x_f0)


#
# スペクトログラムを画像に表示・保存
#
plt.xlabel('sample')					# x軸のラベルを設定
plt.ylabel('frequency [Hz]')		# y軸のラベルを設定

plt.plot(frequency_zero)
plt.ylim([0, 300])
plt.show()




