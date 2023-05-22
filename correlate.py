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

# 配列 a の index 番目の要素がピーク（両隣よりも大きい）であれば True を返す
def is_peak(a, index):
	# 
	if index == 0 or index == len(a) -1:
		return False
	else:
		return  a[index] >= max(a[index-1], a[index+1])


# サンプリングレート
SR = 16000

# 音声ファイルの読み込み
x, _ = librosa.load('a.wav', sr=SR)

# 自己相関が格納された，長さが len(x)*2-1 の対称な配列を得る
autocorr = np.correlate(x, x, 'full')

# 不要な前半を捨てる
autocorr = autocorr [len (autocorr ) // 2 : ]

# ピークのインデックスを抽出する
peakindices = [i for i in range (len (autocorr )) if is_peak (autocorr, i)]

# インデックス0 がピークに含まれていれば捨てる
peakindices = [i for i in peakindices if i != 0]

# 自己相関が最大となるインデックスを得る

max_peak_index = max (peakindices , key=lambda index: autocorr [index])

# インデックスに対応する周波数を得る
# （自分で実装すること）
max_peak_frequency = SR / max_peak_index
print(max_peak_frequency)