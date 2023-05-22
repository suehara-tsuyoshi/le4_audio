#
# 計算機科学実験及演習 4「音響信号処理」課題1 ソースコード
#
# 音声ファイルを読み込み，スペクトル，スペクトログラム，基本周波数，音量の計算と母音推定を行い図示する．
#

# ライブラリの読み込み
from tkinter.constants import LEFT, SOLID, TOP
from typing import List
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import tkinter
import subprocess

from numpy.core.fromnumeric import size
import librosa
import time

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

# ケプストラム係数を計算する関数
def cepstrum(a):
  # フレームサイズ
  size_frame = 2048 # 2のべき乗

  # フレームサイズに合わせてハミング窓を作成
  hamming_window = np.hamming(size_frame)

  # シフトサイズ
  size_shift = 16000 / 100	# 0.01 秒 (10 msec)

  # ケプストラム係数を保存するlist
  cepstrogram = []

  # size_shift分ずらしながらsize_frame分のデータを取得
  for i in np.arange(0, len(a)-size_frame, size_shift):

    # 該当フレームのデータを取得
    idx = int(i)	# arangeのインデクスはfloatなのでintに変換
    a_frame = a[idx : idx+size_frame]
    fft_spec = np.fft.rfft(a_frame * hamming_window) # 窓関数をかけた後にフーリエ変換

    # 複素スペクトログラムを対数振幅スペクトログラムに
    fft_log_abs_spec = np.log(np.abs(fft_spec))

    # 対数振幅スペクトルをケプストラム係数に(フーリエ変換により抽出)
    fft_log_abs_ceps = np.fft.fft(fft_log_abs_spec)

    # ケプストラム係数をリストに追加
    # dimension は抽出するケプストラム係数の次元(今回は13)
    cepstrogram.append(np.real(fft_log_abs_ceps[0:dimension]))

  return cepstrogram

# 各母音の対数尤度を計算する関数
# x_vowelには各母音に対応する学習データが格納
def log_likelihood(x_vowel, a):
  # 各母音に対応するケプストラム係数、その平均、その共分散(と対数をとったもの)を格納するlist
  ceps_vowel = []
  mean_vowel = []
  cov_vowel = []
  log_cov_vowel = []

  # 計算して格納
  for i in range(6):
    # 各母音のケプストラム係数
    ceps_vowel.append(cepstrum(x_vowel[i]))
    # 各母音のケプストラム係数の平均
    mean_vowel.append(np.mean(np.array(ceps_vowel[i]), axis = 0))
    # 各母音のケプストラム係数の共分散
    cov_vowel.append(np.var(np.array(ceps_vowel[i]), axis = 0))
    # 各母音のケプストラム係数の共分散の対数をとる
    log_cov_vowel.append(np.log(cov_vowel[i]) // 2)

  # 対数尤度のリスト
  L = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

  # 「あ」～「お」の対数尤度を計算
  # 確率分布は正規分布を仮定
  for d in range(dimension):
    for i in range(6):
      L[i] -= log_cov_vowel[i][d]
      L[i] -= ((a[d]-mean_vowel[i][d]) ** 2) / (2 * cov_vowel[i][d])

  # 対数尤度が最大となるインデックスを返す(0:a 1:i 2:u 3:e 4:o)
  return L.index(max(L))

# 母音推定のスムージングを行う関数
def vowel_smoothing(a):
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

# 基本周波数のスムージングを行う関数
def f_zero_smoothing(a):
  global vowel_presumption
  
  # 無音なら値を0に変換
  for i in range(len(a)):
    if vowel_presumption[i] == 0:
      a[i] = 0

  return a


# サンプリングレート
SR = 16000

# 抽出するケプストラム係数の次元
dimension = 13

# 音声ファイルの読み込み
model, _ = librosa.load('aiueo_model.wav', sr=SR)
x, _ = librosa.load('aiueo_divided.wav', sr=SR)

# 各母音に分割
model_vowel = np.array([model[0:10000], model[12000:22000], model[40000:50000], model[70000:80000], model[100000:110000], model[130000:140000]]) 

# ファイルサイズ(秒)
duration = len(x) / SR

# フレームサイズ
size_frame = 2048	# 2のべき乗

# フレームサイズに合わせてハミング窓を作成
hamming_window = np.hamming(size_frame)

# シフトサイズ
size_shift = 16000 / 100	# 0.01 秒 (10 msec)

# 基本周波数を保存するlist
frequency_zero = []

# スペクトログラムを保存するlist
spectrogram = []

# 音量を保存するlist
volume = []

# 母音のリスト
vowel = ['x', 'a', 'i', 'u', 'e', 'o']

# 推定した母音に対応するindexを保存するlist
vowel_presumption = []



# size_shift分ずらしながらsize_frame分のデータを取得
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

  # ゼロ交差数がある閾値より大きければ無声音とみなしf0を0とする
  if x_zero_cross > 300:
    frequency_zero.append(0)
  else:
    frequency_zero.append(x_f0)
  # 
  # 対数振幅スペクトログラムの計算
  # 
  # 窓掛けしたデータをFFT
  fft_spec = np.fft.rfft(x_frame * hamming_window)

  # 複素スペクトログラムを対数振幅スペクトログラムに
  fft_log_abs_spec = np.log(np.abs(fft_spec))

  # 計算した対数振幅スペクトログラムを配列に保存
  spectrogram.append(fft_log_abs_spec[0:128])

  # 音量(デシベルに変換)
  vol = 20 * np.log10(np.mean(x_frame ** 2))
  volume.append(vol)

  # 対数振幅スペクトルをケプストラム係数に
  fft_log_abs_ceps = np.fft.fft(fft_log_abs_spec)

  # 母音推定
  recognition_id = log_likelihood(model_vowel, (np.real(fft_log_abs_ceps[0:dimension])))
  vowel_presumption.append(recognition_id)


# 母音推定のリストのスムージング
vowel_presumption = vowel_smoothing(vowel_presumption)

# 基本周波数のリストのスムージング
frequency_zero = f_zero_smoothing(frequency_zero)

# Tkinterを初期化
# メインフレーム
root = tkinter.Tk()
root.title(u"EXP4-AUDIO-TASK1")
root.geometry("1000x600")

# Tkinterのウィジェットを階層的に管理するためにFrameを使用
# frame1 ... スペクトログラムを表示
# frame2 ... 表示するグラフの選択画面を表示
# frame3 ... Scale（スライドバー）とスペクトルを表示
frame1 = tkinter.Frame(root, bd = 1, relief = SOLID, width = 800, height = 500, background="white", pady = 10)
frame1.pack(side=tkinter.LEFT, padx = 10, pady=10)

frame2 = tkinter.Frame(root, bd = 1, relief = SOLID, width = 600, height = 200, padx = 50, background="white")
frame2.pack(side=tkinter.TOP, padx = 10, pady=10)

frame3 = tkinter.Frame(root, bd = 1, relief = SOLID, width = 600, height = 800, background="white")
frame3.pack(side=tkinter.TOP, padx = 10)

# 
# frame1の内容
# 

# 再生ボタンの作成

# 再生ボタンが押された際に呼び出されるコールバック関数
def play_back():
  cmd = "sox aiueo_divided.wav -d"
  p = subprocess.Popen(cmd, shell=True)

# 再生ボタンの描画
bt_pb = tkinter.Button(frame1, text="再生", font=("Times New Roman", 20), background="white", command = play_back)
bt_pb.pack(side=TOP)

# スペクトログラムの描画
fig, ax1 = plt.subplots()
# 2つ目の縦軸を作成
ax2 = ax1.twinx()
# masterに対象とするframeを指定
canvas = FigureCanvasTkAgg(fig, master=frame1)	
# ラベル設定
ax1.set_xlabel('sec')
ax1.set_ylabel('frequency [Hz]')
# 表示の設定
ax1.imshow(
	np.flipud(np.array(spectrogram).T),
	extent=[0, duration, 0, 1000],
	aspect='auto',
	interpolation='nearest'
)
# 最後にFrameに追加する処理
canvas.get_tk_widget().pack(side="left")	

# ラジオボタン「基本周波数」が押された際に呼び出されるコールバック関数
def draw_frequency():
  ax2.cla()
  ax2.set_ylabel('frequency [Hz]')
  x_data = np.linspace(0, duration, len(frequency_zero))
  ax2.plot(x_data, frequency_zero, c = '#ff7f00')
  ax2.set_ylim([0, 1000])
  canvas.draw()

# ラジオボタン「音量」が押された際に呼び出されるコールバック関数
def draw_volume():
  ax2.cla()
  ax2.set_ylabel('volume [db]')
  x_data = np.linspace(0, duration, len(volume))
  ax2.plot(x_data, volume, c = '#ff7f00')
  canvas.draw()

# ラジオボタン「母音推定」が押された際に呼び出されるコールバック関数
def draw_vowel():
  ax2.cla()
  ax2.set_ylabel('vowel')
  x_data = np.linspace(0, duration, len(vowel_presumption))
  ax2.plot(x_data, vowel_presumption, c = '#ff7f00')
  ax2.set_yticks([0, 1, 2, 3, 4, 5])
  ax2.set_yticklabels(['x', 'a', 'i', 'u', 'e', 'o'])
  canvas.draw()

# 
# frame2の内容
# 

# ラベルの生成
label = tkinter.Label(frame2, text="表示するグラフの選択", font=("Times New Roman", 24), anchor="w", background="white")

# ラジオボタンで使用する値の変数を生成
radiovalue = tkinter.IntVar()

# ラジオボタン「基本周波数」の描画
bt_f0 = tkinter.Radiobutton(frame2, variable=radiovalue, value=1, text="基本周波数", font=("Times New Roman", 20), width=50, anchor="w", background="white", command = draw_frequency)
# ラジオボタン「音量」の描画
bt_power = tkinter.Radiobutton(frame2, variable=radiovalue, value=2, text="音量", font=("Times New Roman", 20), width = 50, anchor="w", background="white", command = draw_volume)
# ラジオボタン「母音推定」の描画
bt_vowel = tkinter.Radiobutton(frame2, variable=radiovalue, value=3, text="母音推定", font=("Times New Roman", 20), width = 50, anchor="w", background="white", command = draw_vowel)

label.pack(pady = 10)
bt_f0.pack()
bt_power.pack()
bt_vowel.pack()

# 
# frame3の内容
# 

# スペクトルを表示する領域を確保
# ax2, canvs2 を使って上記のコールバック関数でグラフを描画する
fig2 = plt.figure(figsize=(5, 5), dpi = 60)
ax3 = fig2.add_subplot(111)
ax3.set_ylabel('amblitude')
ax3.set_xlabel('frequency [Hz]')
ax3.set_title("Spectrum")
canvas2 = FigureCanvasTkAgg(fig2, master=frame3)

# スライドバーで選択された時間の基本周波数、音量、母音推定の結果を表示するラベルを作成
label2 = tkinter.Label(frame3, width=100, text=u"基本周波数:   Hz  音量:   db  母音推定:   " , font=("Times New Roman", 20), background="white")

# スライドバーで選択された時間に対応する配列のインデックスを格納する変数
selected_time = 0

# 選択された時間に対応するインデックスを更新する関数
def update_selected_time(v):
  global selected_time
  selected_time =  int(len(frequency_zero)*float(v)/duration)
  # インデックスが範囲内に収まるようにする
  selected_time = min(max(0, selected_time), len(frequency_zero)-1)

# スライドバーの値が変更されたときに呼び出されるコールバック関数
# vはスライドバーの値
def _draw_graph(v):
  # スライドバーの値が変更されたらselected_timeの値を更新
  update_selected_time(v)
  # スライドバーの値が変更されたらラベルの内容を更新
  label2["text"] = "基本周波数: %s Hz  音量: %s db  母音推定: %s" % (int(frequency_zero[selected_time]), int(volume[selected_time]), vowel[vowel_presumption[selected_time]])
  # スライドバーの値からスペクトルのインデクスおよびそのスペクトルを取得
  index = int((len(spectrogram)-1) * (float(v) / duration))

  # 直前のスペクトル描画を削除し，新たなスペクトルを描画
  ax3.cla()
  ax3.set_ylabel('amblitude')
  ax3.set_xlabel('frequency [Hz]')
  x_data = np.linspace(0, 1000, len(spectrogram[index]))
  ax3.plot(x_data, spectrogram[index])
  ax3.set_ylim(-10, 5)
  ax3.set_xlim(0, 1000)
  ax3.set_title("Spectrum")
  canvas2.draw()

# スライドバーを作成
scale = tkinter.Scale(
	command=_draw_graph,		# ここにコールバック関数を指定
	master=frame3,				# 表示するフレーム
  background="white",
	from_=0,					# 最小値
	to=duration,				# 最大値
	resolution=size_shift/SR,	# 刻み幅
	label=u'スペクトルを表示する時間[sec]',
	orient=tkinter.HORIZONTAL,	# 横方向にスライド
	length=600,					# 横サイズ
	width=30,					# 縦サイズ
	font=("", 20)				# フォントサイズは20pxに設定
)
scale.pack(side="top")

label2.pack(side=TOP, pady=8)

# "top"は上部方向にウィジェットを積むことを意味する
canvas2.get_tk_widget().pack(side="right")	

root.mainloop()





