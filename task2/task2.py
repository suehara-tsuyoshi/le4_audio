#
# 計算機科学実験及演習 4「音響信号処理」
# 課題2ソースコード
#
# 簡易カラオケシステム
#
# mp3ファイルを別スレッドで再生しつつ
# マイクからの音声入力に対してスペクトログラムとパワーを計算して表示する
# 上記をリアルタイムで逐次処理を行う
#

# ライブラリの読み込み
import math
from tkinter.constants import SOLID, TOP
from xml.dom.minidom import Element
from openpyxl import Workbook
import pyaudio
import numpy as np
import threading
import time

# matplotlib関連
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# GUI関連
import tkinter
from matplotlib.backends.backend_tkagg import (
	FigureCanvasTkAgg, NavigationToolbar2Tk)

# mp3ファイルを読み込んで再生
from pydub import AudioSegment
from pydub.utils import make_chunks

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
    max_peak_frequency = SAMPLING_RATE / max_peak_index
  
  return max_peak_frequency

# 音高の配列を受け取り，ビブラートの有無を判定する関数
def is_viv(a) :
  global is_viv_state
  # 音高の最大値と最小値の差(レンジ)を計算
  dif = np.amax(a) - np.amin(a)
  # 音高の平均値を計算
  f_mean = np.mean(a, dtype=float)
  # 音高の平均値のラインと交差する回数を保存
  mc = 0

  # 配列内で連続する2つの値と音高の平均値との大小関係が異なれば交差しているとし，mcの値を１増やす
  for i in range(len(a) - 1):
    if(
      (a[i] > f_mean and a[i+1] < f_mean) or
      (a[i] < f_mean and a[i+1] > f_mean)
    ):
      mc += 1
  
  # 音高のレンジが閾値以下かつmcの値が閾値以上であればビブラートであると判定
  if dif <= 50 and mc >= 5:
    # ただしビブラートの連続したカウントを防ぐためis_viv_stateを用いて直前のフレームでビブラートが検出されていなければTrueを返す
    if is_viv_state == False :
      is_viv_state = True
      return True
    else :
      return False
  # ビブラートでなければis_viv_stateの値を1にしてFalseを返す
  else :
    is_viv_state = False
    return False

# 音高の配列を受け取り，しゃくり(グリスアップ)の有無を判定する関数
def is_shakuri(a):
  global is_shakuri_state
  # フレーズの始まりであるかの判定
  if np.all(a[0:4]) or np.any(a[5:] == 0):
    is_shakuri_state = False
    return False
  # 音高の最小値と最大値，そのノートナンバーの差(レンジ)を計算
  f_min = np.amin(a[5:])
  f_max = np.amax(a[5:])
  dif = hz2nn(f_max) - hz2nn(f_min)
  
  # フレーズの最初の音高が最小値と等しいかを判定
  # 配列の最後の音高が最大値と等しいかを判定
  if f_max != a[-1] or f_min != a[5]:
    is_shakuri_state = False
    return False

  # ノートナンバーにして2以上4以下の音高の上昇がなければFalse
  if dif < 2 or dif > 4: 
    is_shakuri_state = False
    return False
  
  # しゃくりの連続カウントを防ぐためis_shakuri_stateを用いて直前のフレームでしゃくりが検出されていなければTrueを返す
  if is_shakuri_state == True:
    return False
  else :
    is_shakuri_state = True
    return True

# ノートナンバーから周波数に変換する関数
def nn2hz(notenum):
  global EPSILON
  # ノートナンバーがある範囲に入っていなければ0を返す
  if notenum <= 40 or notenum >= 75:
    return 0.0
  # ノートナンバーを対応する周波数に変換して返す
  else : 
    return 440.0 * (2.0 ** ((notenum - 69) / 12.0))

# 周波数からノートナンバーに変換する関数
def hz2nn(frequency):
  return int (round (12.0 * ((math.log((frequency + EPSILON) / 440.0)) / math.log(2.0)))) + 69


# スムージングを行う関数
def smoothing(a):
  # XYXならXXXに変換
  for i in range(len(a)-2):
    if a[i+1] > 0 and a[i]!=a[i+1] and a[i+1]!=a[i+2]:
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

  # XYXならXXXに変換
  for i in range(len(a)):
    if a[i] == 0 or a[i] > 500:
      a[i]=np.nan

  return a

# 音程の正解率を計算する関数
# ガイドメロディの音程と入力音声の音高を比較
def calc_similarity(x, y):
  global interval_correct_rate
  # ガイドメロディの音程が不定の部分は現在の正解率をそのまま返す
  if y == 0:
    return interval_correct_rate
  
  # 入力音声の音高が明らかに異常であれば現在の正解率をそのまま返す
  if x < 50 or x > 500:
    return interval_correct_rate
  # ガイドメロディの音程のノートナンバーと入力音声の音高のノートナンバーを比較
  # 絶対値が1違う毎に10%ずつ正解率を引いた値を返す
  else:
    x_nn = hz2nn(x)
    y_nn = hz2nn(y) 
    res = max(0, 100 - abs(x_nn-y_nn)*10)
    return res

# 採点結果を算出する関数
def calc_score():
  global interval_correct_rate, vibrato_count, shakuri_count, interval_score, technique_score, final_score, now_playing_sec
  # 音程正解率を0.9倍した値を音程の得点とする(90点満点)
  interval_score = int(interval_correct_rate * 0.9)

  # 5秒に1回ビブラートあるいはしゃくりがあれば満点となる基準で技術の得点を設定
  # ただし，音程正解率を乗算しており音程の正確さも考慮に入れるようにした
  technique_score = min(10, int(interval_correct_rate*(shakuri_count+vibrato_count)/(2*now_playing_sec)))

  # 音程の得点と技術の得点の合計が最終スコア
  final_score =  min(100, interval_score+technique_score)


# サンプリングレート
SAMPLING_RATE = 16000

# フレームサイズ
FRAME_SIZE = 2048

# サイズシフト
SHIFT_SIZE = int(SAMPLING_RATE / 50)	# 今回は0.02秒

# スペクトルをカラー表示する際に色の範囲を正規化するために
# スペクトルの最小値と最大値を指定
# スペクトルの値がこの範囲を超えると，同じ色になってしまう
SPECTRUM_MIN = -5
SPECTRUM_MAX = 1

# 音量を表示する際の値の範囲
VOLUME_MIN = -200
VOLUME_MAX = -60

# log10を計算する際に，引数が0にならないようにするためにこの値を足す
EPSILON = 1e-10

# ハミング窓
hamming_window = np.hamming(FRAME_SIZE)

# グラフに表示する縦軸方向のデータ数
MAX_NUM_SPECTROGRAM = int(FRAME_SIZE / 2)

# グラフに表示する横軸方向のデータ数
NUM_DATA_SHOWN = 100

# 周波数表示の拡大比率
SCALE_UP_RATE = 16

# GUIの開始フラグ（まだGUIを開始していないので、ここではFalseに）
is_gui_running = False

# 楽曲の開始フラグ(まだ楽曲を開始していないので、ここではFalseに)
is_music_running = False

# 直前のフレームでビブラートがあったかどうかを保持(False:なし True:あり)
is_viv_state = False

# 直前のフレームでしゃくりがあったかどうかを保持(False:なし True:あり)
is_shakuri_state = False

# ビブラートの回数
vibrato_count = 0

# しゃくりの回数
shakuri_count= 0

# 音程正解率
interval_correct_rate = 0

# 各フレームの音程正解率の総和(平均を求める際に使用)
interval_correct_sum = 0

# 音声の最低音高
frequency_zero_min = 1e-10

# 音声の最高音高
frequency_zero_max = 1e-10

# 音程の得点
interval_score = 0

# 技術の得点
technique_score = 0

# 最終スコア
final_score = 0

#
# (1) GUI / グラフ描画の処理
#

# ここでは matplotlib animation を用いて描画する
# 毎回 figure や ax を初期化すると処理時間がかかるため
# データを更新したら，それに従って必要な部分のみ再描画することでリアルタイム処理を実現する

# matplotlib animation によって呼び出される関数
# ここでは最新のスペクトログラムと音量のデータを格納する
# 再描画はmatplotlib animationが行う
def animate(frame_index):

	# ax1を用いて楽曲のスペクトログラム，ガイドメロディ，音声の音高を表示
  ax1_sub_1.set_array(spectrogram_data_music)
  ax1_sub_2.set_data(time_x_data, guide_melody)
  ax1_sub_3.set_data(time_x_data, frequency_zero_data_for_graph)
  # ax2を用いて音声の音量を表示
  ax2_sub.set_data(time_x_data, volume_data) 
  return ax1_sub_1, ax1_sub_2, ax1_sub_3, ax2_sub



# GUIで表示するための処理（Tkinter）
root = tkinter.Tk()
root.wm_title("EXP4-AUDIO-TASK2")
root.geometry("1920x1080")

# フレーム1
frame1 = tkinter.Frame(root, bd = 1, relief = SOLID, width = 1300, height = 1000, background="white", pady = 10)
frame1.propagate(False)
frame1.pack(side=tkinter.LEFT, padx = (5, 0), pady=10)

# フレーム2
frame2 = tkinter.Frame(root, bd = 1, relief = SOLID, width = 590, height = 400, padx = 10, background="white")
frame2.propagate(False)
frame2.pack(side=tkinter.TOP, padx = (5, 5), pady=(25, 5))

# フレーム3
frame3 = tkinter.Frame(root, bd = 1, relief = SOLID, width = 590, height = 590, background="white")
frame3.propagate(False)
frame3.pack(side=tkinter.TOP, padx = 5, pady = 5)




# 横軸の値のデータ
time_x_data = np.linspace(0, NUM_DATA_SHOWN * (SHIFT_SIZE/SAMPLING_RATE), NUM_DATA_SHOWN)
# 縦軸の値のデータ
freq_y_data = np.linspace(8000/MAX_NUM_SPECTROGRAM, 8000/SCALE_UP_RATE, MAX_NUM_SPECTROGRAM//SCALE_UP_RATE)
# 縦軸の値のデータ
freq_y_data_full = np.linspace(8000/MAX_NUM_SPECTROGRAM, 8000, MAX_NUM_SPECTROGRAM)


# とりあえず初期値（ゼロ）のスペクトログラム，音量のデータ，音高のデータ，グラフ用の音高データ，スペクトルのデータ，スペクトル包絡のデータを作成
# この numpy array にデータが更新されていく
spectrogram_data = np.zeros((len(freq_y_data), len(time_x_data)))
volume_data = np.zeros(len(time_x_data))
frequency_zero_data = np.zeros(len(time_x_data))
frequency_zero_data_for_graph = np.zeros(len(time_x_data))
spectrum_data = np.zeros(len(freq_y_data_full))
spectrum_envelope = np.zeros(len(freq_y_data_full))

# 楽曲のスペクトログラム，基本周波数，スペクトル，スペクトル包絡を格納するデータ
spectrogram_data_music = np.zeros((len(freq_y_data), len(time_x_data)))
spectrum_data_music = np.zeros(len(freq_y_data_full))
spectrum_envelope_music = np.zeros(len(freq_y_data_full))

# グラフとして表示するガイドメロディを格納するデータ
guide_melody = np.zeros(len(time_x_data))

# 楽曲「lemon」のガイドメロディを基本周波数の値で格納するデータ
lemon_data = []

# ノートナンバーによってガイドメロディを作成するための関数
def add_to_guide(num, count):
  global lemon_data
  # 第1引数のノートナンバーを周波数に変換し第2引数の数だけガイドメロディとして加える
  new = np.full(count, nn2hz(num))
  if num == 0 : new = np.full(count, 0)
  lemon_data = np.concatenate([lemon_data, new])

# 以下は楽曲「lemon」のガイドメロディを楽譜から作成
add_to_guide(59, 2)
add_to_guide(61, 12)
add_to_guide(63, 12)
add_to_guide(59, 6)
add_to_guide(56, 14)
add_to_guide(0, 16)
add_to_guide(61, 19)
add_to_guide(58, 10)
add_to_guide(54, 13)
add_to_guide(51, 14)
add_to_guide(0, 11)
add_to_guide(58, 18)
add_to_guide(56, 14)
add_to_guide(0, 4)
add_to_guide(54, 7)
add_to_guide(0, 9)
add_to_guide(47, 7)
add_to_guide(0, 16)
add_to_guide(54, 10)
add_to_guide(51, 18)
add_to_guide(0, 27)
add_to_guide(49, 8)
add_to_guide(0, 12)
add_to_guide(51, 5)
add_to_guide(52, 20)
add_to_guide(0, 26)
add_to_guide(59, 9)
add_to_guide(58, 8)
add_to_guide(59, 6)
add_to_guide(54, 27)
add_to_guide(0, 9)
add_to_guide(52, 9)
add_to_guide(0, 11)
add_to_guide(51, 6)
add_to_guide(52, 8)
add_to_guide(0, 9)
add_to_guide(53, 24)
add_to_guide(0, 10)
add_to_guide(59, 8)
add_to_guide(0, 3)
add_to_guide(58, 9)
add_to_guide(56, 5)
add_to_guide(0, 3)
add_to_guide(55, 18)
add_to_guide(0, 39)
add_to_guide(59, 2)
add_to_guide(61, 12)
add_to_guide(63, 12)
add_to_guide(59, 6)
add_to_guide(56, 16)
add_to_guide(0, 14)
add_to_guide(61, 19)
add_to_guide(58, 13)
add_to_guide(54, 13)
add_to_guide(51, 11)
add_to_guide(0, 11)
add_to_guide(58, 18)
add_to_guide(56, 14)
add_to_guide(0, 4)
add_to_guide(54, 7)
add_to_guide(0, 9)
add_to_guide(47, 7)
add_to_guide(0, 16)
add_to_guide(54, 10)
add_to_guide(51, 18)
add_to_guide(0, 24)
add_to_guide(49, 11)
add_to_guide(51, 11)
add_to_guide(52, 25)
add_to_guide(0, 7)
add_to_guide(54, 9)
add_to_guide(52, 15)
add_to_guide(54, 13)
add_to_guide(51, 25)
add_to_guide(54, 12)
add_to_guide(59, 17)
add_to_guide(63, 21)
add_to_guide(61, 50)
add_to_guide(59, 70)
add_to_guide(0, 300)


# スペクトログラムを描画する際に横軸と縦軸のデータを行列にしておく必要がある
# これは下記の matplotlib の pcolormesh の仕様のため
X = np.zeros(spectrogram_data.shape)
Y = np.zeros(spectrogram_data.shape)
for idx_f, f_v in enumerate(freq_y_data):
	for idx_t, t_v in enumerate(time_x_data):
		X[idx_f, idx_t] = t_v
		Y[idx_f, idx_t] = f_v
#
# フレーム1のグラフ表示部分
#
# スペクトログラムを描画
fig, ax1 = plt.subplots(1, 1)
canvas = FigureCanvasTkAgg(fig, master=frame1)

# pcolormeshを用いてスペクトログラムを描画
# 戻り値はデータの更新 & 再描画のために必要
ax1_sub_1 = ax1.pcolormesh(
	X,
	Y,
	spectrogram_data,
	shading='nearest',	# 描画スタイル
	cmap='jet',			# カラーマップ
	norm=Normalize(SPECTRUM_MIN, SPECTRUM_MAX)	# 値の最小値と最大値を指定して，それに色を合わせる
)

# 音量を表示するために反転した軸を作成
ax2 = ax1.twinx()

# ガイドメロディ，音声の音高，音量をプロットする
# 戻り値はデータの更新 & 再描画のために必要
ax1_sub_2, = ax1.plot(time_x_data, guide_melody, c= '0.9', linewidth=5)
ax1_sub_3, = ax1.plot(time_x_data, frequency_zero_data_for_graph, c='y', linewidth=5)
ax2_sub, = ax2.plot(time_x_data, volume_data, c='b')

# ラベルの設定
ax1.set_xlabel('sec')				# x軸のラベルを設定
ax1.set_ylabel('frequency [Hz]')	# y軸のラベルを設定
ax2.set_ylabel('volume [dB]')		# 反対側のy軸のラベルを設定

# 音高と音量を表示する際の値の範囲を設定
ax1.set_ylim([0, 500])
ax2.set_ylim([VOLUME_MIN, VOLUME_MAX])
plt.grid(linewidth=0.5)

# maplotlib animationを設定
ani = animation.FuncAnimation(
	fig,
	animate,		# 再描画のために呼び出される関数
	interval=400,	# 100ミリ秒間隔で再描画を行う（PC環境によって処理が追いつかない場合はこの値を大きくするとよい）
	blit=True		# blitting処理を行うため描画処理が速くなる
)

#
# フレーム1の再生時間と歌詞の表示部分
#

# 終了ボタンが押されたときに呼び出される関数
# ここではGUIを終了する
def _quit():
	root.quit()
	root.destroy()

# 再生時間と歌詞を表示する領域のフレームをフレーム1内に作成
frame1_sub = tkinter.Frame(master=frame1, width=1050, height=120, background="white")
frame1_sub.propagate(False)

# 再生時間を表示するラベルフレーム
playing_sec_frame = tkinter.LabelFrame(master=frame1_sub, text="再生時間", font=("Arial", 25), width=350, height=120, background="white")
playing_sec_frame.propagate(False)
playing_sec_label = tkinter.Label(master=playing_sec_frame, text="0.0 sec", font=("Times New Roman", 30), background="white")

# 終了ボタンを作成
bt_fin = tkinter.Button(master=playing_sec_frame, text="終了", command=_quit, font=("", 25))

# 歌詞を表示するラベルフレーム
lyrics_frame = tkinter.LabelFrame(master=frame1_sub, text="歌詞", font=("Arial", 25), width=600, height=120, background="white")
lyrics_frame.propagate(False)
lyrics_label = tkinter.Label(master=lyrics_frame, text="", font=("", 30), background="white")

#
# フレーム2
#
# 音域を表示するラベルフレーム
range_frame = tkinter.LabelFrame(master=frame2, text="音域", font=("Arial", 25), width=640, height=120, background="white")
range_frame.propagate(False)
range_label = tkinter.Label(master=range_frame, text="最低音:   Hz  最高音:   Hz", font=("Times New Roman", 30), background="white")

# テクニック(ビブラートとしゃくりの回数)を表示するラベルフレーム
technique_frame = tkinter.LabelFrame(master=frame2, text="テクニック", font=("Arial", 25), width=640, height=120, background="white")
technique_frame.propagate(False)
technique_label = tkinter.Label(master=technique_frame, text="ビブラート:   回  しゃくり:   回", font=("Times New Roman", 30), background="white")

# 音程の正解率と採点結果を表示するラベルフレーム
interval_frame = tkinter.LabelFrame(master=frame2, text="音程", font=("Arial", 25), width=640, height=120, background="white")
interval_frame.propagate(False)
interval_label = tkinter.Label(master=interval_frame, text="音程正解率:   ％", font=("Times New Roman", 30), background="white")

# 「採点結果を表示」のボタンが押されたときに呼び出される関数
# 最終スコアを表示
def show_score():
  global interval_correct_rate, interval_score, technique_score, final_score
  bt_score.pack_forget()
  interval_label.pack_forget()
  calc_score()
  interval_frame["text"] = "採点結果"
  interval_label["padx"] = 5
  interval_label["text"] = "音程: %s 点 +  技術: %s 点 = %s 点" % (interval_score, technique_score, final_score)
  interval_label.pack()
  return 

# 採点結果を表示するボタン
bt_score = tkinter.Button(interval_frame, text="採点結果", font=("Times New Roman", 20), background="white", command = show_score, padx=10, state=tkinter.DISABLED)

frame1_sub.pack(side=TOP, pady=(10, 0))
playing_sec_frame.pack(side="left", padx=(30, 15))
playing_sec_label.pack(side="left", padx=(30, 15))
bt_fin.pack(side="right", padx=(15, 30))
lyrics_frame.pack(side="right", padx=(15, 30))
lyrics_label.pack()
range_frame.pack(side=TOP, pady= (10, 5))
range_label.pack()
technique_frame.pack(side=TOP, pady= 5)
technique_label.pack()
interval_frame.pack(side=TOP, pady= (5, 10))
interval_label.pack(side="left", padx=(50, 10))
bt_score.pack(side="right", padx=(20, 50))
canvas.get_tk_widget().pack(side="left")

# 音声のスペクトル，音声と楽曲(ボーカル)のスペクトル包絡を描画する関数
def draw_graph():
  global is_gui_running, is_music_running
  while is_gui_running and is_music_running:
    ax3.cla()
    ax4.cla()
    
    ax3.plot(freq_y_data_full, spectrum_data, c='b')
    ax3.plot(freq_y_data_full, spectrum_envelope, c='r')
    ax3.set_ylabel('amblitude')
    ax3.set_xlabel('frequency [Hz]')
    ax3.set_title("Spectrum")
    ax3.set_ylim(-10, 5)
    ax3.set_xlim(0, 2000)

    ax4.plot(freq_y_data_full, spectrum_envelope_music, c='green')
    ax4.set_ylim(-10, 5)
    ax4.set_xlim(0, 2000)
    canvas2.draw()
    time.sleep(1.0)

# スペクトルを表示する領域を確保
# ax3, canvs2 を使って上記の関数でグラフを描画する
fig2 = plt.figure(figsize=(5, 5), dpi = 60)
canvas2 = FigureCanvasTkAgg(fig2, master=frame3)
canvas2.get_tk_widget().pack()
ax3 = fig2.add_subplot(111)
ax3.plot(freq_y_data_full, spectrum_data, c='b')
ax3.plot(freq_y_data_full, spectrum_envelope, c='r')
ax3.set_ylabel('amblitude')
ax3.set_xlabel('frequency [Hz]')
ax3.set_title("Spectrum")
ax3.set_ylim(-10, 5)
ax3.set_xlim(0, 2000)

ax4 = ax3.twinx()
ax4.plot(freq_y_data_full, spectrum_envelope_music, c='green')
ax4.set_ylim(-10, 5)
ax4.set_xlim(0, 2000)
canvas2.draw()



#
# (2) マイク入力のための処理
#

# 入力音声データ
x_stacked_data = np.array([])

# 表示するガイドメロディのインデックス
idx_guide = 0

# ガイド表示の開始フラグ
is_guide_start = False

# フレーム毎に呼び出される関数
def input_callback(in_data, frame_count, time_info, status_flags):

  # この関数は別スレッドで実行するため
  # メインスレッドで定義した以下の２つの numpy array を利用できるように global 宣言する
  global x_stacked_data, spectrogram_data, volume_data, frequency_zero_data, frequency_zero_data_for_graph, vibrato_count, shakuri_count, spectrum_data, now_playing_sec, spectrum_envelope, lemon_data, guide_melody, idx_guide, is_guide_start, now_playing_sec, frequency_zero_min, frequency_zero_max, interval_correct_rate, interval_correct_sum

  # 現在のフレームの音声データをnumpy arrayに変換
  x_current_frame = np.frombuffer(in_data, dtype=np.float32)

  # 現在のフレームとこれまでに入力されたフレームを連結
  x_stacked_data = np.concatenate([x_stacked_data, x_current_frame])

  # 抽出するケプストラム係数の次元
  dimension = 13


  # フレームサイズ分のデータがあれば処理を行う
  if len(x_stacked_data) >= FRAME_SIZE:
    
    # フレームサイズからはみ出した過去のデータは捨てる
    x_stacked_data = x_stacked_data[len(x_stacked_data)-FRAME_SIZE:]
    
    # スペクトルを計算
    fft_spec = np.fft.rfft(x_stacked_data * hamming_window)
    fft_log_abs_spec = np.log10(np.abs(fft_spec) + EPSILON)[:-1]

    # スペクトル包絡を抽出
    spectrum_data = np.log(np.abs(fft_spec) + EPSILON)[:-1]
    fft_log_abs_ceps = np.fft.fft(spectrum_data)
    fft_log_abs_ceps[dimension:len(fft_log_abs_ceps)-dimension] = 0
    spectrum_envelope = (np.fft.ifft(fft_log_abs_ceps)).real
    

    # ２次元配列上で列方向（時間軸方向）に１つずらし（戻し）
    # 最後の列（＝最後の時刻のスペクトルがあった位置）に最新のスペクトルデータを挿入
    spectrogram_data = np.roll(spectrogram_data, -1, axis=1)
    spectrogram_data[:, -1] = fft_log_abs_spec[0:(FRAME_SIZE//2)//SCALE_UP_RATE]

    # 音量も同様の処理
    vol = 20 * np.log10(np.mean(x_current_frame ** 2) + EPSILON)
    volume_data = np.roll(volume_data, -1)
    volume_data[-1] = vol

    # 基本周波数も同様の処理
    # x_f0は自己相関によって推定された基本周波数，x_f0_for_graphはそれを正規化したもの
    x_f0 = correlate(x_stacked_data)
    nn = hz2nn(x_f0)
    x_f0_for_graph = nn2hz(nn)
    frequency_zero_data = np.roll(frequency_zero_data, -1)
    frequency_zero_data_for_graph = np.roll(frequency_zero_data_for_graph, -1)

    # 音量がある閾値より小さければ無音であるとみなす
    if vol < -80 : 
      frequency_zero_data[-1] = 0
      frequency_zero_data_for_graph[-1] = 0
    # 発声があった際に行う処理
    else : 
      frequency_zero_data[-1] = x_f0
      frequency_zero_data_for_graph[-1] = x_f0_for_graph

      # 最低音高と最高音高を更新
      # 音高がある範囲に入っていれば一番最初は無条件でその値にし，それ以降は比較により更新するかどうかを決定
      if frequency_zero_min == 1e-10 and x_f0_for_graph > 50 and x_f0_for_graph < 500:
        frequency_zero_min = x_f0_for_graph
        frequency_zero_max = x_f0_for_graph
      elif x_f0_for_graph > 50 and x_f0_for_graph < 500:
        frequency_zero_min = min(frequency_zero_min, x_f0_for_graph)
        frequency_zero_max = max(frequency_zero_max, x_f0_for_graph)
      
      # ビブラートの判定
      # 直近0.5秒間の入力音声に対して判定
      if is_viv(frequency_zero_data[75:]) :
        vibrato_count += 1  
      
      # しゃくりの判定
      # 直近0.5秒間の入力音声に対して判定
      if is_shakuri(frequency_zero_data[75:]):
        shakuri_count += 1

    # 再生時間が3.95秒になったらガイドメロディを開始
    if now_playing_sec >= 3.95 : is_guide_start = True

    # ガイドメロディが始まっていれば処理を行う
    if is_guide_start:
    # 配列上で１つずらし（戻し）
    # 最後の列（＝最後の時刻のスペクトルがあった位置）に最新のガイドメロディデータを挿入
      guide_melody = np.roll(guide_melody, -1)
      guide_melody[-1] = lemon_data[idx_guide]
      # ガイドメロディの値が0であればグラフには表示しないようにする
      if guide_melody[-1] == 0 : guide_melody[-1] = np.nan
      idx_guide += 1
      # ガイドメロディと音声の音高から音程正解率を算出
      # interval_correct_sumに得られた値を加える
      interval_correct_sum += calc_similarity(x_f0, lemon_data[idx_guide])
      # interval_correct_sumをidx_guideで割り，音程正解率の平均値を算出
      interval_correct_rate = int(interval_correct_sum/idx_guide)  

  # グラフのスムージング処理
  frequency_zero_data_for_graph = smoothing(frequency_zero_data_for_graph)
  
	# 戻り値は pyaudio の仕様に従うこと
  return None, pyaudio.paContinue

# マイクからの音声入力にはpyaudioを使用
# ここではpyaudioの再生ストリームを作成
# 【注意】シフトサイズごとに指定された関数が呼び出される
p = pyaudio.PyAudio()
stream = p.open(
	format = pyaudio.paFloat32,
	channels = 1,
	rate = SAMPLING_RATE,
	input = True,						# ここをTrueにするとマイクからの入力になる 
	frames_per_buffer = SHIFT_SIZE,		# シフトサイズ
	stream_callback = input_callback	# ここでした関数がマイク入力の度に呼び出される（frame_per_bufferで指定した単位で）
)


#
# (3) wavファイル音楽を再生する処理
#

# wavファイル名
filename = 'lemon_vocal_1.wav'

# pydubを使用して音楽ファイルを読み込む
audio_data = AudioSegment.from_wav(filename)

# 音声ファイルの再生にはpyaudioを使用
# ここではpyaudioの再生ストリームを作成
p_play = pyaudio.PyAudio()
stream_play = p_play.open(
	format = p.get_format_from_width(audio_data.sample_width),	# ストリームを読み書きするときのデータ型
	channels = audio_data.channels,								# チャネル数
	rate = audio_data.frame_rate,								# サンプリングレート
	output = True												# 出力モードに設定
)

# 楽曲のデータを格納
x_stacked_data_music = np.array([])

# pydubで読み込んだ音楽ファイルを再生する部分のみ関数化する
# 別スレッドで実行するため
def play_music():

  # この関数は別スレッドで実行するため
  # メインスレッドで定義した以下の２つの変数を利用できるように global 宣言する
  global is_gui_running, is_music_running, audio_data, now_playing_sec, x_stacked_data_music, spectrogram_data_music,  spectrum_data_music, spectrum_envelope_music, guide_melody, lemon_data, ani, stream

  # pydubのmake_chunksを用いて音楽ファイルのデータを切り出しながら読み込む
  # 第二引数には何ミリ秒毎に読み込むかを指定
  # ここでは20ミリ秒ごとに読み込む
  size_frame_music = 20	# 20ミリ秒毎に読み込む

  idx = 0

  # make_chunks関数を使用して一定のフレーム毎に音楽ファイルを読み込む
  for chunk in make_chunks(audio_data, size_frame_music):
    
    # GUIが終了してれば，この関数の処理も終了する
    if is_gui_running == False:
      break

    # pyaudioの再生ストリームに切り出した音楽データを流し込む
    # 再生が完了するまで処理はここでブロックされる
    stream_play.write(chunk._data)
    
    # 現在の再生位置を計算（単位は秒）
    now_playing_sec = (idx * size_frame_music) / 1000.
    
    idx += 1

    # データの取得
    data_music = np.array(chunk.get_array_of_samples())

    # 正規化
    data_music = data_music / np.iinfo(np.int16).max	

    #
    # 以下はマイク入力のときと同様
    #

    # 現在のフレームとこれまでに入力されたフレームを連結
    x_stacked_data_music = np.concatenate([x_stacked_data_music, data_music])

    # 抽出するケプストラム係数の次元
    dimension = 13

    # フレームサイズ分のデータがあれば処理を行う
    if len(x_stacked_data_music) >= FRAME_SIZE:
    # フレームサイズからはみ出した過去のデータは捨てる
      x_stacked_data_music = x_stacked_data_music[len(x_stacked_data_music)-FRAME_SIZE:]
      
      # 音量も同様の処理
      vol = 20 * np.log10(np.mean(data_music ** 2) + EPSILON)

      # 
      # 基本周波数の推定
      # 
      # 自己相関から基本周波数を推定
      x_f0 = correlate(x_stacked_data_music)
      nn = hz2nn(x_f0)

      # スペクトルを計算
      fft_spec = np.fft.rfft(x_stacked_data_music * hamming_window)
      fft_log_abs_spec = np.log10(np.abs(fft_spec) + EPSILON)[:-1]
      
      # スペクトル包絡を抽出
      spectrum_data_music = np.log(np.abs(fft_spec) + EPSILON)[:-1]
      fft_log_abs_ceps = np.fft.fft(spectrum_data_music)
      fft_log_abs_ceps[dimension:len(fft_log_abs_ceps)-dimension] = 0
      spectrum_envelope_music = (np.fft.ifft(fft_log_abs_ceps)).real

      # ２次元配列上で列方向（時間軸方向）に１つずらし（戻し）
      # 最後の列（＝最後の時刻のスペクトルがあった位置）に最新のスペクトルデータを挿入
      spectrogram_data_music = np.roll(spectrogram_data_music, -1, axis=1)
      spectrogram_data_music[:, -1] = fft_log_abs_spec[0:(FRAME_SIZE//2)//SCALE_UP_RATE]

  # 楽曲が終了したらフラグをFalseにし，「採点結果を表示」のボタンをアクティブにする
  is_music_running = False
  bt_score["state"] = tkinter.NORMAL
  ani.event_source.stop()
  stream.stop_stream()

# 各テキストの表示を随時更新する関数
def update_gui_text():

  global is_gui_running, is_music_running, now_playing_sec, frequency_zero_min, frequency_zero_max, vibrato_count, shakuri_count, interval_correct_rate

  while True:
    # 歌詞の更新
    if now_playing_sec >= 20.0:
      lyrics_label["text"] = "古びた思い出の埃を払う"
    elif now_playing_sec >= 14.0:
      lyrics_label["text"] = "忘れたものをとりに帰るように"
    elif now_playing_sec >= 8.0:
      lyrics_label["text"] = "未だにあなたのことを夢に見る"
    elif now_playing_sec >=0.0:
      lyrics_label["text"] = "夢ならばどれほど良かったでしょう"
      
    # GUIが表示されているかつ音楽が再生されていれば再生位置（秒），音域，各種歌唱テクニックの回数，音程正解率をテキストとしてGUI上に表示
    if is_gui_running and is_music_running:
      playing_sec_label["text"] = '%.1f sec' % now_playing_sec
      range_label["text"] = "最低音: %s Hz  最高音: %s Hz" % (int(frequency_zero_min), int(frequency_zero_max))
      technique_label["text"] = "ビブラート: %s 回  しゃくり: %s 回" % (vibrato_count, shakuri_count)
      interval_label["text"] = "音程正解率: %s ％" % int(interval_correct_rate)

    # 0.1秒ごとに更新
    time.sleep(0.1)


# 再生時間を表す
now_playing_sec = 0.0

# 音楽を再生するパートを関数化したので，それを別スレッドで（GUIのため）再生開始
t_play_music = threading.Thread(target=play_music)
t_play_music.setDaemon(True)	# GUIが消されたときにこの別スレッドの処理も終了されるようにするため

# 再生時間の表示を随時更新する関数を別スレッドで開始
t_update_gui = threading.Thread(target=update_gui_text)
t_update_gui.setDaemon(True)	# GUIが消されたときにこの別スレッドの処理も終了されるようにするため

# スペクトルとスペクトル包絡の描画を別スレッドで開始
t_update_spectrum_graph = threading.Thread(target=draw_graph)
t_update_spectrum_graph.setDaemon(True)

#
# (4) 全体の処理を実行
#

# GUIの開始フラグをTrueに
is_gui_running = True

# 楽曲の開始フラグをTrueに
is_music_running = True

# 上記で設定したスレッドを開始（直前のフラグを立ててから）
t_play_music.start()
t_update_gui.start()
t_update_spectrum_graph.start()

# GUIを開始，GUIが表示されている間は処理はここでストップ
tkinter.mainloop()

# GUIの開始フラグをFalseに = 音楽再生スレッドのループを終了
is_gui_running = False

# 終了処理
stream_play.stop_stream()
stream_play.close()
p_play.terminate()
