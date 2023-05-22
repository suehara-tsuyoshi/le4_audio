import matplotlib.pyplot as plt
import numpy as np
import math
import librosa


def nn2hz(notenum):
  return 440.0 * (2.0 ** ((notenum - 69) / 12.0))

def hz2nn(frequency):
  return int (round (12.0 * (math.log((frequency + 0.001) / 440.0) / math.log(2.0)))) + 69

def chroma_vector(spectrum, frequencies):
  cv = np.zeros(12)

  for s, f in zip(spectrum, frequencies):
    nn = hz2nn(f)
    cv[nn % 12] += np.abs(s)
  
  return cv

# サンプリング周波数
SR = 16000

# 音声ファイルの読み込み
x, _ = librosa.load("easy_chords.wav", sr = SR)

size_frame = 2048

hamming_window = np.hamming(size_frame)

size_shift = 16000/100

frequency_list = np.zeros(size_frame // 2)

for i in range(len(frequency_list)):
  frequency_list[i] = (SR / size_frame) * i

chroma_list = []

for i in np.arange(0, len(x)-size_frame, size_shift):

  idx = int(i)
  x_frame = x[idx : idx+size_frame]

  fft_spec = np.fft.rfft(x_frame * hamming_window)
  cv_list = chroma_vector(fft_spec, frequency_list)

  chroma_list.append(cv_list)

def chord_presumption(chromagram):
  major_template = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
  minor_template = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
  templates = np.array([np.roll(major_template, i) for i in range(0, 12)] + [np.roll(minor_template, i) for i in range(0, 12)])
  chords = []
  for ch in chromagram:
    idx = np.argmax(np.dot(templates, ch))
    chords.append(idx)
  
  return chords

chord_list = chord_presumption(chroma_list)

# # クロマグラムを描画
# plt.xlabel('time[s]')					# x軸のラベルを設定
# plt.ylabel('chord')		# y軸のラベルを設定
# plt.imshow(
# 	np.flipud(np.array(chroma_list).T),		# 画像とみなすために，データを転地して上下反転
# 	extent=[0, len(x) / SR, 0, 12],			# (横軸の原点の値，横軸の最大値，縦軸の原点の値，縦軸の最大値)
# 	aspect='auto',
# 	interpolation='nearest'
# )
# plt.show()

# スペクトログラムを描画
plt.xlabel('time [s]')		# x軸のラベルを設定
plt.ylabel('chord')				# y軸のラベルを設定
plt.xlim([0, len(x) / SR])					# x軸の範囲を設定
x_data = np.arange(0, len(chord_list)/100 , 0.01)
plt.plot(x_data, chord_list)			# 描画
plt.show()











