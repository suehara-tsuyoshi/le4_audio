import matplotlib.pyplot as plt
import numpy as np
import math
import librosa

# サンプリング周波数
SR = 16000
# 音声データを取得
x, _ = librosa.load("aiueo.wav", sr = SR)

size_frame = 2048

hamming_window = np.hamming(size_frame)

size_shift = 16000 / 100

def nn2hz(notenum):
  return int(440.0 * (2.0 ** ((notenum - 69) / 12.0)))

def shs_presumption(spectrum):

  f_max = SR // 2
  f_list = np.zeros(f_max + 5)

  for i in range(len(spectrum)):
    f_list[f_max * i // len(spectrum)] = np.abs(spectrum[i])

  nn_list = np.arange(36, 60, 1)
  power_sum_list = np.zeros(len(nn_list))
  # 5倍音の振幅の総和を計算
  for i in range(len(nn_list)):
    hz = nn2hz(nn_list[i])

    power_sum_list[i] = f_list[hz] + f_list[2*hz] + f_list[3*hz] + f_list[4*hz] + f_list[5*hz]
  
  idx = np.argmax(power_sum_list)
  return nn2hz(nn_list[idx])

frequency_zero = []

for i in np.arange(0, len(x)-size_frame, size_shift):
  idx = int(i)
  x_frame = x[idx : idx + size_frame]
  
  fft_spec = np.fft.rfft(x_frame * hamming_window)

  frequency_zero.append(shs_presumption(fft_spec))

plt.xlabel('sample')					# x軸のラベルを設定
plt.ylabel('frequency [Hz]')		# y軸のラベルを設定

plt.plot(frequency_zero)
plt.ylim([0, 300])
plt.show()



